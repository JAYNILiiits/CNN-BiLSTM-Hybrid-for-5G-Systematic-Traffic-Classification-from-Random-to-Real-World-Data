import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from collections import deque

# ============================================
# MODEL ARCHITECTURE
# ============================================
class CNNBiLSTM(nn.Module):
    def __init__(self, classes=3):
        super().__init__()
        self.embed = nn.Linear(4, 64)
        self.do = nn.Dropout(0.3)
        self.conv = nn.ModuleList([
            nn.Conv1d(64, 128, k, padding=k//2) for k in (3, 5, 7)
        ])
        self.bn = nn.ModuleList([nn.BatchNorm1d(128) for _ in range(3)])
        self.lstm = nn.LSTM(384, 128, 2, bidirectional=True,
                            batch_first=True, dropout=0.3)
        self.cls = nn.Sequential(
            nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, classes)
        )
    
    def forward(self, x):
        x = self.do(F.relu(self.embed(x)))
        x = x.transpose(1, 2)
        conv_outs = [F.relu(self.bn[i](self.conv[i](x))) for i in range(3)]
        x = torch.cat(conv_outs, dim=1)
        x = x.transpose(1, 2)
        lstm_out, _ = self.lstm(x)
        return self.cls(lstm_out[:, -1, :])


# ============================================
# DATASET GENERATION
# ============================================
def generate_datasets(N=10000, L=100):
    print(f"\n{'='*70}")
    print(f"Generating {N} samples with {L} timesteps...")
    print(f"{'='*70}\n")
    
    X_list, y_list = [], []
    counts = [int(N * 0.588), int(N * 0.294)]
    counts.append(N - counts[0] - counts[1])
    
    for label, cnt in enumerate(counts):
        print(f"Class {label}: Generating {cnt} samples...", flush=True)
        for _ in range(cnt):
            t = np.linspace(0, 4*np.pi, L)
            f = np.random.uniform(0.8, 1.2)
            ph = np.random.uniform(0, 2*np.pi)
            
            if label == 0:
                base = 80 + 40*np.sin(0.5*f*t + ph)
                noise = np.random.normal(0, 30, L)
            elif label == 1:
                base = 1200 + 200*np.sin(2*f*t + ph)
                noise = np.random.normal(0, 100, L)
            else:
                base = 300 + 100*np.sin(4*f*t + ph)
                noise = np.random.normal(0, 20, L)
            
            p = (base + noise).clip(40,1500)
            inter = (10+5*np.sin(f*t+ph)+np.random.normal(0,2,L)).clip(1,200)
            port = np.random.choice([1883,443,5000], L) % 256
            pr = np.random.choice([17,6], L, p=[0.8,0.2])
            
            feat = np.stack([p, inter, port, pr], axis=1)
            X_list.append(feat)
            y_list.append(label)
    
    X = np.stack(X_list)
    y = np.array(y_list)
    
    print("\nApplying global normalization...")
    ch_min = X.min(axis=(0,1), keepdims=True)
    ch_max = X.max(axis=(0,1), keepdims=True)
    X_norm = ((X - ch_min) / (ch_max - ch_min) * 255).astype(np.uint8)
    
    print(f"Dataset shape: {X_norm.shape}")
    print(f"Min: {X_norm.min()}, Max: {X_norm.max()}")
    print(f"Class distribution: {counts}")
    print(f"{'='*70}\n")
    
    X_t = torch.tensor(X_norm, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)
    
    return TensorDataset(X_t, y_t)


# ============================================
# PLOTTING
# ============================================
def save_plot(epochs, accs, losses, tag, out_dir):
    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(14,5))
    fig.suptitle(f'{tag} Training Progress', fontsize=16, fontweight='bold')
    
    ax1.plot(epochs, accs, 'o-', color='#2ecc71', linewidth=2, markersize=4)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('Validation Accuracy')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(epochs, losses,'s-', color='#e74c3c', linewidth=2, markersize=4)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title('Validation Loss')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f'{tag}_progress.png')
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"📊 Plot saved: {path}")


# ============================================
# TRAINING
# ============================================
def train(ds, tag, epochs=500_000, save_every=5000, device='cpu'):
    print(f"\n{'='*70}")
    print(f"TRAINING: {tag}")
    print(f"{'='*70}")
    
    tr, va, _ = random_split(ds, [int(len(ds)*0.7), int(len(ds)*0.15), 
                                   len(ds)-int(len(ds)*0.85)])
    
    dl = DataLoader(tr, batch_size=256, shuffle=True, pin_memory=True)
    vl = DataLoader(va, batch_size=256, pin_memory=True)
    
    print(f"Train: {len(tr)} | Val: {len(va)} | Test: {len(ds)-len(tr)-len(va)}")
    print(f"Device: {device.upper()}")
    print(f"Target epochs: {epochs:,} | Save interval: {save_every:,}")
    print(f"{'='*70}\n")
    
    model = CNNBiLSTM().to(device)
    opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    
    epochs_list, acc_list, loss_list = [], [], []
    times = deque(maxlen=100)
    start = time.time()
    
    print("Starting training...\n")
    
    for e in range(1, epochs+1):
        t0 = time.time()
        
        # Training
        model.train()
        for x,y in dl:
            x,y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
        
        times.append(time.time()-t0)
        
        # Print progress dots every 10 epochs
        if e % 10 == 0:
            print('.', end='', flush=True)
        
        # Newline every 100 epochs
        if e % 100 == 0:
            print(f' [{e}]', flush=True)
        
        # Validation every 500 epochs
        if e % 500 == 0:
            model.eval()
            correct, total, vloss = 0, 0, 0.0
            with torch.no_grad():
                for x,y in vl:
                    x,y = x.to(device), y.to(device)
                    out = model(x)
                    vloss += crit(out,y).item()
                    pred = out.argmax(1)
                    correct += (pred==y).sum().item()
                    total += y.size(0)
            
            acc = correct/total
            loss_avg = vloss/len(vl)
            epochs_list.append(e)
            acc_list.append(acc)
            loss_list.append(loss_avg)
        
        # Detailed log every 1000 epochs
        if e % 1000 == 0:
            elapsed = time.time() - start
            eps = np.mean(times)
            eta = eps * (epochs - e) / 3600
            
            print(f"\n{'='*70}")
            print(f"Epoch {e:,}/{epochs:,} ({100*e/epochs:.2f}%)")
            print(f"Accuracy: {acc:.4f} | Loss: {loss_avg:.4f}")
            print(f"Speed: {eps:.3f}s/epoch | Elapsed: {elapsed/3600:.2f}h | ETA: {eta:.1f}h")
            print(f"{'='*70}\n")
        
        # Save plot
        if e % save_every == 0 and len(epochs_list) > 0:
            save_plot(epochs_list, acc_list, loss_list, tag, f'results_{tag}')
    
    # Final plot
    if len(epochs_list) > 0:
        save_plot(epochs_list, acc_list, loss_list, tag, f'results_{tag}')
    
    total_time = time.time() - start
    print(f"\n✅ {tag} training complete in {total_time/3600:.2f} hours\n")
    
    return epochs_list, acc_list, loss_list


# ============================================
# MAIN
# ============================================
if __name__ == '__main__':
    print("\n" + "="*70)
    print("HARDWARE DETECTION")
    print("="*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device.upper()}")
    
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB")
    
    print("="*70)
    
    # Generate dataset
    ds = generate_datasets(N=10000, L=100)
    
    # Train
    train(ds, 'Adaptive', epochs=500_000, save_every=5000, device=device)
