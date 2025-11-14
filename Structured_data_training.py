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
from tqdm import trange

# ============================================
# 1. MODEL ARCHITECTURE (UNCHANGED)
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
        x = F.relu(self.embed(x))
        x = self.do(x)
        x = x.transpose(1, 2)
        convs = [F.relu(self.bn[i](self.conv[i](x))) for i in range(3)]
        x = torch.cat(convs, dim=1)
        x = x.transpose(1, 2)
        lstm_out, _ = self.lstm(x)
        return self.cls(lstm_out[:, -1, :])

# ============================================
# 2. DATA GENERATION (15K samples for speed)
# ============================================
def generate_datasets(N=15000, L=100):
    ratios = [0.588, 0.294, 0.118]
    counts = [int(N*r) for r in ratios]
    counts[-1] = N - sum(counts[:-1])
    X = np.zeros((N, L, 4), dtype=np.float32)
    y = np.zeros(N, dtype=np.int64)
    idx = 0
    for label, cnt in enumerate(counts):
        for _ in range(cnt):
            t = np.linspace(0, 4*np.pi, L)
            f = np.random.uniform(0.8, 1.2)
            ph = np.random.uniform(0, 2*np.pi)
            if label == 0:
                base = 80 + 40*np.sin(0.5*f*t+ph)
                noise = np.random.normal(0, 30, L)
                inter = 10+5*np.sin(f*t+ph)+np.random.exponential(2,L)
                ports, prot = [80,443,8080,8443], [0.6,0.4]
            elif label == 1:
                base = 1200+200*np.sin(2*f*t+ph)
                noise = np.random.normal(0,100,L)
                inter = 2+1*np.sin(3*f*t+ph)+np.random.exponential(1,L)
                ports, prot = [21,23,445,3389], [0.9,0.1]
            else:
                base = 300+100*np.sin(4*f*t+ph)
                noise = np.random.normal(0,20,L)
                inter = 15+10*np.sin(2*f*t+ph)+np.random.uniform(0,5,L)
                ports, prot = [1433,3306,5432,27017], [0.5,0.5]
            p = np.clip(base+noise, 40, 1500)
            X[idx,:,0] = p
            X[idx,:,1] = np.clip(inter, 1, 200)
            X[idx,:,2] = np.random.choice(ports, L) % 256
            X[idx,:,3] = np.random.choice([6,17], L, p=prot)
            y[idx] = label
            idx += 1
    X = ((X - X.min())/(X.max()-X.min())*255).astype(np.uint8)
    return TensorDataset(torch.tensor(X, dtype=torch.float32),
                         torch.tensor(y, dtype=torch.long))

# ============================================
# 3. PLOTTING
# ============================================
def save_plot(epochs, accs, losses, tag, out_dir):
    if not epochs:
        return
    fig, (a1,a2) = plt.subplots(1, 2, figsize=(14, 5))
    a1.plot(epochs, accs, 'o-', color='#2ecc71')
    a1.set_title('Val Accuracy'); a1.set_xlabel('Epoch'); a1.grid(alpha=0.3)
    a2.plot(epochs, losses, 's-', color='#e74c3c')
    a2.set_title('Val Loss'); a2.set_xlabel('Epoch'); a2.grid(alpha=0.3)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f'{tag}_{epochs[-1]}.png')
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"📊 Plot saved: {path}")

# ============================================
# 4. TRAINING WITH PROGRESS BAR
# ============================================
def train(ds, tag='Adaptive', epochs=5000, save_every=500, patience=500, device='cpu'):
    tr, va, _ = random_split(ds, [int(len(ds)*0.7), int(len(ds)*0.15), len(ds)-int(len(ds)*0.85)])
    batch_size = 1024
    dl = DataLoader(tr, batch_size=batch_size, shuffle=True, num_workers=0)
    vl = DataLoader(va, batch_size=batch_size*2, shuffle=False, num_workers=0)

    model = CNNBiLSTM().to(device)
    opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=200, min_lr=1e-6)

    best_acc, best_ep = 0.0, 0
    wait = 0
    os.makedirs(f'results_{tag}', exist_ok=True)
    start = time.time()
    epochs_list, acc_list, loss_list = [], [], []

    print(f"🔍 Training {epochs} epochs | batch {batch_size} | device {device.upper()}\n")

    # PROGRESS BAR
    pbar = trange(1, epochs+1, desc='Training', unit='epoch')
    
    for e in pbar:
        # TRAINING
        model.train()
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = crit(out, y)
            loss.backward()
            opt.step()

        # VALIDATION & EARLY STOPPING
        if e % 50 == 0:
            model.eval()
            correct, total, vloss = 0, 0, 0.0
            with torch.no_grad():
                for x, y in vl:
                    x, y = x.to(device), y.to(device)
                    out = model(x)
                    vloss += crit(out, y).item()
                    correct += (out.argmax(1) == y).sum().item()
                    total += y.size(0)
            acc = correct / total
            val_loss = vloss / len(vl)
            sched.step(acc)

            epochs_list.append(e)
            acc_list.append(acc)
            loss_list.append(val_loss)

            # Update progress bar with metrics
            pbar.set_postfix({
                'Acc': f'{acc:.4f}',
                'Loss': f'{val_loss:.4f}',
                'Best': f'{best_acc:.4f}'
            })

            if acc > best_acc:
                best_acc, best_ep = acc, e
                wait = 0
                torch.save(model.state_dict(), f'results_{tag}/best.pt')
            else:
                wait += 50

            if wait >= patience:
                pbar.write(f"⏹ Early stopping at epoch {e}")
                break

        # SAVE PLOTS
        if e % save_every == 0 and epochs_list:
            save_plot(epochs_list, acc_list, loss_list, tag, f'results_{tag}')

    # FINAL PLOT
    if epochs_list:
        save_plot(epochs_list, acc_list, loss_list, tag, f'results_{tag}')

    dur = (time.time() - start) / 3600
    print(f"\n✅ Training complete in {dur:.2f}h | Best Acc={best_acc:.4f} @epoch {best_ep}")

# ============================================
# 5. MAIN
# ============================================
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    ds = generate_datasets(N=15000)
    train(ds, 'Adaptive', epochs=5000, save_every=500, patience=500, device=device)
