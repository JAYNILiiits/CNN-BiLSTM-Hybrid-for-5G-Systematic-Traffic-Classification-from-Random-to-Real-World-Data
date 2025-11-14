from flask import Flask, render_template_string, request, jsonify
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os, io, base64, threading
from datetime import datetime
import json, warnings
warnings.filterwarnings('ignore')


# ============================================================================
# MAXIMUM CPU OPTIMIZATION
# ============================================================================
torch.set_num_threads(os.cpu_count())
torch.set_num_interop_threads(os.cpu_count())


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads'
app.config['RESULTS_FOLDER'] = './training_results'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)


torch.manual_seed(42)
np.random.seed(42)


uploaded_files = {}
model_state = {'trained': False, 'training': False, 'progress': 0, 'status': 'idle', 'history': None, 'error': None, 'label_map': None, 'save_path': None, 'current_phase': 'Idle', 'phase_progress': 0, 'eta': 'N/A', 'duration': '0m'}


CONFIG = {
    'sequence_length': 10000,
    'num_epochs':500,
    'batch_size': 32,
    'learning_rate': 0.001,
    'weight_decay': 0.0005,
    'gradient_accumulation_steps': 2,
    'dropout_rate': 0.35,
    'label_smoothing': 0.05,
    'mixed_precision': False,
    'validation_frequency': 1,
}


print("\n" + "="*80)
print("MAXIMUM SPEED OPTIMIZED - AGGRESSIVE POOLING + TORCH.COMPILE")
print("="*80)
print(f"✓ Sequence Length: {CONFIG['sequence_length']:,}")
print(f"✓ LSTM Input Reduced: 400K → 2.5K (40x reduction)")
print(f"✓ Validation: Every {CONFIG['validation_frequency']} epochs")
print(f"✓ CPU Threads: {torch.get_num_threads()}")
print("="*80 + "\n")


# ============================================================================
# ATTENTION LAYER CLASS
# ============================================================================
class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )
    
    def forward(self, lstm_output):
        attention_weights = self.attention(lstm_output)
        attention_weights = F.softmax(attention_weights, dim=1)
        weighted = torch.sum(attention_weights * lstm_output, dim=1)
        return weighted


# ============================================================================
# FOCAL LOSS CLASS
# ============================================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.alpha is not None:
            focal_loss = self.alpha[targets] * focal_loss
        return focal_loss.mean()


# ============================================================================
# ULTRA-OPTIMIZED ARCHITECTURE
# ============================================================================
class UltraOptimizedPacketClassifier(nn.Module):
    def __init__(self, n_classes=6):
        super(UltraOptimizedPacketClassifier, self).__init__()
        
        # ---- CONV PATHS ----
        self.conv_path1 = nn.Sequential(
            nn.Conv1d(1, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(96),
            nn.ReLU(inplace=True),
            nn.Dropout(CONFIG['dropout_rate']),
            nn.MaxPool1d(2)
        )
        
        self.conv_path2 = nn.Sequential(
            nn.Conv1d(1, 96, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(96),
            nn.ReLU(inplace=True),
            nn.Dropout(CONFIG['dropout_rate']),
            nn.MaxPool1d(2)
        )
        
        self.conv_path3 = nn.Sequential(
            nn.Conv1d(1, 96, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(96),
            nn.ReLU(inplace=True),
            nn.Dropout(CONFIG['dropout_rate']),
            nn.MaxPool1d(2)
        )
        
        # ---- COMBINATION + POOLING ----
        self.conv_combine = nn.Sequential(
            nn.Conv1d(288, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(CONFIG['dropout_rate']),
            nn.MaxPool1d(2)
        )
        
        # ---- MODIFIED AGGRESSIVE POOLING ----
        self.aggressive_pool = nn.Sequential(
            nn.MaxPool1d(4),
            nn.Conv1d(256, 384, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(384),
            nn.ReLU(inplace=True),
            nn.Dropout(CONFIG['dropout_rate']),
            nn.MaxPool1d(5),
            nn.Conv1d(384, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(CONFIG['dropout_rate']),
            nn.MaxPool1d(2)
        )
        
        # ---- bi LSTM ----
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        
        # ---- ATTENTION ----
        self.attention = AttentionLayer(256)
        
        # ---- CLASSIFIER ----
        self.classifier = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256, 512),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.LayerNorm(512),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.LayerNorm(128),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, n_classes)
        )
    
    def forward(self, x):
        x = x.unsqueeze(1)
        
        # Parallel convolution
        c1 = self.conv_path1(x)
        c2 = self.conv_path2(x)
        c3 = self.conv_path3(x)
        
        # Concatenate and combine
        x = torch.cat([c1, c2, c3], dim=1).contiguous()
        x = self.conv_combine(x)
        
        # Aggressive pooling
        x = self.aggressive_pool(x)
        
        # LSTM processing
        x = x.transpose(1, 2).contiguous()
        lstm_out, _ = self.lstm(x)
        
        # Attention mechanism
        context = self.attention(lstm_out)
        
        return self.classifier(context)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def update_progress(phase, overall_progress, phase_progress, status, eta="N/A", duration="0m"):
    model_state.update({'progress': overall_progress, 'status': status, 'current_phase': phase, 'phase_progress': phase_progress, 'eta': eta, 'duration': duration})


def load_data(file_path, label):
    """Optimized data loading with data augmentation"""
    try:
        df = pd.read_csv(file_path)
        if 'Length' in df.columns:
            lengths = pd.to_numeric(df['Length'], errors='coerce')
        else:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                return [], []
            lengths = df[numeric_cols[0]]
        
        lengths = lengths.dropna().values
        
        q1, q3 = np.percentile(lengths, [25, 75])
        iqr = q3 - q1
        mask = (lengths >= q1 - 1.5*iqr) & (lengths <= q3 + 1.5*iqr)
        lengths = lengths[mask]
        
        lengths = np.clip(lengths, 0, 2000).astype(np.float32)
        
        if len(lengths) < CONFIG['sequence_length']:
            return [], []
        
        sequences = []
        labels = []
        
        # Calculate step size for overlapping windows
        step = CONFIG['sequence_length'] // 2  # 50% overlap
        
        for i in range(0, len(lengths) - CONFIG['sequence_length'] + 1, step):
            seq = lengths[i:i + CONFIG['sequence_length']]
            if len(seq) == CONFIG['sequence_length']:
                # Add original sequence
                sequences.append(seq)
                labels.append(label)
                                
        print(f"  → Generated {len(sequences)} sequences from {file_path}")
        return sequences, labels
        
        
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return [], []


def save_results(save_dir, model, history, label_map, cm, report):
    """Save results"""
    torch.save(model.state_dict(), os.path.join(save_dir, 'model_weights.pth'))
    
    with open(os.path.join(save_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=4)
    
    with open(os.path.join(save_dir, 'label_map.json'), 'w') as f:
        json.dump({str(k): v for k, v in label_map.items()}, f, indent=4)
    
    cm_df = pd.DataFrame(cm, index=list(label_map.values()), columns=list(label_map.values()))
    cm_df.to_csv(os.path.join(save_dir, 'confusion_matrix.csv'))
    
    with open(os.path.join(save_dir, 'classification_report.json'), 'w') as f:
        json.dump(report, f, indent=4)
    
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(CONFIG, f, indent=4)
    
    # Generate plots
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    epochs = range(1, len(history['loss']) + 1)
    
    axes[0, 0].plot(epochs, history['loss'], 'b-', lw=2)
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', lw=2)
    axes[0, 0].set_title('Loss', fontweight='bold')
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].legend(['Train', 'Val'])
    
    axes[0, 1].plot(epochs, history['acc'], 'g-', lw=2)
    axes[0, 1].plot(epochs, history['val_acc'], 'orange', lw=2)
    axes[0, 1].set_title('Accuracy', fontweight='bold')
    axes[0, 1].set_ylim([0, 101])
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].legend(['Train', 'Val'])
    
    axes[0, 2].semilogy(epochs, history['loss'], 'b-', lw=2)
    axes[0, 2].semilogy(epochs, history['val_loss'], 'r-', lw=2)
    axes[0, 2].set_title('Loss (Log)', fontweight='bold')
    axes[0, 2].grid(alpha=0.3, which='both')
    axes[0, 2].legend(['Train', 'Val'])
    
    axes[1, 0].plot(epochs, history['val_acc'], 'purple', lw=3)
    axes[1, 0].fill_between(epochs, history['val_acc'], alpha=0.2, color='purple')
    axes[1, 0].set_title('Val Accuracy', fontweight='bold')
    axes[1, 0].set_ylim([0, 101])
    axes[1, 0].grid(alpha=0.3)
    
    axes[1, 1].semilogy(history['lr'], 'navy', lw=2)
    axes[1, 1].set_title('Learning Rate', fontweight='bold')
    axes[1, 1].grid(alpha=0.3, which='both')
    
    axes[1, 2].plot(history['grad_norm'], 'darkred', lw=2)
    axes[1, 2].set_title('Gradient Norm', fontweight='bold')
    axes[1, 2].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=list(label_map.values()),
                yticklabels=list(label_map.values()),
                cbar_kws={'label': 'Count'}, ax=ax,
                annot_kws={"size": 12}, linewidths=0.5)
    ax.set_title('Confusion Matrix', fontweight='bold', fontsize=15)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()


def train_model_background():
    """ULTRA-OPTIMIZED TRAINING"""
    start_time = datetime.now()
    try:
        update_progress("Init", 5, 10, "Initializing...")
        print(f"\n✓ Device: CPU")
        print(f"✓ CPU Threads: {torch.get_num_threads()}\n")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_dir = os.path.join(app.config['RESULTS_FOLDER'], f'training_{timestamp}')
        os.makedirs(save_dir, exist_ok=True)
        model_state['save_path'] = save_dir
        
        if len(uploaded_files) < 2:
            raise Exception("Need 2+ categories")
        
        # ========== DATA LOADING ==========
        all_sequences = []
        all_labels = []
        label_map = {}
        
        for cat_idx, (cat, files) in enumerate(uploaded_files.items()):
            label_map[cat_idx] = cat
            for f in files:
                seqs, labs = load_data(f['path'], cat_idx)
                if seqs:
                    all_sequences.extend(seqs)
                    all_labels.extend(labs)
        
        if not all_sequences:
            raise Exception("No sequences created")
        
        print(f"✓ Loaded {len(all_sequences)} sequences")
        
        # ========== PREPROCESSING ==========
        update_progress("Preprocessing", 40, 50, "Scaling...")
        
        X = np.array(all_sequences, dtype=np.float32)
        X_flat = X.reshape(-1, 1)
        scaler = RobustScaler(quantile_range=(5.0, 95.0))
        scaler.fit(X_flat)
        X_scaled = scaler.transform(X_flat).reshape(X.shape).astype(np.float32)
        del X_flat, X
        
        y = np.array(all_labels, dtype=np.int64)
        
        update_progress("Preprocessing", 50, 100, "Split...")
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=0.15, random_state=42, stratify=y
        )
        del X_scaled
        
        print(f"✓ Train: {len(X_train)}, Val: {len(X_val)}")
        
        # ========== MODEL SETUP ==========
        update_progress("Training", 55, 0, "Initializing model...")
        device = torch.device('cpu')
        
        train_ds = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
        val_ds = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
        
        train_loader = DataLoader(
            train_ds, batch_size=CONFIG['batch_size'], shuffle=True,
            num_workers=4, pin_memory=False, persistent_workers=True, prefetch_factor=2
        )
        
        val_loader = DataLoader(
            val_ds, batch_size=CONFIG['batch_size'],
            num_workers=2, pin_memory=False, persistent_workers=True
        )
        
        model = UltraOptimizedPacketClassifier(n_classes=len(label_map)).to(device)
        
        # ========== TORCH.COMPILE FOR PYTORCH 2.0+ ==========
        try:
            if hasattr(torch, 'compile'):
                print("✓ Using torch.compile() for additional speedup...")
                compile_mode = 'default' if str(device) == 'cpu' else 'max-autotune'
                model = torch.compile(model, mode=compile_mode)
                print(f"✓ Model compiled with mode: {compile_mode}!")
        except Exception as e:
            print(f"✓ torch.compile failed ({e}), using regular model")
        
        # Class weighting
        class_counts = np.bincount(y_train)
        total_samples = len(y_train)
        
        # Calculate balanced class weights
        class_weights = total_samples / (len(class_counts) * class_counts)
        
        # apply clipping to prevent extra weights
        class_weights = np.clip(class_weights, 0.5, 2.0)
        
        # Normalize weights
        class_weights = class_weights / class_weights.sum() * len(class_counts)
        
        # Convert to tensor
        class_weights = torch.FloatTensor(class_weights).to(device)
        
        print(f"✓ Class weights (smoothed): {class_weights.cpu().numpy()}")
        
        criterion = FocalLoss(alpha=class_weights, gamma=2.0, 
                            label_smoothing=CONFIG['label_smoothing'])
        
        optimizer = optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'],
                               weight_decay=CONFIG['weight_decay'])
        
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=CONFIG['learning_rate'] * 10,
            epochs=CONFIG['num_epochs'],
            steps_per_epoch=len(train_loader),
            pct_start=0.1,
            anneal_strategy='cos',
            final_div_factor=1000
        )
        
        history = {'loss': [], 'acc': [], 'val_loss': [], 'val_acc': [], 'lr': [], 'grad_norm': []}
        
        # Early stopping variables
        best_val_acc = 0
        best_model_state = None
        patience_counter = 0
        patience_limit = 75
        min_delta = 0.0005
        
        print(f"✓ Training started (validation every {CONFIG['validation_frequency']} epochs)...\n")
        
        # ========== OPTIMIZED TRAINING LOOP ==========
        for ep in range(1, CONFIG['num_epochs'] + 1):
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            total_grad_norm = 0.0
            
            for batch_idx, (x, y_true) in enumerate(train_loader):
                x = x.to(device, non_blocking=True)
                y_true = y_true.to(device, non_blocking=True)
                
                out = model(x)
                loss = criterion(out, y_true) / CONFIG['gradient_accumulation_steps']
                loss.backward()
                
                if (batch_idx + 1) % CONFIG['gradient_accumulation_steps'] == 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    total_grad_norm += grad_norm.item()
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                
                train_loss += loss.item() * CONFIG['gradient_accumulation_steps']
                with torch.no_grad():
                    train_correct += (out.argmax(1) == y_true).sum().item()
                    train_total += len(y_true)
                
                del x, y_true, out, loss
            
            train_loss /= len(train_loader)
            train_acc = 100.0 * train_correct / train_total
            
            # ========== VALIDATE ONLY EVERY N EPOCHS ==========
            val_loss = 0.0
            val_acc = 0.0
            
            if ep % CONFIG['validation_frequency'] == 0 or ep == 1:
                model.eval()
                val_correct = 0
                val_total = 0
                all_preds = []
                all_targets = []
                
                with torch.inference_mode():
                    for x, y_true in val_loader:
                        x = x.to(device, non_blocking=True)
                        y_true = y_true.to(device, non_blocking=True)
                        
                        out = model(x)
                        loss = criterion(out, y_true)
                        val_loss += loss.item()
                        
                        preds = out.argmax(1)
                        val_correct += (preds == y_true).sum().item()
                        val_total += len(y_true)
                        
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(y_true.cpu().numpy())
                        
                        del x, y_true, out, loss, preds
                
                val_loss /= len(val_loader)
                val_acc = 100.0 * val_correct / val_total
                
                # Early stopping logic
                if val_acc > (best_val_acc + min_delta):
                    best_val_acc = val_acc
                    best_model_state = model.state_dict().copy() if not hasattr(model, '_orig_mod') else model._orig_mod.state_dict().copy()
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                # Early stopping check
                if patience_counter >= patience_limit:
                    print(f"\n⚠️  Early stopping triggered at epoch {ep}")
                    print(f"    No improvement for {patience_limit} epochs")
                    print(f"    Best validation accuracy: {best_val_acc:.4f}%")
                    break
            
            else:
                # Use last validation values
                if history['val_loss']:
                    val_loss = history['val_loss'][-1]
                    val_acc = history['val_acc'][-1]
            
            scheduler.step()
            
            # Record history
            history['loss'].append(train_loss)
            history['acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['lr'].append(optimizer.param_groups[0]['lr'])
            history['grad_norm'].append(total_grad_norm / max(len(train_loader) // CONFIG['gradient_accumulation_steps'], 1))
            
            # Progress update every 5 epochs
            if ep % 5 == 0 or ep == 1:
                elapsed = (datetime.now() - start_time).total_seconds()
                duration_str = f"{int(elapsed//60)}m {int(elapsed%60)}s"
                eta_secs = (elapsed / ep) * (CONFIG['num_epochs'] - ep)
                eta_str = f"{int(eta_secs/60)}m {int(eta_secs%60)}s"
                progress = 55 + int((ep / CONFIG['num_epochs']) * 35)
                update_progress("Training", progress, int((ep / CONFIG['num_epochs']) * 100), 
                               f"Epoch {ep} | Train: {train_acc:.2f}% | Val: {val_acc:.2f}%", 
                               eta_str, duration_str)
        
        # ========== FINAL EVALUATION ==========
        if best_model_state is not None:
            if hasattr(model, '_orig_mod'):
                model._orig_mod.load_state_dict(best_model_state)
            else:
                model.load_state_dict(best_model_state)
        
        update_progress("Saving", 90, 50, "Saving...")
        
        model.eval()
        all_preds = []
        all_targets = []
        
        with torch.inference_mode():
            for x, y_true in val_loader:
                x, y_true = x.to(device), y_true.to(device)
                out = model(x)
                preds = out.argmax(1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(y_true.cpu().numpy())
        
        cm = confusion_matrix(all_targets, all_preds)
        report = classification_report(all_targets, all_preds, target_names=list(label_map.values()), 
                                      output_dict=True, zero_division=0)
        
        # Save original model if compiled
        save_model = model._orig_mod if hasattr(model, '_orig_mod') else model
        save_results(save_dir, save_model, history, label_map, cm, report)
        
        final_val_acc = max(history['val_acc'])
        elapsed = (datetime.now() - start_time).total_seconds()
        duration_str = f"{int(elapsed//60)}m {int(elapsed%60)}s"
        
        model_state.update({
            'trained': True,
            'training': False,
            'progress': 100,
            'status': f'Complete | Accuracy: {final_val_acc:.2f}% | Duration: {duration_str}',
            'history': history,
            'label_map': label_map,
            'confusion_matrix': cm.tolist(),
            'classification_report': report,
            'duration': duration_str
        })
        
        print(f"\n✓ Training complete!")
        print(f"✓ Best Val Accuracy: {final_val_acc:.2f}%")
        print(f"✓ Duration: {duration_str}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        model_state.update({'training': False, 'status': f'Error: {str(e)}', 'error': str(e)})



# ============================================================================
# FLASK ROUTES
# ============================================================================
HTML = """<!DOCTYPE html>
<html>
<head>
    <title>ULTRA-OPTIMIZED Packet Classifier</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', Arial; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; }
        .container { max-width: 1200px; margin: 0 auto; background: white; border-radius: 15px; padding: 30px; box-shadow: 0 25px 80px rgba(0,0,0,0.4); }
        h1 { color: #667eea; text-align: center; margin-bottom: 10px; font-size: 2.4em; }
        .subtitle { text-align: center; color: #666; margin-bottom: 20px; font-size: 0.95em; }
        .warning { background: #d4edda; border: 2px solid #28a745; padding: 15px; border-radius: 8px; margin: 15px 0; font-size: 0.9em; }
        .tab { display: inline-block; padding: 12px 28px; cursor: pointer; background: #f0f0f0; margin: 5px; border-radius: 8px; font-weight: 600; border: none; transition: all 0.3s; }
        .tab:hover { background: #e0e0e0; }
        .tab.active { background: #667eea; color: white; }
        .tab-content { display: none; padding: 20px 0; }
        .tab-content.active { display: block; }
        input, button { padding: 11px; margin: 5px; border-radius: 8px; border: 2px solid #ddd; font-size: 0.95em; }
        button { background: #667eea; color: white; border: none; cursor: pointer; font-weight: 600; }
        button:hover { background: #5568d3; }
        button:disabled { background: #ccc; }
        .category { background: #f8f9fa; padding: 18px; margin: 12px 0; border-radius: 10px; border: 2px solid #e0e0e0; }
        .alert { padding: 13px; margin: 12px 0; border-radius: 8px; border-left: 4px solid; }
        .alert-info { background: #d1ecf1; color: #0c5460; border-color: #0c5460; }
        .alert-success { background: #d4edda; color: #155724; border-color: #155724; }
        .stats { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; margin: 18px 0; }
        .stat { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 18px; text-align: center; border-radius: 10px; }
        .stat-value { font-size: 2em; font-weight: bold; }
        .stat-label { font-size: 0.9em; margin-top: 5px; opacity: 0.9; }
        .progress { width: 100%; height: 28px; background: #e9ecef; border-radius: 15px; overflow: hidden; margin: 18px 0; }
        .progress-bar { height: 100%; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); text-align: center; color: white; line-height: 28px; font-weight: 600; transition: width 0.3s; }
        .phase-info { background: #f0f7ff; padding: 12px; margin: 10px 0; border-radius: 8px; border-left: 4px solid #0066cc; }
        img { max-width: 100%; margin: 18px 0; border-radius: 10px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>⚡ ULTRA-OPTIMIZED Packet Classifier</h1>
        <div class="subtitle">MAXIMUM SPEED | LSTM Input: 400K→2.5K (40x reduction) | torch.compile</div>
        <div class="warning">
            ⚡ <strong>EXTREME OPTIMIZATIONS:</strong> Aggressive pooling (40x reduction) | Validation every 5 epochs | torch.compile | 2-layer BiLSTM + Attention
        </div>
        
        <div>
            <button class="tab active" onclick="switchTab(0)">📁 Upload</button>
            <button class="tab" onclick="switchTab(1)">🏋️ Train</button>
            <button class="tab" onclick="switchTab(2)">📊 Results</button>
        </div>
        
        <div class="tab-content active" id="tab0">
            <div style="display: flex; gap: 10px; margin-top: 15px;">
                <input type="text" id="cat" placeholder="Category name (e.g., TCP, UDP)" style="flex: 1;">
                <button onclick="add()">Add Category</button>
            </div>
            <div id="cats"></div>
        </div>
        
        <div class="tab-content" id="tab1">
            <div id="info"></div>
            <button onclick="train()" id="btn" style="width: 100%; font-size: 1.2em; padding: 16px;">Start Training</button>
            <div id="prog" style="display: none;">
                <div class="phase-info">
                    <strong>Phase:</strong> <span id="phase">Idle</span> | <strong>Progress:</strong> <span id="phaseprog">0%</span> | <strong>Duration:</strong> <span id="duration">0m</span> | <strong>ETA:</strong> <span id="eta">N/A</span>
                </div>
                <div class="progress"><div class="progress-bar" id="bar" style="width: 0%;">0%</div></div>
                <p id="stat" style="text-align: center; margin-top: 10px; font-weight: 600; color: #667eea;">Starting...</p>
            </div>
        </div>
        
        <div class="tab-content" id="tab2">
            <div id="res"></div>
        </div>
    </div>
    
    <script>
        let cats = {}, int = null;
        
        function switchTab(i) {
            document.querySelectorAll('.tab').forEach((t, idx) => t.classList.toggle('active', idx === i));
            document.querySelectorAll('.tab-content').forEach((c, idx) => c.classList.toggle('active', idx === i));
            if (i === 1) updateInfo();
            if (i === 2) loadRes();
        }
        
        function add() {
            const n = document.getElementById('cat').value.trim();
            if (!n || cats[n]) { alert('Invalid or duplicate!'); return; }
            cats[n] = [];
            document.getElementById('cat').value = '';
            render();
        }
        
        function render() {
            document.getElementById('cats').innerHTML = Object.keys(cats).map(c => 
                `<div class="category">
                    <strong>📁 ${c}</strong> <span>(${cats[c].length} files)</span>
                    <input type="file" multiple accept=".csv" id="f-${c}" style="display:none" onchange="up('${c}')">
                    <button onclick="document.getElementById('f-${c}').click()">Upload Files</button>
                    <button onclick="del('${c}')" style="background:#ff6b6b;">Delete</button>
                </div>`
            ).join('');
        }
        
        function up(c) {
            const fd = new FormData();
            fd.append('category', c);
            for (let f of document.getElementById('f-' + c).files) fd.append('files', f);
            fetch('/api/upload', {method: 'POST', body: fd})
            .then(() => fetch('/api/categories').then(r => r.json()).then(d => { cats = d; render(); }));
        }
        
        function del(c) {
            fetch('/api/remove_category', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({category: c})})
            .then(() => { delete cats[c]; render(); });
        }
        
        function updateInfo() {
            const catCount = Object.keys(cats).length;
            const fileCount = Object.values(cats).reduce((s, f) => s + f.length, 0);
            document.getElementById('info').innerHTML = catCount < 2 ? 
                '<div class="alert alert-info">⚠️ Need 2+ categories</div>' :
                `<div class="alert alert-success">✓ Ready (${catCount} categories, ${fileCount} files)</div>
                <div class="stats">
                    <div class="stat"><div class="stat-value">${catCount}</div><div class="stat-label">Categories</div></div>
                    <div class="stat"><div class="stat-value">${fileCount}</div><div class="stat-label">Files</div></div>
                    <div class="stat"><div class="stat-value">40x</div><div class="stat-label">Speedup</div></div>
                </div>`;
        }
        
        function train() {
            if (Object.keys(cats).length < 2) { alert('Need 2+ categories!'); return; }
            document.getElementById('btn').disabled = true;
            document.getElementById('prog').style.display = 'block';
            fetch('/api/train', {method: 'POST'});
            int = setInterval(check, 2000);
        }
        
        function check() {
            fetch('/api/progress').then(r => r.json()).then(d => {
                document.getElementById('bar').style.width = d.progress + '%';
                document.getElementById('bar').textContent = d.progress + '%';
                document.getElementById('stat').textContent = d.status;
                document.getElementById('phase').textContent = d.current_phase;
                document.getElementById('phaseprog').textContent = d.phase_progress + '%';
                document.getElementById('eta').textContent = d.eta;
                document.getElementById('duration').textContent = d.duration;
                
                if (!d.training && d.progress === 100) {
                    clearInterval(int);
                    alert('✓ Complete!');
                    document.getElementById('btn').disabled = false;
                    switchTab(2);
                }
            });
        }
        
        function loadRes() {
            fetch('/api/results').then(r => r.json()).then(d => {
                document.getElementById('res').innerHTML = !d.trained ? 
                    '<div class="alert alert-info">Train first</div>' :
                    `<div class="alert alert-success">✓ Complete</div>
                    <div class="stats">
                        <div class="stat"><div class="stat-value">${d.train_acc.toFixed(2)}%</div><div class="stat-label">Train</div></div>
                        <div class="stat"><div class="stat-value">${d.val_acc.toFixed(2)}%</div><div class="stat-label">Val</div></div>
                        <div class="stat"><div class="stat-value">${d.epochs}</div><div class="stat-label">Epochs</div></div>
                    </div>
                    ${d.plots.map(p => `<img src="data:image/png;base64,${p}">`).join('')}
                    <p style="margin-top:15px;"><strong>📂 Saved:</strong> ${d.save_path}</p>`;
            });
        }
        
        fetch('/api/categories').then(r => r.json()).then(d => {cats = d; render();});
    </script>
</body>
</html>"""



@app.route('/')
def index():
    return render_template_string(HTML)



@app.route('/api/upload', methods=['POST'])
def upload():
    cat = request.form.get('category')
    if cat not in uploaded_files:
        uploaded_files[cat] = []
    for f in request.files.getlist('files'):
        if f.filename.endswith('.csv'):
            path = os.path.join(app.config['UPLOAD_FOLDER'], f"{cat}_{f.filename}")
            f.save(path)
            uploaded_files[cat].append({'name': f.filename, 'path': path})
    return jsonify({'success': True})



@app.route('/api/categories')
def categories():
    return jsonify(uploaded_files)



@app.route('/api/remove_category', methods=['POST'])
def remove_cat():
    cat = request.json.get('category')
    if cat in uploaded_files:
        del uploaded_files[cat]
    return jsonify({'success': True})



@app.route('/api/train', methods=['POST'])
def train():
    model_state['training'] = True
    threading.Thread(target=train_model_background, daemon=True).start()
    return jsonify({'success': True})



@app.route('/api/progress')
def progress():
    return jsonify({
        'progress': model_state['progress'],
        'status': model_state['status'],
        'current_phase': model_state.get('current_phase', 'Idle'),
        'phase_progress': model_state.get('phase_progress', 0),
        'eta': model_state.get('eta', 'N/A'),
        'duration': model_state.get('duration', '0m'),
        'training': model_state['training'],
        'error': model_state['error']
    })



@app.route('/api/results')
def results():
    if not model_state['trained']:
        return jsonify({'trained': False})
    
    hist = model_state['history']
    label_map = model_state.get('label_map', {})
    cm = model_state.get('confusion_matrix')
    save_path = model_state.get('save_path')
    
    plots = []
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    epochs = range(1, len(hist['loss']) + 1)
    
    axes[0, 0].plot(epochs, hist['loss'], 'b-', lw=2)
    axes[0, 0].plot(epochs, hist['val_loss'], 'r-', lw=2)
    axes[0, 0].set_title('Loss', fontweight='bold')
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].legend(['Train', 'Val'])
    
    axes[0, 1].plot(epochs, hist['acc'], 'g-', lw=2)
    axes[0, 1].plot(epochs, hist['val_acc'], 'orange', lw=2)
    axes[0, 1].set_title('Accuracy', fontweight='bold')
    axes[0, 1].set_ylim([0, 101])
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].legend(['Train', 'Val'])
    
    axes[0, 2].semilogy(epochs, hist['loss'], 'b-', lw=2)
    axes[0, 2].semilogy(epochs, hist['val_loss'], 'r-', lw=2)
    axes[0, 2].set_title('Loss (Log)', fontweight='bold')
    axes[0, 2].grid(alpha=0.3, which='both')
    axes[0, 2].legend(['Train', 'Val'])
    
    axes[1, 0].plot(epochs, hist['val_acc'], 'purple', lw=3)
    axes[1, 0].fill_between(epochs, hist['val_acc'], alpha=0.2, color='purple')
    axes[1, 0].set_title('Val Accuracy', fontweight='bold')
    axes[1, 0].set_ylim([0, 101])
    axes[1, 0].grid(alpha=0.3)
    
    axes[1, 1].semilogy(hist['lr'], 'navy', lw=2)
    axes[1, 1].set_title('Learning Rate', fontweight='bold')
    axes[1, 1].grid(alpha=0.3, which='both')
    
    axes[1, 2].plot(hist['grad_norm'], 'darkred', lw=2)
    axes[1, 2].set_title('Gradient Norm', fontweight='bold')
    axes[1, 2].grid(alpha=0.3)
    
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150)
    buf.seek(0)
    plots.append(base64.b64encode(buf.read()).decode('utf-8'))
    plt.close()
    
    if cm:
        fig, ax = plt.subplots(figsize=(14, 12))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=list(label_map.values()),
                   yticklabels=list(label_map.values()),
                   cbar_kws={'label': 'Count'}, ax=ax,
                   annot_kws={"size": 12}, linewidths=0.5)
        ax.set_title('Confusion Matrix', fontweight='bold', fontsize=15)
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150)
        buf.seek(0)
        plots.append(base64.b64encode(buf.read()).decode('utf-8'))
        plt.close()
    
    best_val_acc = max(hist['val_acc']) if hist['val_acc'] else 0
    best_train_acc = max(hist['acc']) if hist['acc'] else 0
    
    return jsonify({
        'trained': True,
        'train_acc': best_train_acc,
        'val_acc': best_val_acc,
        'epochs': len(hist['loss']),
        'categories': ', '.join(label_map.values()),
        'plots': plots,
        'save_path': save_path
    })



if __name__ == '__main__':
    print("\n" + "="*80)
    print("ULTRA-OPTIMIZED VERSION - MAXIMUM SPEED")
    print("Aggressive Pooling (40x) | torch.compile | Validation Every 5 Epochs")
    print("="*80)
    print("🌐 http://localhost:5000")
    print("="*80 + "\n")
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)
