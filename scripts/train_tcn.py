"""
Step 1: TCN (Temporal Convolutional Network) classifier for stuttering.
Works on sequential embeddings T x 1024 from wav2vec2.

Architecture: 4 residual dilated 1D conv blocks (dilation 1,2,4,8)
-> Global Average Pooling -> Linear -> sigmoid.
"""

import os
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from utils import (
    FocalLoss, SequentialDataset, MAX_SEQ_LEN,
    compute_metrics, print_metrics, get_project_root
)

# ============================================================
# TCN Architecture
# ============================================================

class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=1, dropout=0.2):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2 

        self.conv1 = nn.Conv1d(channels, channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=1)
        self.bn2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        out = self.dropout(F.relu(self.bn1(self.conv1(x))))
        out = self.bn2(self.conv2(out))
        if out.shape[-1] != residual.shape[-1]:
            out = out[..., :residual.shape[-1]]
        return F.relu(out + residual)


class StutterTCN(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=256, num_blocks=4,
                 kernel_size=5, dropout=0.3):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )

        dilations = [2 ** i for i in range(num_blocks)] 
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, kernel_size, dilation=d, dropout=dropout)
            for d in dilations
        ])

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x, mask=None):
        x = x.permute(0, 2, 1)
        x = self.input_proj(x) 

        for block in self.blocks:
            x = block(x) 

        x = x.permute(0, 2, 1) 
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1) 
            x = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            x = x.mean(dim=1) 

        logits = self.classifier(x).squeeze(-1) 
        return logits


# ============================================================
# Training Loop
# ============================================================

def train_tcn(args):
    base_dir = get_project_root()
    embeddings_dir = base_dir / "data" / args.embeddings_subdir
    csv_path = base_dir / "data" / "sep28k_audio" / "metadata.csv"
    save_path = base_dir / "scripts" / f"best_tcn_{args.encoder_name}.pth"

    print("\n" + "=" * 60)
    print(f"[INFO] TCN Training | Encoder: {args.encoder_name}")
    print(f"[INFO] Embeddings: {embeddings_dir}")
    print(f"[INFO] Input dim: {args.input_dim}")
    print("=" * 60)

    df = pd.read_csv(csv_path)

    df['npy_exists'] = df['file_name'].apply(
        lambda x: (embeddings_dir / x.replace('.wav', '.npy')).exists()
    )
    df = df[df['npy_exists']].drop(columns=['npy_exists'])
    print(f"[DATA] Valid files with embeddings: {len(df)}")

    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df['is_stutter']
    )
    print(f"[DATA] Train: {len(train_df)} | Val: {len(val_df)}")

    train_dataset = SequentialDataset(train_df, embeddings_dir,
                                      max_seq_len=args.max_seq_len, desc="Train")
    val_dataset = SequentialDataset(val_df, embeddings_dir,
                                    max_seq_len=args.max_seq_len, desc="Valid")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=0, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[DEVICE] Compute device: {device}")

    model = StutterTCN(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        num_blocks=args.num_blocks,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[MODEL] Trainable parameters: {param_count:,}")

    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # CRITICAL FIX: Track ROC-AUC instead of F1 to avoid mode collapse
    best_roc_auc = 0.0
    patience_counter = 0

    print(f"\n{'='*60}")
    print(f"[START] Training for {args.epochs} epochs")
    print(f"{'='*60}\n")

    for epoch in range(args.epochs):
        epoch_start = time.time()

        model.train()
        train_loss = 0.0
        train_steps = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:02d}/{args.epochs} [Train]",
                     leave=False, colour='cyan')
        for embeddings, masks, labels in pbar:
            embeddings = embeddings.to(device)
            masks = masks.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(embeddings, masks)
            loss = criterion(logits, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            train_loss += loss.item()
            train_steps += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()

        model.eval()
        val_preds = []
        val_probs = []
        val_true = []

        with torch.no_grad():
            for embeddings, masks, labels in val_loader:
                embeddings = embeddings.to(device)
                masks = masks.to(device)

                logits = model(embeddings, masks)
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()

                val_probs.extend(probs.cpu().numpy())
                val_preds.extend(preds.cpu().numpy())
                val_true.extend(labels.numpy())

        metrics = compute_metrics(val_true, val_preds, val_probs)
        avg_loss = train_loss / max(train_steps, 1)
        duration = time.time() - epoch_start
        lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch+1:02d}/{args.epochs} [{duration:.1f}s] | "
              f"Loss: {avg_loss:.4f} | LR: {lr:.2e} | ", end="")
        print_metrics(metrics)

        # Evaluate based on ROC-AUC
        current_roc = metrics.get('roc_auc', 0.0)
        
        if current_roc > best_roc_auc:
            best_roc_auc = current_roc
            patience_counter = 0
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_roc_auc': best_roc_auc,
                'best_f1': metrics.get('f1', 0.0),
                'metrics': metrics,
                'args': vars(args),
            }, save_path)
            print(f"  [SAVE] New best model! ROC-AUC: {best_roc_auc:.4f} (F1: {metrics.get('f1', 0.0):.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n[STOP] Early stopping: {args.patience} epochs without ROC-AUC improvement.")
                break

    print(f"\n{'='*60}")
    print(f"[DONE] Training complete. Best val ROC-AUC: {best_roc_auc:.4f}")
    print(f"[INFO] Model saved to: {save_path}")
    print(f"{'='*60}")

    return best_roc_auc


def parse_args():
    parser = argparse.ArgumentParser(description="Train TCN classifier for stuttering detection")
    parser.add_argument('--encoder_name', type=str, default='wav2vec2',
                        choices=['wav2vec2', 'wavlm', 'whisper'])
    parser.add_argument('--embeddings_subdir', type=str, default=None)
    parser.add_argument('--max_seq_len', type=int, default=MAX_SEQ_LEN)
    parser.add_argument('--input_dim', type=int, default=None)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--num_blocks', type=int, default=4)
    parser.add_argument('--kernel_size', type=int, default=5)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--patience', type=int, default=10)

    args = parser.parse_args()

    from utils import ENCODER_DIMS
    if args.input_dim is None:
        args.input_dim = ENCODER_DIMS[args.encoder_name]

    if args.embeddings_subdir is None:
        subdir_map = {
            'wav2vec2': 'sep28k_seq_embeddings',
            'wavlm': 'sep28k_seq_wavlm',
            'whisper': 'sep28k_seq_whisper',
        }
        args.embeddings_subdir = subdir_map[args.encoder_name]

    return args

if __name__ == "__main__":
    args = parse_args()
    train_tcn(args)