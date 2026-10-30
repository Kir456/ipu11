"""
Common utilities for the multimodal stuttering detection project.
Contains: metrics, datasets, loss functions, constants.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score,
    confusion_matrix
)

MAX_SEQ_LEN = 150
SAMPLE_RATE = 16000

ENCODER_DIMS = {
    "wav2vec2": 1024,
    "wavlm": 768,
    "whisper": 768,
}

def get_project_root():
    return Path(__file__).resolve().parent.parent


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce)
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        return focal.mean()


class MeanPooledDataset(Dataset):
    def __init__(self, df, embeddings_dir, desc="Loading"):
        self.data = []
        self.labels = []
        df = df.reset_index(drop=True)
        for idx in tqdm(range(len(df)), desc=f"RAM [{desc}]", leave=False):
            fname = df.loc[idx, 'file_name'].replace(".wav", ".npy")
            emb = np.load(str(embeddings_dir / fname))
            self.data.append(emb.astype(np.float16))
            self.labels.append(df.loc[idx, 'is_stutter'])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.data[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.float32),
        )


class SequentialDataset(Dataset):
    def __init__(self, df, embeddings_dir, max_seq_len=MAX_SEQ_LEN, desc="Loading"):
        self.data = []
        self.labels = []
        self.masks = []
        self.max_seq_len = max_seq_len
        df = df.reset_index(drop=True)

        skipped = 0
        for idx in tqdm(range(len(df)), desc=f"RAM [{desc}]", leave=False):
            fname = df.loc[idx, 'file_name'].replace(".wav", ".npy")
            path = embeddings_dir / fname
            if not path.exists():
                skipped += 1
                continue

            emb = np.load(str(path))  # (T, D)
            T, D = emb.shape

            if T >= max_seq_len:
                emb_padded = emb[:max_seq_len]
                mask = np.ones(max_seq_len, dtype=np.float32)
            else:
                pad_len = max_seq_len - T
                emb_padded = np.pad(emb, ((0, pad_len), (0, 0)), mode='constant')
                mask = np.concatenate([np.ones(T, dtype=np.float32),
                                       np.zeros(pad_len, dtype=np.float32)])

            # Store in float16 to prevent RAM overflow on 16GB systems
            self.data.append(emb_padded.astype(np.float16))
            self.masks.append(mask)
            self.labels.append(df.loc[idx, 'is_stutter'])

        if skipped > 0:
            print(f"  [WARN] Skipped {skipped} files (missing .npy)")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.data[idx], dtype=torch.float32),
            torch.tensor(self.masks[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.float32),
        )


def compute_metrics(y_true, y_pred, y_prob=None):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    metrics = {
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'specificity': specificity,
    }

    if y_prob is not None:
        y_prob = np.asarray(y_prob)
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
        except ValueError:
            metrics['roc_auc'] = 0.0
        try:
            metrics['ap'] = average_precision_score(y_true, y_prob)
        except ValueError:
            metrics['ap'] = 0.0

    return metrics


def print_metrics(metrics, prefix=""):
    parts = []
    for key in ['f1', 'precision', 'recall', 'specificity', 'roc_auc', 'ap']:
        if key in metrics:
            parts.append(f"{key.upper()}: {metrics[key]:.4f}")
    print(f"{prefix}{' | '.join(parts)}")


def metrics_to_row(metrics, method_name, **extra):
    row = {'method': method_name}
    row.update(metrics)
    row.update(extra)
    return row