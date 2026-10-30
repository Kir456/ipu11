"""
Step 2: Comparative metrics table across all encoders.
Loads saved checkpoints of the best TCN models for each encoder
and evaluates them on the validation set.

Run: python scripts/compare_encoders.py
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tabulate import tabulate

from utils import (
    SequentialDataset, MeanPooledDataset, compute_metrics, MAX_SEQ_LEN,
    get_project_root
)
from train_tcn import StutterTCN


ENCODER_CONFIGS = {
    'wav2vec2': {
        'embeddings_subdir': 'sep28k_seq_embeddings',
        'checkpoint': 'best_tcn_wav2vec2.pth',
        'input_dim': 1024,
    },
    'wavlm': {
        'embeddings_subdir': 'sep28k_seq_wavlm',
        'checkpoint': 'best_tcn_wavlm.pth',
        'input_dim': 768,
    },
    'whisper': {
        'embeddings_subdir': 'sep28k_seq_whisper',
        'checkpoint': 'best_tcn_whisper.pth',
        'input_dim': 768,
    },
}


def evaluate_tcn(encoder_name, config, base_dir, device, df_val):
    ckpt_path = base_dir / "scripts" / config['checkpoint']
    emb_dir = base_dir / "data" / config['embeddings_subdir']

    if not ckpt_path.exists():
        print(f"  [WARN] Checkpoint {ckpt_path.name} not found - skipping {encoder_name}")
        return None

    if not emb_dir.exists():
        print(f"  [WARN] Embeddings {emb_dir} not found - skipping {encoder_name}")
        return None

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    saved_args = checkpoint.get('args', {})

    model = StutterTCN(
        input_dim=config['input_dim'],
        hidden_dim=saved_args.get('hidden_dim', 256),
        num_blocks=saved_args.get('num_blocks', 4),
        kernel_size=saved_args.get('kernel_size', 5),
        dropout=saved_args.get('dropout', 0.3),
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    val_dataset = SequentialDataset(
        df_val, emb_dir,
        max_seq_len=saved_args.get('max_seq_len', MAX_SEQ_LEN),
        desc=f"Val-{encoder_name}"
    )
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)

    all_probs, all_preds, all_true = [], [], []

    with torch.no_grad():
        for embeddings, masks, labels in val_loader:
            embeddings = embeddings.to(device)
            masks = masks.to(device)
            logits = model(embeddings, masks)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_true.extend(labels.numpy())

    metrics = compute_metrics(all_true, all_preds, all_probs)
    metrics['encoder'] = f"TCN ({encoder_name})"
    return metrics


def evaluate_mlp(base_dir, device, df_val):
    from train_mlp import AudioClassifier

    ckpt_path = base_dir / "scripts" / "best_mlp_model.pth"
    emb_dir = base_dir / "data" / "sep28k_embeddings"

    if not ckpt_path.exists() or not emb_dir.exists():
        print("  [WARN] MLP checkpoint or embeddings not found - skipping")
        return None

    model = AudioClassifier(input_dim=1024).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False))
    model.eval()

    val_dataset = MeanPooledDataset(df_val, emb_dir, desc="Val-MLP")
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)

    all_probs, all_preds, all_true = [], [], []

    with torch.no_grad():
        for embeddings, labels in val_loader:
            embeddings = embeddings.to(device)
            logits = model(embeddings)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_true.extend(labels.numpy())

    metrics = compute_metrics(all_true, all_preds, all_probs)
    metrics['encoder'] = 'MLP (wav2vec2 mean-pooled)'
    return metrics


def main():
    base_dir = get_project_root()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    csv_path = base_dir / "data" / "sep28k_audio" / "metadata.csv"
    df = pd.read_csv(csv_path)
    _, df_val = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df['is_stutter']
    )

    print("\n" + "=" * 70)
    print("  [INFO] Encoder Comparison: Unified Validation Evaluation")
    print("=" * 70 + "\n")

    results = []

    print("[EVAL] MLP Baseline (mean-pooled wav2vec2)...")
    mlp_metrics = evaluate_mlp(base_dir, device, df_val)
    if mlp_metrics:
        results.append(mlp_metrics)

    for encoder_name, config in ENCODER_CONFIGS.items():
        print(f"[EVAL] TCN + {encoder_name}...")
        metrics = evaluate_tcn(encoder_name, config, base_dir, device, df_val)
        if metrics:
            results.append(metrics)

    if results:
        print("\n" + "=" * 70)
        print("  [RESULTS] COMPARATIVE TABLE")
        print("=" * 70 + "\n")

        df_results = pd.DataFrame(results)
        columns_order = ['encoder', 'roc_auc', 'ap', 'f1', 'precision', 'recall', 'specificity']
        columns_present = [c for c in columns_order if c in df_results.columns]
        df_display = df_results[columns_present].copy()

        for col in columns_present:
            if col != 'encoder':
                df_display[col] = df_display[col].apply(lambda x: f"{x:.4f}")

        print(tabulate(df_display, headers='keys', tablefmt='grid', showindex=False))

        out_path = base_dir / "scripts" / "encoder_comparison.csv"
        df_results[columns_present].to_csv(out_path, index=False)
        print(f"\n[SAVE] Table saved to: {out_path}")
    else:
        print("[ERROR] No encoders were evaluated. Train the models first.")


if __name__ == "__main__":
    main()