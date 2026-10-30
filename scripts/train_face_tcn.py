"""
Step 3: TCN classifier on visual kinematic features.
Run: python scripts/train_face_tcn.py --features_dir <path> --labels_json <path>
"""

import argparse
import json
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupKFold
from tqdm import tqdm

from utils import FocalLoss, compute_metrics, get_project_root
from train_tcn import StutterTCN


class VideoKinematicsDataset(Dataset):
    def __init__(self, video_list, features_dir, labels_json,
                 window_frames=450, stride_frames=150, desc="Loading"):
        self.data, self.labels, self.masks, self.video_ids = [], [], [], []
        features_dir = Path(features_dir)

        with open(labels_json, 'r') as f:
            annotations = json.load(f)

        for video_name in tqdm(video_list, desc=f"[{desc}]", leave=False):
            npy_path = features_dir / f"{video_name}.npy"
            if not npy_path.exists():
                continue

            features = np.load(str(npy_path))
            T, _ = features.shape

            frame_labels = np.zeros(T, dtype=np.float32)
            if video_name in annotations:
                for start, end in annotations[video_name]:
                    frame_labels[max(0, int(start)):min(T, int(end))] = 1.0

            for win_start in range(0, T, stride_frames):
                win_end = min(win_start + window_frames, T)
                segment = features[win_start:win_end]
                seg_labels = frame_labels[win_start:win_end]

                actual_len = segment.shape[0]
                if actual_len < window_frames:
                    pad_len = window_frames - actual_len
                    segment = np.pad(segment, ((0, pad_len), (0, 0)), mode='constant')
                    seg_labels = np.pad(seg_labels, (0, pad_len), mode='constant')
                    mask = np.concatenate([
                        np.ones(actual_len, dtype=np.float32),
                        np.zeros(pad_len, dtype=np.float32)
                    ])
                else:
                    mask = np.ones(window_frames, dtype=np.float32)

                stutter_ratio = seg_labels.sum() / mask.sum() if mask.sum() > 0 else 0
                binary_label = 1.0 if stutter_ratio >= 0.3 else 0.0

                self.data.append(segment)
                self.masks.append(mask)
                self.labels.append(binary_label)
                self.video_ids.append(video_name)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.data[idx], dtype=torch.float32),
            torch.tensor(self.masks[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.float32),
        )


def train_face_tcn(args):
    base_dir = get_project_root()
    features_dir = Path(args.features_dir)
    save_path = base_dir / "scripts" / "best_face_tcn.pth"

    print("\n" + "=" * 60)
    print("  [INFO] Face TCN Training (Visual Kinematics)")
    print(f"  [INFO] Features: {features_dir}")
    print("=" * 60)

    with open(args.labels_json, 'r') as f:
        annotations = json.load(f)

    if args.patient_map_json:
        with open(args.patient_map_json, 'r') as f:
            patient_map = json.load(f)
    else:
        patient_map = {v: v for v in annotations.keys()}

    video_list = sorted([v for v in annotations.keys() if (features_dir / f"{v}.npy").exists()])
    patients = [patient_map.get(v, v) for v in video_list]

    print(f"[DATA] Videos: {len(video_list)}, Unique patients: {len(set(patients))}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] Compute device: {device}")

    sample = np.load(str(features_dir / f"{video_list[0]}.npy"))
    input_dim = sample.shape[1]
    print(f"[MODEL] Input dim: {input_dim} (from {video_list[0]}.npy)")

    n_folds = min(args.n_folds, len(set(patients)))
    gkf = GroupKFold(n_splits=n_folds)
    groups = np.array(patients)
    video_array = np.array(video_list)

    fold_results = []
    best_overall_roc = 0.0

    for fold, (train_idx, val_idx) in enumerate(gkf.split(video_array, groups=groups)):
        print(f"\n--- Fold {fold+1}/{n_folds} ---")
        train_videos = video_array[train_idx].tolist()
        val_videos = video_array[val_idx].tolist()
        print(f"  [SPLIT] Train: {len(train_videos)} videos, Val: {len(val_videos)} videos")

        train_dataset = VideoKinematicsDataset(
            train_videos, features_dir, args.labels_json,
            window_frames=args.window_frames, stride_frames=args.stride_frames,
            desc=f"Fold-{fold+1} Train"
        )
        val_dataset = VideoKinematicsDataset(
            val_videos, features_dir, args.labels_json,
            window_frames=args.window_frames, stride_frames=args.stride_frames,
            desc=f"Fold-{fold+1} Val"
        )

        if len(train_dataset) == 0 or len(val_dataset) == 0:
            print("  [WARN] Empty dataset - skipping fold")
            continue

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

        model = StutterTCN(
            input_dim=input_dim, hidden_dim=128, num_blocks=3, kernel_size=5, dropout=0.3
        ).to(device)

        criterion = FocalLoss(alpha=0.25, gamma=2.0)
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

        best_fold_roc = 0.0

        for epoch in range(args.epochs):
            model.train()
            for emb, mask, labels in train_loader:
                emb, mask, labels = emb.to(device), mask.to(device), labels.to(device)
                optimizer.zero_grad()
                logits = model(emb, mask)
                loss = criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            scheduler.step()

            model.eval()
            all_probs, all_preds, all_true = [], [], []
            with torch.no_grad():
                for emb, mask, labels in val_loader:
                    emb, mask = emb.to(device), mask.to(device)
                    logits = model(emb, mask)
                    probs = torch.sigmoid(logits)
                    all_probs.extend(probs.cpu().numpy())
                    all_preds.extend((probs > 0.5).float().cpu().numpy())
                    all_true.extend(labels.numpy())

            metrics = compute_metrics(all_true, all_preds, all_probs)
            current_roc = metrics.get('roc_auc', 0.0)

            if current_roc > best_fold_roc:
                best_fold_roc = current_roc
                if current_roc > best_overall_roc:
                    best_overall_roc = current_roc
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'fold': fold,
                        'epoch': epoch,
                        'metrics': metrics,
                        'input_dim': input_dim,
                    }, save_path)

        print(f"  [INFO] Fold {fold+1} best ROC-AUC: {best_fold_roc:.4f}")
        fold_results.append({'fold': fold + 1, 'best_roc_auc': best_fold_roc})

    print(f"\n{'='*60}")
    print(f"  [DONE] Training complete")
    if fold_results:
        print(f"  [RESULTS] Mean ROC-AUC across folds: {np.mean([r['best_roc_auc'] for r in fold_results]):.4f}")
        print(f"  [RESULTS] Best overall ROC-AUC: {best_overall_roc:.4f}")
    if save_path.exists():
        print(f"  [SAVE] Model saved to: {save_path}")
    print(f"{'='*60}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train Face TCN on visual kinematics features")
    parser.add_argument('--features_dir', type=str, required=True)
    parser.add_argument('--labels_json', type=str, required=True)
    parser.add_argument('--patient_map_json', type=str, default=None)
    parser.add_argument('--window_frames', type=int, default=450)
    parser.add_argument('--stride_frames', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--n_folds', type=int, default=6)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_face_tcn(args)