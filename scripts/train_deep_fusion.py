"""
Step 4: Deep Fusion - multimodal TCN with early/deep audio-visual fusion.
Run: python scripts/train_deep_fusion.py --audio_dir <> --video_dir <> --labels_json <>
"""

import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupKFold
from tqdm import tqdm

from utils import FocalLoss, compute_metrics, get_project_root


class ResidualBlock1D(nn.Module):
    def __init__(self, channels, kernel_size=5, dilation=1, dropout=0.2):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
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


class StreamEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_blocks=4, kernel_size=5, dropout=0.3):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
        dilations = [2 ** i for i in range(num_blocks)]
        self.blocks = nn.ModuleList([
            ResidualBlock1D(hidden_dim, kernel_size, dilation=d, dropout=dropout)
            for d in dilations
        ])

    def forward(self, x):
        x = self.proj(x)
        for block in self.blocks:
            x = block(x)
        return x


class DeepFusionTCN(nn.Module):
    def __init__(self, audio_dim=1024, video_dim=42, hidden_dim=256,
                 audio_blocks=4, video_blocks=3, fusion_blocks=2,
                 kernel_size=5, dropout=0.3, modality_dropout=0.2):
        super().__init__()
        self.modality_dropout = modality_dropout

        self.audio_encoder = StreamEncoder(audio_dim, hidden_dim, audio_blocks, kernel_size, dropout)
        self.video_encoder = StreamEncoder(video_dim, hidden_dim, video_blocks, kernel_size, dropout)

        fusion_dim = hidden_dim * 2
        self.fusion_proj = nn.Sequential(
            nn.Conv1d(fusion_dim, hidden_dim, kernel_size=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
        dilations = [2 ** i for i in range(fusion_blocks)]
        self.fusion_blocks = nn.ModuleList([
            ResidualBlock1D(hidden_dim, kernel_size, dilation=d, dropout=dropout)
            for d in dilations
        ])

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, audio, video, mask=None):
        audio = audio.permute(0, 2, 1)
        video = video.permute(0, 2, 1)

        audio_feat = self.audio_encoder(audio)
        video_feat = self.video_encoder(video)

        if self.training and self.modality_dropout > 0:
            r = torch.rand(1).item()
            if r < self.modality_dropout / 2:
                audio_feat = torch.zeros_like(audio_feat)
            elif r < self.modality_dropout:
                video_feat = torch.zeros_like(video_feat)

        fused = torch.cat([audio_feat, video_feat], dim=1)
        fused = self.fusion_proj(fused)
        for block in self.fusion_blocks:
            fused = block(fused)

        fused = fused.permute(0, 2, 1)
        if mask is not None:
            mask_exp = mask.unsqueeze(-1)
            fused = (fused * mask_exp).sum(dim=1) / mask_exp.sum(dim=1).clamp(min=1)
        else:
            fused = fused.mean(dim=1)

        return self.classifier(fused).squeeze(-1)


class MultimodalDataset(Dataset):
    def __init__(self, video_list, audio_dir, video_dir, labels_json,
                 target_len=450, stride=150, desc="Loading"):
        self.data_audio, self.data_video, self.labels, self.masks = [], [], [], []
        audio_dir = Path(audio_dir)
        video_dir = Path(video_dir)

        with open(labels_json, 'r') as f:
            annotations = json.load(f)

        for vname in tqdm(video_list, desc=f"[{desc}]", leave=False):
            audio_path = audio_dir / f"{vname}.npy"
            video_path = video_dir / f"{vname}.npy"

            if not audio_path.exists() or not video_path.exists():
                continue

            audio_feat = np.load(str(audio_path))
            video_feat = np.load(str(video_path))

            T_target = max(audio_feat.shape[0], video_feat.shape[0])
            if audio_feat.shape[0] != T_target:
                audio_feat = self._resample(audio_feat, T_target)
            if video_feat.shape[0] != T_target:
                video_feat = self._resample(video_feat, T_target)

            T = audio_feat.shape[0]
            frame_labels = np.zeros(T, dtype=np.float32)
            if vname in annotations:
                for start, end in annotations[vname]:
                    frame_labels[max(0, int(start)):min(T, int(end))] = 1.0

            for ws in range(0, T, stride):
                we = min(ws + target_len, T)
                a_seg = audio_feat[ws:we]
                v_seg = video_feat[ws:we]
                l_seg = frame_labels[ws:we]

                actual = a_seg.shape[0]
                if actual < target_len:
                    pad = target_len - actual
                    a_seg = np.pad(a_seg, ((0, pad), (0, 0)))
                    v_seg = np.pad(v_seg, ((0, pad), (0, 0)))
                    l_seg = np.pad(l_seg, (0, pad))
                    mask = np.concatenate([np.ones(actual), np.zeros(pad)]).astype(np.float32)
                else:
                    mask = np.ones(target_len, dtype=np.float32)

                stutter_ratio = l_seg.sum() / mask.sum() if mask.sum() > 0 else 0
                binary_label = 1.0 if stutter_ratio >= 0.3 else 0.0

                self.data_audio.append(a_seg.astype(np.float32))
                self.data_video.append(v_seg.astype(np.float32))
                self.masks.append(mask)
                self.labels.append(binary_label)

    @staticmethod
    def _resample(arr, target_len):
        T_src, D = arr.shape
        x_src = np.linspace(0, 1, T_src)
        x_tgt = np.linspace(0, 1, target_len)
        result = np.zeros((target_len, D), dtype=arr.dtype)
        for d in range(D):
            result[:, d] = np.interp(x_tgt, x_src, arr[:, d])
        return result

    def __len__(self):
        return len(self.data_audio)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.data_audio[idx]),
            torch.tensor(self.data_video[idx]),
            torch.tensor(self.masks[idx]),
            torch.tensor(self.labels[idx]),
        )


def train_fusion(args):
    base_dir = get_project_root()
    save_path = base_dir / "scripts" / "best_deep_fusion.pth"

    print("\n" + "=" * 60)
    print("  [INFO] Deep Fusion Training (Audio + Video)")
    print("=" * 60)

    with open(args.labels_json, 'r') as f:
        annotations = json.load(f)

    if args.patient_map_json:
        with open(args.patient_map_json, 'r') as f:
            patient_map = json.load(f)
    else:
        patient_map = {v: v for v in annotations.keys()}

    audio_dir = Path(args.audio_dir)
    video_dir = Path(args.video_dir)

    video_list = sorted([
        v for v in annotations.keys()
        if (audio_dir / f"{v}.npy").exists() and (video_dir / f"{v}.npy").exists()
    ])
    patients = [patient_map.get(v, v) for v in video_list]

    print(f"[DATA] Videos with both modalities: {len(video_list)}")
    print(f"[DATA] Unique patients: {len(set(patients))}")

    sample_a = np.load(str(audio_dir / f"{video_list[0]}.npy"))
    sample_v = np.load(str(video_dir / f"{video_list[0]}.npy"))
    audio_dim = sample_a.shape[1]
    video_dim = sample_v.shape[1]
    print(f"[MODEL] Audio dim: {audio_dim}, Video dim: {video_dim}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] Compute device: {device}")

    n_folds = min(args.n_folds, len(set(patients)))
    gkf = GroupKFold(n_splits=n_folds)
    groups = np.array(patients)
    video_array = np.array(video_list)

    fold_results = []
    best_overall_roc = 0.0

    for fold, (train_idx, val_idx) in enumerate(gkf.split(video_array, groups=groups)):
        print(f"\n--- Fold {fold+1}/{n_folds} ---")
        train_vids = video_array[train_idx].tolist()
        val_vids = video_array[val_idx].tolist()

        train_dataset = MultimodalDataset(
            train_vids, audio_dir, video_dir, args.labels_json,
            target_len=args.window_frames, stride=args.stride_frames,
            desc=f"F{fold+1} Train"
        )
        val_dataset = MultimodalDataset(
            val_vids, audio_dir, video_dir, args.labels_json,
            target_len=args.window_frames, stride=args.stride_frames,
            desc=f"F{fold+1} Val"
        )

        if len(train_dataset) == 0 or len(val_dataset) == 0:
            continue

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

        model = DeepFusionTCN(
            audio_dim=audio_dim, video_dim=video_dim,
            hidden_dim=args.hidden_dim, modality_dropout=args.modality_dropout,
        ).to(device)

        criterion = FocalLoss(alpha=0.25, gamma=2.0)
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

        best_fold_roc = 0.0

        for epoch in range(args.epochs):
            model.train()
            for audio, video, mask, labels in train_loader:
                audio, video, mask, labels = audio.to(device), video.to(device), mask.to(device), labels.to(device)
                optimizer.zero_grad()
                logits = model(audio, video, mask)
                loss = criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            scheduler.step()

            model.eval()
            all_probs, all_preds, all_true = [], [], []
            with torch.no_grad():
                for audio, video, mask, labels in val_loader:
                    audio, video, mask = audio.to(device), video.to(device), mask.to(device)
                    logits = model(audio, video, mask)
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
                        'audio_dim': audio_dim,
                        'video_dim': video_dim,
                        'hidden_dim': args.hidden_dim,
                    }, save_path)

        print(f"  [INFO] Fold {fold+1} best ROC-AUC: {best_fold_roc:.4f}")
        fold_results.append(best_fold_roc)

    print(f"\n{'='*60}")
    print(f"  [DONE] Deep Fusion training complete")
    if fold_results:
        print(f"  [RESULTS] Mean ROC-AUC: {np.mean(fold_results):.4f} +/- {np.std(fold_results):.4f}")
        print(f"  [RESULTS] Best ROC-AUC: {best_overall_roc:.4f}")
    if save_path.exists():
        print(f"  [SAVE] Model saved to: {save_path}")
    print(f"{'='*60}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train Deep Fusion TCN")
    parser.add_argument('--audio_dir', type=str, required=True)
    parser.add_argument('--video_dir', type=str, required=True)
    parser.add_argument('--labels_json', type=str, required=True)
    parser.add_argument('--patient_map_json', type=str, default=None)
    parser.add_argument('--window_frames', type=int, default=450)
    parser.add_argument('--stride_frames', type=int, default=150)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--modality_dropout', type=float, default=0.2)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--n_folds', type=int, default=6)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_fusion(args)