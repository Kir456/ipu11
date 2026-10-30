"""
Step 4: Cross-lingual validation - evaluating all models on Russian data.
Run: python scripts/evaluate_crosslingual.py --russian_audio_dir <> --russian_video_dir <> --labels_json <> --output_dir <>
"""

import argparse
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from tabulate import tabulate

from utils import compute_metrics, get_project_root
from train_tcn import StutterTCN
from train_deep_fusion import DeepFusionTCN


def load_frame_labels(labels_json, video_name, T):
    with open(labels_json, 'r') as f:
        annotations = json.load(f)

    frame_labels = np.zeros(T, dtype=np.float32)
    if video_name in annotations:
        for start, end in annotations[video_name]:
            s, e = max(0, int(start)), min(T, int(end))
            frame_labels[s:e] = 1.0
    return frame_labels


def evaluate_audio_tcn(audio_dir, labels_json, video_list, device, encoder_name='wav2vec2'):
    base_dir = get_project_root()
    ckpt_path = base_dir / "scripts" / f"best_tcn_{encoder_name}.pth"

    if not ckpt_path.exists():
        return None, None

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    saved_args = checkpoint.get('args', {})

    model = StutterTCN(
        input_dim=saved_args.get('input_dim', 1024),
        hidden_dim=saved_args.get('hidden_dim', 256),
        num_blocks=saved_args.get('num_blocks', 4),
        kernel_size=saved_args.get('kernel_size', 5),
        dropout=0.0,
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    all_true, all_probs = [], []
    per_video_results = {}
    window_len = saved_args.get('max_seq_len', 150)
    stride = window_len // 2
    audio_dir = Path(audio_dir)

    for vname in video_list:
        npy_path = audio_dir / f"{vname}.npy"
        if not npy_path.exists():
            continue

        features = np.load(str(npy_path))
        T = features.shape[0]
        frame_labels = load_frame_labels(labels_json, vname, T)

        frame_scores = np.zeros(T)
        frame_counts = np.zeros(T)

        with torch.no_grad():
            for ws in range(0, T, stride):
                we = min(ws + window_len, T)
                segment = features[ws:we]
                actual = segment.shape[0]

                if actual < window_len:
                    segment = np.pad(segment, ((0, window_len - actual), (0, 0)))
                    mask = np.concatenate([np.ones(actual), np.zeros(window_len - actual)])
                else:
                    mask = np.ones(window_len)

                seg_t = torch.tensor(segment, dtype=torch.float32).unsqueeze(0).to(device)
                mask_t = torch.tensor(mask, dtype=torch.float32).unsqueeze(0).to(device)

                logit = model(seg_t, mask_t)
                prob = torch.sigmoid(logit).item()

                frame_scores[ws:we] += prob
                frame_counts[ws:we] += 1

        frame_counts = np.maximum(frame_counts, 1)
        frame_probs = frame_scores / frame_counts

        all_true.extend(frame_labels.tolist())
        all_probs.extend(frame_probs.tolist())
        per_video_results[vname] = {'labels': frame_labels, 'probs': frame_probs}

    all_preds = [1 if p > 0.5 else 0 for p in all_probs]
    metrics = compute_metrics(all_true, all_preds, all_probs)
    return metrics, per_video_results


def evaluate_deep_fusion(audio_dir, video_dir, labels_json, video_list, device):
    base_dir = get_project_root()
    ckpt_path = base_dir / "scripts" / "best_deep_fusion.pth"

    if not ckpt_path.exists():
        return None, None

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    model = DeepFusionTCN(
        audio_dim=checkpoint['audio_dim'],
        video_dim=checkpoint['video_dim'],
        hidden_dim=checkpoint.get('hidden_dim', 256),
        modality_dropout=0.0,
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    audio_dir = Path(audio_dir)
    video_dir = Path(video_dir)

    all_true, all_probs = [], []
    per_video_results = {}
    window_len = 450
    stride = 150

    for vname in video_list:
        a_path = audio_dir / f"{vname}.npy"
        v_path = video_dir / f"{vname}.npy"
        if not a_path.exists() or not v_path.exists():
            continue

        audio_feat = np.load(str(a_path))
        video_feat = np.load(str(v_path))

        T = max(audio_feat.shape[0], video_feat.shape[0])
        if audio_feat.shape[0] != T:
            audio_feat = _resample_array(audio_feat, T)
        if video_feat.shape[0] != T:
            video_feat = _resample_array(video_feat, T)

        frame_labels = load_frame_labels(labels_json, vname, T)
        frame_scores = np.zeros(T)
        frame_counts = np.zeros(T)

        with torch.no_grad():
            for ws in range(0, T, stride):
                we = min(ws + window_len, T)
                a_seg = audio_feat[ws:we]
                v_seg = video_feat[ws:we]
                actual = a_seg.shape[0]

                if actual < window_len:
                    pad = window_len - actual
                    a_seg = np.pad(a_seg, ((0, pad), (0, 0)))
                    v_seg = np.pad(v_seg, ((0, pad), (0, 0)))
                    mask = np.concatenate([np.ones(actual), np.zeros(pad)])
                else:
                    mask = np.ones(window_len)

                a_t = torch.tensor(a_seg, dtype=torch.float32).unsqueeze(0).to(device)
                v_t = torch.tensor(v_seg, dtype=torch.float32).unsqueeze(0).to(device)
                m_t = torch.tensor(mask, dtype=torch.float32).unsqueeze(0).to(device)

                logit = model(a_t, v_t, m_t)
                prob = torch.sigmoid(logit).item()

                frame_scores[ws:we] += prob
                frame_counts[ws:we] += 1

        frame_counts = np.maximum(frame_counts, 1)
        frame_probs = frame_scores / frame_counts

        all_true.extend(frame_labels.tolist())
        all_probs.extend(frame_probs.tolist())
        per_video_results[vname] = {'labels': frame_labels, 'probs': frame_probs}

    all_preds = [1 if p > 0.5 else 0 for p in all_probs]
    metrics = compute_metrics(all_true, all_preds, all_probs)
    return metrics, per_video_results


def _resample_array(arr, target_len):
    T_src, D = arr.shape
    x_src = np.linspace(0, 1, T_src)
    x_tgt = np.linspace(0, 1, target_len)
    result = np.zeros((target_len, D), dtype=arr.dtype)
    for d in range(D):
        result[:, d] = np.interp(x_tgt, x_src, arr[:, d])
    return result


def plot_timeline(video_name, labels, probs, threshold=0.5, save_path=None):
    fig, ax = plt.subplots(1, 1, figsize=(14, 3))
    T = len(labels)
    x = np.arange(T)

    in_event = False
    for i in range(T):
        if labels[i] > 0.5 and not in_event:
            start = i
            in_event = True
        elif (labels[i] <= 0.5 or i == T - 1) and in_event:
            ax.axvspan(start, i, alpha=0.3, color='green')
            in_event = False

    preds = (np.array(probs) > threshold).astype(float)
    in_det = False
    for i in range(T):
        if preds[i] > 0.5 and not in_det:
            start = i
            in_det = True
        elif (preds[i] <= 0.5 or i == T - 1) and in_det:
            ax.axvspan(start, i, alpha=0.2, color='red', hatch='///')
            in_det = False

    ax.plot(x, probs, color='blue', linewidth=0.8, alpha=0.8)
    ax.axhline(y=threshold, color='gray', linestyle='--', alpha=0.5, linewidth=0.5)

    ax.set_xlim(0, T)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Frame')
    ax.set_ylabel('Score')
    ax.set_title(f'{video_name}', fontsize=11)

    handles = [
        mpatches.Patch(color='green', alpha=0.3, label='Annotation'),
        mpatches.Patch(color='red', alpha=0.2, hatch='///', label='Detection'),
    ]
    ax.legend(handles=handles, loc='upper right', fontsize=8)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Cross-lingual evaluation on Russian data")
    parser.add_argument('--russian_audio_dir', type=str, required=True)
    parser.add_argument('--russian_video_dir', type=str, default=None)
    parser.add_argument('--labels_json', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--encoders', type=str, default='wav2vec2,wavlm,whisper')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.labels_json, 'r') as f:
        annotations = json.load(f)
    video_list = sorted(annotations.keys())

    print("\n" + "=" * 70)
    print("  [INFO] Cross-Lingual Validation (RU)")
    print(f"  [INFO] Videos: {len(video_list)}")
    print("=" * 70)

    results = []
    all_per_video = {}

    for enc in args.encoders.split(','):
        enc = enc.strip()
        print(f"\n[EVAL] Audio TCN ({enc})...")
        metrics, per_vid = evaluate_audio_tcn(
            args.russian_audio_dir, args.labels_json, video_list, device, enc
        )
        if metrics:
            metrics['method'] = f'TCN ({enc})'
            results.append(metrics)
            all_per_video[f'TCN_{enc}'] = per_vid
            print(f"  ROC-AUC: {metrics['roc_auc']:.4f} | F1: {metrics['f1']:.4f} | "
                  f"Prec: {metrics['precision']:.4f} | Rec: {metrics['recall']:.4f} | Spec: {metrics['specificity']:.4f}")
        else:
            print(f"  [WARN] Model not found")

    if args.russian_video_dir:
        print(f"\n[EVAL] Deep Fusion (audio + video)...")
        metrics, per_vid = evaluate_deep_fusion(
            args.russian_audio_dir, args.russian_video_dir,
            args.labels_json, video_list, device
        )
        if metrics:
            metrics['method'] = 'Deep Fusion'
            results.append(metrics)
            all_per_video['DeepFusion'] = per_vid
            print(f"  ROC-AUC: {metrics['roc_auc']:.4f} | F1: {metrics['f1']:.4f} | "
                  f"Prec: {metrics['precision']:.4f} | Rec: {metrics['recall']:.4f} | Spec: {metrics['specificity']:.4f}")

    if results:
        print("\n" + "=" * 70)
        print("  [RESULTS] SUMMARY TABLE (Russian Data)")
        print("=" * 70 + "\n")

        df = pd.DataFrame(results)
        cols = ['method', 'roc_auc', 'ap', 'f1', 'precision', 'recall', 'specificity']
        cols = [c for c in cols if c in df.columns]
        df_disp = df[cols].copy()
        for c in cols:
            if c != 'method':
                df_disp[c] = df_disp[c].apply(lambda x: f"{x:.4f}")
        print(tabulate(df_disp, headers='keys', tablefmt='grid', showindex=False))

        print("\n  [INFO] Laboratory Baselines (for reference):")
        print("  Vocametrix:     ROC-AUC=0.701 | F1=0.487 | Prec=0.327 | Rec=0.957 | Spec=0.300")
        print("  StutterLens:    ROC-AUC=0.632 | F1=0.269 | Prec=0.391 | Rec=0.205 | Spec=0.886")
        print("  Lab Late Fus.:  ROC-AUC=0.710 | F1=0.282 | Prec=0.504 | Rec=0.195 | Spec=0.929")

        out_path = get_project_root() / "scripts" / "crosslingual_results.csv"
        df[cols].to_csv(out_path, index=False)
        print(f"\n  [SAVE] Results saved to: {out_path}")

    if args.output_dir and all_per_video:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n  [INFO] Generating timeline plots...")

        for method_key, per_vid in all_per_video.items():
            if per_vid is None:
                continue
            for vname, data in per_vid.items():
                plot_path = output_dir / f"timeline_{method_key}_{vname}.png"
                plot_timeline(
                    f"{vname} ({method_key})",
                    data['labels'], data['probs'],
                    save_path=str(plot_path)
                )
        print(f"  [DONE] Plots saved to: {output_dir}")


if __name__ == "__main__":
    main()