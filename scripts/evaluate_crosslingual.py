"""
Step 5: Full cross-lingual evaluation on Russian video corpus.
Evaluates 12 configurations:
  1. Lab baselines (Ivankin et al., 2023)
  2. Single-modal Audio Zero-Shot:
     - Baseline MLP (Wav2Vec2 pooled)
     - Audio TCN (Wav2Vec2-XLSR-53)
     - Audio TCN (WavLM-Base-Plus)
     - Audio TCN (Whisper-Small)
     - Hybrid TCN + Multi-Head Attention (Whisper-Small)
  3. Visual Ablation:
     - Face TCN (42 kinematic features)
  4. Multimodal Deep Fusion:
     - Deep Fusion (Wav2Vec2 + Video)
     - Deep Fusion (WavLM + Video)
     - Deep Fusion (Whisper + Video)

Outputs Table 3 to console, writes scripts/crosslingual_results.csv,
and saves segmentation timeline plots to results/plots/.
"""

import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tabulate import tabulate
import torch

from train_deep_fusion import DeepFusionTCN
from train_hybrid_tcn import StutterHybridTCN
from train_mlp import AudioClassifier
from train_tcn import StutterTCN
from utils import compute_metrics, get_project_root

TARGET_FPS = 15


def load_ground_truth(labels_json, video_list, video_dir):
    with open(labels_json, 'r', encoding='utf-8') as f:
        annotations = json.load(f)

    gt_dict = {}
    for vname in video_list:
        v_feat = np.load(str(Path(video_dir) / f"{vname}.npy"))
        T = v_feat.shape[0]
        y = np.zeros(T, dtype=np.float32)
        if vname in annotations:
            for s, e in annotations[vname]:
                y[max(0, int(s)):min(T, int(e))] = 1.0
        gt_dict[vname] = y
    return gt_dict


def predict_sliding_window_1d(model, feat, window_len, stride, device):
    T, _ = feat.shape
    prob_acc = np.zeros(T, dtype=np.float32)
    count_acc = np.zeros(T, dtype=np.float32)

    with torch.no_grad():
        for ws in range(0, max(1, T - window_len // 2), stride):
            we = min(ws + window_len, T)
            seg = feat[ws:we]
            actual = seg.shape[0]

            if actual < window_len:
                pad = window_len - actual
                seg = np.pad(seg, ((0, pad), (0, 0)))
                mask = np.concatenate([np.ones(actual), np.zeros(pad)]).astype(np.float32)
            else:
                mask = np.ones(window_len, dtype=np.float32)

            t_feat = torch.tensor(seg, dtype=torch.float32).unsqueeze(0).to(device)
            t_mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0).to(device)

            logit = model(t_feat, t_mask).item()
            prob = 1.0 / (1.0 + np.exp(-logit))

            prob_acc[ws:we] += prob
            count_acc[ws:we] += 1.0

    return prob_acc / np.maximum(count_acc, 1.0)


def predict_sliding_window_fusion(model, a_feat, v_feat, window_len, stride, device):
    T = a_feat.shape[0]
    prob_acc = np.zeros(T, dtype=np.float32)
    count_acc = np.zeros(T, dtype=np.float32)

    with torch.no_grad():
        for ws in range(0, max(1, T - window_len // 2), stride):
            we = min(ws + window_len, T)
            a_seg = a_feat[ws:we]
            v_seg = v_feat[ws:we]
            actual = a_seg.shape[0]

            if actual < window_len:
                pad = window_len - actual
                a_seg = np.pad(a_seg, ((0, pad), (0, 0)))
                v_seg = np.pad(v_seg, ((0, pad), (0, 0)))
                mask = np.concatenate([np.ones(actual), np.zeros(pad)]).astype(np.float32)
            else:
                mask = np.ones(window_len, dtype=np.float32)

            t_a = torch.tensor(a_seg, dtype=torch.float32).unsqueeze(0).to(device)
            t_v = torch.tensor(v_seg, dtype=torch.float32).unsqueeze(0).to(device)
            t_mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0).to(device)

            logit = model(t_a, t_v, t_mask).item()
            prob = 1.0 / (1.0 + np.exp(-logit))

            prob_acc[ws:we] += prob
            count_acc[ws:we] += 1.0

    return prob_acc / np.maximum(count_acc, 1.0)


def resample_1d(arr, target_len):
    T = len(arr)
    if T == target_len:
        return arr
    x_src = np.linspace(0, 1, T)
    x_tgt = np.linspace(0, 1, target_len)
    return np.interp(x_tgt, x_src, arr)


def resample_2d(arr, target_len):
    T, D = arr.shape
    if T == target_len:
        return arr
    x_src = np.linspace(0, 1, T)
    x_tgt = np.linspace(0, 1, target_len)
    out = np.zeros((target_len, D), dtype=arr.dtype)
    for d in range(D):
        out[:, d] = np.interp(x_tgt, x_src, arr[:, d])
    return out


def evaluate_mlp(video_list, gt_dict, base_dir, device):
    ckpt_path = base_dir / "scripts" / "best_mlp_model.pth"
    if not ckpt_path.exists():
        print("  [WARN] best_mlp_model.pth not found, skipping MLP baseline")
        return None, {}

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    in_dim = 1024
    model = AudioClassifier(input_dim=in_dim).to(device)
    state_dict = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()

    a_dir = base_dir / "data" / "russian_seq_wav2vec2"
    all_y_true, all_y_prob = [], []
    per_vid = {}

    with torch.no_grad():
        for v in video_list:
            a_feat = np.load(str(a_dir / f"{v}.npy"))
            T = a_feat.shape[0]
            prob_audio = np.zeros(T, dtype=np.float32)
            count_audio = np.zeros(T, dtype=np.float32)

            window_len = 150
            stride = 50
            for ws in range(0, max(1, T - window_len // 2), stride):
                we = min(ws + window_len, T)
                seg = a_feat[ws:we]
                pooled = torch.tensor(seg.mean(axis=0), dtype=torch.float32).unsqueeze(0).to(device)
                logit = model(pooled).item()
                prob = 1.0 / (1.0 + np.exp(-logit))

                prob_audio[ws:we] += prob
                count_audio[ws:we] += 1.0

            prob_audio = prob_audio / np.maximum(count_audio, 1.0)
            y_true = gt_dict[v]
            prob_video_grid = resample_1d(prob_audio, len(y_true))

            all_y_true.extend(y_true)
            all_y_prob.extend(prob_video_grid)
            per_vid[v] = prob_video_grid

    y_true_np = np.array(all_y_true)
    y_prob_np = np.array(all_y_prob)
    y_pred_np = (y_prob_np >= 0.5).astype(np.float32)
    m = compute_metrics(y_true_np, y_pred_np, y_prob_np)
    return m, per_vid


def evaluate_audio_tcn(encoder_name, video_list, gt_dict, base_dir, device):
    ckpt_path = base_dir / "scripts" / f"best_tcn_{encoder_name}.pth"
    if not ckpt_path.exists():
        print(f"  [WARN] {ckpt_path.name} not found, skipping")
        return None, {}

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    in_dim = ckpt.get('input_dim', 768 if encoder_name in ['wavlm', 'whisper'] else 1024)
    model = StutterTCN(input_dim=in_dim, hidden_dim=256, num_blocks=4, kernel_size=5, dropout=0.0).to(device)
    state_dict = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()

    a_dir = base_dir / "data" / f"russian_seq_{encoder_name}"
    all_y_true, all_y_prob = [], []
    per_vid = {}

    for v in video_list:
        a_feat = np.load(str(a_dir / f"{v}.npy"))
        prob_audio = predict_sliding_window_1d(model, a_feat, window_len=150, stride=50, device=device)
        y_true = gt_dict[v]
        prob_video_grid = resample_1d(prob_audio, len(y_true))

        all_y_true.extend(y_true)
        all_y_prob.extend(prob_video_grid)
        per_vid[v] = prob_video_grid

    y_true_np = np.array(all_y_true)
    y_prob_np = np.array(all_y_prob)
    y_pred_np = (y_prob_np >= 0.5).astype(np.float32)
    m = compute_metrics(y_true_np, y_pred_np, y_prob_np)
    return m, per_vid


def evaluate_hybrid_whisper(video_list, gt_dict, base_dir, device):
    ckpt_path = base_dir / "scripts" / "best_hybrid_whisper.pth"
    if not ckpt_path.exists():
        print("  [WARN] best_hybrid_whisper.pth not found, skipping Hybrid TCN")
        return None, {}

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    in_dim = ckpt.get('input_dim', 768)
    model = StutterHybridTCN(input_dim=in_dim).to(device)
    state_dict = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()

    a_dir = base_dir / "data" / "russian_seq_whisper"
    all_y_true, all_y_prob = [], []
    per_vid = {}

    for v in video_list:
        a_feat = np.load(str(a_dir / f"{v}.npy"))
        prob_audio = predict_sliding_window_1d(model, a_feat, window_len=150, stride=50, device=device)
        y_true = gt_dict[v]
        prob_video_grid = resample_1d(prob_audio, len(y_true))

        all_y_true.extend(y_true)
        all_y_prob.extend(prob_video_grid)
        per_vid[v] = prob_video_grid

    y_true_np = np.array(all_y_true)
    y_prob_np = np.array(all_y_prob)
    y_pred_np = (y_prob_np >= 0.5).astype(np.float32)
    m = compute_metrics(y_true_np, y_pred_np, y_prob_np)
    return m, per_vid


def evaluate_face_tcn(video_list, gt_dict, base_dir, device):
    ckpt_path = base_dir / "scripts" / "best_face_tcn.pth"
    if not ckpt_path.exists():
        print("  [WARN] best_face_tcn.pth not found, skipping visual ablation")
        return None, {}

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    in_dim = ckpt.get('input_dim', 42)
    model = StutterTCN(input_dim=in_dim, hidden_dim=128, num_blocks=3, kernel_size=5, dropout=0.0).to(device)
    state_dict = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()

    v_dir = base_dir / "data" / "russian_face_kinematics"
    all_y_true, all_y_prob = [], []
    per_vid = {}

    for v in video_list:
        v_feat = np.load(str(v_dir / f"{v}.npy"))
        prob = predict_sliding_window_1d(model, v_feat, window_len=45, stride=15, device=device)
        y_true = gt_dict[v]

        all_y_true.extend(y_true)
        all_y_prob.extend(prob)
        per_vid[v] = prob

    y_true_np = np.array(all_y_true)
    y_prob_np = np.array(all_y_prob)
    y_pred_np = (y_prob_np >= 0.5).astype(np.float32)
    m = compute_metrics(y_true_np, y_pred_np, y_prob_np)
    return m, per_vid


def evaluate_deep_fusion(encoder_name, video_list, gt_dict, base_dir, device):
    ckpt_path = base_dir / "scripts" / f"best_deep_fusion_{encoder_name}.pth"
    if not ckpt_path.exists():
        print(f"  [WARN] {ckpt_path.name} not found, skipping Deep Fusion")
        return None, {}

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    a_dim = ckpt.get('audio_dim', 768 if encoder_name in ['wavlm', 'whisper'] else 1024)
    v_dim = ckpt.get('video_dim', 42)
    h_dim = ckpt.get('hidden_dim', 256)

    model = DeepFusionTCN(audio_dim=a_dim, video_dim=v_dim, hidden_dim=h_dim, modality_dropout=0.0).to(device)
    state_dict = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()

    a_dir = base_dir / "data" / f"russian_seq_{encoder_name}"
    v_dir = base_dir / "data" / "russian_face_kinematics"
    all_y_true, all_y_prob = [], []
    per_vid = {}

    for v in video_list:
        a_feat = np.load(str(a_dir / f"{v}.npy"))
        v_feat = np.load(str(v_dir / f"{v}.npy"))
        y_true = gt_dict[v]

        t_target = max(a_feat.shape[0], v_feat.shape[0])
        a_feat_rs = resample_2d(a_feat, t_target)
        v_feat_rs = resample_2d(v_feat, t_target)

        prob_fused = predict_sliding_window_fusion(model, a_feat_rs, v_feat_rs, window_len=150, stride=50, device=device)
        prob_video_grid = resample_1d(prob_fused, len(y_true))

        all_y_true.extend(y_true)
        all_y_prob.extend(prob_video_grid)
        per_vid[v] = prob_video_grid

    y_true_np = np.array(all_y_true)
    y_prob_np = np.array(all_y_prob)
    y_pred_np = (y_prob_np >= 0.5).astype(np.float32)
    m = compute_metrics(y_true_np, y_pred_np, y_prob_np)
    return m, per_vid


def plot_timelines(video_list, gt_dict, per_vid_whisper_df, per_vid_whisper_tcn, per_vid_hybrid, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    for v in video_list:
        gt = gt_dict[v]
        p_df = per_vid_whisper_df.get(v, None)
        p_tcn = per_vid_whisper_tcn.get(v, None)
        p_hyb = per_vid_hybrid.get(v, None)

        time_axis = np.arange(len(gt)) / TARGET_FPS

        plt.figure(figsize=(12, 3.8))
        plt.fill_between(time_axis, 0, gt, color='green', alpha=0.25, label='Reference labeling (stuttering)')
        if p_tcn is not None:
            plt.plot(time_axis, p_tcn, color='gray', linestyle='--', linewidth=1.2, label='Audio TCN (Whisper)')
        if p_hyb is not None:
            plt.plot(time_axis, p_hyb, color='darkorange', linestyle='-.', linewidth=1.3, label='Hybrid TCN+MHA (Whisper)')
        if p_df is not None:
            plt.plot(time_axis, p_df, color='royalblue', linewidth=1.8, label='Deep Fusion (Whisper + Video)')

        plt.axhline(0.5, color='crimson', linestyle=':', label='Threshold (0.5)')
        plt.title(f"Frame detection of stuttering: {v}.mp4", fontsize=11, fontweight='bold')
        plt.xlabel("Time, seconds", fontsize=10)
        plt.ylabel("Probability P(stuttering)", fontsize=10)
        plt.ylim(-0.05, 1.05)
        plt.xlim(0, time_axis[-1])
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend(loc='upper right', framealpha=0.9, fontsize=9)
        plt.tight_layout()

        plot_file = out_dir / f"timeline_{v}.png"
        plt.savefig(plot_file, dpi=200)
        plt.close()
    print(f"  [OK] Saved {len(video_list)} timeline plots to: {out_dir}")


def main():
    base_dir = get_project_root()
    video_dir = base_dir / "data" / "russian_face_kinematics"
    labels_json = base_dir / "data" / "russian_labels.json"
    plots_dir = base_dir / "results" / "plots"

    with open(labels_json, 'r', encoding='utf-8') as f:
        annotations = json.load(f)
    video_list = sorted(list(annotations.keys()))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 70)
    print("  [BENCHMARK] Full Cross-Lingual & Multimodal Evaluation on Russian Data")
    print(f"  [BENCHMARK] Total videos: {len(video_list)}, Compute device: {device}")
    print("=" * 70)

    gt_dict = load_ground_truth(labels_json, video_list, video_dir)

    rows = []

    rows.append({
        'Category': 'Baseline', 'Model': 'Vocametrix (Audio Wav2Vec2)',
        'ROC-AUC': 0.7010, 'AP': None, 'F1': 0.4870, 'Precision': 0.3270, 'Recall': 0.9570, 'Specificity': 0.3000
    })
    rows.append({
        'Category': 'Baseline', 'Model': 'StutterLens (Audio)',
        'ROC-AUC': 0.6320, 'AP': None, 'F1': 0.2690, 'Precision': 0.3910, 'Recall': 0.2050, 'Specificity': 0.8860
    })
    rows.append({
        'Category': 'Baseline', 'Model': 'Lab Late Fusion (Vocametrix + FAU)',
        'ROC-AUC': 0.7100, 'AP': None, 'F1': 0.2820, 'Precision': 0.5040, 'Recall': 0.1950, 'Specificity': 0.9290
    })

    print("\n[EVAL 1/4] Baseline MLP (Wav2Vec2 Pooled)...")
    m_mlp, _ = evaluate_mlp(video_list, gt_dict, base_dir, device)
    if m_mlp:
        rows.append({
            'Category': 'Unimodal (Zero-Shot)', 'Model': 'Baseline MLP (Wav2Vec2 Pool)',
            'ROC-AUC': m_mlp['roc_auc'], 'AP': m_mlp['ap'], 'F1': m_mlp['f1'],
            'Precision': m_mlp['precision'], 'Recall': m_mlp['recall'], 'Specificity': m_mlp['specificity']
        })

    all_audio_preds = {}
    print("\n[EVAL 2/4] Audio TCNs in Zero-Shot mode...")
    for enc in ['wav2vec2', 'wavlm', 'whisper']:
        m, p = evaluate_audio_tcn(enc, video_list, gt_dict, base_dir, device)
        if m:
            all_audio_preds[enc] = p
            rows.append({
                'Category': 'Unimodal (Zero-Shot)', 'Model': f'Audio TCN ({enc})',
                'ROC-AUC': m['roc_auc'], 'AP': m['ap'], 'F1': m['f1'],
                'Precision': m['precision'], 'Recall': m['recall'], 'Specificity': m['specificity']
            })

    per_vid_hybrid = {}
    m_hyb, p_hyb = evaluate_hybrid_whisper(video_list, gt_dict, base_dir, device)
    if m_hyb:
        per_vid_hybrid = p_hyb
        rows.append({
            'Category': 'Unimodal (Zero-Shot)', 'Model': 'Hybrid TCN+MHA (Whisper)',
            'ROC-AUC': m_hyb['roc_auc'], 'AP': m_hyb['ap'], 'F1': m_hyb['f1'],
            'Precision': m_hyb['precision'], 'Recall': m_hyb['recall'], 'Specificity': m_hyb['specificity']
        })

    print("\n[EVAL 3/4] Visual Kinematics Ablation (Face TCN)...")
    m_face, _ = evaluate_face_tcn(video_list, gt_dict, base_dir, device)
    if m_face:
        rows.append({
            'Category': 'Visual Ablation', 'Model': "Face TCN (42 kinametics' pairs)",
            'ROC-AUC': m_face['roc_auc'], 'AP': m_face['ap'], 'F1': m_face['f1'],
            'Precision': m_face['precision'], 'Recall': m_face['recall'], 'Specificity': m_face['specificity']
        })

    all_fusion_preds = {}
    print("\n[EVAL 4/4] Multimodal Deep Fusion (Audio + Face Kinematics)...")
    for enc in ['wav2vec2', 'wavlm', 'whisper']:
        m, p = evaluate_deep_fusion(enc, video_list, gt_dict, base_dir, device)
        if m:
            all_fusion_preds[enc] = p
            rows.append({
                'Category': 'Multimodal (Early Fusion)', 'Model': f'Deep Fusion ({enc} + Video)',
                'ROC-AUC': m['roc_auc'], 'AP': m['ap'], 'F1': m['f1'],
                'Precision': m['precision'], 'Recall': m['recall'], 'Specificity': m['specificity']
            })

    p_df_wh = all_fusion_preds.get('whisper', {})
    p_tcn_wh = all_audio_preds.get('whisper', {})
    if p_df_wh or p_tcn_wh:
        print("\n[PLOTS] Generating timeline detection plots...")
        plot_timelines(video_list, gt_dict, p_df_wh, p_tcn_wh, per_vid_hybrid, plots_dir)

    df = pd.DataFrame(rows)
    csv_path = base_dir / "scripts" / "crosslingual_results.csv"
    df.to_csv(csv_path, index=False, encoding='utf-8')

    table_data = []
    for r in rows:
        table_data.append([
            r['Category'], r['Model'],
            f"{r['ROC-AUC']:.4f}" if r['ROC-AUC'] is not None else "---",
            f"{r['AP']:.4f}" if r['AP'] is not None else "---",
            f"{r['F1']:.4f}" if r['F1'] is not None else "---",
            f"{r['Precision']:.4f}" if r['Precision'] is not None else "---",
            f"{r['Recall']:.4f}" if r['Recall'] is not None else "---",
            f"{r['Specificity']:.4f}" if r['Specificity'] is not None else "---",
        ])

    headers = ["Category", "Model", "ROC-AUC", "AP", "F1", "Precision", "Recall", "Specificity"]
    print("\n" + tabulate(table_data, headers=headers, tablefmt="grid"))
    print(f"\n[DONE] Full benchmark complete! Results saved to: {csv_path}")


if __name__ == "__main__":
    main()