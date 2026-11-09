"""
Auto-inspector & pre-labeler for Russian stuttering videos using system FFmpeg.
Run: python scripts/auto_inspect_videos.py
"""

import json
import subprocess
import cv2
import torch
import numpy as np
import soundfile as sf
import mediapipe as mp
from pathlib import Path
from transformers import WhisperFeatureExtractor, WhisperModel

from train_tcn import StutterTCN
from utils import get_project_root


def extract_wav_system_ffmpeg(video_files, audio_dir: Path):
    audio_dir.mkdir(parents=True, exist_ok=True)
    print("\n[1/3] Extracting 16kHz mono WAV via FFmpeg...")
    for vf in video_files:
        out_wav = audio_dir / f"{vf.stem}.wav"
        cmd = [
            "ffmpeg", "-y", "-i", str(vf),
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            str(out_wav)
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"  [OK] {out_wav.name}")


def check_face_visibility(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    step = max(1, int(round(fps / 5)))

    mp_face = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)
    sec_status = {}
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            sec = int(idx / fps)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            has_face = mp_face.process(rgb).multi_face_landmarks is not None
            sec_status.setdefault(sec, []).append(has_face)
        idx += 1
    cap.release()
    mp_face.close()

    bad_seconds = [s for s, vals in sec_status.items() if sum(vals) / len(vals) < 0.5]
    face_rate = 100.0 * (1.0 - len(bad_seconds) / max(len(sec_status), 1))
    return duration, face_rate, bad_seconds


def suggest_stutter_intervals(wav_path: Path, fe, whisper_model, tcn_model, device):
    speech, sr = sf.read(str(wav_path))
    win_samples = int(3.0 * sr)
    hop_samples = int(0.5 * sr)
    duration_s = len(speech) / sr

    sec_probs = np.zeros(int(np.ceil(duration_s * 10)))
    sec_counts = np.zeros_like(sec_probs)

    with torch.no_grad():
        for start in range(0, max(1, len(speech) - int(0.5 * sr)), hop_samples):
            chunk = speech[start:start + win_samples]
            inputs = fe(
                chunk, sampling_rate=16000,
                return_tensors="pt", padding="max_length", max_length=16000 * 30
            )
            hidden = whisper_model.encoder(inputs.input_features.to(device)).last_hidden_state
            real_frames = min(int(np.ceil(len(chunk) / 320)), 150)
            emb = hidden[:, :real_frames, :]
            if real_frames < 150:
                pad = torch.zeros((1, 150 - real_frames, 768), device=device)
                emb = torch.cat([emb, pad], dim=1)
            mask = torch.zeros((1, 150), device=device)
            mask[:, :real_frames] = 1.0

            prob = torch.sigmoid(tcn_model(emb, mask)).item()
            t0 = int((start / sr) * 10)
            t1 = min(len(sec_probs), int(((start + len(chunk)) / sr) * 10))
            sec_probs[t0:t1] += prob
            sec_counts[t0:t1] += 1

    avg_probs = sec_probs / np.maximum(sec_counts, 1)
    is_stutter = avg_probs >= 0.55

    intervals = []
    in_seg = False
    for i, val in enumerate(is_stutter):
        if val and not in_seg:
            s_start = i / 10.0
            in_seg = True
        elif not val and in_seg:
            if (i / 10.0) - s_start >= 0.5:
                intervals.append([round(s_start, 1), round(i / 10.0, 1)])
            in_seg = False
    if in_seg and (len(is_stutter) / 10.0 - s_start >= 0.5):
        intervals.append([round(s_start, 1), round(len(is_stutter) / 10.0, 1)])

    return intervals


def main():
    base_dir = get_project_root()
    video_dir = base_dir / "data" / "russian_videos"
    audio_dir = base_dir / "data" / "russian_audio"

    video_files = sorted(video_dir.glob("ru_*.mp4"))
    if not video_files:
        print(f"[ERROR] No ru_*.mp4 files found in {video_dir}")
        return

    extract_wav_system_ffmpeg(video_files, audio_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[2/3] Loading Whisper-small & best_tcn_whisper.pth on {device}...")
    fe = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
    whisper_model = WhisperModel.from_pretrained("openai/whisper-small").to(device).eval()

    ckpt = torch.load(base_dir / "scripts" / "best_tcn_whisper.pth", map_location=device, weights_only=False)
    tcn_model = StutterTCN(input_dim=768, hidden_dim=256, num_blocks=4, kernel_size=5, dropout=0.0).to(device)
    tcn_model.load_state_dict(ckpt["model_state_dict"])
    tcn_model.eval()

    print("\n[3/3] Inspecting videos & detecting stuttering intervals:\n" + "=" * 70)
    suggested_seconds = {}

    for vf in video_files:
        dur, face_rate, bad_secs = check_face_visibility(vf)
        wav_path = audio_dir / f"{vf.stem}.wav"
        intervals = suggest_stutter_intervals(wav_path, fe, whisper_model, tcn_model, device)
        suggested_seconds[vf.stem] = intervals

        trim_note = "OK (no trim needed)" if not bad_secs else f"No face at sec: {bad_secs}"
        print(f"Video: {vf.name} | Duration: {dur:.1f}s | Face visible: {face_rate:.1f}% ({trim_note})")
        print(f"  Suggested stutter intervals (sec): {intervals}\n")

    print("=" * 70)
    print("ANNOTATIONS_SECONDS =", json.dumps(suggested_seconds, indent=4))


if __name__ == "__main__":
    main()