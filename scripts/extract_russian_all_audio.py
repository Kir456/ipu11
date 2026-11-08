"""
Extract frame-level embeddings for Russian audio across ALL 3 encoders:
1. wav2vec2-large-xlsr-53 -> data/russian_seq_wav2vec2/ (T, 1024)
2. wavlm-base-plus        -> data/russian_seq_wavlm/    (T, 768)
3. whisper-small          -> data/russian_seq_whisper/  (T, 768)

Run: python scripts/extract_russian_all_audio.py
"""

import torch
import numpy as np
import soundfile as sf
from pathlib import Path
from transformers import (
    Wav2Vec2FeatureExtractor, Wav2Vec2Model,
    WavLMModel,
    WhisperFeatureExtractor, WhisperModel
)

SAMPLE_RATE = 16000


def extract_wav2vec2(wav_files, out_dir, device):
    out_dir.mkdir(parents=True, exist_ok=True)
    print("\n[1/3] Loading Wav2Vec2-XLSR-53...")
    model_name = "facebook/wav2vec2-large-xlsr-53"
    fe = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
    model = Wav2Vec2Model.from_pretrained(model_name).to(device).eval()

    with torch.no_grad():
        for wav_path in wav_files:
            speech, _ = sf.read(str(wav_path))
            inputs = fe(speech, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            hidden = model(**inputs).last_hidden_state.squeeze(0).cpu().numpy()
            np.save(str(out_dir / f"{wav_path.stem}.npy"), hidden.astype(np.float32))
            print(f"  [Wav2Vec2] {wav_path.stem}: {hidden.shape}")

    del model
    torch.cuda.empty_cache()


def extract_wavlm(wav_files, out_dir, device):
    out_dir.mkdir(parents=True, exist_ok=True)
    print("\n[2/3] Loading WavLM-Base-Plus...")
    model_name = "microsoft/wavlm-base-plus"
    fe = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
    model = WavLMModel.from_pretrained(model_name).to(device).eval()

    with torch.no_grad():
        for wav_path in wav_files:
            speech, _ = sf.read(str(wav_path))
            inputs = fe(speech, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            hidden = model(**inputs).last_hidden_state.squeeze(0).cpu().numpy()
            np.save(str(out_dir / f"{wav_path.stem}.npy"), hidden.astype(np.float32))
            print(f"  [WavLM]    {wav_path.stem}: {hidden.shape}")

    del model
    torch.cuda.empty_cache()


def extract_whisper(wav_files, out_dir, device):
    out_dir.mkdir(parents=True, exist_ok=True)
    print("\n[3/3] Loading Whisper-Small encoder...")
    model_name = "openai/whisper-small"
    fe = WhisperFeatureExtractor.from_pretrained(model_name)
    model = WhisperModel.from_pretrained(model_name).to(device).eval()

    chunk_samples = SAMPLE_RATE * 30  # 30-second chunks for Whisper

    with torch.no_grad():
        for wav_path in wav_files:
            speech, _ = sf.read(str(wav_path))
            chunks_emb = []

            for start in range(0, len(speech), chunk_samples):
                chunk = speech[start:start + chunk_samples]
                inputs = fe(
                    chunk, sampling_rate=SAMPLE_RATE,
                    return_tensors="pt", padding="max_length", max_length=chunk_samples
                )
                input_features = inputs.input_features.to(device)
                hidden = model.encoder(input_features).last_hidden_state

                real_frames = min(int(np.ceil(len(chunk) / 320)), hidden.shape[1])
                chunks_emb.append(hidden[0, :real_frames, :].cpu().numpy())

            full_emb = np.concatenate(chunks_emb, axis=0)
            np.save(str(out_dir / f"{wav_path.stem}.npy"), full_emb.astype(np.float32))
            print(f"  [Whisper]  {wav_path.stem}: {full_emb.shape}")

    del model
    torch.cuda.empty_cache()


def main():
    base_dir = Path(__file__).resolve().parent.parent
    audio_dir = base_dir / "data" / "russian_audio"
    wav_files = sorted(list(audio_dir.glob("*.wav")))

    if not wav_files:
        print(f"[ERROR] No .wav files found in {audio_dir}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Found {len(wav_files)} Russian audio files. Device: {device}")

    extract_wav2vec2(wav_files, base_dir / "data" / "russian_seq_wav2vec2", device)
    extract_wavlm(wav_files, base_dir / "data" / "russian_seq_wavlm", device)
    extract_whisper(wav_files, base_dir / "data" / "russian_seq_whisper", device)

    print("\n[DONE] All 3 acoustic feature sets extracted successfully!")


if __name__ == "__main__":
    main()