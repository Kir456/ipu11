"""
Step 2: Extract frame-level embeddings from Whisper encoder (small).
Whisper is a multilingual model trained on 680k hours of audio.
The encoder captures prosodic anomalies and rhythmic disruptions.

Output: data/sep28k_seq_whisper/<clip_name>.npy — array shape (T, 768)
"""

import torch
import pandas as pd
import numpy as np
import soundfile as sf
from transformers import WhisperModel, WhisperFeatureExtractor
from pathlib import Path
from tqdm import tqdm


def extract_whisper_embeddings():
    script_dir = Path(__file__).resolve().parent
    base_dir = script_dir.parent
    audio_dir = base_dir / "data" / "sep28k_audio"
    csv_path = audio_dir / "metadata.csv"
    output_dir = base_dir / "data" / "sep28k_seq_whisper"

    output_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        print(f"[ERROR] File {csv_path} not found!")
        return

    df = pd.read_csv(csv_path)

    already_done = sum(
        1 for _, row in df.iterrows()
        if (output_dir / row['file_name'].replace(".wav", ".npy")).exists()
    )
    remaining = len(df) - already_done
    print(f"[INFO] Total files: {len(df)}, already processed: {already_done}, remaining: {remaining}")

    if remaining == 0:
        print("[OK] All files have been processed!")
        return

    print("[LOAD] Loading Whisper-small encoder...")
    model_name = "openai/whisper-small"

    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
    model = WhisperModel.from_pretrained(model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    for param in model.parameters():
        param.requires_grad = False

    print(f"[DEVICE] {device}")
    print(f"[START] Starting Whisper sequential extraction for {remaining} files...\n")

    processed = 0
    errors = 0

    with torch.no_grad():
        pbar = tqdm(df.iterrows(), total=len(df), desc="Whisper Extraction", colour='yellow')
        for _, row in pbar:
            file_name = row['file_name']
            output_file = output_dir / file_name.replace(".wav", ".npy")

            if output_file.exists():
                continue

            file_path = audio_dir / file_name
            if not file_path.exists():
                errors += 1
                continue

            try:
                speech_array, sr = sf.read(str(file_path))

                inputs = feature_extractor(
                    speech_array,
                    sampling_rate=16000,
                    return_tensors="pt",
                    padding="max_length",
                    max_length=16000 * 30,
                )
                input_features = inputs.input_features.to(device)  # (1, 80, 3000)

                encoder_outputs = model.encoder(input_features)
                hidden_states = encoder_outputs.last_hidden_state  # (1, T, 768)

                real_duration_s = len(speech_array) / 16000
                real_frames = int(np.ceil(real_duration_s * 16000 / 320))
                real_frames = min(real_frames, hidden_states.shape[1])

                seq_embedding = hidden_states[0, :real_frames, :].cpu().numpy()
                np.save(str(output_file), seq_embedding)

                processed += 1

            except Exception as e:
                errors += 1
                tqdm.write(f"[WARN] Error processing {file_name}: {e}")

            pbar.set_postfix(done=processed, err=errors)

    print(f"\n{'='*50}")
    print(f"[DONE] Whisper extraction complete!")
    print(f"   Processed: {processed}")
    print(f"   Errors: {errors}")
    print(f"   Saved to: {output_dir}")
    print(f"{'='*50}")


if __name__ == "__main__":
    extract_whisper_embeddings()