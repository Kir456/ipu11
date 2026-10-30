"""
Step 2: Extract frame-level embeddings from WavLM-Base-Plus.
WavLM was trained considering paralinguistics and noise, providing
robust features for disfluency detection.

Output: data/sep28k_seq_wavlm/<clip_name>.npy — array shape (T, 768)
"""

import torch
import pandas as pd
import numpy as np
import soundfile as sf
from transformers import WavLMModel, Wav2Vec2FeatureExtractor
from pathlib import Path
from tqdm import tqdm


def extract_wavlm_embeddings():
    script_dir = Path(__file__).resolve().parent
    base_dir = script_dir.parent
    audio_dir = base_dir / "data" / "sep28k_audio"
    csv_path = audio_dir / "metadata.csv"
    output_dir = base_dir / "data" / "sep28k_seq_wavlm"

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

    print("[LOAD] Loading WavLM-Base-Plus...")
    model_name = "microsoft/wavlm-base-plus"

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
    model = WavLMModel.from_pretrained(model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    for param in model.parameters():
        param.requires_grad = False

    print(f"[DEVICE] {device}")
    print(f"[START] Starting WavLM sequential extraction for {remaining} files...\n")

    processed = 0
    errors = 0

    with torch.no_grad():
        pbar = tqdm(df.iterrows(), total=len(df), desc="WavLM Extraction", colour='blue')
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
                    padding=True
                )
                inputs = {key: val.to(device) for key, val in inputs.items()}

                outputs = model(**inputs)
                hidden_states = outputs.last_hidden_state  # (1, T, 768)

                seq_embedding = hidden_states.squeeze(0).cpu().numpy()
                np.save(str(output_file), seq_embedding)

                processed += 1

            except Exception as e:
                errors += 1
                tqdm.write(f"[WARN] Error processing {file_name}: {e}")

            pbar.set_postfix(done=processed, err=errors)

    print(f"\n{'='*50}")
    print(f"[DONE] WavLM extraction complete!")
    print(f"   Processed: {processed}")
    print(f"   Errors: {errors}")
    print(f"   Saved to: {output_dir}")
    print(f"{'='*50}")


if __name__ == "__main__":
    extract_wavlm_embeddings()