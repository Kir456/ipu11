import os
import torch
import pandas as pd
import numpy as np
import soundfile as sf
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
from pathlib import Path
from tqdm import tqdm

def extract_embeddings():
    script_dir = Path(__file__).resolve().parent
    base_dir = script_dir.parent
    audio_dir = base_dir / "data" / "sep28k_audio"
    csv_path = audio_dir / "metadata.csv"
    output_dir = base_dir / "data" / "sep28k_embeddings"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not csv_path.exists():
        print(f"File {csv_path} not found!")
        return

    df = pd.read_csv(csv_path)
    
    print("Loading Wav2Vec2-XLSR-53 model...")
    model_name = "facebook/wav2vec2-large-xlsr-53"
    
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
    model = Wav2Vec2Model.from_pretrained(model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    print(f"Model loaded on device: {device}")
    print(f"Starting feature extraction for {len(df)} files...")

    with torch.no_grad():
        for index, row in tqdm(df.iterrows(), total=len(df)):
            file_name = row['file_name']
            file_path = audio_dir / file_name
            
            output_file = output_dir / file_name.replace(".wav", ".npy")
            if output_file.exists():
                continue
            
            if not file_path.exists():
                continue

            speech_array, sr = sf.read(str(file_path))

            inputs = feature_extractor(
                speech_array, 
                sampling_rate=16000, 
                return_tensors="pt", 
                padding=True
            )
            
            inputs = {key: val.to(device) for key, val in inputs.items()}

            outputs = model(**inputs)
            
            hidden_states = outputs.last_hidden_state
            embedding = hidden_states.mean(dim=1).squeeze().cpu().numpy()
            
            np.save(str(output_file), embedding)
            
    print(f"\nAll embeddings saved to: {output_dir}")

if __name__ == "__main__":
    extract_embeddings()