import os
import io
import pandas as pd
import soundfile as sf
import librosa
from datasets import load_dataset, Audio
from pathlib import Path
from dotenv import load_dotenv

def download_and_prepare_sep28k():
    script_dir = Path(__file__).resolve().parent
    base_dir = script_dir.parent
    
    cache_dir = base_dir / "hf_cache"
    output_dir = base_dir / "data" / "sep28k_audio"
    
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    load_dotenv()
    
    hf_token = os.getenv('HF_TOKEN')

    if not hf_token:
        raise ValueError('Token error')
    
    print(f"Project root folder: {base_dir}")
    print(f"Audio files will be saved to: {output_dir}")
    print("Starting SEP-28k dataset download...")
    
    dataset = load_dataset(
        "saeedzou/sep28k-fluencybank-stutter-dataset", 
        split="train",
        cache_dir=str(cache_dir),
        token=hf_token
    )
    
    dataset = dataset.cast_column("audio", Audio(decode=False))
    
    metadata = []
    
    for i, item in enumerate(dataset):
        audio_bytes = item['audio']['bytes']

        audio_array, orig_sr = sf.read(io.BytesIO(audio_bytes))

        if len(audio_array.shape) > 1:
            audio_array = audio_array.mean(axis=1)
            
        if orig_sr != 16000:
            audio_array = librosa.resample(y=audio_array, orig_sr=orig_sr, target_sr=16000)
        
        file_name = f"clip_{i:05d}.wav"
        file_path = output_dir / file_name
        
        sf.write(str(file_path), audio_array, 16000)
        
        row = {
            'file_name': file_name,
            'prolongation': item.get('Prolongation', 0),
            'block': item.get('Block', 0),
            'sound_repetition': item.get('SoundRep', 0),
            'word_repetition': item.get('WordRep', 0),
            'interjection': item.get('Interjection', 0),
            'is_stutter': 1 if any([
                item.get('Prolongation', 0) > 0,
                item.get('Block', 0) > 0,
                item.get('SoundRep', 0) > 0,
                item.get('WordRep', 0) > 0
            ]) else 0
        }
        metadata.append(row)
        
        if (i + 1) % 1000 == 0:
            print(f"Processed {i + 1} files...")
            
    df = pd.DataFrame(metadata)
    df.to_csv(output_dir / "metadata.csv", index=False)
    
    print(f"Saved {len(df)} files. Metadata written to {output_dir / 'metadata.csv'}")
    print(f"\nYou can delete the cache folder: {cache_dir}")
    print(f"Only the following folder will be used for training: {output_dir}")

if __name__ == "__main__":
    download_and_prepare_sep28k()