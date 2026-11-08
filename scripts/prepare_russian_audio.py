import subprocess
from pathlib import Path

def extract_audio():
    base_dir = Path(__file__).resolve().parent.parent
    video_dir = base_dir / "data" / "russian_videos"
    audio_dir = base_dir / "data" / "russian_audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    video_files = list(video_dir.glob("*.mp4")) + list(video_dir.glob("*.avi")) + list(video_dir.glob("*.mkv"))
    print(f"[INFO] Found {len(video_files)} video files.")

    for vf in video_files:
        out_wav = audio_dir / f"{vf.stem}.wav"
        cmd = [
            "ffmpeg", "-y", "-i", str(vf),
            "-vn", "-acodec", "pcm_s16le",
            "-ar", "16000", "-ac", "1",
            str(out_wav)
        ]
        res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if res.returncode == 0:
            print(f"  [OK] {out_wav.name}")
        else:
            print(f"  [WARN] ffmpeg error on {vf.name}, check ffmpeg installation.")

if __name__ == "__main__":
    extract_audio()