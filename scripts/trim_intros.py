"""
Trim intros from Russian videos, re-extract audio, and shift stutter labels.
Run: python scripts/trim_intros.py
"""

import json
import shutil
import subprocess
from pathlib import Path

# Exact intro durations detected by MediaPipe (seconds to cut from start)
CUT_OFFSETS = {
    "ru_01": 6.0,
    "ru_02": 5.0,
    "ru_03": 6.0,
    "ru_04": 6.0,
    "ru_05": 5.0,
    "ru_06": 6.0,
}

# Raw detected intervals from inspect script
RAW_INTERVALS = {
    "ru_01": [[7.0, 12.0], [16.5, 27.0], [30.5, 41.0], [47.0, 49.0]],
    "ru_02": [[4.5, 9.0], [10.5, 45.5], [47.5, 78.0], [83.5, 120.5]],
    "ru_03": [[7.5, 24.5], [26.5, 29.5], [32.5, 47.0]],
    "ru_04": [[6.0, 28.0], [29.5, 36.5]],
    "ru_05": [[6.0, 12.0], [13.5, 46.5]],
    "ru_06": [[8.0, 10.5], [15.0, 18.0], [18.5, 22.5], [33.0, 42.0], [45.5, 50.0]],
}

TARGET_FPS = 15


def main():
    base_dir = Path(__file__).resolve().parent.parent
    video_dir = base_dir / "data" / "russian_videos"
    backup_dir = base_dir / "data" / "russian_videos_raw"
    audio_dir = base_dir / "data" / "russian_audio"
    labels_path = base_dir / "data" / "russian_labels.json"

    backup_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    print("\n[1/3] Backing up original videos to data/russian_videos_raw/...")
    for vf in sorted(video_dir.glob("ru_*.mp4")):
        backup_file = backup_dir / vf.name
        if not backup_file.exists():
            shutil.copy2(vf, backup_file)

    print("\n[2/3] Trimming intros via FFmpeg (video + 16kHz mono audio)...")
    for stem, offset in CUT_OFFSETS.items():
        src_video = backup_dir / f"{stem}.mp4"
        out_video = video_dir / f"{stem}.mp4"
        out_audio = audio_dir / f"{stem}.wav"

        # Frame-accurate cut with fast re-encoding
        cmd_video = [
            "ffmpeg", "-y", "-ss", str(offset), "-i", str(src_video),
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-c:a", "aac", str(out_video)
        ]
        subprocess.run(cmd_video, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # 16kHz mono WAV for transformer encoders
        cmd_audio = [
            "ffmpeg", "-y", "-ss", str(offset), "-i", str(src_video),
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            str(out_audio)
        ]
        subprocess.run(cmd_audio, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"  [OK] {stem}: cut {offset}s intro -> video & audio updated")

    print("\n[3/3] Synchronizing stutter intervals into frame indices...")
    shifted_seconds = {}
    frame_labels = {}

    for stem, intervals in RAW_INTERVALS.items():
        offset = CUT_OFFSETS[stem]
        shifted_list = []
        frame_list = []

        for start_s, end_s in intervals:
            new_start = max(0.0, round(start_s - offset, 2))
            new_end = max(new_start + 0.3, round(end_s - offset, 2))
            shifted_list.append([new_start, new_end])

            start_f = int(round(new_start * TARGET_FPS))
            end_f = int(round(new_end * TARGET_FPS))
            frame_list.append([start_f, end_f])

        shifted_seconds[stem] = shifted_list
        frame_labels[stem] = frame_list

    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(frame_labels, f, indent=2)

    print(f"  [OK] Labels written to: {labels_path}")
    print("\nShifted intervals in clean video timeline (seconds):")
    print(json.dumps(shifted_seconds, indent=4))
    print("\n[DONE] Video trimming and label synchronization complete!")


if __name__ == "__main__":
    main()