import json
from pathlib import Path

# Пример разметки в секундах: [начало_сек, конец_сек], видео еще не подготовлены
ANNOTATIONS_SECONDS = {
    "ru_01": [
        [2.4, 4.1],
        [11.5, 13.0]
    ],
    "ru_02": [
        [5.0, 7.8]
    ],
    "ru_03": [
        [1.2, 3.0],
        [8.4, 10.2],
        [15.1, 16.5]
    ],
}

def build_labels_json(target_fps=15):
    base_dir = Path(__file__).resolve().parent.parent
    labels_file = base_dir / "data" / "russian_labels.json"

    frame_annotations = {}
    for video_name, intervals in ANNOTATIONS_SECONDS.items():
        frame_intervals = []
        for start_sec, end_sec in intervals:
            start_frame = int(round(start_sec * target_fps))
            end_frame = int(round(end_sec * target_fps))
            frame_intervals.append([start_frame, end_frame])
        frame_annotations[video_name] = frame_intervals

    with open(labels_file, "w", encoding="utf-8") as f:
        json.dump(frame_annotations, f, indent=2)

    print(f"[OK] Russian labels saved to: {labels_file}")

if __name__ == "__main__":
    build_labels_json()