"""
Step 3: Extract facial kinematic features from video via MediaPipe Face Mesh.
Run: python scripts/extract_face_kinematics.py --video_dir <path> --output_dir <path>
"""

import argparse
import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from tqdm import tqdm

UPPER_LIP_INNER = [82, 13, 312]
LOWER_LIP_INNER = [87, 14, 317]
LEFT_LIP_CORNER = 61
RIGHT_LIP_CORNER = 291
CHIN = 152
LEFT_JAW = 234
RIGHT_JAW = 454
LEFT_BROW_INNER = 70
RIGHT_BROW_INNER = 300
NOSE_TIP = 1
NOSE_BRIDGE = 6
LEFT_EYE_TOP = 159
LEFT_EYE_BOTTOM = 145
RIGHT_EYE_TOP = 386
RIGHT_EYE_BOTTOM = 374


def euclidean(p1, p2):
    return np.sqrt(np.sum((np.array(p1) - np.array(p2)) ** 2))


def get_landmark_coords(landmarks, idx, w=1.0, h=1.0):
    lm = landmarks[idx]
    return np.array([lm.x * w, lm.y * h, lm.z * w])


def compute_frame_features(landmarks, w, h):
    def lm(idx):
        return get_landmark_coords(landmarks, idx, w, h)

    face_scale = euclidean(lm(NOSE_BRIDGE), lm(NOSE_TIP))
    if face_scale < 1e-6:
        face_scale = 1.0

    upper_lip_pts = [lm(i) for i in UPPER_LIP_INNER]
    lower_lip_pts = [lm(i) for i in LOWER_LIP_INNER]
    vertical_dist = np.mean([euclidean(u, l) for u, l in zip(upper_lip_pts, lower_lip_pts)])
    horizontal_dist = euclidean(lm(LEFT_LIP_CORNER), lm(RIGHT_LIP_CORNER))
    mar = vertical_dist / max(horizontal_dist, 1e-6)

    lip_spread = horizontal_dist / face_scale
    jaw_disp = lm(CHIN)[1] / face_scale
    brow_dist = euclidean(lm(LEFT_BROW_INNER), lm(RIGHT_BROW_INNER)) / face_scale

    left_brow_h = (lm(LEFT_BROW_INNER)[1] - lm(LEFT_EYE_TOP)[1]) / face_scale
    right_brow_h = (lm(RIGHT_BROW_INNER)[1] - lm(RIGHT_EYE_TOP)[1]) / face_scale
    avg_brow_height = (left_brow_h + right_brow_h) / 2

    left_eye_ap = euclidean(lm(LEFT_EYE_TOP), lm(LEFT_EYE_BOTTOM)) / face_scale
    right_eye_ap = euclidean(lm(RIGHT_EYE_TOP), lm(RIGHT_EYE_BOTTOM)) / face_scale
    avg_eye_aperture = (left_eye_ap + right_eye_ap) / 2

    left_jaw_dist = euclidean(lm(LEFT_JAW), lm(CHIN))
    right_jaw_dist = euclidean(lm(RIGHT_JAW), lm(CHIN))
    jaw_asymmetry = abs(left_jaw_dist - right_jaw_dist) / face_scale

    return {
        'mar': mar,
        'lip_spread': lip_spread,
        'jaw_disp': jaw_disp,
        'brow_tension': brow_dist,
        'brow_height': avg_brow_height,
        'eye_aperture': avg_eye_aperture,
        'jaw_asymmetry': jaw_asymmetry,
    }


def compute_temporal_derivatives(features_array, fps):
    T, N = features_array.shape
    dt = 1.0 / fps

    velocity = np.zeros_like(features_array)
    velocity[1:] = (features_array[1:] - features_array[:-1]) / dt

    acceleration = np.zeros_like(features_array)
    acceleration[1:] = (velocity[1:] - velocity[:-1]) / dt

    abs_velocity = np.abs(velocity)

    window_size = max(int(0.3 * fps), 1)
    rolling_energy = np.zeros_like(features_array)
    for i in range(T):
        start = max(0, i - window_size // 2)
        end = min(T, i + window_size // 2 + 1)
        rolling_energy[i] = np.mean(abs_velocity[start:end], axis=0)

    tremor_window = max(int(0.3 * fps), 3)
    tremor = np.zeros_like(features_array)
    for i in range(T):
        start = max(0, i - tremor_window // 2)
        end = min(T, i + tremor_window // 2 + 1)
        if end - start > 1:
            tremor[i] = np.std(velocity[start:end], axis=0)

    return np.concatenate([
        features_array, velocity, acceleration, abs_velocity, rolling_energy, tremor
    ], axis=1)


def extract_video_features(video_path, target_fps=15):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None, 0

    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    if actual_fps <= 0:
        actual_fps = 30.0

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_skip = max(1, int(round(actual_fps / target_fps)))

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    frame_features = []
    frame_idx = 0
    missing_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_skip != 0:
            frame_idx += 1
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            feats = compute_frame_features(landmarks, w, h)
            frame_features.append(list(feats.values()))
        else:
            frame_features.append([np.nan] * 7)
            missing_frames += 1

        frame_idx += 1

    cap.release()
    face_mesh.close()

    if len(frame_features) == 0:
        return None, actual_fps

    features_array = np.array(frame_features, dtype=np.float64)

    for col in range(features_array.shape[1]):
        mask = np.isnan(features_array[:, col])
        if mask.any() and not mask.all():
            valid_idx = np.where(~mask)[0]
            features_array[mask, col] = np.interp(
                np.where(mask)[0], valid_idx, features_array[valid_idx, col]
            )
        elif mask.all():
            features_array[:, col] = 0.0

    for col in range(features_array.shape[1]):
        median = np.median(features_array[:, col])
        q75, q25 = np.percentile(features_array[:, col], [75, 25])
        iqr = q75 - q25
        if iqr < 1e-8:
            iqr = np.std(features_array[:, col])
        if iqr < 1e-8:
            iqr = 1.0
        features_array[:, col] = np.clip((features_array[:, col] - median) / iqr, -6, 6)

    effective_fps = actual_fps / frame_skip
    full_features = compute_temporal_derivatives(features_array, effective_fps)

    if missing_frames > 0:
        total = len(frame_features)
        pct = missing_frames / total * 100
        if pct > 30:
            tqdm.write(f"  [WARN] {video_path.name}: {pct:.0f}% frames without detected face")

    return full_features, effective_fps


FEATURE_NAMES_BASE = [
    'mar', 'lip_spread', 'jaw_disp', 'brow_tension',
    'brow_height', 'eye_aperture', 'jaw_asymmetry'
]
DERIVATIVE_SUFFIXES = ['_raw', '_vel', '_acc', '_abs_vel', '_roll_energy', '_tremor']


def get_all_feature_names():
    return [base + suffix for suffix in DERIVATIVE_SUFFIXES for base in FEATURE_NAMES_BASE]


def main():
    parser = argparse.ArgumentParser(description="Extract facial kinematics features from video files")
    parser.add_argument('--video_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--target_fps', type=int, default=15)
    parser.add_argument('--extensions', type=str, default='.mp4,.avi,.mov,.mkv')
    args = parser.parse_args()

    video_dir = Path(args.video_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    extensions = tuple(args.extensions.split(','))
    video_files = sorted([f for f in video_dir.iterdir() if f.suffix.lower() in extensions])

    print(f"\n{'='*60}")
    print(f"  [INFO] Facial Kinematics Extraction")
    print(f"  [INFO] Video directory: {video_dir} ({len(video_files)} files)")
    print(f"  [INFO] Target FPS: {args.target_fps}")
    print(f"  [INFO] Features per frame: {len(get_all_feature_names())}")
    print(f"{'='*60}\n")

    processed = 0
    for vf in tqdm(video_files, desc="Processing videos", colour='green'):
        npy_path = (output_dir / vf.stem).with_suffix('.npy')
        if npy_path.exists():
            processed += 1
            continue

        features, _ = extract_video_features(vf, args.target_fps)
        if features is not None:
            np.save(str(npy_path), features.astype(np.float32))
            processed += 1
            tqdm.write(f"  [OK] {vf.name}: {features.shape[0]} frames x {features.shape[1]} features")
        else:
            tqdm.write(f"  [ERROR] {vf.name}: failed to process")

    print(f"\n[DONE] Processed: {processed}/{len(video_files)}")
    print(f"[INFO] Saved to: {output_dir}")


if __name__ == "__main__":
    main()