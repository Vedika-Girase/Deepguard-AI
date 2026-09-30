import csv
import cv2
from pathlib import Path
from PIL import Image
from facenet_pytorch import MTCNN


# -----------------------------
# Configuration
# -----------------------------
INPUT_ROOT = Path("data/external/celebdf")
OUTPUT_ROOT = Path("data/external/celebdf_faces")

TARGET_FACES_PER_VIDEO = 10
MAX_SAMPLE_FRAMES = 40
IMAGE_SIZE = 224

# MTCNN used only for face detection/cropping.
mtcnn = MTCNN(
    image_size=IMAGE_SIZE,
    margin=0,
    min_face_size=40,
    thresholds=[0.6, 0.7, 0.7],
    factor=0.709,
    post_process=False,
    keep_all=True,
    device="cpu"
)


def extract_faces_from_video(video_path, label, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"[ERROR] Could not open: {video_path}")
        return {
            "label": label,
            "video": video_path.name,
            "frames_checked": 0,
            "faces_detected": 0,
            "faces_saved": 0,
            "status": "OPEN_FAILED"
        }

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        return {
            "label": label,
            "video": video_path.name,
            "frames_checked": 0,
            "faces_detected": 0,
            "faces_saved": 0,
            "status": "NO_FRAMES"
        }

    # Deterministic evenly-spaced frame selection.
    sample_count = min(MAX_SAMPLE_FRAMES, total_frames)

    if sample_count == 1:
        frame_indices = [0]
    else:
        frame_indices = [
            round(i * (total_frames - 1) / (sample_count - 1))
            for i in range(sample_count)
        ]

    saved = 0
    frames_checked = 0
    faces_detected = 0

    for frame_idx in frame_indices:

        if saved >= TARGET_FACES_PER_VIDEO:
            break

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        success, frame = cap.read()

        if not success:
            continue

        frames_checked += 1

        # OpenCV BGR -> RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)

        try:
            boxes, probs = mtcnn.detect(pil_image)
        except Exception as e:
            print(f"[WARNING] MTCNN failed on {video_path.name}, frame {frame_idx}: {e}")
            continue

        if boxes is None or probs is None:
            continue

        # Keep the highest-confidence detected face.
        valid_faces = [
            (box, float(prob))
            for box, prob in zip(boxes, probs)
            if prob is not None and float(prob) >= 0.90
        ]

        if not valid_faces:
            continue

        valid_faces.sort(key=lambda x: x[1], reverse=True)
        box, confidence = valid_faces[0]

        faces_detected += 1

        x1, y1, x2, y2 = [int(round(v)) for v in box]

        # Small padding around the face.
        width = x2 - x1
        height = y2 - y1

        pad_x = int(width * 0.15)
        pad_y = int(height * 0.15)

        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(pil_image.width, x2 + pad_x)
        y2 = min(pil_image.height, y2 + pad_y)

        if x2 <= x1 or y2 <= y1:
            continue

        face = pil_image.crop((x1, y1, x2, y2))
        face = face.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)

        output_name = (
            f"{video_path.stem}_frame{frame_idx:06d}"
            f"_face{saved + 1:02d}.jpg"
        )

        output_path = output_dir / output_name
        face.save(output_path, quality=95)

        saved += 1

    cap.release()

    status = "OK" if saved == TARGET_FACES_PER_VIDEO else "PARTIAL"

    print(
        f"[{label.upper():8}] {video_path.name:25} "
        f"checked={frames_checked:2d} "
        f"detected={faces_detected:2d} "
        f"saved={saved:2d} "
        f"{status}"
    )

    return {
        "label": label,
        "video": video_path.name,
        "frames_checked": frames_checked,
        "faces_detected": faces_detected,
        "faces_saved": saved,
        "status": status
    }


def main():
    real_dir = INPUT_ROOT / "real"
    deepfake_dir = INPUT_ROOT / "deepfake"

    real_output = OUTPUT_ROOT / "real"
    deepfake_output = OUTPUT_ROOT / "deepfake"

    real_output.mkdir(parents=True, exist_ok=True)
    deepfake_output.mkdir(parents=True, exist_ok=True)

    results = []

    print("\n==========================================")
    print(" DeepGuard - Celeb-DF Face Extraction")
    print("==========================================")
    print(f"Target faces/video : {TARGET_FACES_PER_VIDEO}")
    print(f"Max sampled frames : {MAX_SAMPLE_FRAMES}")
    print(f"Face image size    : {IMAGE_SIZE}x{IMAGE_SIZE}")
    print("Device             : CPU")
    print("==========================================\n")

    # Real videos
    for video_path in sorted(real_dir.glob("*.mp4")):
        results.append(
            extract_faces_from_video(
                video_path,
                "real",
                real_output
            )
        )

    # DeepFake videos
    for video_path in sorted(deepfake_dir.glob("*.mp4")):
        results.append(
            extract_faces_from_video(
                video_path,
                "deepfake",
                deepfake_output
            )
        )

    # Save manifest
    manifest_path = OUTPUT_ROOT / "manifest.csv"

    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "label",
                "video",
                "frames_checked",
                "faces_detected",
                "faces_saved",
                "status"
            ]
        )
        writer.writeheader()
        writer.writerows(results)

    total_saved = sum(r["faces_saved"] for r in results)
    real_saved = sum(
        r["faces_saved"] for r in results
        if r["label"] == "real"
    )
    deepfake_saved = sum(
        r["faces_saved"] for r in results
        if r["label"] == "deepfake"
    )

    print("\n==========================================")
    print(" Extraction complete")
    print("==========================================")
    print(f"Videos processed : {len(results)}")
    print(f"Real faces       : {real_saved}")
    print(f"DeepFake faces   : {deepfake_saved}")
    print(f"Total faces      : {total_saved}")
    print(f"Manifest         : {manifest_path}")
    print("==========================================\n")


if __name__ == "__main__":
    main()
