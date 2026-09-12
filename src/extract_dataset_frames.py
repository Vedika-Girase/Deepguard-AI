import cv2
from pathlib import Path


# Video locations
REAL_DIR = Path("data/faceforensics/original_sequences/youtube/c23/videos")
FAKE_DIR = Path("data/faceforensics/manipulated_sequences/Deepfakes/c23/videos")

# Output locations
REAL_OUTPUT = Path("data/processed/real_frames")
FAKE_OUTPUT = Path("data/processed/deepfake_frames")

# Extract one frame every N frames
FRAME_INTERVAL = 10


def extract_frames(video_dir, output_dir, label):
    output_dir.mkdir(parents=True, exist_ok=True)

    videos = list(video_dir.glob("*.mp4"))

    print(f"\n{label} videos found: {len(videos)}")

    total_frames = 0

    for video in videos:

        print(f"Processing: {video.name}")

        cap = cv2.VideoCapture(str(video))

        frame_number = 0
        saved = 0

        while True:

            success, frame = cap.read()

            if not success:
                break

            if frame_number % FRAME_INTERVAL == 0:

                filename = (
                    f"{video.stem}_frame_{saved:04d}.jpg"
                )

                output_path = output_dir / filename

                cv2.imwrite(
                    str(output_path),
                    frame
                )

                saved += 1
                total_frames += 1

            frame_number += 1

        cap.release()

        print(f"  Saved {saved} frames")

    print(f"\nTotal {label} frames: {total_frames}")


# Extract REAL frames
extract_frames(
    REAL_DIR,
    REAL_OUTPUT,
    "REAL"
)

# Extract DEEPFAKE frames
extract_frames(
    FAKE_DIR,
    FAKE_OUTPUT,
    "DEEPFAKE"
)

print("\nFrame extraction completed!")