import cv2
from pathlib import Path


def extract_frames(video_path, output_dir, frame_interval=5):

    video = cv2.VideoCapture(str(video_path))

    if not video.isOpened():
        print(f"Could not open video: {video_path}")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_number = 0
    saved_frames = 0

    while True:

        success, frame = video.read()

        if not success:
            break

        # Save every 5th frame
        if frame_number % frame_interval == 0:

            frame_path = output_dir / f"frame_{saved_frames:05d}.jpg"

            cv2.imwrite(
                str(frame_path),
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, 95]
            )

            saved_frames += 1

        frame_number += 1

    video.release()

    print(f"Total video frames: {frame_number}")
    print(f"Frames saved: {saved_frames}")


if __name__ == "__main__":

    video_path = "data/raw/sample.mp4"
    output_dir = "data/processed/sample_frames"

    extract_frames(
        video_path,
        output_dir,
        frame_interval=5
    )