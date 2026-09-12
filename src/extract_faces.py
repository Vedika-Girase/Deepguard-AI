from pathlib import Path
from PIL import Image
from facenet_pytorch import MTCNN


# Input folders
REAL_INPUT = Path("data/processed/real_frames")
FAKE_INPUT = Path("data/processed/deepfake_frames")

# Output folders
REAL_OUTPUT = Path("data/processed/real_faces")
FAKE_OUTPUT = Path("data/processed/deepfake_faces")


# Create output folders
REAL_OUTPUT.mkdir(parents=True, exist_ok=True)
FAKE_OUTPUT.mkdir(parents=True, exist_ok=True)


# Face detector
mtcnn = MTCNN(
    image_size=224,
    margin=20,
    min_face_size=40,
    keep_all=False,
    post_process=False
)


def extract_faces(input_dir, output_dir, label):

    images = list(input_dir.glob("*.jpg"))

    print(f"\nProcessing {label} images: {len(images)}")

    detected = 0
    not_detected = 0

    for i, image_path in enumerate(images):

        try:
            image = Image.open(image_path).convert("RGB")

            face = mtcnn(image)

            if face is not None:

                # Convert tensor to PIL image
                face = face.permute(1, 2, 0).byte().numpy()
                face_image = Image.fromarray(face)

                output_path = output_dir / image_path.name

                face_image.save(output_path)

                detected += 1

            else:
                not_detected += 1

        except Exception as e:
            print(f"Error processing {image_path.name}: {e}")

        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{len(images)}")


    print(f"\n{label} completed")
    print(f"Faces detected: {detected}")
    print(f"Faces not detected: {not_detected}")


# Process both classes
extract_faces(
    REAL_INPUT,
    REAL_OUTPUT,
    "REAL"
)

extract_faces(
    FAKE_INPUT,
    FAKE_OUTPUT,
    "DEEPFAKE"
)

print("\nFace extraction completed!")