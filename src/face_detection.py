import cv2
from pathlib import Path
from facenet_pytorch import MTCNN


# Create face detector
mtcnn = MTCNN(
    image_size=224,
    margin=20,
    keep_all=False
)


def extract_face(image_path, output_path):
    image = cv2.imread(str(image_path))

    if image is None:
        
        print(f"Could not read: {image_path}")
        return False

    # OpenCV uses BGR, MTCNN expects RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    face = mtcnn(image_rgb)

    if face is None:
        print(f"No face found: {image_path}")
        return False

    # Convert PyTorch tensor to NumPy image
    face = face.permute(1, 2, 0).byte().numpy()

    # RGB → BGR for OpenCV
    face = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(output_path), face)

    return True


if __name__ == "__main__":

    input_dir = Path("data/processed/sample_frames")
    output_dir = Path("data/processed/sample_faces")

    output_dir.mkdir(parents=True, exist_ok=True)

    count = 0

    for image_path in input_dir.glob("*.jpg"):

        output_path = output_dir / image_path.name

        if extract_face(image_path, output_path):
            count += 1

    print(f"\nFaces extracted: {count}")