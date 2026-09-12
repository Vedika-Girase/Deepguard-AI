import cv2
from pathlib import Path
from facenet_pytorch import MTCNN


mtcnn = MTCNN(
    image_size=224,
    margin=20,
    keep_all=False
)


def process_folder(input_dir, output_dir):

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    detected = 0

    for image_path in input_dir.glob("*.jpg"):

        total += 1

        image = cv2.imread(str(image_path))

        if image is None:
            continue

        image_rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        face = mtcnn(image_rgb)

        if face is None:
            continue

        face = face.permute(
            1, 2, 0
        ).byte().numpy()

        face = cv2.cvtColor(
            face,
            cv2.COLOR_RGB2BGR
        )

        output_path = output_dir / image_path.name

        cv2.imwrite(
            str(output_path),
            face
        )

        detected += 1

    print(f"Total images: {total}")
    print(f"Faces detected: {detected}")


if __name__ == "__main__":

    process_folder(
        "data/processed/sample_frames",
        "data/faces/real"
    )