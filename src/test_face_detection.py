import cv2
from facenet_pytorch import MTCNN


image_path = "data/test/test_face.jpg"
output_path = "data/test/detected_face.jpg"


# Create face detector
mtcnn = MTCNN(
    image_size=224,
    margin=20,
    keep_all=False
)


# Read image
image = cv2.imread(image_path)

if image is None:
    print("Could not read image.")
    exit()


# Convert BGR → RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# Detect face
face = mtcnn(image_rgb)


if face is None:
    print("No face detected.")
else:

    # Convert tensor → image
    face = face.permute(1, 2, 0).byte().numpy()

    # RGB → BGR
    face = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)

    cv2.imwrite(output_path, face)

    print("Face detected successfully!")
    print("Saved:", output_path)