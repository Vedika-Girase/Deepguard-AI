import sys
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
from facenet_pytorch import MTCNN

# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SRC_DIR = PROJECT_ROOT / "src"
MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "production"
    / "deepguard_resnet18.pth"
)

# Allow imports from src/
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models.resnet18_model import build_resnet18


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="DeepGuard",
    page_icon="🛡️",
    layout="centered"
)


# ---------------------------------------------------------
# Constants
# ---------------------------------------------------------

IMAGE_SIZE = 224
DEVICE = torch.device("cpu")

CLASS_NAMES = {
    0: "Deepfake",
    1: "Real"
}

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


# ---------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=MEAN,
        std=STD
    )
])


# ---------------------------------------------------------
# Load face detector
# ---------------------------------------------------------

@st.cache_resource
def load_face_detector():

    return MTCNN(
        image_size=IMAGE_SIZE,
        margin=0,
        min_face_size=40,
        thresholds=[0.6, 0.7, 0.7],
        factor=0.709,
        post_process=False,
        keep_all=True,
        device="cpu"
    )


# ---------------------------------------------------------
# Load trained model
# ---------------------------------------------------------

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found:\n{MODEL_PATH}"
        )

    model = build_resnet18(
        num_classes=2,
        pretrained=False
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]

        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]

        else:
            state_dict = checkpoint

    else:
        state_dict = checkpoint

    cleaned_state_dict = {}

    for key, value in state_dict.items():
        cleaned_state_dict[
            key.replace("module.", "", 1)
        ] = value

    model.load_state_dict(cleaned_state_dict)

    model.to(DEVICE)
    model.eval()

    return model


# ---------------------------------------------------------
# Detect and crop face
# ---------------------------------------------------------

def detect_face(image):

    detector = load_face_detector()

    image_rgb = image.convert("RGB")

    boxes, probabilities = detector.detect(
        image_rgb
    )

    if boxes is None or probabilities is None:
        return None, None

    valid_faces = []

    for box, probability in zip(boxes, probabilities):

        if probability is None:
            continue

        probability = float(probability)

        if probability >= 0.90:
            valid_faces.append(
                (box, probability)
            )

    if not valid_faces:
        return None, None

    # Highest-confidence face
    valid_faces.sort(
        key=lambda x: x[1],
        reverse=True
    )

    box, confidence = valid_faces[0]

    x1, y1, x2, y2 = [
        int(round(value))
        for value in box
    ]

    width = x2 - x1
    height = y2 - y1

    # Small padding around face
    pad_x = int(width * 0.15)
    pad_y = int(height * 0.15)

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)

    x2 = min(image_rgb.width, x2 + pad_x)
    y2 = min(image_rgb.height, y2 + pad_y)

    if x2 <= x1 or y2 <= y1:
        return None, None

    face = image_rgb.crop(
        (x1, y1, x2, y2)
    )

    return face, confidence


# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------

def predict(face):

    model = load_model()

    tensor = transform(face)
    tensor = tensor.unsqueeze(0).to(DEVICE)

    start_time = time.perf_counter()

    with torch.no_grad():

        output = model(tensor)

        probabilities = torch.softmax(
            output,
            dim=1
        )[0]

        prediction = int(
            torch.argmax(probabilities).item()
        )

    inference_time = (
        time.perf_counter() - start_time
    )

    return (
        prediction,
        probabilities.cpu().numpy(),
        inference_time
    )


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------

st.title("🛡️ DeepGuard")

st.subheader(
    "Digital Media Authenticity Detection System"
)

st.write(
    "Upload a facial image to analyze whether "
    "the detected face appears Real or Deepfake."
)

st.divider()




# ---------------------------------------------------------
# Upload
# ---------------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload a facial image",
    type=[
        "jpg",
        "jpeg",
        "png",
        "webp"
    ]
)


if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    st.divider()

    if st.button(
        "🔍 Analyze Image",
        type="primary",
        use_container_width=True
    ):

        with st.spinner(
            "Detecting face and analyzing..."
        ):

            try:

                # Face detection
                face, face_confidence = detect_face(
                    image
                )

                if face is None:

                    st.error(
                        "No clear face was detected."
                    )

                    st.info(
                        "Please upload a clear facial "
                        "image with the face visible."
                    )

                else:

                    # Show detected face
                    st.image(
                        face,
                        caption=(
                            f"Detected Face "
                            f"(MTCNN confidence: "
                            f"{face_confidence:.2%})"
                        ),
                        width=300
                    )

                    # Prediction
                    prediction, probabilities, inference_time = (
                        predict(face)
                    )

                    predicted_class = CLASS_NAMES[
                        prediction
                    ]

                    deepfake_probability = float(
                        probabilities[0]
                    )

                    real_probability = float(
                        probabilities[1]
                    )

                    st.divider()

                    st.subheader(
                        "Analysis Result"
                    )

                    # Main prediction
                    if predicted_class == "Deepfake":

                        st.error(
                            f"⚠️ Prediction: {predicted_class.upper()}"
                        )

                    else:

                        st.success(
                            f"✓ Prediction: {predicted_class.upper()}"
                        )

                    confidence = float(
                        probabilities[prediction]
                    )

                    st.metric(
                        "Prediction Confidence",
                        f"{confidence:.2%}"
                    )

                    # Probabilities
                    col1, col2 = st.columns(2)

                    with col1:

                        st.metric(
                            "Deepfake Probability",
                            f"{deepfake_probability:.2%}"
                        )

                    with col2:

                        st.metric(
                            "Real Probability",
                            f"{real_probability:.2%}"
                        )

                    st.progress(
                        confidence
                    )

                    st.caption(
                        f"Inference time: "
                        f"{inference_time:.3f} seconds"
                    )

            except Exception as error:

                st.error(
                    "An error occurred during analysis."
                )

                st.exception(error)


# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------

st.divider()

st.caption(
    "DeepGuard provides an automated authenticity "
    "prediction and should not be treated as definitive "
    "proof of manipulation."
)

st.caption(
    "Research prototype — image-based facial "
    "deepfake detection."
)
