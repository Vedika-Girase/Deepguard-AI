from pathlib import Path
from typing import Any
import time

import torch
from PIL import Image
from torchvision import transforms

from .config import IMAGE_SIZE, MODEL_PATH, MODEL_TYPE
from src.models.custom_cnn import CustomCNN

try:
    from src.models.resnet18_model import build_resnet18
except Exception:
    build_resnet18 = None


class ModelService:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.loaded_path: Path | None = None
        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def _build(self):
        if MODEL_TYPE == "resnet18":
            if build_resnet18 is None:
                raise RuntimeError("ResNet18 implementation is unavailable")
            return build_resnet18(num_classes=2, pretrained=False)
        return CustomCNN(num_classes=2)

    def load(self) -> bool:
        if not MODEL_PATH.exists():
            self.model = None
            return False
        model = self._build()
        state = torch.load(MODEL_PATH, map_location=self.device)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state)
        model.to(self.device)
        model.eval()
        self.model = model
        self.loaded_path = MODEL_PATH
        return True

    @property
    def available(self) -> bool:
        return self.model is not None

    def predict(self, image: Image.Image) -> dict[str, Any]:
        if not self.available and not self.load():
            raise FileNotFoundError(f"Production model not found: {MODEL_PATH}")
        image = image.convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        start = time.perf_counter()
        with torch.no_grad():
            logits = self.model(tensor)
            probabilities = torch.softmax(logits, dim=1)[0]
        elapsed = time.perf_counter() - start
        idx = int(probabilities.argmax().item())
        labels = {0: "deepfake", 1: "real"}
        return {
            "label": labels[idx],
            "confidence": round(float(probabilities[idx].item()), 6),
            "probabilities": {labels[i]: round(float(probabilities[i].item()), 6) for i in range(2)},
            "inference_time_seconds": round(elapsed, 6),
            "model": MODEL_TYPE,
            "device": str(self.device),
        }


model_service = ModelService()
