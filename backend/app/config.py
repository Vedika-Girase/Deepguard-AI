from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = Path(os.getenv("DEEPGUARD_MODEL_PATH", ROOT / "models" / "production" / "deepguard_model.pth"))
MODEL_TYPE = os.getenv("DEEPGUARD_MODEL_TYPE", "custom_cnn")
IMAGE_SIZE = int(os.getenv("DEEPGUARD_IMAGE_SIZE", "224"))
MAX_UPLOAD_MB = int(os.getenv("DEEPGUARD_MAX_UPLOAD_MB", "10"))
