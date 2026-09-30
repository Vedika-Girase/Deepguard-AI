from pathlib import Path
import json
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

from .config import MAX_UPLOAD_MB, MODEL_PATH, MODEL_TYPE
from .model_service import model_service

ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = ROOT / "results"

app = FastAPI(title="DeepGuard API", version="1.0.0", description="Image-only facial deepfake detection API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "model_available": model_service.available or MODEL_PATH.exists(),
        "model_path": str(MODEL_PATH),
        "model_type": MODEL_TYPE,
    }

@app.get("/api/experiments")
def experiments():
    records = []
    for path in RESULTS_ROOT.rglob("experiment_record.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            data["run_directory"] = str(path.parent.relative_to(ROOT))
            records.append(data)
        except Exception:
            continue
    return {"count": len(records), "experiments": records}

@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")
    raw = await file.read()
    if len(raw) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"Image exceeds {MAX_UPLOAD_MB} MB limit.")
    try:
        image = Image.open(__import__('io').BytesIO(raw))
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid image.")
    try:
        result = model_service.predict(image)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")
    result["filename"] = file.filename
    result["image_size"] = list(image.size)
    return result
