# DeepGuard — Digital Media Authenticity Detection System

Image-only facial deepfake detection research + demo platform.

## Project scope
- Input: a single facial image
- Output: Real / Deepfake prediction with confidence
- Research dataset: FaceForensics++-derived facial images, evaluated with source-video-independent splits
- Models: Custom CNN and ResNet18
- External validation planned: Celeb-DF image subset, used for testing only
- No video inference or temporal modeling

## Repository structure
```text
DeepGuard/
├── backend/              # FastAPI inference + experiment APIs
├── frontend/             # React + Vite dashboard
├── src/                  # ML training, evaluation, splitting, plotting
├── models/production/    # Put the selected trained checkpoint here
├── data/                 # Dataset is intentionally NOT bundled
├── results/              # Experiment records and generated plots
├── docs/                 # Research notes and experiment log
└── scripts/              # Windows helper scripts
```

## 1. Environment
Create/activate the existing project environment:
```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 2. Dataset
Place the source-independent image split at:
```text
data/processed/source_independent/
├── train/deepfake
├── train/real
├── validation/deepfake
├── validation/real
├── test/deepfake
└── test/real
```
The dataset itself is not included because it is large and subject to dataset licensing/terms.

## 3. Run research experiments
```powershell
python src/leakage_audit.py
python src/train_cnn.py
python src/compare_models.py
```
The training pipeline creates a unique run directory under `results/runs/` and saves model checkpoints, JSON records, curves, confusion matrices, and metric plots.

For ResNet18:
```powershell
python src/train_resnet.py
```

## 4. Production checkpoint
After selecting a verified model, copy its checkpoint to:
```text
models/production/deepguard_model.pth
```
The backend will automatically use that file. Set `DEEPGUARD_MODEL_PATH` to override it.

## 5. Start backend
```powershell
uvicorn backend.app.main:app --reload --port 8000
```
API docs: http://127.0.0.1:8000/docs

## 6. Start frontend
In another terminal:
```powershell
cd frontend
npm install
npm run dev
```
Open the Vite URL shown in the terminal.

## 7. One-command Windows launcher
From the repository root:
```powershell
.\scripts\start_backend.ps1
```
Use a second terminal for the frontend.

## Important research rule
Do not tune the model on the test set. Use validation data for model selection and keep the source-independent test set for the final evaluation. External Celeb-DF testing must be performed without retraining/fine-tuning on the external test images.
