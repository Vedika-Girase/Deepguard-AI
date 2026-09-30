# DeepGuard — Optimized Image-Only Research Pipeline

## Scope
DeepGuard is an image-based facial deepfake detector. FaceForensics++ videos are only the source of extracted facial images; source-video IDs are retained for leakage-safe grouping. No video/temporal training is used.

## Core commands

### 1. Verify source-independent split
```powershell
python src/leakage_audit.py
```

### 2. Experiment 2F — regularized Custom CNN
```powershell
python src/train_cnn.py
```
This defaults to AdamW, lr=0.0003, weight_decay=0.0001, 25 max epochs, early stopping, ReduceLROnPlateau, and moderate ColorJitter.

### 3. ResNet18 comparison
```powershell
python src/train_resnet.py --epochs 10 --lr 0.001
```
Use `--pretrained` only as a separately documented experiment.

### 4. Generate model comparison graphs
```powershell
python src/compare_models.py
```

## Per-run outputs
Each run creates a unique timestamped directory under `results/runs/` containing:
- `best_model.pth`
- `experiment_record.json`
- `test_metrics.json`
- `classification_report.txt`
- `loss_curve.png`
- `accuracy_curve.png`
- `learning_rate_curve.png`
- `generalization_gap.png`
- `confusion_matrix.png`
- `test_metrics.png`

## Comparison outputs
`results/comparison/` contains:
- `experiment_records.csv`
- `experiment_records.json`
- `validation_accuracy_comparison.png`
- `test_accuracy_comparison.png`
- `test_metrics_comparison.png`
- `training_time_comparison.png`
- `inference_time_comparison.png`
- `validation_vs_test_accuracy.png`

## Research discipline
- Never tune on the test set.
- Keep source-video IDs disjoint across train/validation/test.
- Never overwrite a previous experiment record.
- Keep each model/configuration as a separate run ID.
- Treat validation results as model-selection evidence and test results as final held-out evaluation.
