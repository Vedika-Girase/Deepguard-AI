import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from models.resnet18_model import build_resnet18


# ============================================================
# Configuration
# ============================================================

DATA_DIR = Path("data/external/celebdf_faces")

CHECKPOINT = Path(
    "results/runs/"
    "resnet18_resnet18_source_independent_20260930_204802/"
    "best_model.pth"
)

OUTPUT_DIR = Path(
    "results/external/celebdf_resnet18"
)

IMAGE_SIZE = 224
BATCH_SIZE = 16

DEVICE = torch.device("cpu")

# Must match DeepGuard evaluation preprocessing.
TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


# ============================================================
# Main
# ============================================================

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n==============================================")
    print(" DeepGuard - Celeb-DF External Evaluation")
    print("==============================================")
    print(f"Dataset    : {DATA_DIR}")
    print(f"Checkpoint : {CHECKPOINT}")
    print(f"Device     : {DEVICE}")
    print("Training   : NO")
    print("Fine-tune  : NO")
    print("==============================================\n")

    # --------------------------------------------------------
    # Check paths
    # --------------------------------------------------------

    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"External dataset not found: {DATA_DIR}"
        )

    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT}"
        )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset = datasets.ImageFolder(
        root=str(DATA_DIR),
        transform=TRANSFORM
    )

    print("Class mapping:")
    print(dataset.class_to_idx)

    print(f"\nExternal images: {len(dataset)}")

    if len(dataset) != 200:
        print(
            f"[WARNING] Expected approximately 200 images, "
            f"but found {len(dataset)}."
        )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    model = build_resnet18(
        num_classes=2,
        pretrained=False
    )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE
    )

    # Handle common checkpoint formats.
    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]

        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]

        else:
            state_dict = checkpoint

    else:
        state_dict = checkpoint

    # Remove DataParallel prefix if present.
    cleaned_state_dict = {}

    for key, value in state_dict.items():
        new_key = key.replace("module.", "", 1)
        cleaned_state_dict[new_key] = value

    model.load_state_dict(cleaned_state_dict)

    model.to(DEVICE)
    model.eval()

    print("\nCheckpoint loaded successfully.")
    print("Model is now FROZEN for external evaluation.\n")

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    all_labels = []
    all_predictions = []
    all_probabilities = []

    start_time = time.perf_counter()

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(DEVICE)

            outputs = model(images)

            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1)

            all_labels.extend(labels.cpu().numpy().tolist())
            all_predictions.extend(
                predictions.cpu().numpy().tolist()
            )
            all_probabilities.extend(
                probabilities.cpu().numpy().tolist()
            )

    total_time = time.perf_counter() - start_time

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    precision = precision_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )

    cm = confusion_matrix(
        all_labels,
        all_predictions
    )

    report = classification_report(
        all_labels,
        all_predictions,
        target_names=dataset.classes,
        digits=4,
        zero_division=0
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print("==============================================")
    print(" CELEB-DF EXTERNAL RESULTS")
    print("==============================================")

    print(f"Images evaluated : {len(all_labels)}")
    print(f"Accuracy         : {accuracy * 100:.2f}%")
    print(f"Macro Precision  : {precision * 100:.2f}%")
    print(f"Macro Recall     : {recall * 100:.2f}%")
    print(f"Macro F1         : {f1 * 100:.2f}%")
    print(f"Inference time   : {total_time:.2f}s")

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(report)

    print("==============================================\n")

    # --------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------

    metrics = {
        "experiment": "Celeb-DF external evaluation",
        "dataset": "Celeb-DF v2",
        "dataset_size": len(dataset),
        "classes": dataset.classes,
        "class_mapping": dataset.class_to_idx,
        "checkpoint": str(CHECKPOINT),
        "training_performed": False,
        "fine_tuning_performed": False,
        "accuracy": float(accuracy),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "inference_time_seconds": float(total_time),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }

    with open(
        OUTPUT_DIR / "external_metrics.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            metrics,
            f,
            indent=4
        )

    with open(
        OUTPUT_DIR / "classification_report.txt",
        "w",
        encoding="utf-8"
    ) as f:
        f.write(report)

    print(
        f"Results saved to: {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
