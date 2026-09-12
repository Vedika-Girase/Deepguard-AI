"""
DeepGuard - Experiment 2
Video-Level Split Custom CNN Training

IMPORTANT:
This is EXPERIMENT 2.
It uses the VIDEO-LEVEL SPLIT dataset and saves all outputs
separately from Experiment 1.

Dataset:
data/processed/video_dataset/

Output:
models/experiment2/cnn/
results/experiment2/cnn/
"""

import os
import json
import time
import csv
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Video-level dataset
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "video_dataset"

TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "validation"
TEST_DIR = DATA_DIR / "test"

# EXPERIMENT 2 OUTPUT DIRECTORIES
MODEL_DIR = PROJECT_ROOT / "models" / "experiment2" / "cnn"
RESULT_DIR = PROJECT_ROOT / "results" / "experiment2" / "cnn"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)


# Model configuration
IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 0.001

NUM_WORKERS = 0       # Windows-safe
SEED = 42


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# 3. PRINT EXPERIMENT INFORMATION
# ============================================================

print("=" * 70)
print("DEEPGUARD - EXPERIMENT 2")
print("CUSTOM CNN - VIDEO-LEVEL SPLIT")
print("=" * 70)

print(f"Device       : {DEVICE}")
print(f"Dataset      : {DATA_DIR}")
print(f"Batch Size   : {BATCH_SIZE}")
print(f"Epochs       : {EPOCHS}")
print(f"Learning Rate: {LEARNING_RATE}")
print()


# ============================================================
# 4. VERIFY DATASET
# ============================================================

required_dirs = [
    TRAIN_DIR / "deepfake",
    TRAIN_DIR / "real",
    VAL_DIR / "deepfake",
    VAL_DIR / "real",
    TEST_DIR / "deepfake",
    TEST_DIR / "real",
]

print("Checking dataset directories...")

for directory in required_dirs:
    if not directory.exists():
        raise FileNotFoundError(
            f"\nERROR: Required directory does not exist:\n{directory}"
        )

print("All dataset directories found.")
print()


# ============================================================
# 5. IMAGE TRANSFORMS
# ============================================================

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.RandomHorizontalFlip(p=0.5),

    transforms.RandomRotation(10),

    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# 6. LOAD DATASETS
# ============================================================

print("Loading datasets...")

train_dataset = datasets.ImageFolder(
    root=str(TRAIN_DIR),
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    root=str(VAL_DIR),
    transform=eval_transform
)

test_dataset = datasets.ImageFolder(
    root=str(TEST_DIR),
    transform=eval_transform
)


print(f"Train samples      : {len(train_dataset)}")
print(f"Validation samples : {len(val_dataset)}")
print(f"Test samples       : {len(test_dataset)}")

print(f"Classes             : {train_dataset.classes}")

print()


# ============================================================
# 7. VERIFY CLASS CONSISTENCY
# ============================================================

if train_dataset.classes != val_dataset.classes:
    raise ValueError(
        "Train and validation class mappings are different!"
    )

if train_dataset.classes != test_dataset.classes:
    raise ValueError(
        "Train and test class mappings are different!"
    )

CLASS_NAMES = train_dataset.classes
CLASS_TO_INDEX = train_dataset.class_to_idx

print("Class mapping:")
print(CLASS_TO_INDEX)
print()


# ============================================================
# 8. DATASET DISTRIBUTION
# ============================================================

def get_distribution(dataset):
    counter = Counter(dataset.targets)

    result = {}

    for class_name, class_index in dataset.class_to_idx.items():
        result[class_name] = counter[class_index]

    return result


train_distribution = get_distribution(train_dataset)
val_distribution = get_distribution(val_dataset)
test_distribution = get_distribution(test_dataset)


print("Dataset distribution:")
print("----------------------------------------")

print("TRAIN")
for name, count in train_distribution.items():
    print(f"  {name:10s}: {count}")

print()

print("VALIDATION")
for name, count in val_distribution.items():
    print(f"  {name:10s}: {count}")

print()

print("TEST")
for name, count in test_distribution.items():
    print(f"  {name:10s}: {count}")

print()


# ============================================================
# 9. DATA LOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)


# ============================================================
# 10. CUSTOM CNN MODEL
# ============================================================

class DeepGuardCNN(nn.Module):

    def __init__(self, num_classes=2):

        super(DeepGuardCNN, self).__init__()

        self.features = nn.Sequential(

            # Block 1
            nn.Conv2d(
                3, 32,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 2
            nn.Conv2d(
                32, 64,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 3
            nn.Conv2d(
                64, 128,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 4
            nn.Conv2d(
                128, 256,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),

        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(256, 128),

            nn.ReLU(),

            nn.Dropout(0.5),

            nn.Linear(128, num_classes)

        )

    def forward(self, x):

        x = self.features(x)

        x = self.global_pool(x)

        x = self.classifier(x)

        return x


# ============================================================
# 11. CREATE MODEL
# ============================================================

print("=" * 70)
print("CREATING CUSTOM CNN")
print("=" * 70)

model = DeepGuardCNN(
    num_classes=len(CLASS_NAMES)
)

model = model.to(DEVICE)

print(model)
print()


# ============================================================
# 12. LOSS AND OPTIMIZER
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# 13. TRAINING FUNCTION
# ============================================================

def train_one_epoch(model, loader, criterion, optimizer):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item() * images.size(0)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total

    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


# ============================================================
# 14. VALIDATION FUNCTION
# ============================================================

def evaluate(model, loader, criterion):

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    all_labels = []
    all_predictions = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)

            correct += (predicted == labels).sum().item()

            all_labels.extend(
                labels.cpu().numpy().tolist()
            )

            all_predictions.extend(
                predicted.cpu().numpy().tolist()
            )

    loss_value = running_loss / total

    accuracy = correct / total

    return (
        loss_value,
        accuracy,
        all_labels,
        all_predictions
    )


# ============================================================
# 15. TRAINING
# ============================================================

print("=" * 70)
print("STARTING TRAINING")
print("=" * 70)
print()

training_history = []

best_val_accuracy = 0.0

best_model_path = (
    MODEL_DIR /
    "deepguard_cnn_video_best.pth"
)

training_start = time.time()


for epoch in range(EPOCHS):

    epoch_start = time.time()

    train_loss, train_accuracy = train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer
    )

    val_loss, val_accuracy, _, _ = evaluate(
        model,
        val_loader,
        criterion
    )

    epoch_time = time.time() - epoch_start

    epoch_record = {

        "epoch": epoch + 1,

        "train_loss": train_loss,

        "train_accuracy": train_accuracy,

        "validation_loss": val_loss,

        "validation_accuracy": val_accuracy,

        "epoch_time_seconds": epoch_time

    }

    training_history.append(epoch_record)

    print(f"Epoch [{epoch + 1}/{EPOCHS}]")

    print(f"Train Loss       : {train_loss:.4f}")

    print(f"Train Accuracy   : {train_accuracy:.4f}")

    print(f"Validation Loss  : {val_loss:.4f}")

    print(f"Validation Acc   : {val_accuracy:.4f}")

    print(f"Epoch Time       : {epoch_time:.2f}s")

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        torch.save(
            model.state_dict(),
            best_model_path
        )

        print("✓ Best model saved.")

    print()


training_time = time.time() - training_start


# ============================================================
# 16. LOAD BEST MODEL
# ============================================================

print("=" * 70)
print("LOADING BEST MODEL")
print("=" * 70)

model.load_state_dict(
    torch.load(
        best_model_path,
        map_location=DEVICE
    )
)

print(f"Best validation accuracy: {best_val_accuracy:.4f}")
print()


# ============================================================
# 17. FINAL VALIDATION
# ============================================================

(
    final_val_loss,
    final_val_accuracy,
    val_labels,
    val_predictions
) = evaluate(
    model,
    val_loader,
    criterion
)


# ============================================================
# 18. FINAL TEST
# ============================================================

print("=" * 70)
print("EVALUATING ON TEST DATASET")
print("=" * 70)

inference_start = time.time()

(
    test_loss,
    test_accuracy,
    test_labels,
    test_predictions
) = evaluate(
    model,
    test_loader,
    criterion
)

inference_time = time.time() - inference_start


# ============================================================
# 19. METRICS
# ============================================================

precision = precision_score(
    test_labels,
    test_predictions,
    average="binary",
    zero_division=0
)

recall = recall_score(
    test_labels,
    test_predictions,
    average="binary",
    zero_division=0
)

f1 = f1_score(
    test_labels,
    test_predictions,
    average="binary",
    zero_division=0
)


# ============================================================
# 20. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    test_labels,
    test_predictions
)


# ============================================================
# 21. CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    test_labels,
    test_predictions,
    target_names=CLASS_NAMES,
    zero_division=0
)


# ============================================================
# 22. PRINT RESULTS
# ============================================================

print()

print("=" * 70)
print("DEEPGUARD CNN - EXPERIMENT 2 RESULTS")
print("=" * 70)

print(
    f"Validation Accuracy : {final_val_accuracy:.4f}"
)

print(
    f"Test Accuracy       : {test_accuracy:.4f}"
)

print(
    f"Precision           : {precision:.4f}"
)

print(
    f"Recall              : {recall:.4f}"
)

print(
    f"F1 Score            : {f1:.4f}"
)

print()

print(
    f"Training Time       : {training_time:.2f} seconds"
)

print(
    f"Inference Time      : {inference_time:.2f} seconds"
)

print()

print("Confusion Matrix:")
print(cm)

print()

print("Classification Report:")
print(report)


# ============================================================
# 23. AUTOMATIC DIAGNOSTIC
# ============================================================

print("=" * 70)
print("AUTOMATIC DIAGNOSTIC")
print("=" * 70)

if test_accuracy >= 0.90:

    print(
        "Strong performance on video-level test split."
    )

elif test_accuracy >= 0.75:

    print(
        "Moderate performance. Further model improvement may help."
    )

elif test_accuracy >= 0.55:

    print(
        "Weak performance. Dataset/model investigation required."
    )

else:

    print(
        "WARNING: Model performance is close to random guessing."
    )

    print(
        "Investigate data quality, labels, preprocessing, "
        "and model learning."
    )

print()


# ============================================================
# 24. SAVE TRAINING HISTORY CSV
# ============================================================

history_csv = (
    RESULT_DIR /
    "cnn_video_training_history.csv"
)

with open(
    history_csv,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "epoch",
            "train_loss",
            "train_accuracy",
            "validation_loss",
            "validation_accuracy",
            "epoch_time_seconds"
        ]
    )

    writer.writeheader()

    writer.writerows(training_history)


# ============================================================
# 25. SAVE CONFUSION MATRIX JSON
# ============================================================

cm_json_path = (
    RESULT_DIR /
    "cnn_video_confusion_matrix.json"
)

cm_data = {

    "class_names": CLASS_NAMES,

    "class_to_index": CLASS_TO_INDEX,

    "confusion_matrix": cm.tolist()

}

with open(
    cm_json_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        cm_data,
        file,
        indent=4
    )


# ============================================================
# 26. SAVE CLASSIFICATION REPORT
# ============================================================

report_path = (
    RESULT_DIR /
    "cnn_video_classification_report.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "DEEPGUARD - EXPERIMENT 2\n"
    )

    file.write(
        "CUSTOM CNN - VIDEO LEVEL SPLIT\n"
    )

    file.write("=" * 60 + "\n\n")

    file.write(
        f"Validation Accuracy: {final_val_accuracy:.6f}\n"
    )

    file.write(
        f"Test Accuracy: {test_accuracy:.6f}\n"
    )

    file.write(
        f"Precision: {precision:.6f}\n"
    )

    file.write(
        f"Recall: {recall:.6f}\n"
    )

    file.write(
        f"F1 Score: {f1:.6f}\n"
    )

    file.write(
        f"Training Time: {training_time:.2f} seconds\n"
    )

    file.write(
        f"Inference Time: {inference_time:.2f} seconds\n\n"
    )

    file.write(
        "Confusion Matrix:\n"
    )

    file.write(
        str(cm)
    )

    file.write(
        "\n\nClassification Report:\n"
    )

    file.write(report)


# ============================================================
# 27. SAVE COMPLETE JSON
# ============================================================

results_json_path = (
    RESULT_DIR /
    "cnn_video_results.json"
)

results = {

    "experiment": "Experiment 2",

    "experiment_type": "Video-Level Split",

    "model": "DeepGuard Custom CNN",

    "dataset": "FaceForensics++",

    "dataset_path": str(DATA_DIR),

    "device": str(DEVICE),

    "image_size": IMAGE_SIZE,

    "batch_size": BATCH_SIZE,

    "epochs": EPOCHS,

    "learning_rate": LEARNING_RATE,

    "class_names": CLASS_NAMES,

    "class_to_index": CLASS_TO_INDEX,

    "dataset_distribution": {

        "train": train_distribution,

        "validation": val_distribution,

        "test": test_distribution

    },

    "sample_counts": {

        "train": len(train_dataset),

        "validation": len(val_dataset),

        "test": len(test_dataset)

    },

    "best_validation_accuracy":
        best_val_accuracy,

    "validation_accuracy":
        final_val_accuracy,

    "test_accuracy":
        test_accuracy,

    "precision":
        precision,

    "recall":
        recall,

    "f1_score":
        f1,

    "training_time_seconds":
        training_time,

    "inference_time_seconds":
        inference_time,

    "confusion_matrix":
        cm.tolist(),

    "training_history":
        training_history,

    "model_path":
        str(best_model_path),

    "classification_report":
        report

}


with open(
    results_json_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        results,
        file,
        indent=4
    )


# ============================================================
# 28. FINAL OUTPUT
# ============================================================

print("=" * 70)
print("EXPERIMENT 2 COMPLETED")
print("=" * 70)

print()

print("Model saved:")
print(best_model_path)

print()

print("Results saved:")
print(results_json_path)

print()

print("Classification report saved:")
print(report_path)

print()

print("Confusion matrix saved:")
print(cm_json_path)

print()

print("Training history saved:")
print(history_csv)

print()

print("=" * 70)
print("EXPERIMENT 2 FILES")
print("=" * 70)

print("1. Model:")
print("   models/experiment2/cnn/deepguard_cnn_video_best.pth")

print()

print("2. Complete results:")
print("   results/experiment2/cnn/cnn_video_results.json")

print()

print("3. Classification report:")
print("   results/experiment2/cnn/cnn_video_classification_report.txt")

print()

print("4. Confusion matrix:")
print("   results/experiment2/cnn/cnn_video_confusion_matrix.json")

print()

print("5. Training history:")
print("   results/experiment2/cnn/cnn_video_training_history.csv")

print()

print("=" * 70)
print("READY FOR EXPERIMENT 2 MODEL COMPARISON")
print("=" * 70)