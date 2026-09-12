# ============================================================
# DeepGuard
# Experiment 1: Custom CNN on FaceForensics++
# ============================================================

import os
import json
import time
import random
from pathlib import Path

import numpy as np
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
# 1. REPRODUCIBILITY
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# 2. CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_DIR / "data" / "processed" / "dataset"

TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "validation"
TEST_DIR = DATA_DIR / "test"

MODEL_DIR = PROJECT_DIR / "models"
RESULT_DIR = PROJECT_DIR / "results"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)


# Image configuration
IMAGE_SIZE = 224

# Training configuration
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 0.001

# Number of workers
NUM_WORKERS = 0

# Device
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# 3. PRINT CONFIGURATION
# ============================================================

print("\n" + "=" * 60)
print("DEEPGUARD - CUSTOM CNN EXPERIMENT")
print("=" * 60)

print(f"Device       : {DEVICE}")
print(f"Image size   : {IMAGE_SIZE}x{IMAGE_SIZE}")
print(f"Batch size   : {BATCH_SIZE}")
print(f"Epochs       : {EPOCHS}")
print(f"Learning rate: {LEARNING_RATE}")

print("\nDataset paths:")
print(f"Train      : {TRAIN_DIR}")
print(f"Validation : {VAL_DIR}")
print(f"Test       : {TEST_DIR}")


# ============================================================
# 4. CHECK DATASET DIRECTORIES
# ============================================================

for directory in [TRAIN_DIR, VAL_DIR, TEST_DIR]:

    if not directory.exists():
        raise FileNotFoundError(
            f"\nDataset directory does not exist:\n{directory}"
        )


# ============================================================
# 5. IMAGE TRANSFORMS
# ============================================================

train_transform = transforms.Compose([

    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.RandomHorizontalFlip(),

    transforms.RandomRotation(5),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


val_test_transform = transforms.Compose([

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

print("\nLoading datasets...")

train_dataset = datasets.ImageFolder(
    root=str(TRAIN_DIR),
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    root=str(VAL_DIR),
    transform=val_test_transform
)

test_dataset = datasets.ImageFolder(
    root=str(TEST_DIR),
    transform=val_test_transform
)


# ============================================================
# 7. IMPORTANT: CLASS MAPPING
# ============================================================

print("\n" + "=" * 60)
print("CLASS MAPPING")
print("=" * 60)

print("Train      :", train_dataset.class_to_idx)
print("Validation :", val_dataset.class_to_idx)
print("Test       :", test_dataset.class_to_idx)


# Make sure all datasets use the same class mapping
if (
    train_dataset.class_to_idx
    != val_dataset.class_to_idx
    or
    train_dataset.class_to_idx
    != test_dataset.class_to_idx
):

    raise ValueError(
        "\nERROR: Class mappings are different between datasets!"
    )


class_names = train_dataset.classes

print("\nClasses:", class_names)


# ============================================================
# 8. DATASET SIZES
# ============================================================

print("\n" + "=" * 60)
print("DATASET SIZES")
print("=" * 60)

print(f"Train      : {len(train_dataset)}")
print(f"Validation : {len(val_dataset)}")
print(f"Test       : {len(test_dataset)}")


# ============================================================
# 9. CLASS DISTRIBUTION
# ============================================================

def get_class_distribution(dataset):

    counts = {}

    for class_name in dataset.classes:

        class_index = dataset.class_to_idx[class_name]

        count = sum(
            1 for target in dataset.targets
            if target == class_index
        )

        counts[class_name] = count

    return counts


train_distribution = get_class_distribution(train_dataset)
val_distribution = get_class_distribution(val_dataset)
test_distribution = get_class_distribution(test_dataset)


print("\n" + "=" * 60)
print("CLASS DISTRIBUTION")
print("=" * 60)

print("Train      :", train_distribution)
print("Validation :", val_distribution)
print("Test       :", test_distribution)


# ============================================================
# 10. DATA LOADERS
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
# 11. CUSTOM CNN MODEL
# ============================================================

class DeepGuardCNN(nn.Module):

    def __init__(self, num_classes=2):

        super(DeepGuardCNN, self).__init__()

        self.features = nn.Sequential(

            # Block 1
            nn.Conv2d(
                3,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(32),

            nn.ReLU(),

            nn.MaxPool2d(2),

            # Block 2
            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(64),

            nn.ReLU(),

            nn.MaxPool2d(2),

            # Block 3
            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(128),

            nn.ReLU(),

            nn.MaxPool2d(2),

            # Block 4
            nn.Conv2d(
                128,
                256,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(256),

            nn.ReLU(),

            nn.MaxPool2d(2)
        )


        # Global Adaptive Pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))


        # Classifier
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
# 12. CREATE MODEL
# ============================================================

model = DeepGuardCNN(
    num_classes=len(class_names)
)

model = model.to(DEVICE)


print("\n" + "=" * 60)
print("MODEL ARCHITECTURE")
print("=" * 60)

print(model)


# ============================================================
# 13. LOSS FUNCTION AND OPTIMIZER
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# Learning-rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=2
)


# ============================================================
# 14. TRAINING FUNCTION
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer
):

    model.train()

    running_loss = 0.0

    correct = 0
    total = 0


    for images, labels in loader:

        images = images.to(DEVICE)

        labels = labels.to(DEVICE)


        optimizer.zero_grad()


        outputs = model(images)


        loss = criterion(
            outputs,
            labels
        )


        loss.backward()

        optimizer.step()


        running_loss += (
            loss.item() * images.size(0)
        )


        _, predicted = torch.max(
            outputs,
            1
        )


        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()


    epoch_loss = (
        running_loss / total
    )

    epoch_accuracy = (
        correct / total
    )


    return epoch_loss, epoch_accuracy


# ============================================================
# 15. VALIDATION FUNCTION
# ============================================================

def validate(
    model,
    loader,
    criterion
):

    model.eval()

    running_loss = 0.0

    correct = 0
    total = 0


    with torch.no_grad():

        for images, labels in loader:

            images = images.to(DEVICE)

            labels = labels.to(DEVICE)


            outputs = model(images)


            loss = criterion(
                outputs,
                labels
            )


            running_loss += (
                loss.item() * images.size(0)
            )


            _, predicted = torch.max(
                outputs,
                1
            )


            total += labels.size(0)

            correct += (
                predicted == labels
            ).sum().item()


    epoch_loss = (
        running_loss / total
    )

    epoch_accuracy = (
        correct / total
    )


    return epoch_loss, epoch_accuracy


# ============================================================
# 16. TRAIN MODEL
# ============================================================

print("\n" + "=" * 60)
print("STARTING TRAINING")
print("=" * 60)

training_start = time.time()


best_val_accuracy = 0.0

training_history = []


for epoch in range(EPOCHS):

    epoch_start = time.time()


    # -------------------------
    # Training
    # -------------------------

    train_loss, train_accuracy = train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer
    )


    # -------------------------
    # Validation
    # -------------------------

    val_loss, val_accuracy = validate(
        model,
        val_loader,
        criterion
    )


    # -------------------------
    # Scheduler
    # -------------------------

    scheduler.step(val_loss)


    epoch_time = time.time() - epoch_start


    # -------------------------
    # Save history
    # -------------------------

    training_history.append({

        "epoch": epoch + 1,

        "train_loss": train_loss,

        "train_accuracy": train_accuracy,

        "validation_loss": val_loss,

        "validation_accuracy": val_accuracy,

        "epoch_time_seconds": epoch_time

    })


    # -------------------------
    # Print results
    # -------------------------

    print(
        f"\nEpoch [{epoch + 1}/{EPOCHS}]"
    )

    print(
        f"Train Loss      : {train_loss:.4f}"
    )

    print(
        f"Train Accuracy  : {train_accuracy:.4f}"
    )

    print(
        f"Validation Loss : {val_loss:.4f}"
    )

    print(
        f"Validation Acc  : {val_accuracy:.4f}"
    )

    print(
        f"Epoch Time      : {epoch_time:.2f}s"
    )


    # -------------------------
    # Save best model
    # -------------------------

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy


        torch.save(
            model.state_dict(),
            MODEL_DIR / "deepguard_cnn_best.pth"
        )


        print(
            "✓ Best model saved."
        )


training_time = (
    time.time() - training_start
)


# ============================================================
# 17. LOAD BEST MODEL
# ============================================================

best_model_path = (
    MODEL_DIR / "deepguard_cnn_best.pth"
)


if best_model_path.exists():

    model.load_state_dict(
        torch.load(
            best_model_path,
            map_location=DEVICE
        )
    )

    print(
        "\nBest model loaded for final testing."
    )


# ============================================================
# 18. TEST FUNCTION
# ============================================================

def evaluate_model(
    model,
    loader
):

    model.eval()


    all_labels = []

    all_predictions = []


    inference_start = time.time()


    with torch.no_grad():

        for images, labels in loader:

            images = images.to(DEVICE)


            outputs = model(images)


            predictions = torch.argmax(
                outputs,
                dim=1
            )


            all_labels.extend(
                labels.numpy()
            )


            all_predictions.extend(
                predictions.cpu().numpy()
            )


    inference_time = (
        time.time() - inference_start
    )


    return (
        np.array(all_labels),
        np.array(all_predictions),
        inference_time
    )


# ============================================================
# 19. FINAL TESTING
# ============================================================

print("\n" + "=" * 60)
print("EVALUATING ON TEST DATASET")
print("=" * 60)


y_true, y_pred, inference_time = evaluate_model(
    model,
    test_loader
)


# ============================================================
# 20. CALCULATE METRICS
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)


precision = precision_score(
    y_true,
    y_pred,
    zero_division=0
)


recall = recall_score(
    y_true,
    y_pred,
    zero_division=0
)


f1 = f1_score(
    y_true,
    y_pred,
    zero_division=0
)


# ============================================================
# 21. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred
)


print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print(cm)


# ============================================================
# 22. PREDICTION DISTRIBUTION
# ============================================================

prediction_distribution = {}

for class_name, class_index in train_dataset.class_to_idx.items():

    count = int(
        np.sum(y_pred == class_index)
    )

    prediction_distribution[class_name] = count


print("\n" + "=" * 60)
print("PREDICTION DISTRIBUTION")
print("=" * 60)

print(
    prediction_distribution
)


# ============================================================
# 23. CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    zero_division=0
)


print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(report)


# ============================================================
# 24. FINAL RESULTS
# ============================================================

print("\n" + "=" * 60)
print("DEEPGUARD CNN RESULTS")
print("=" * 60)

print(
    f"Validation Accuracy : {best_val_accuracy:.4f}"
)

print(
    f"Test Accuracy       : {accuracy:.4f}"
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

print(
    f"\nTraining Time       : {training_time:.2f} seconds"
)

print(
    f"Inference Time      : {inference_time:.2f} seconds"
)


# ============================================================
# 25. INTERPRETATION
# ============================================================

print("\n" + "=" * 60)
print("AUTOMATIC DIAGNOSTIC")
print("=" * 60)


if accuracy <= 0.55:

    print(
        "WARNING: Model performance is close to random guessing."
    )

    print(
        "We will investigate data quality, labels, preprocessing,"
    )

    print(
        "and model learning before comparing algorithms."
    )

elif accuracy >= 0.90:

    print(
        "Strong baseline performance."
    )

else:

    print(
        "Moderate baseline performance."
    )


if len(set(y_pred)) == 1:

    predicted_class = class_names[
        int(y_pred[0])
    ]

    print(
        f"\nWARNING: Model predicted only one class: "
        f"{predicted_class}"
    )

    print(
        "This is important for our research analysis."
    )


# ============================================================
# 26. SAVE COMPLETE EXPERIMENT RESULTS
# ============================================================

results = {

    "experiment": "Experiment 1 - Custom CNN",

    "project": "DeepGuard",

    "dataset": "FaceForensics++",

    "model": "Custom CNN",

    "seed": SEED,

    "device": str(DEVICE),

    "image_size": IMAGE_SIZE,

    "batch_size": BATCH_SIZE,

    "epochs": EPOCHS,

    "learning_rate": LEARNING_RATE,

    "class_mapping":
        train_dataset.class_to_idx,

    "classes":
        class_names,

    "dataset_sizes": {

        "train":
            len(train_dataset),

        "validation":
            len(val_dataset),

        "test":
            len(test_dataset)

    },

    "class_distribution": {

        "train":
            train_distribution,

        "validation":
            val_distribution,

        "test":
            test_distribution

    },

    "prediction_distribution":
        prediction_distribution,

    "validation_accuracy":
        float(best_val_accuracy),

    "test_accuracy":
        float(accuracy),

    "precision":
        float(precision),

    "recall":
        float(recall),

    "f1_score":
        float(f1),

    "confusion_matrix":
        cm.tolist(),

    "training_time_seconds":
        float(training_time),

    "inference_time_seconds":
        float(inference_time),

    "training_history":
        training_history

}


results_path = (
    RESULT_DIR /
    "cnn_faceforensics_results.json"
)


with open(
    results_path,
    "w"
) as file:

    json.dump(
        results,
        file,
        indent=4
    )


# ============================================================
# 27. SAVE CLASSIFICATION REPORT
# ============================================================

report_path = (
    RESULT_DIR /
    "cnn_classification_report.txt"
)


with open(
    report_path,
    "w"
) as file:

    file.write(report)


# ============================================================
# 28. FINAL FILE LOCATIONS
# ============================================================

print("\n" + "=" * 60)
print("EXPERIMENT COMPLETED")
print("=" * 60)

print("\nModel saved:")
print(
    "models/deepguard_cnn_best.pth"
)

print("\nResults saved:")
print(
    "results/cnn_faceforensics_results.json"
)

print("\nClassification report saved:")
print(
    "results/cnn_classification_report.txt"
)

print("\nTraining history is also stored in the JSON file.")

print("\n" + "=" * 60)
print("READY FOR NEXT EXPERIMENT")
print("=" * 60)