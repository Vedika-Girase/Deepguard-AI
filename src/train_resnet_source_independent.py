import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms, models
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
# CONFIGURATION
# ============================================================

DATA_DIR = "data/processed/source_independent"
MODEL_DIR = "models/experiment2"
RESULTS_DIR = "results/experiment2"

IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 0.001

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================
# DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ============================================================
# TRANSFORMS
# ============================================================

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
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
# DATASETS
# ============================================================

train_dataset = datasets.ImageFolder(
    os.path.join(DATA_DIR, "train"),
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    os.path.join(DATA_DIR, "validation"),
    transform=eval_transform
)

test_dataset = datasets.ImageFolder(
    os.path.join(DATA_DIR, "test"),
    transform=eval_transform
)

print("\nClass mapping:")
print(train_dataset.class_to_idx)

print("\nDataset sizes:")
print("Train:", len(train_dataset))
print("Validation:", len(val_dataset))
print("Test:", len(test_dataset))

# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

# ============================================================
# RESNET18 MODEL
# ============================================================

model = models.resnet18(weights=None)

num_features = model.fc.in_features

model.fc = nn.Linear(num_features, 2)

model = model.to(device)

print("\nModel:")
print(model)

# ============================================================
# LOSS + OPTIMIZER
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

# ============================================================
# TRAINING
# ============================================================

best_val_acc = 0.0

history = {
    "train_loss": [],
    "train_accuracy": [],
    "val_loss": [],
    "val_accuracy": []
}

model_path = os.path.join(
    MODEL_DIR,
    "resnet18_source_independent_best.pth"
)

start_time = time.time()

for epoch in range(EPOCHS):

    # -------------------------
    # TRAIN
    # -------------------------

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item() * images.size(0)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / total
    train_acc = correct / total

    # -------------------------
    # VALIDATION
    # -------------------------

    model.eval()

    val_running_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            val_running_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, 1)

            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_loss = val_running_loss / val_total
    val_acc = val_correct / val_total

    history["train_loss"].append(train_loss)
    history["train_accuracy"].append(train_acc)

    history["val_loss"].append(val_loss)
    history["val_accuracy"].append(val_acc)

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_acc:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_acc:.4f}"
    )

    # -------------------------
    # SAVE BEST MODEL
    # -------------------------

    if val_acc > best_val_acc:

        best_val_acc = val_acc

        torch.save(model.state_dict(), model_path)

        print(
            f"  Best model saved "
            f"(Val Acc = {val_acc:.4f})"
        )

training_time = time.time() - start_time

# ============================================================
# LOAD BEST MODEL
# ============================================================

model.load_state_dict(
    torch.load(model_path, map_location=device)
)

model.eval()

# ============================================================
# TEST
# ============================================================

all_labels = []
all_predictions = []

inference_start = time.time()

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(images)

        _, predictions = torch.max(outputs, 1)

        all_labels.extend(labels.numpy())
        all_predictions.extend(
            predictions.cpu().numpy()
        )

inference_time = time.time() - inference_start

# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

precision = precision_score(
    all_labels,
    all_predictions,
    zero_division=0
)

recall = recall_score(
    all_labels,
    all_predictions,
    zero_division=0
)

f1 = f1_score(
    all_labels,
    all_predictions,
    zero_division=0
)

cm = confusion_matrix(
    all_labels,
    all_predictions
)

report = classification_report(
    all_labels,
    all_predictions,
    target_names=train_dataset.classes,
    zero_division=0
)

# ============================================================
# FINAL RESULTS
# ============================================================

print("\n==========================================")
print("FINAL TEST RESULTS")
print("==========================================")

print(f"Accuracy : {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall   : {recall * 100:.2f}%")
print(f"F1 Score : {f1 * 100:.2f}%")

print(f"\nTraining time: {training_time:.2f} seconds")
print(f"Inference time: {inference_time:.2f} seconds")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(report)

# ============================================================
# SAVE RESULTS
# ============================================================

metrics = {
    "model": "ResNet18",
    "experiment": "Experiment 2 - Source Independent",
    "dataset": "FaceForensics++",
    "train_images": len(train_dataset),
    "validation_images": len(val_dataset),
    "test_images": len(test_dataset),
    "validation_accuracy": best_val_acc,
    "test_accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "training_time_seconds": training_time,
    "inference_time_seconds": inference_time,
    "confusion_matrix": cm.tolist()
}

with open(
    os.path.join(
        RESULTS_DIR,
        "resnet18_source_independent_metrics.json"
    ),
    "w"
) as f:
    json.dump(metrics, f, indent=4)

with open(
    os.path.join(
        RESULTS_DIR,
        "resnet18_source_independent_classification_report.txt"
    ),
    "w"
) as f:
    f.write(report)

with open(
    os.path.join(
        RESULTS_DIR,
        "resnet18_source_independent_training_history.json"
    ),
    "w"
) as f:
    json.dump(history, f, indent=4)

print("\nResults saved to:")
print(RESULTS_DIR)

print("\nModel saved to:")
print(model_path)