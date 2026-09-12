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
    confusion_matrix
)

from pathlib import Path
import json
import time


# ============================================================
# 1. CONFIGURATION
# ============================================================

TRAIN_DIR = "data/processed/dataset/train"
VAL_DIR = "data/processed/dataset/validation"
TEST_DIR = "data/processed/dataset/test"

MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 224
BATCH_SIZE = 16

EPOCHS = 10

# Small learning rate for pretrained layers
BACKBONE_LR = 1e-5

# Higher learning rate for classifier
CLASSIFIER_LR = 1e-4

MODEL_PATH = MODEL_DIR / "deepguard_resnet18_finetuned_best.pth"

RESULT_PATH = (
    RESULTS_DIR /
    "resnet18_finetuned_faceforensics_results.json"
)


# ============================================================
# 2. DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 65)
print("DEEPGUARD - RESNET18 FINE-TUNING")
print("=" * 65)

print(f"Device: {device}")


# ============================================================
# 3. DATA TRANSFORMS
# ============================================================

train_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.RandomRotation(
        10
    ),

    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


test_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# 4. LOAD DATASETS
# ============================================================

print("\nLoading datasets...")

train_dataset = datasets.ImageFolder(
    TRAIN_DIR,
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    VAL_DIR,
    transform=test_transform
)

test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=test_transform
)

print(
    f"Train samples      : {len(train_dataset)}"
)

print(
    f"Validation samples : {len(val_dataset)}"
)

print(
    f"Test samples       : {len(test_dataset)}"
)

print(
    f"Classes             : {train_dataset.classes}"
)


# ============================================================
# 5. DATA LOADERS
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
# 6. CREATE PRETRAINED RESNET18
# ============================================================

print("\nCreating pretrained ResNet18...")

model = models.resnet18(
    weights=models.ResNet18_Weights.DEFAULT
)


# ============================================================
# 7. FREEZE ALL LAYERS FIRST
# ============================================================

for param in model.parameters():
    param.requires_grad = False


# ============================================================
# 8. UNFREEZE LAYER4
# ============================================================

print("\nFine-tuning layers:")

for param in model.layer4.parameters():

    param.requires_grad = True

    print(
        "  ResNet18 layer4"
    )


# ============================================================
# 9. REPLACE CLASSIFIER
# ============================================================

num_features = model.fc.in_features

model.fc = nn.Sequential(

    nn.Dropout(
        p=0.4
    ),

    nn.Linear(
        num_features,
        2
    )
)


# Classifier must be trainable
for param in model.fc.parameters():

    param.requires_grad = True


print(
    "  ResNet18 classifier"
)


model = model.to(device)


# ============================================================
# 10. DISPLAY TRAINABLE PARAMETERS
# ============================================================

print("\nTrainable parameters:")

trainable_params = 0
total_params = 0

for name, param in model.named_parameters():

    total_params += param.numel()

    if param.requires_grad:

        trainable_params += param.numel()

        print(
            f"  {name}"
        )


print(
    f"\nTrainable parameters: {trainable_params:,}"
)

print(
    f"Total parameters    : {total_params:,}"
)


# ============================================================
# 11. LOSS FUNCTION
# ============================================================

criterion = nn.CrossEntropyLoss()


# ============================================================
# 12. DIFFERENT LEARNING RATES
# ============================================================

optimizer = optim.Adam([

    {
        "params":
            model.layer4.parameters(),

        "lr":
            BACKBONE_LR
    },

    {
        "params":
            model.fc.parameters(),

        "lr":
            CLASSIFIER_LR
    }

])


# ============================================================
# 13. SCHEDULER
# ============================================================

scheduler = optim.lr_scheduler.ReduceLROnPlateau(

    optimizer,

    mode="max",

    factor=0.5,

    patience=2
)


# ============================================================
# 14. TRAIN FUNCTION
# ============================================================

def train_one_epoch():

    model.train()

    running_loss = 0.0

    correct = 0

    total = 0


    for images, labels in train_loader:

        images = images.to(device)

        labels = labels.to(device)


        optimizer.zero_grad()


        outputs = model(images)


        loss = criterion(
            outputs,
            labels
        )


        loss.backward()


        optimizer.step()


        running_loss += (
            loss.item()
        )


        _, predicted = torch.max(
            outputs,
            1
        )


        total += labels.size(0)


        correct += (
            predicted == labels
        ).sum().item()


    loss = (
        running_loss /
        len(train_loader)
    )


    accuracy = (
        correct /
        total
    )


    return loss, accuracy


# ============================================================
# 15. VALIDATION FUNCTION
# ============================================================

def validate():

    model.eval()

    running_loss = 0.0

    correct = 0

    total = 0


    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)

            labels = labels.to(device)


            outputs = model(images)


            loss = criterion(
                outputs,
                labels
            )


            running_loss += (
                loss.item()
            )


            _, predicted = torch.max(
                outputs,
                1
            )


            total += labels.size(0)


            correct += (
                predicted == labels
            ).sum().item()


    loss = (
        running_loss /
        len(val_loader)
    )


    accuracy = (
        correct /
        total
    )


    return loss, accuracy


# ============================================================
# 16. TRAINING
# ============================================================

print("\nStarting fine-tuning...")

best_val_accuracy = 0.0

training_start = time.time()

history = []


for epoch in range(EPOCHS):

    epoch_start = time.time()


    train_loss, train_accuracy = (
        train_one_epoch()
    )


    val_loss, val_accuracy = (
        validate()
    )


    scheduler.step(
        val_accuracy
    )


    epoch_time = (
        time.time() -
        epoch_start
    )


    print("\n" + "-" * 60)

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}]"
    )

    print(
        f"Train Loss      : {train_loss:.4f}"
    )

    print(
        f"Train Accuracy  : {train_accuracy:.4f}"
    )

    print(
        f"Val Loss        : {val_loss:.4f}"
    )

    print(
        f"Val Accuracy    : {val_accuracy:.4f}"
    )

    print(
        f"Epoch Time      : {epoch_time:.2f}s"
    )


    history.append({

        "epoch":
            epoch + 1,

        "train_loss":
            train_loss,

        "train_accuracy":
            train_accuracy,

        "validation_loss":
            val_loss,

        "validation_accuracy":
            val_accuracy,

        "epoch_time":
            epoch_time
    })


    # Save best model
    if val_accuracy > best_val_accuracy:

        best_val_accuracy = (
            val_accuracy
        )


        torch.save(

            model.state_dict(),

            MODEL_PATH
        )


        print(
            "Best fine-tuned model saved."
        )


training_time = (
    time.time() -
    training_start
)


# ============================================================
# 17. LOAD BEST MODEL
# ============================================================

print(
    "\nLoading best fine-tuned model..."
)

model.load_state_dict(

    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.eval()


# ============================================================
# 18. TEST MODEL
# ============================================================

print(
    "\nEvaluating on test dataset..."
)


all_labels = []

all_predictions = []


inference_start = time.time()


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)


        outputs = model(images)


        _, predictions = torch.max(
            outputs,
            1
        )


        all_labels.extend(
            labels.numpy()
        )


        all_predictions.extend(
            predictions.cpu().numpy()
        )


inference_time = (
    time.time() -
    inference_start
)


# ============================================================
# 19. CALCULATE METRICS
# ============================================================

test_accuracy = accuracy_score(

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


# ============================================================
# 20. RESULTS
# ============================================================

print("\n")

print("=" * 65)

print(
    "DEEPGUARD RESNET18 FINE-TUNED RESULTS"
)

print("=" * 65)


print(
    f"Best Validation Accuracy : "
    f"{best_val_accuracy:.4f}"
)


print(
    f"Test Accuracy            : "
    f"{test_accuracy:.4f}"
)


print(
    f"Precision                : "
    f"{precision:.4f}"
)


print(
    f"Recall                   : "
    f"{recall:.4f}"
)


print(
    f"F1 Score                 : "
    f"{f1:.4f}"
)


print()

print(
    f"Training Time            : "
    f"{training_time:.2f} seconds"
)


print(
    f"Inference Time           : "
    f"{inference_time:.2f} seconds"
)


print()

print(
    "Confusion Matrix:"
)

print(cm)


# ============================================================
# 21. SAVE RESULTS
# ============================================================

results = {

    "experiment":
        "DeepGuard ResNet18 Fine-Tuned",

    "dataset":
        "FaceForensics++",

    "model":
        "ResNet18 Transfer Learning - Layer4 Fine-Tuning",

    "device":
        str(device),

    "image_size":
        IMAGE_SIZE,

    "batch_size":
        BATCH_SIZE,

    "epochs":
        EPOCHS,

    "backbone_learning_rate":
        BACKBONE_LR,

    "classifier_learning_rate":
        CLASSIFIER_LR,

    "trainable_layers":
        [
            "layer4",
            "fc"
        ],

    "train_samples":
        len(train_dataset),

    "validation_samples":
        len(val_dataset),

    "test_samples":
        len(test_dataset),

    "best_validation_accuracy":
        best_val_accuracy,

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
        history,

    "model_path":
        str(MODEL_PATH)
}


with open(

    RESULT_PATH,

    "w"
) as f:

    json.dump(

        results,

        f,

        indent=4
    )


# ============================================================
# 22. DIAGNOSTIC
# ============================================================

print("\n")

print("=" * 65)

print(
    "AUTOMATIC DIAGNOSTIC"
)

print("=" * 65)


if test_accuracy >= 0.90:

    print(
        "Excellent performance."
    )

elif test_accuracy >= 0.80:

    print(
        "Good performance."
    )

elif test_accuracy >= 0.70:

    print(
        "Moderate performance."
    )

else:

    print(
        "Performance remains weak."
    )


print("\n")

print("=" * 65)

print(
    "EXPERIMENT COMPLETED"
)

print("=" * 65)


print("\nModel saved:")

print(
    MODEL_PATH
)


print("\nResults saved:")

print(
    RESULT_PATH
)