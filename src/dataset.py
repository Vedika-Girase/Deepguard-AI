from torchvision import datasets, transforms
from torch.utils.data import DataLoader


IMAGE_SIZE = 224
BATCH_SIZE = 16


transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])


train_dataset = datasets.ImageFolder(
    "data/processed/train",
    transform=transform
)

validation_dataset = datasets.ImageFolder(
    "data/processed/validation",
    transform=transform
)

test_dataset = datasets.ImageFolder(
    "data/processed/test",
    transform=transform
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


print("Classes:", train_dataset.classes)
print("Training images:", len(train_dataset))
print("Validation images:", len(validation_dataset))
print("Test images:", len(test_dataset))