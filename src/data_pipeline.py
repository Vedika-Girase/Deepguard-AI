from collections import Counter
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

IMAGE_SIZE = 224
BATCH_SIZE = 16
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(train=False, image_size=IMAGE_SIZE, color_jitter=False):
    if train:
        ops = [transforms.Resize((image_size, image_size)), transforms.RandomHorizontalFlip(), transforms.RandomRotation(10)]
        if color_jitter:
            ops.append(transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02))
        ops += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
        return transforms.Compose(ops)
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def build_datasets(data_dir, image_size=IMAGE_SIZE, color_jitter=False):
    root = Path(data_dir)
    train_ds = datasets.ImageFolder(root / "train", transform=get_transforms(True, image_size, color_jitter))
    val_ds = datasets.ImageFolder(root / "validation", transform=get_transforms(False, image_size))
    test_ds = datasets.ImageFolder(root / "test", transform=get_transforms(False, image_size))
    mappings = [train_ds.class_to_idx, val_ds.class_to_idx, test_ds.class_to_idx]
    if not (mappings[0] == mappings[1] == mappings[2]):
        raise ValueError(f"Class mappings differ: {mappings}")
    if len(train_ds.classes) != 2:
        raise ValueError(f"Expected binary classification, found {train_ds.classes}")
    return train_ds, val_ds, test_ds


def build_loaders(data_dir, batch_size=BATCH_SIZE, num_workers=0, image_size=IMAGE_SIZE, color_jitter=False):
    train_ds, val_ds, test_ds = build_datasets(data_dir, image_size, color_jitter)
    loaders = {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=False),
        "validation": DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False),
        "test": DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False),
    }
    return loaders, {"train": train_ds, "validation": val_ds, "test": test_ds}


def dataset_summary(datasets_map):
    result = {}
    for split, ds in datasets_map.items():
        counts = Counter(ds.targets)
        result[split] = {"total": len(ds), "classes": {name: int(counts[idx]) for name, idx in ds.class_to_idx.items()}}
    return result
