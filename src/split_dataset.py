from pathlib import Path
import random
import shutil


# Source folders
REAL_SOURCE = Path("data/processed/real_faces")
FAKE_SOURCE = Path("data/processed/deepfake_faces")

# Dataset output
BASE_OUTPUT = Path("data/dataset")


# Split ratios
TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

random.seed(42)


def split_class(source_dir, class_name):

    images = list(source_dir.glob("*.jpg"))

    random.shuffle(images)

    total = len(images)

    train_end = int(total * TRAIN_RATIO)
    validation_end = train_end + int(total * VALIDATION_RATIO)

    train_images = images[:train_end]
    validation_images = images[train_end:validation_end]
    test_images = images[validation_end:]

    splits = {
        "train": train_images,
        "validation": validation_images,
        "test": test_images
    }

    for split_name, split_images in splits.items():

        output_dir = BASE_OUTPUT / split_name / class_name
        output_dir.mkdir(parents=True, exist_ok=True)

        for image in split_images:
            shutil.copy2(
                image,
                output_dir / image.name
            )

        print(
            f"{class_name} | "
            f"{split_name}: {len(split_images)}"
        )


# Split both classes
split_class(REAL_SOURCE, "real")
split_class(FAKE_SOURCE, "deepfake")


print("\nDataset splitting completed!")