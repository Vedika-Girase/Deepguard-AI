from pathlib import Path
from collections import defaultdict

# ============================================================
# DeepGuard - Experiment 2 Video-Level Dataset Audit
# ============================================================

DATASET_ROOT = Path("data/processed/video_dataset")

SPLITS = ["train", "validation", "test"]
CLASSES = ["deepfake", "real"]

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def extract_video_id(filename):
    """
    Extract video ID from face-frame filename.

    Expected examples:
        183_0001.jpg
        253_0045.jpg

    The first part before '_' is treated as the video ID.
    """

    stem = Path(filename).stem

    if "_" in stem:
        return stem.split("_")[0]

    return stem


def collect_video_ids(split_path):
    """
    Collect unique video IDs for a dataset split.
    """

    video_ids = set()
    image_count = 0
    class_counts = {}

    for class_name in CLASSES:

        class_path = split_path / class_name

        if not class_path.exists():
            print(f"WARNING: Missing folder: {class_path}")
            class_counts[class_name] = 0
            continue

        count = 0

        for file in class_path.iterdir():

            if file.is_file() and file.suffix.lower() in VALID_EXTENSIONS:

                count += 1
                image_count += 1

                video_id = extract_video_id(file.name)
                video_ids.add(video_id)

        class_counts[class_name] = count

    return video_ids, image_count, class_counts


def main():

    print("=" * 70)
    print("DeepGuard - Experiment 2 Video-Level Dataset Audit")
    print("=" * 70)

    if not DATASET_ROOT.exists():
        print("\nERROR: Dataset folder not found:")
        print(DATASET_ROOT)
        return

    split_data = {}

    # --------------------------------------------------------
    # Collect information
    # --------------------------------------------------------

    for split in SPLITS:

        split_path = DATASET_ROOT / split

        video_ids, image_count, class_counts = collect_video_ids(split_path)

        split_data[split] = {
            "video_ids": video_ids,
            "image_count": image_count,
            "class_counts": class_counts
        }

        print(f"\n{split.upper()}")
        print("-" * 70)

        print(f"Images       : {image_count}")
        print(f"Unique videos: {len(video_ids)}")

        print(f"Deepfake     : {class_counts['deepfake']}")
        print(f"Real         : {class_counts['real']}")

        print("Video IDs:")
        print(sorted(video_ids))

    # --------------------------------------------------------
    # Check overlap
    # --------------------------------------------------------

    train_ids = split_data["train"]["video_ids"]
    val_ids = split_data["validation"]["video_ids"]
    test_ids = split_data["test"]["video_ids"]

    train_val = train_ids.intersection(val_ids)
    train_test = train_ids.intersection(test_ids)
    val_test = val_ids.intersection(test_ids)

    print("\n" + "=" * 70)
    print("VIDEO-LEVEL LEAKAGE CHECK")
    print("=" * 70)

    print(f"\nTrain ∩ Validation : {len(train_val)}")
    print(f"Train ∩ Test       : {len(train_test)}")
    print(f"Validation ∩ Test  : {len(val_test)}")

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    if len(train_val) == 0 and len(train_test) == 0 and len(val_test) == 0:

        print("\n✓ PASS")
        print("No video IDs overlap between train, validation and test.")
        print("The dataset satisfies the video-level split requirement.")

    else:

        print("\n✗ FAIL")
        print("Video-level leakage detected!")

        if train_val:
            print("\nTrain/Validation overlapping IDs:")
            print(sorted(train_val))

        if train_test:
            print("\nTrain/Test overlapping IDs:")
            print(sorted(train_test))

        if val_test:
            print("\nValidation/Test overlapping IDs:")
            print(sorted(val_test))

    # --------------------------------------------------------
    # Check class folders
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)

    total_images = 0

    for split in SPLITS:

        count = split_data[split]["image_count"]
        total_images += count

        print(
            f"{split:<12}: "
            f"{count:>4} images | "
            f"Deepfake = {split_data[split]['class_counts']['deepfake']:>3} | "
            f"Real = {split_data[split]['class_counts']['real']:>3} | "
            f"Videos = {len(split_data[split]['video_ids']):>3}"
        )

    print("-" * 70)
    print(f"{'TOTAL':<12}: {total_images:>4} images")

    print("\nAudit completed.")


if __name__ == "__main__":
    main()