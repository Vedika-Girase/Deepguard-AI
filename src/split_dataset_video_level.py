import shutil
import random
import re
from pathlib import Path
from collections import defaultdict

# ============================================================
# DEEPGUARD - VIDEO LEVEL DATASET SPLITTING
# Experiment 2: Leakage-Free Video-Level Split
# ============================================================

# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

# Source face datasets
REAL_DIR = PROJECT_DIR / "data" / "processed" / "real_faces"
DEEPFAKE_DIR = PROJECT_DIR / "data" / "processed" / "deepfake_faces"

# New dataset - DO NOT overwrite Experiment 1
OUTPUT_DIR = PROJECT_DIR / "data" / "processed" / "video_dataset"

# Split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Reproducibility
RANDOM_SEED = 42

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def extract_video_id(filename):
    """
    Extract the original video ID from filenames such as:

        183_frame_0001.jpg
        183_253_frame_0001.jpg

    The part before '_frame_' is treated as the video ID.
    """

    match = re.match(r"^(.*?)_frame_\d+\.[^.]+$", filename)

    if match:
        return match.group(1)

    # Fallback:
    # Remove extension and frame-like suffix
    stem = Path(filename).stem

    if "_frame_" in stem:
        return stem.split("_frame_")[0]

    return stem


def collect_images_by_video(source_dir):
    """
    Read all images and group them according to original video ID.
    """

    videos = defaultdict(list)

    if not source_dir.exists():
        raise FileNotFoundError(
            f"\nSource directory does not exist:\n{source_dir}"
        )

    images = [
        p for p in source_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    for image_path in images:
        video_id = extract_video_id(image_path.name)
        videos[video_id].append(image_path)

    return videos


def split_video_ids(video_ids):
    """
    Split VIDEO IDs into train / validation / test.
    """

    video_ids = list(video_ids)

    random.shuffle(video_ids)

    total = len(video_ids)

    train_count = round(total * TRAIN_RATIO)
    val_count = round(total * VAL_RATIO)

    # Make sure at least one video remains for test
    if train_count + val_count >= total:
        val_count = max(1, total - train_count - 1)

    train_ids = video_ids[:train_count]

    val_ids = video_ids[
        train_count:train_count + val_count
    ]

    test_ids = video_ids[
        train_count + val_count:
    ]

    return train_ids, val_ids, test_ids


def copy_images(video_groups, selected_video_ids, destination):
    """
    Copy every frame belonging to selected videos.
    """

    destination.mkdir(parents=True, exist_ok=True)

    count = 0

    for video_id in selected_video_ids:

        for image_path in video_groups[video_id]:

            target = destination / image_path.name

            shutil.copy2(image_path, target)

            count += 1

    return count


# ============================================================
# DATASET INFORMATION
# ============================================================

def print_dataset_info(class_name, video_groups):

    total_frames = sum(
        len(images)
        for images in video_groups.values()
    )

    print(f"\n{class_name.upper()}")
    print("-" * 50)
    print(f"Videos : {len(video_groups)}")
    print(f"Frames : {total_frames}")

    print("\nVideo IDs:")

    for video_id in sorted(video_groups.keys()):
        print(
            f"  {video_id:<20} "
            f"{len(video_groups[video_id])} frames"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("DEEPGUARD - VIDEO LEVEL DATASET SPLITTING")
    print("=" * 70)

    print("\nExperiment 2")
    print("Creating leakage-free video-level dataset.")

    random.seed(RANDOM_SEED)

    # --------------------------------------------------------
    # Check source directories
    # --------------------------------------------------------

    print("\nChecking source directories...")

    print(f"REAL     : {REAL_DIR}")
    print(f"DEEPFAKE : {DEEPFAKE_DIR}")

    # --------------------------------------------------------
    # Collect images
    # --------------------------------------------------------

    print("\nCollecting REAL images...")

    real_videos = collect_images_by_video(
        REAL_DIR
    )

    print("Collecting DEEPFAKE images...")

    deepfake_videos = collect_images_by_video(
        DEEPFAKE_DIR
    )

    # --------------------------------------------------------
    # Display source information
    # --------------------------------------------------------

    print_dataset_info(
        "Real",
        real_videos
    )

    print_dataset_info(
        "Deepfake",
        deepfake_videos
    )

    # --------------------------------------------------------
    # Split each class independently
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("CREATING VIDEO-LEVEL SPLIT")
    print("=" * 70)

    real_train, real_val, real_test = split_video_ids(
        real_videos.keys()
    )

    fake_train, fake_val, fake_test = split_video_ids(
        deepfake_videos.keys()
    )

    # --------------------------------------------------------
    # Print split information
    # --------------------------------------------------------

    print("\nREAL VIDEO SPLIT")
    print("-" * 50)

    print("Train:", sorted(real_train))
    print("Validation:", sorted(real_val))
    print("Test:", sorted(real_test))

    print("\nDEEPFAKE VIDEO SPLIT")
    print("-" * 50)

    print("Train:", sorted(fake_train))
    print("Validation:", sorted(fake_val))
    print("Test:", sorted(fake_test))

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    if OUTPUT_DIR.exists():

        print("\nRemoving previous Experiment 2 dataset...")

        shutil.rmtree(OUTPUT_DIR)

    # --------------------------------------------------------
    # Create directory structure
    # --------------------------------------------------------

    splits = {
        "train": {
            "real": real_train,
            "deepfake": fake_train
        },

        "validation": {
            "real": real_val,
            "deepfake": fake_val
        },

        "test": {
            "real": real_test,
            "deepfake": fake_test
        }
    }

    total_counts = {}

    # --------------------------------------------------------
    # Copy frames
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("COPYING FRAMES")
    print("=" * 70)

    for split_name, classes in splits.items():

        total_counts[split_name] = {}

        for class_name, video_ids in classes.items():

            if class_name == "real":
                video_groups = real_videos
            else:
                video_groups = deepfake_videos

            destination = (
                OUTPUT_DIR
                / split_name
                / class_name
            )

            count = copy_images(
                video_groups,
                video_ids,
                destination
            )

            total_counts[split_name][class_name] = count

            print(
                f"{split_name:<12} "
                f"{class_name:<10} "
                f"videos={len(video_ids):<3} "
                f"frames={count}"
            )

    # --------------------------------------------------------
    # Final dataset summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("VIDEO-LEVEL DATASET SUMMARY")
    print("=" * 70)

    for split_name in ["train", "validation", "test"]:

        real_count = total_counts[split_name]["real"]
        fake_count = total_counts[split_name]["deepfake"]

        print(
            f"{split_name.upper():<12} "
            f"Real: {real_count:<4} "
            f"Deepfake: {fake_count:<4} "
            f"Total: {real_count + fake_count}"
        )

    # --------------------------------------------------------
    # Leakage verification
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("VIDEO-LEVEL LEAKAGE CHECK")
    print("=" * 70)

    train_videos = set(real_train + fake_train)
    val_videos = set(real_val + fake_val)
    test_videos = set(real_test + fake_test)

    train_val_overlap = train_videos & val_videos
    train_test_overlap = train_videos & test_videos
    val_test_overlap = val_videos & test_videos

    print(
        f"\nTrain ∩ Validation : "
        f"{len(train_val_overlap)} videos"
    )

    print(
        f"Train ∩ Test       : "
        f"{len(train_test_overlap)} videos"
    )

    print(
        f"Validation ∩ Test  : "
        f"{len(val_test_overlap)} videos"
    )

    # --------------------------------------------------------
    # Final verification
    # --------------------------------------------------------

    if (
        len(train_val_overlap) == 0
        and
        len(train_test_overlap) == 0
        and
        len(val_test_overlap) == 0
    ):

        print("\n" + "=" * 70)
        print("SUCCESS")
        print("=" * 70)

        print(
            "\nNo video-level overlap detected."
        )

        print(
            "The dataset is ready for Experiment 2."
        )

    else:

        print("\n" + "=" * 70)
        print("WARNING")
        print("=" * 70)

        print(
            "\nVideo overlap still exists."
        )

        print(
            "DO NOT train the models yet."
        )

    # --------------------------------------------------------
    # Output location
    # --------------------------------------------------------

    print("\nOutput dataset:")
    print(OUTPUT_DIR)

    print("\nDirectory structure:")

    print(
        """
video_dataset/
├── train/
│   ├── real/
│   └── deepfake/
│
├── validation/
│   ├── real/
│   └── deepfake/
│
└── test/
    ├── real/
    └── deepfake/
"""
    )

    print("=" * 70)
    print("VIDEO-LEVEL SPLITTING COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()