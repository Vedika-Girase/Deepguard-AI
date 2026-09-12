from pathlib import Path
import shutil
from collections import defaultdict, Counter


# ============================================================
# DeepGuard
# Experiment 2: Source-Video-Independent Image Split
# ============================================================

SOURCE_DIR = Path("data/processed/dataset")
OUTPUT_DIR = Path("data/processed/source_independent")


# ------------------------------------------------------------
# 1. Fixed source-video assignment
# ------------------------------------------------------------
# 10 source videos are divided into:
# 6 train, 2 validation, 2 test
#
# This assignment is fixed so the experiment is reproducible.

SPLIT_IDS = {
    "train": {"183", "253", "469", "481", "585", "599"},
    "validation": {"672", "720"},
    "test": {"866", "878"},
}


# ------------------------------------------------------------
# 2. Collect images
# ------------------------------------------------------------

images = list(SOURCE_DIR.rglob("*.jpg"))

print(f"Total images found: {len(images)}")


# ------------------------------------------------------------
# 3. Determine source-video ID
# ------------------------------------------------------------

source_images = defaultdict(list)

for image_path in images:
    filename = image_path.name

    # Example:
    # 481_469_frame_0010.jpg
    #
    # Source-video ID = first part = 481

    parts = filename.split("_")

    if len(parts) < 3:
        print(f"WARNING: unexpected filename: {filename}")
        continue

    source_id = parts[0]

    source_images[source_id].append(image_path)


print(f"Unique source videos found: {len(source_images)}")


# ------------------------------------------------------------
# 4. Print source-video distribution
# ------------------------------------------------------------

print("\nSource-video distribution:")

for source_id in sorted(source_images, key=lambda x: int(x)):
    print(
        f"Video {source_id}: "
        f"{len(source_images[source_id])} images"
    )


# ------------------------------------------------------------
# 5. Check every source video has an assignment
# ------------------------------------------------------------

all_assigned_ids = set().union(*SPLIT_IDS.values())
found_ids = set(source_images.keys())

missing_assignments = found_ids - all_assigned_ids

if missing_assignments:
    raise ValueError(
        f"ERROR: These source videos have no split assignment: "
        f"{sorted(missing_assignments)}"
    )


# Check that no source video is assigned twice

for split1 in SPLIT_IDS:
    for split2 in SPLIT_IDS:
        if split1 >= split2:
            continue

        overlap = SPLIT_IDS[split1] & SPLIT_IDS[split2]

        if overlap:
            raise ValueError(
                f"ERROR: Source-video overlap between "
                f"{split1} and {split2}: {overlap}"
            )


# ------------------------------------------------------------
# 6. Create output directories
# ------------------------------------------------------------

for split in ["train", "validation", "test"]:
    for class_name in ["deepfake", "real"]:
        path = OUTPUT_DIR / split / class_name
        path.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# 7. Copy images
# ------------------------------------------------------------

counts = Counter()


for source_id, image_list in source_images.items():

    # Find split
    split = None

    for split_name, ids in SPLIT_IDS.items():
        if source_id in ids:
            split = split_name
            break

    if split is None:
        raise ValueError(
            f"No split found for source video {source_id}"
        )

    for image_path in image_list:

        # Original dataset structure:
        # .../train/deepfake/image.jpg
        # .../train/real/image.jpg

        class_name = image_path.parent.name

        if class_name not in {"deepfake", "real"}:
            raise ValueError(
                f"Unexpected class folder: {class_name}"
            )

        destination = (
            OUTPUT_DIR
            / split
            / class_name
            / image_path.name
        )

        shutil.copy2(image_path, destination)

        counts[(split, class_name)] += 1


# ------------------------------------------------------------
# 8. Print final distribution
# ------------------------------------------------------------

print("\n==========================================")
print("SOURCE-INDEPENDENT DATASET CREATED")
print("==========================================")

total = 0

for split in ["train", "validation", "test"]:

    deepfake_count = counts[(split, "deepfake")]
    real_count = counts[(split, "real")]

    split_total = deepfake_count + real_count
    total += split_total

    print(f"\n{split.upper()}")
    print(f"  Deepfake : {deepfake_count}")
    print(f"  Real     : {real_count}")
    print(f"  Total    : {split_total}")


print("\n------------------------------------------")
print(f"TOTAL IMAGES: {total}")
print("------------------------------------------")

print("\nSource-video assignment:")

for split, ids in SPLIT_IDS.items():
    print(f"{split}: {sorted(ids, key=lambda x: int(x))}")


print("\nNo source video is shared between splits.")
print("Original dataset was NOT modified.")
print("Experiment 1 dataset remains preserved.")