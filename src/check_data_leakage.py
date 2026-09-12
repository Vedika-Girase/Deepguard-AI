from pathlib import Path
from collections import defaultdict
import re


# ============================================================
# DEEPGUARD - DATA LEAKAGE AUDIT
# ============================================================

DATASET_DIR = Path("data/processed/dataset")

TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "validation"
TEST_DIR = DATASET_DIR / "test"


# ============================================================
# EXTRACT ORIGINAL VIDEO ID
# ============================================================

def get_video_id(filename):
    """
    Example:
        183_frame_0004.jpg
        183_frame_0014.jpg

    Both belong to video ID: 183
    """

    name = Path(filename).stem

    # Pattern: videoID_frame_number
    match = re.match(r"(.+?)_frame_\d+$", name)

    if match:
        return match.group(1)

    # Fallback
    return name


# ============================================================
# COLLECT VIDEO IDs
# ============================================================

def collect_video_ids(folder):

    video_ids = set()
    files = []

    if not folder.exists():
        print(f"WARNING: Folder does not exist: {folder}")
        return video_ids, files

    for file in folder.rglob("*"):

        if file.suffix.lower() in [".jpg", ".jpeg", ".png"]:

            video_id = get_video_id(file.name)

            video_ids.add(video_id)
            files.append(file)

    return video_ids, files


# ============================================================
# LOAD DATA
# ============================================================

print("\n" + "=" * 70)
print("DEEPGUARD - DATA LEAKAGE AUDIT")
print("=" * 70)

print("\nScanning dataset...\n")


splits = {
    "TRAIN": TRAIN_DIR,
    "VALIDATION": VAL_DIR,
    "TEST": TEST_DIR
}

video_sets = {}
file_sets = {}

for split_name, folder in splits.items():

    video_ids, files = collect_video_ids(folder)

    video_sets[split_name] = video_ids
    file_sets[split_name] = files

    print(f"{split_name}")
    print("-" * 40)
    print(f"Images     : {len(files)}")
    print(f"Video IDs  : {len(video_ids)}")


# ============================================================
# CHECK OVERLAP
# ============================================================

print("\n" + "=" * 70)
print("VIDEO-LEVEL OVERLAP ANALYSIS")
print("=" * 70)


train_ids = video_sets["TRAIN"]
val_ids = video_sets["VALIDATION"]
test_ids = video_sets["TEST"]


train_val_overlap = train_ids & val_ids
train_test_overlap = train_ids & test_ids
val_test_overlap = val_ids & test_ids


print("\nTrain ∩ Validation:")
print(f"Overlapping videos: {len(train_val_overlap)}")

print("\nTrain ∩ Test:")
print(f"Overlapping videos: {len(train_test_overlap)}")

print("\nValidation ∩ Test:")
print(f"Overlapping videos: {len(val_test_overlap)}")


# ============================================================
# SHOW OVERLAPPING VIDEO IDs
# ============================================================

def show_examples(name, overlap):

    if not overlap:
        print(f"\n{name}: NONE")
        return

    print(f"\n{name} examples:")

    for video_id in sorted(overlap)[:20]:
        print(f"  {video_id}")


show_examples(
    "TRAIN / VALIDATION overlap",
    train_val_overlap
)

show_examples(
    "TRAIN / TEST overlap",
    train_test_overlap
)

show_examples(
    "VALIDATION / TEST overlap",
    val_test_overlap
)


# ============================================================
# FILE-LEVEL DUPLICATES
# ============================================================

print("\n" + "=" * 70)
print("EXACT FILE DUPLICATE CHECK")
print("=" * 70)


train_files = {file.name for file in file_sets["TRAIN"]}
val_files = {file.name for file in file_sets["VALIDATION"]}
test_files = {file.name for file in file_sets["TEST"]}


train_test_files = train_files & test_files
train_val_files = train_files & val_files
val_test_files = val_files & test_files


print(f"\nTrain / Validation duplicate files : {len(train_val_files)}")
print(f"Train / Test duplicate files       : {len(train_test_files)}")
print(f"Validation / Test duplicate files  : {len(val_test_files)}")


# ============================================================
# CLASS-WISE ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("CLASS-WISE VIDEO ANALYSIS")
print("=" * 70)


for split_name, folder in splits.items():

    print(f"\n{split_name}")

    for class_name in ["real", "deepfake"]:

        class_folder = folder / class_name

        if not class_folder.exists():
            print(f"  {class_name}: folder not found")
            continue

        files = [
            f for f in class_folder.rglob("*")
            if f.suffix.lower() in [".jpg", ".jpeg", ".png"]
        ]

        videos = {
            get_video_id(f.name)
            for f in files
        }

        print(
            f"  {class_name:8} | "
            f"frames: {len(files):4} | "
            f"videos: {len(videos):4}"
        )


# ============================================================
# FINAL DIAGNOSIS
# ============================================================

print("\n" + "=" * 70)
print("LEAKAGE AUDIT DIAGNOSIS")
print("=" * 70)


if len(train_test_overlap) > 0:

    print("""
WARNING: POTENTIAL DATA LEAKAGE DETECTED

The same original videos appear in both TRAIN and TEST.

Your current test accuracy may therefore be overly optimistic.

Recommended action:
Use VIDEO-LEVEL splitting and retrain the models.
""")

elif len(train_val_overlap) > 0:

    print("""
WARNING: POTENTIAL DATA LEAKAGE DETECTED

The same original videos appear in TRAIN and VALIDATION.

Recommended action:
Use VIDEO-LEVEL splitting and retrain the models.
""")

elif len(val_test_overlap) > 0:

    print("""
WARNING: Validation/Test overlap detected.

The same original videos appear in validation and test.
Use video-level splitting before final reporting.
""")

else:

    print("""
GOOD NEWS:

No video-level overlap was detected between
TRAIN, VALIDATION and TEST.

The current split appears to be video-independent.
""")


# ============================================================
# FINAL RESEARCH NOTE
# ============================================================

print("\n" + "=" * 70)
print("RESEARCH RECOMMENDATION")
print("=" * 70)

print("""
Regardless of the result, keep the current experiment unchanged.

These results will be treated as EXPERIMENT 1.

Next:
1. Verify dataset independence.
2. If necessary, create a video-level split.
3. Retrain the models.
4. Compare performance.
5. Test the best model on Celeb-DF.
""")


print("\nLeakage audit completed.")