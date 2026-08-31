import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

# Config
INPUT_DIR_IMAGES = Path(
    r"C:\Data\soucemma\spot-detection-benchmark\datasets\FISH_spots\2d\image"
)
INPUT_DIR_CSV = Path(
    r"C:\Data\soucemma\spot-detection-benchmark\datasets\FISH_spots\2d\csv"
)
OUTPUT_DIR = Path(
    r"C:\Data\soucemma\spot-detection-benchmark\datasets\train_val_test_split"
)
IMAGE_EXTS = {".tif", ".tiff"}
VALIDATION_SPLIT = 0.15
TEST_SPLIT = 0.05
SEED = 42
DRY_RUN = True  # Set to False to actually copy files

TRAIN_DIR = OUTPUT_DIR / "train"
VALIDATION_DIR = OUTPUT_DIR / "val"
TEST_DIR = OUTPUT_DIR / "test"


def get_type(stem):
    return stem.split("_")[0]


# Find the label
def find_labels(stem):
    labels = []
    exact = INPUT_DIR_CSV / f"{stem}.csv"
    if exact.exists():
        labels.append(exact)
    labels += sorted(INPUT_DIR_CSV.glob(f"{stem}_*.csv"))
    return labels


# Create training and validation directories
TRAIN_DIR.mkdir(exist_ok=True)
VALIDATION_DIR.mkdir(exist_ok=True)
TEST_DIR.mkdir(exist_ok=True)

# Collect all images and group them by type
images = [p for p in INPUT_DIR_IMAGES.iterdir() if p.suffix.lower() in IMAGE_EXTS]
by_type = defaultdict(list)
for img in images:
    img_type = get_type(img.stem)
    by_type[img_type].append(img)


# Stratified split of images into training and validation sets
rng = random.Random(SEED)
plan = []
for typ, files in sorted(by_type.items()):
    files = sorted(files)
    rng.shuffle(files)
    n_val = round(len(files) * VALIDATION_SPLIT)
    valid_set = set(files[:n_val])
    n_test = round(len(files) * TEST_SPLIT)
    test_set = set(files[n_val : n_val + n_test])
    for f in files:
        if f in test_set:
            split = "test"
        elif f in valid_set:
            split = "validation"
        else:
            split = "training"
        plan.append((f, find_labels(f.stem), split))

# copy images and csv files
for img, labels, split in plan:
    dest = (
        TRAIN_DIR
        if split == "training"
        else VALIDATION_DIR
        if split == "validation"
        else TEST_DIR
    )
    print(f"{split:<11}{img.name}  (+{len(labels)} csv)")
    if not DRY_RUN:
        shutil.copy2(str(img), str(dest / img.name))
        for lab in labels:
            shutil.copy2(str(lab), str(dest / lab.name))

# Report split statistics
counts = Counter((get_type(img.stem), split) for img, _, split in plan)
print(f"{'type':<20} {'training':>7} {'validation':>7} {'test':>7}")
for typ in sorted(by_type):
    t = counts[(typ, "training")]
    v = counts[(typ, "validation")]
    test = counts[(typ, "test")]
    print(f"{typ:<20}{t:>7}{v:>7}{test:>7}")
print(f"\nTotal: {len(images)} images   DRY_RUN={DRY_RUN}\n")
