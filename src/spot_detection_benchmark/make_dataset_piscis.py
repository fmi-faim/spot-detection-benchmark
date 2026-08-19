import shutil
from pathlib import Path
import pandas as pd

# --- paths ---
CANON  = Path(r"C:\Data\soucemma\spot-detection-benchmark\datasets\train_val_test_split")
PISCIS = Path(r"C:\Data\soucemma\spot-detection-benchmark\datasets\train_val_test_split_piscis")

SPLITS = ("train", "val", "test")

def build_piscis_dataset(canon, dst, splits):
    for split in splits:
        src = canon / split
        if not src.is_dir():
            print(f"skip: {src} not found")
            continue

        x_dir = dst / split / "x"     # images
        y_dir = dst / split / "y"     # labels
        x_dir.mkdir(parents=True, exist_ok=True)
        y_dir.mkdir(parents=True, exist_ok=True)

        n = 0
        for tif in sorted(src.glob("*.tif")):
            csv = tif.with_suffix(".csv")
            if not csv.exists():
                print(f"  WARNING: no label for {tif.name}, skipping")
                continue

            # image: straight copy
            shutil.copy(tif, x_dir / tif.name)

            # label: canonical has an axis-0,axis-1 header; Piscis wants (row,col) NO header
            coords = pd.read_csv(csv)[["axis-0", "axis-1"]]
            coords.to_csv(y_dir / f"{tif.stem}.csv", index=False, header=False)
            n += 1

        print(f"{split:6s}  ({n} pairs)")

build_piscis_dataset(CANON, PISCIS, SPLITS)
print("\nDone. Piscis dataset at:", PISCIS)

