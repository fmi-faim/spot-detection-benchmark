import numpy as np
import tifffile
from pathlib import Path

# create the splits 
def load_split(folder):
    imgs, coords = [], []
    for tif in sorted(Path(folder).glob("*.tif")):
        csv = tif.with_suffix(".csv")
        imgs.append(tifffile.imread(tif))
        c = np.loadtxt(csv, delimiter=",", skiprows=1)   # drop the axis-0,axis-1 header
        c = np.atleast_2d(c).reshape(-1, 2)              # handle 0- and 1-spot files
        coords.append(c)
    return imgs, coords

splits = {
    'train': load_split(r"C:\Data\soucemma\spot-detection-benchmark\datasets\train_val_test_split\train"),
    'valid': load_split(r"C:\Data\soucemma\spot-detection-benchmark\datasets\train_val_test_split\val"),
    'test':  load_split(r"C:\Data\soucemma\spot-detection-benchmark\datasets\train_val_test_split\test"),
}

out_path = r"C:\Data\soucemma\spot-detection-benchmark\datasets\deepblink_dataset.npz"

def build_deepblink_npz(out_path, splits):
    """splits = {'train': (imgs, coords), 'valid': (...), 'test': (...)}
       imgs: list of (H,W) arrays, all same square power-of-two size
       coords: list of (M,2) arrays in (row, col) pixels"""
    arrays = {}
    for split in ('train', 'valid', 'test'): 
        imgs, coords = splits[split]
        arrays[f'x_{split}'] = np.array(imgs)
        # object array of per-image coordinate arrays
        y = np.empty(len(coords), dtype=object)
        for i, c in enumerate(coords):
            y[i] = np.asarray(c, dtype=np.float64).reshape(-1, 2)
        arrays[f'y_{split}'] = y

    np.savez_compressed(out_path, **arrays)

# generate the npz file
build_deepblink_npz(out_path, splits)

# verify it loads:
from deepblink.io import load_npz

x_train, y_train, x_valid, y_valid, x_test, y_test = load_npz(out_path) 

print("x_train shape:", x_train.shape, "y_train shape:", y_train.shape)
print("x_valid shape:", x_valid.shape, "y_valid shape:", y_valid.shape)
print("x_test shape:", x_test.shape, "y_test shape:", y_test.shape)

data = np.load(out_path, allow_pickle=True)
print(data)