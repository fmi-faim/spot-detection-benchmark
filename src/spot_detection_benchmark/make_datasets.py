import argparse
import logging
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from spot_detection_benchmark.utils import (
    list_images,
    load_coords,
    setup_logging,
    subset_of,
)

log = logging.getLogger(__name__)

SPLITS = ("train", "val", "test")


def list_pairs(images_dir, csv_dir=None):
    csv_dir = Path(csv_dir or images_dir)
    pairs = []
    for image in list_images(images_dir):
        csv = csv_dir / f"{image.stem}.csv"
        if not csv.exists():
            log.warning(f"no CSV for {image.name}, skipped")
            continue
        pairs.append((image, csv))
    return pairs


def assign_splits(pairs, val_fraction=0.15, test_fraction=0.05, seed=42):
    by_type = defaultdict(list)
    for pair in pairs:
        by_type[subset_of(pair[0].stem)].append(pair)

    rng = random.Random(seed)
    split_of = {}
    for _, type_pairs in sorted(by_type.items()):
        type_pairs = sorted(type_pairs)
        rng.shuffle(type_pairs)
        n_val = round(len(type_pairs) * val_fraction)
        n_test = round(len(type_pairs) * test_fraction)
        for i, pair in enumerate(type_pairs):
            if i < n_val:
                split_of[pair] = "val"
            elif i < n_val + n_test:
                split_of[pair] = "test"
            else:
                split_of[pair] = "train"
    return split_of


def split_dataset(
    images_dir,
    csv_dir,
    output_dir,
    val_fraction=0.15,
    test_fraction=0.05,
    seed=42,
    dry_run=False,
):
    output_dir = Path(output_dir)
    if not dry_run and output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"{output_dir} exists and is not empty: choose another --output_dir or delete it to split again"
        )
    pairs = list_pairs(images_dir, csv_dir)
    split_of = assign_splits(pairs, val_fraction, test_fraction, seed)
    table = pd.DataFrame(
        [
            {"image": image.stem, "subset": subset_of(image.stem), "split": split}
            for (image, _), split in split_of.items()
        ]
    ).sort_values(["split", "image"])

    counts = Counter(zip(table["subset"], table["split"]))
    log.info(f"{'type':<20}" + "".join(f"{s:>8}" for s in SPLITS))
    for image_type in sorted(table["subset"].unique()):
        log.info(
            f"{image_type:<20}"
            + "".join(f"{counts[(image_type, s)]:>8}" for s in SPLITS)
        )
    log.info(f"total: {len(table)} images (dry run: {dry_run})")

    if not dry_run:
        for split in SPLITS:
            (output_dir / split).mkdir(parents=True, exist_ok=True)
        for (image, csv), split in split_of.items():
            shutil.copy2(image, output_dir / split / image.name)
            shutil.copy2(csv, output_dir / split / csv.name)
        table.to_csv(output_dir / "split.csv", index=False)
        log.info(f"split written to {output_dir}")
    return table


def to_piscis(split_dir, output_dir):
    split_dir, output_dir = Path(split_dir), Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"{output_dir} exists and is not empty: choose another --output_dir or delete it to build Piscis dataset again"
        )
    for split in SPLITS:
        x_dir = output_dir / split / "x"
        y_dir = output_dir / split / "y"
        x_dir.mkdir(parents=True, exist_ok=True)
        y_dir.mkdir(parents=True, exist_ok=True)
        pairs = list_pairs(split_dir / split)
        for image, csv in pairs:
            shutil.copy(image, x_dir / image.name)
            spots = pd.read_csv(csv)[["axis-0", "axis-1"]]
            spots.to_csv(y_dir / f"{image.stem}.csv", index=False, header=False)
        log.info(f"{split:<6} {len(pairs)} pairs")
    log.info(f"Piscis dataset written to {output_dir}")


def to_deepblink(split_dir, output_path):
    split_dir = Path(split_dir)
    arrays = {}
    for split, npz_name in zip(SPLITS, ("train", "valid", "test")):
        pairs = list_pairs(split_dir / split)
        images = [tifffile.imread(image) for image, _ in pairs]
        shapes = {image.shape for image in images}
        if len(shapes) > 1:
            raise ValueError(
                f"deepBlink needs images of one size, {split}/ has {sorted(shapes)}"
            )
        spots = np.empty(len(pairs), dtype=object)  # one (n, 2) array per image
        for i, (_, csv) in enumerate(pairs):
            spots[i] = load_coords(csv).reshape(-1, 2)
        arrays[f"x_{npz_name}"] = np.array(images)
        arrays[f"y_{npz_name}"] = spots
        log.info(f"{split:<6} {len(pairs)} images {shapes.pop() if shapes else ''}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays)
    log.info(f"deepBlink dataset written to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Build the reference split and each tool's training dataset."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    split = commands.add_parser("split", help="stratified train / val / test split")
    split.add_argument("--images_dir", required=True, help="Folder of .tif images")
    split.add_argument(
        "--csv_dir", default=None, help="Folder of spot CSVs (default: images_dir)"
    )
    split.add_argument("--output_dir", required=True, help="Where the split is written")
    split.add_argument("--val_fraction", type=float, default=0.15)
    split.add_argument("--test_fraction", type=float, default=0.05)
    split.add_argument("--seed", type=int, default=42)
    split.add_argument(
        "--dry_run", action="store_true", help="Only print the per-type counts"
    )

    piscis = commands.add_parser("piscis", help="split -> Piscis layout")
    piscis.add_argument("--split_dir", required=True, help="Folder made by 'split'")
    piscis.add_argument("--output_dir", required=True, help="Piscis dataset folder")

    deepblink = commands.add_parser("deepblink", help="split -> deepBlink .npz")
    deepblink.add_argument("--split_dir", required=True, help="Folder made by 'split'")
    deepblink.add_argument("--output", required=True, help="Path of the .npz file")

    args = parser.parse_args()
    setup_logging()
    if args.command == "split":
        split_dataset(
            args.images_dir,
            args.csv_dir,
            args.output_dir,
            val_fraction=args.val_fraction,
            test_fraction=args.test_fraction,
            seed=args.seed,
            dry_run=args.dry_run,
        )
    elif args.command == "piscis":
        to_piscis(args.split_dir, args.output_dir)
    else:
        to_deepblink(args.split_dir, args.output)


if __name__ == "__main__":
    main()
