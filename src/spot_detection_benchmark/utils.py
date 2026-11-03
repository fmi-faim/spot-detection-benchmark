"""Small helpers shared by several modules.

This module imports no detection tool and no yaml, so that every pixi environment
(the default one included, where evaluate.py and make_datasets.py run) can import it.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

REPO_ROOT = Path(__file__).resolve().parents[2]

MODES = ["2d", "2d_stack", "3d"]
IMAGE_EXTS = {".tif", ".tiff"}


def setup_logging(level=logging.INFO):
    logging.basicConfig(level=level, format="%(message)s")


def get_device():
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def set_piscis_dir(piscis_dir=None):
    import piscis.convert
    import piscis.core
    import piscis.downloads
    import piscis.paths
    import piscis.training

    piscis_dir = Path(
        piscis_dir or os.environ.get("PISCIS_DIR", REPO_ROOT / "models" / "piscis")
    )
    models_dir = piscis_dir / "models"
    checkpoints_dir = piscis_dir / "checkpoints"
    models_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    piscis.paths.PISCIS_DIR = piscis_dir
    for module in (
        piscis.paths,
        piscis.core,
        piscis.training,
        piscis.downloads,
        piscis.convert,
    ):
        module.MODELS_DIR = models_dir
    for module in (piscis.paths, piscis.training):
        module.CHECKPOINTS_DIR = checkpoints_dir
    return models_dir, checkpoints_dir


def given_options(**options):
    kept = {}
    for name, value in options.items():
        if value is not None:
            kept[name] = value
    return kept


def list_images(folder):
    return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMAGE_EXTS)


def subset_of(stem):
    return stem.split("_")[0]


def load_coords(csv):
    df = pd.read_csv(csv, float_precision="round_trip")
    cols = sorted(
        (c for c in df.columns if c.startswith("axis-")),
        key=lambda c: int(c.split("-")[1]),
    )
    if not cols:
        raise ValueError(f"{csv}: no axis-* column")
    return df[cols].to_numpy(float) if len(df) else np.empty((0, len(cols)))


def results_dirname(tool, pretrained=False):
    name = f"results_{tool}"
    return f"{name}_pretrained" if pretrained else name


def write_run_info(output_dir, **info):
    info = {
        **info,
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    path = Path(output_dir) / "run_info.json"
    path.write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
    return path
