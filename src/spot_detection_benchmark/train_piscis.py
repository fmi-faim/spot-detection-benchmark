import os
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse

parser = argparse.ArgumentParser(description="Train Piscis model")
parser.add_argument("--input_dir", type=str, help="Path to the input directory")

args = parser.parse_args()
REPO_ROOT = Path(__file__).resolve().parents[2]

models_dir = Path(
    os.environ.get("PISCIS_MODELS_DIR", REPO_ROOT / "models" / ".piscis" / "models")
)
checkpoints_dir = Path(
    os.environ.get(
        "PISCIS_CHECKPOINTS_DIR", REPO_ROOT / "models" / ".piscis" / "checkpoints"
    )
)
models_dir.mkdir(parents=True, exist_ok=True)
checkpoints_dir.mkdir(parents=True, exist_ok=True)


import piscis.core
import piscis.paths
import piscis.training
from piscis.training import train_model

piscis.paths.MODELS_DIR = models_dir
piscis.core.MODELS_DIR = models_dir
piscis.training.MODELS_DIR = models_dir
piscis.training.CHECKPOINTS_DIR = checkpoints_dir

train_model(
    model_name="piscis_customed_512",
    dataset_path=args.input_dir,
    input_size=(512, 512),
    device="cuda",
)
