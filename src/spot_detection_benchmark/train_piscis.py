import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse

parser = argparse.ArgumentParser(description="Train Piscis model")
parser.add_argument("--input_dir", type=str, help="Path to the input directory")

args = parser.parse_args()

import piscis.paths
from piscis.training import train_model

# TODO: add environment variables
piscis.paths.PISCIS_DIR = r"C:\Data\soucemma\spot-detection-benchmark\models\.piscis"
piscis.paths.PISCIS_MODEL_DIR = (
    r"C:\Data\soucemma\spot-detection-benchmark\models\.piscis\models"
)

train_model(
    model_name="piscis_customed",
    dataset_path=args.input_dir,
    input_size=(256, 256),
    device="cuda",
)
