import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse

from spotiflow.model import Spotiflow
from spotiflow.utils import get_data

parser = argparse.ArgumentParser(description="Train Spotiflow model")
parser.add_argument(
    "--input_dir", type=str, help="Path to the input directory containing training data"
)
parser.add_argument(
    "--output_dir",
    type=str,
    help="Path to the output directory where the model will be saved",
)
args = parser.parse_args()

INPUT_DIR = args.input_dir
OUTPUT_DIR = args.output_dir

train_imgs, train_spots, val_imgs, val_spots = get_data(INPUT_DIR)

model = Spotiflow()

model.fit(
    train_imgs,
    train_spots,
    val_imgs,
    val_spots,
    save_dir=OUTPUT_DIR,
)
