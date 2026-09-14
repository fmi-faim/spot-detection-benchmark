import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_MODELS = {
    "deepblink": REPO_ROOT
    / "models"
    / "model_deepblink"
    / "260827_095918_deepBlink_is_sweet.h5",
    "spotiflow": REPO_ROOT / "models" / "model_spotiflow",
    "piscis": "piscis_customed",
    "ufish": REPO_ROOT / "models" / "model_ufish" / "best_model.pth",
}

# Default detection threshold per tool (used when threshold=None)
DEFAULT_THRESHOLDS = {
    "deepblink": 0.5,  # grid-cell probability cutoff
    "spotiflow": None,  # None → uses model's own stored optimal threshold
    "piscis": 0.5,  # pooled-label confidence
    "ufish": 0.5,  # enhanced-image intensity cutoff
}


def _save_output_csv(coords, out_csv):
    """Write (row, col) coords as a napari-ready axis-0,axis-1 CSV."""
    df = pd.DataFrame(np.asarray(coords).reshape(-1, 2), columns=["axis-0", "axis-1"])
    df.to_csv(out_csv, index=False)


def run_inference(tool, input_dir, output_root, model=None, threshold=None):
    """
    Run inference using the specified tool on the input directory and save results to the output directory.

    Args:
        tool (str): The name of the tool to use for inference.
        input_dir (str): Path to the directory containing input images.
        output_root (str): Path to the root directory where csv results will be saved.
        model (str): Path to the trained model file.
            - for deepblink, this is the path to the .h5 model file.
            - for spotiflow, this is the path to the folder containing the model.
            - for piscis, this is the name of the pre-trained model.
            - for ufish, this is the path to the .pth model file.
        threshold (float, optional): Detection threshold. If None, uses the tool's default.
            - deepblink: probability cutoff (default: 0.5)
            - spotiflow: prob_thresh (default: None → model's own stored threshold)
            - piscis: spot detection threshold (default: 0.5)
            - ufish: intensity threshold on enhanced image (default: 0.5)
    """
    if sys.platform.startswith("win"):
        device = "cpu"
    else:
        device = "cuda"

    if model is None:
        model = DEFAULT_MODELS[tool]
    else:
        model = Path(model) if tool != "piscis" else model

    # resolve threshold: use provided value or fall back to tool default
    if threshold is None:
        threshold = DEFAULT_THRESHOLDS[tool]

    input_dir = Path(input_dir)
    output_dir = Path(output_root) / f"results_{tool}"
    output_dir.mkdir(parents=True, exist_ok=True)
    images = sorted(input_dir.glob("*.tif"))

    if not images:
        raise FileNotFoundError(f"No .tif images found in {input_dir}")
    print(f"[{tool}] found {len(images)} images -> {output_dir}")
    print(f"[{tool}] threshold: {threshold}")

    if tool == "deepblink":
        from deepblink.inference import predict
        from deepblink.io import load_model

        m = load_model(model)
        for tif in images:
            image = tifffile.imread(tif)
            coords = predict(image=image, model=m, probability=threshold)
            coords = np.asarray(coords)
            if coords.ndim == 2 and coords.shape[1] > 2:
                coords = coords[:, :2]  # drop probability column
            _save_output_csv(coords, output_dir / f"{tif.stem}.csv")

    elif tool == "spotiflow":
        from spotiflow.model import Spotiflow

        m = Spotiflow.from_folder(model, map_location=device)
        for tif in images:
            image = tifffile.imread(tif)
            # threshold=None → Spotiflow uses its own stored optimal prob_thresh
            coords, _ = m.predict(image, device=device, prob_thresh=threshold)
            _save_output_csv(coords, output_dir / f"{tif.stem}.csv")

    elif tool == "piscis":
        import piscis.core
        import piscis.paths

        models_dir = Path(
            os.environ.get(
                "PISCIS_MODELS_DIR", REPO_ROOT / "models" / ".piscis" / "models"
            )
        )
        models_dir.mkdir(parents=True, exist_ok=True)
        piscis.paths.MODELS_DIR = models_dir
        piscis.core.MODELS_DIR = models_dir
        print(piscis.core.MODELS_DIR)

        from piscis import Piscis

        m = Piscis(model_name=model, device=device)
        for tif in images:
            image = tifffile.imread(tif)
            coords = m.predict(image, threshold=threshold)
            _save_output_csv(coords, output_dir / f"{tif.stem}.csv")

    elif tool == "ufish":
        from ufish.api import UFish

        UFISH_CHUNK_SIZE = 512
        uf = UFish()
        uf.load_weights_from_path(model)
        for tif in images:
            if max(tifffile.imread(tif).shape) > UFISH_CHUNK_SIZE:
                print(
                    f"Warning: {tif.name} is larger than {UFISH_CHUNK_SIZE} pixels. "
                    "UFish will process it in chunks, which may affect performance."
                )
                spots_df, _ = uf.predict_chunks(
                    tifffile.imread(tif),
                    chunk_size=(UFISH_CHUNK_SIZE, UFISH_CHUNK_SIZE),
                    intensity_threshold=threshold,
                )
            else:
                spots_df, _ = uf.predict(
                    tifffile.imread(tif),
                    intensity_threshold=threshold,
                )
            spots_df[["axis-0", "axis-1"]].to_csv(
                output_dir / f"{tif.stem}.csv", index=False
            )

    else:
        raise ValueError(
            f"unknown tool: {tool}. Please choose from 'deepblink', 'spotiflow', 'piscis', or 'ufish'."
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run inference using a specified tool on input images."
    )
    parser.add_argument(
        "--tool",
        type=str,
        choices=["deepblink", "spotiflow", "piscis", "ufish"],
        help="The name of the tool to use for inference.",
    )
    parser.add_argument(
        "--input_dir", type=str, help="Path to the directory containing input images."
    )
    parser.add_argument(
        "--output_root",
        type=str,
        help="Path to the root directory where csv results will be saved.",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Path to the pre-trained model file or folder.",
        required=False,
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        required=False,
        help=(
            "Detection threshold. If omitted, uses each tool's default: "
            "deepblink=0.5, spotiflow=model default, piscis=0.5, ufish=0.5."
        ),
    )

    args = parser.parse_args()
    run_inference(
        args.tool, args.input_dir, args.output_root, args.model, args.threshold
    )
