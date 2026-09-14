import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

REPO_ROOT = Path(__file__).resolve().parents[2]

UFISH_CHUNK_SIZE = 512

DEFAULT_MODELS = {
    "deepblink": REPO_ROOT
    / "models"
    / "model_deepblink"
    / "260827_095918_deepBlink_is_sweet.h5",
    "spotiflow": REPO_ROOT / "models" / "model_spotiflow",
    "piscis": "piscis_customed",
    "ufish": REPO_ROOT / "models" / "model_ufish" / "best_model.pth",
}

DEFAULT_PRETRAINED_MODELS = {
    "deepblink": REPO_ROOT / "models" / "model_deepblink" / "smfish1.h5",
    "spotiflow": "general",
    "piscis": None,
    "ufish": None,
}

DEFAULT_THRESHOLDS = {
    "deepblink": 0.5,
    "spotiflow": None,
    "piscis": 0.5,
    "ufish": 0.5,
}


def _save_output_csv(coords, out_csv):
    """Write (row, col) coords as a napari-ready axis-0,axis-1 CSV."""
    df = pd.DataFrame(np.asarray(coords).reshape(-1, 2), columns=["axis-0", "axis-1"])
    df.to_csv(out_csv, index=False)


def run_inference(
    tool, input_dir, output_root, model=None, threshold=None, pretrained=False
):
    """
    Run inference using the specified tool on the input directory and save results to the output directory.

    Args:
        tool (str): The name of the tool to use for inference.
        input_dir (str): Path to the directory containing input images.
        output_root (str): Path to the root directory where csv results will be saved.
        model (str): Model identifier. If None, uses the tool's default for this mode.
            Trained mode (pretrained=False):
            - deepblink: path to the .h5 model file.
            - spotiflow: path to the folder containing the model.
            - piscis: name of the model in the Piscis registry.
            - ufish: path to the .pth model file.
            Pretrained mode (pretrained=True):
            - deepblink: path to a downloaded .h5 model file (default: smfish1.h5).
            - spotiflow: pretrained model name (default: "general").
            - piscis: model name (default: package default).
            - ufish: ignored, weights are downloaded from the internet.
        threshold (float, optional): Detection threshold. If None, uses the tool's default.
            - deepblink: probability cutoff (default: 0.5)
            - spotiflow: prob_thresh (default: None -> model's own stored threshold)
            - piscis: spot detection threshold (default: 0.5)
            - ufish: intensity threshold on enhanced image (default: 0.5)
        pretrained (bool): If True, use the tool's pretrained model instead of the one
            trained in this project. Results go to results_<tool>_pretrained.
    """
    if sys.platform.startswith("win"):
        device = "cpu"
    else:
        device = "cuda"

    defaults = DEFAULT_PRETRAINED_MODELS if pretrained else DEFAULT_MODELS
    if model is None:
        model = defaults[tool]
    elif tool != "piscis" and not pretrained:
        model = Path(model)
    if threshold is None:
        threshold = DEFAULT_THRESHOLDS[tool]

    out = f"results_{tool}_pretrained" if pretrained else f"results_{tool}"
    input_dir = Path(input_dir)
    output_dir = Path(output_root) / out
    output_dir.mkdir(parents=True, exist_ok=True)
    images = sorted(input_dir.glob("*.tif"))

    if not images:
        raise FileNotFoundError(f"No .tif images found in {input_dir}")
    print(f"[{tool}] found {len(images)} images -> {output_dir}")
    print(f"[{tool}] {'pretrained' if pretrained else 'trained'} model: {model}")
    print(f"[{tool}] threshold: {threshold}")

    if tool == "deepblink":
        from deepblink.inference import predict
        from deepblink.io import load_model

        m = load_model(str(model))
        for tif in images:
            image = tifffile.imread(tif)
            coords = predict(image=image, model=m, probability=threshold)
            # remove the probability column if present, and ensure coords is 2D
            coords = np.asarray(coords)
            if coords.ndim == 2 and coords.shape[1] > 2:
                coords = coords[:, :2]
            _save_output_csv(coords, output_dir / f"{tif.stem}.csv")

    elif tool == "spotiflow":
        from spotiflow.model import Spotiflow

        m = (
            Spotiflow.from_pretrained(model)
            if pretrained
            else Spotiflow.from_folder(model, map_location=device)
        )
        for tif in images:
            image = tifffile.imread(tif)
            coords, _ = m.predict(image, device=device, prob_thresh=threshold)
            _save_output_csv(coords, output_dir / f"{tif.stem}.csv")

    elif tool == "piscis":
        if not pretrained:
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

        from piscis import Piscis

        m = Piscis(model_name=model, device=device) if model else Piscis(device=device)
        for tif in images:
            image = tifffile.imread(tif)
            coords = m.predict(image, threshold=threshold)
            _save_output_csv(coords, output_dir / f"{tif.stem}.csv")

    elif tool == "ufish":
        from ufish.api import UFish

        uf = UFish()
        if pretrained:
            uf.load_weights_from_internet()
        else:
            uf.load_weights_from_path(model)

        for tif in images:
            image = tifffile.imread(tif)
            if max(image.shape) > UFISH_CHUNK_SIZE:
                print(
                    f"Warning: {tif.name} is larger than {UFISH_CHUNK_SIZE} pixels. "
                    "UFish will process it in chunks, which may affect performance."
                )
                spots_df, _ = uf.predict_chunks(
                    image,
                    chunk_size=(UFISH_CHUNK_SIZE, UFISH_CHUNK_SIZE),
                    intensity_threshold=threshold,
                )
            else:
                spots_df, _ = uf.predict(image, intensity_threshold=threshold)
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
        help="Model file, folder or name. Omit to use the tool's default.",
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
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help=(
            "Use the tool's pretrained model instead of the one trained in this "
            "project. Results go to results_<tool>_pretrained."
        ),
    )

    args = parser.parse_args()
    run_inference(
        args.tool,
        args.input_dir,
        args.output_root,
        args.model,
        args.threshold,
        args.pretrained,
    )
