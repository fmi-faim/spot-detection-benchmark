import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

# Default pretrained models per tool
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODELS = {
    "deepblink": REPO_ROOT / "models" / "model_deepblink" / "smfish1.h5",
    "spotiflow": "general",
    "piscis": None,  # None -> package default
    "ufish": None,  # weights downloaded from internet
}


def _save_output_csv(coords, out_csv):
    """Write (row, col) coords as a napari-ready axis-0,axis-1 CSV."""
    df = pd.DataFrame(np.asarray(coords).reshape(-1, 2), columns=["axis-0", "axis-1"])
    df.to_csv(out_csv, index=False)


def run_inference_pretrained(tool, input_dir, output_root, model=None):
    """
    Run inference using the pretrained model of the specified tool.

    Args:
        tool (str): The name of the tool to use for inference.
        input_dir (str): Path to the directory containing input images.
        output_root (str): Path to the root directory where csv results will be saved.
        model (str): Pretrained model identifier. If None, uses the tool's default.
        - for deepblink, path to the downloaded .h5 model file (default: smfish1.h5).
        - for spotiflow, the pretrained model name (default: "general").
        - for piscis, the model name (default: package default).
        - for ufish, ignored (weights downloaded from internet).
    """

    if model is None:
        model = DEFAULT_MODELS[tool]

    input_dir = Path(input_dir)
    output_dir = Path(output_root) / f"results_{tool}_pretrained"
    output_dir.mkdir(parents=True, exist_ok=True)
    images = sorted(input_dir.glob("*.tif"))

    if not images:
        raise FileNotFoundError(f"No .tif images found in {input_dir}")
    print(f"[{tool}] found {len(images)} images -> {output_dir}")
    print(f"[{tool}] pretrained model: {model}")

    if tool == "deepblink":
        from deepblink.inference import predict
        from deepblink.io import load_model

        m = load_model(str(model))
        for tif in images:
            image = tifffile.imread(tif)
            coords = predict(image=image, model=m)
            _save_output_csv(coords, output_dir / f"{tif.stem}.csv")

    elif tool == "spotiflow":
        from spotiflow.model import Spotiflow

        m = Spotiflow.from_pretrained(model)
        for tif in images:
            image = tifffile.imread(tif)
            coords, _ = m.predict(image)
            _save_output_csv(coords, output_dir / f"{tif.stem}.csv")

    elif tool == "piscis":
        from piscis import Piscis

        m = Piscis(model_name=model) if model else Piscis()
        for tif in images:
            image = tifffile.imread(tif)
            coords = m.predict(image)
            _save_output_csv(coords, output_dir / f"{tif.stem}.csv")

    elif tool == "ufish":
        from ufish.api import UFish

        UFISH_CHUNK_SIZE = 512
        uf = UFish()
        uf.load_weights_from_internet()
        for tif in images:
            if max(tifffile.imread(tif).shape) > UFISH_CHUNK_SIZE:
                print(
                    f"Warning: {tif.name} is larger than {UFISH_CHUNK_SIZE} pixels. "
                    "UFish will process it in chunks, which may affect performance."
                )
                spots_df, _ = uf.predict_chunks(
                    tifffile.imread(tif),
                    chunk_size=(UFISH_CHUNK_SIZE, UFISH_CHUNK_SIZE),
                )
            else:
                spots_df, _ = uf.predict(tifffile.imread(tif))
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
        description="Run inference using the pretrained model of a specified tool."
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
        help="Pretrained model identifier. Omit to use the tool's default.",
        required=False,
        default=None,
    )
    args = parser.parse_args()
    run_inference_pretrained(args.tool, args.input_dir, args.output_root, args.model)
