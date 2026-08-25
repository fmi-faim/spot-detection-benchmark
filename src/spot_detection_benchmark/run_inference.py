import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import tifffile
from pathlib import Path
import pandas as pd
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_MODELS = {
    "deepblink": REPO_ROOT / "models" / "model_deepblink",
    "spotiflow": REPO_ROOT / "models" / "model_spotiflow",
    "piscis": "piscis_customed",
    "ufish": REPO_ROOT / "models" / "model_ufish" / "best_model.pth",
}

def _save_output_csv(coords, out_csv):
    """Write (row, col) coords as a napari-ready axis-0,axis-1 CSV."""
    df = pd.DataFrame(np.asarray(coords).reshape(-1, 2), columns=["axis-0", "axis-1"])
    df.to_csv(out_csv, index=False)

def run_inference(tool, input_dir, output_root, model=None):
    """
    Run inference using the specified tool on the input directory and save results to the output directory.

    Args:
        tool (str): The name of the tool to use for inference.
        input_dir (str): Path to the directory containing input images.
        output_root (str): Path to the root directory where csv results will be saved.
        model (str): Path to the trained model file.
        - for deepblink, this is the path to the .h5 model file.
        - for spotiflow, this is the path to the folder containing the model.
        device (str): The device to run inference on ('cpu' (default) or 'cuda').
    """
    if sys.platform.startswith("win"):
        device = "cpu"
    else:
        device = "cuda" 

    if model is None:
        model = DEFAULT_MODELS[tool]
    else:
        model = Path(model) if tool != "piscis" else model  

    input_dir = Path(input_dir)
    output_dir = Path(output_root) / f"results_{tool}"
    output_dir.mkdir(parents=True, exist_ok=True)
    images = sorted(input_dir.glob("*.tif"))

    if tool=="deepblink":
        from deepblink.io import load_model
        from deepblink.inference import predict
        m = load_model(model)
        for tif in images:
            image = tifffile.imread(tif)
            coords = predict(image=image, model=m)
            _save_output_csv(coords, output_dir /  f"{tif.stem}.csv")

    elif tool == "spotiflow":
        from spotiflow.model import Spotiflow
        m = Spotiflow.from_folder(model, map_location=device)
        for tif in images:
            image = tifffile.imread(tif)
            coords, _ = m.predict(image, device=device)
            _save_output_csv(coords, output_dir / f"{tif.stem}.csv")

    elif tool == "piscis":
        import piscis.paths, piscis.core
        models_dir = Path(os.environ.get("PISCIS_MODELS_DIR", REPO_ROOT / "models" / ".piscis" / "models"))
        models_dir.mkdir(parents=True, exist_ok=True)
        piscis.paths.MODELS_DIR = models_dir  
        piscis.core.MODELS_DIR = models_dir
        print(piscis.core.MODELS_DIR)

        from piscis import Piscis
        m = Piscis(model_name=model, device=device)   # loads models_dir/<model>.pt
        for tif in images:
            image = tifffile.imread(tif)
            coords = m.predict(image)
            _save_output_csv(coords, output_dir /  f"{tif.stem}.csv")

    elif tool=="ufish":
        from ufish.api import UFish
        uf = UFish()
        uf.load_weights_from_path(model)
        for tif in images:
            spots_df, _ = uf.predict(tifffile.imread(tif))
            spots_df[["axis-0", "axis-1"]].to_csv(output_dir / f"{tif.stem}.csv", index=False)

    else:
        raise ValueError(f"unknown tool: {tool}. Please choose from 'deepblink', 'spotiflow', 'piscis', or 'ufish'.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run inference using a specified tool on input images.")
    parser.add_argument("--tool", type=str, choices=["deepblink", "spotiflow", "piscis", "ufish"], help="The name of the tool to use for inference.")
    parser.add_argument("--input_dir", type=str, help="Path to the directory containing input images.")
    parser.add_argument("--output_root", type=str, help="Path to the root directory where csv results will be saved.")
    parser.add_argument("--model", type=str, help="Path to the pre-trained model file or folder.", required=False)

    args = parser.parse_args()
    run_inference(args.tool, args.input_dir, args.output_root, args.model)


