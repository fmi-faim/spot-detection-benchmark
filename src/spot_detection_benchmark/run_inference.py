import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

REPO_ROOT = Path(__file__).resolve().parents[2]

# LoG spot size in pixels
LOG_LATERAL_SIGMA = 1.5

DEFAULT_MODELS = {
    "deepblink": REPO_ROOT
    / "models"
    / "deepblink"
    / "261005_162813_deepBlink_is_sweet.h5",
    "spotiflow": REPO_ROOT / "models" / "spotiflow" / "spotiflow_customed",
    "piscis": "piscis_customed_512",
    "ufish": REPO_ROOT / "models" / "ufish" / "best_model.pth",
    "log": None,
}

DEFAULT_PRETRAINED_MODELS = {
    "deepblink": REPO_ROOT / "models" / "deepblink" / "smfish1.h5",
    "spotiflow": REPO_ROOT / "models" / "spotiflow" / "general",
    "piscis": "20251212",
    "ufish": REPO_ROOT / "models" / "ufish" / "v1.0-alldata-ufish_c32.onnx",
    "log": None,
}

DEFAULT_THRESHOLDS = {
    "deepblink": 0.5,
    "spotiflow": None,
    "piscis": 0.5,
    "ufish": 0.5,
    "log": 20,
}


def _save_output_csv(coords, out_csv, scores=None, scores_colname="score"):
    """Write (row, col) coords as a axis-0,axis-1 CSV for napari, and a third column for probability/intensity if available."""
    df = pd.DataFrame(np.asarray(coords).reshape(-1, 2), columns=["axis-0", "axis-1"])
    if scores is not None:
        df[scores_colname] = np.asarray(scores).reshape(-1)
    df.to_csv(out_csv, index=False)


def _results_dirname(tool, pretrained=False):
    """Folder name run_inference writes into, for one tool and mode."""
    name = f"results_{tool}"
    return f"{name}_pretrained" if pretrained else name


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
            - spotiflow: path to the folder containing the pretrained model (default: "general").
            - piscis: pretrained model name (default: "20251212").
            - ufish: path to the pretrained .onnx model file (default: "v1.0-alldata-ufish_c32.onnx").
            Both modes:
            - log: unused (classical method, no model).
        threshold (float, optional): Detection threshold. If None, uses the tool's default.
            - deepblink: probability cutoff (default: 0.5)
            - spotiflow: prob_thresh (default: None, uses model's own stored threshold)
            - piscis: spot detection threshold (default: 0.5)
            - ufish: intensity threshold on enhanced image (default: 0.5)
            - log: h-maxima height, in grey levels of the image dtype (default: 20)
        pretrained (bool): If True, use the tool's pretrained model instead of the one
            trained on the UFISH dataset. Results go to results_<tool>_pretrained.

    """
    if sys.platform.startswith("win"):
        device = "cpu"
    else:
        device = "cuda"

    defaults = DEFAULT_PRETRAINED_MODELS if pretrained else DEFAULT_MODELS
    if model is None:
        model = defaults[tool]
    elif tool != "piscis":
        model = Path(model)
    if threshold is None:
        threshold = DEFAULT_THRESHOLDS[tool]

    input_dir = Path(input_dir)
    output_dir = Path(output_root) / _results_dirname(tool, pretrained)
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
            coords = np.asarray(predict(image=image, model=m, probability=threshold))
            probs = coords[:, 2] if coords.ndim == 2 and coords.shape[1] > 2 else None
            _save_output_csv(
                coords[:, :2] if coords.ndim == 2 and coords.shape[1] > 2 else coords,
                output_dir / f"{tif.stem}.csv",
                scores=probs,
                scores_colname="probability",
            )

    elif tool == "spotiflow":
        from spotiflow.model import Spotiflow

        m = Spotiflow.from_folder(model, map_location=device)
        for tif in images:
            image = tifffile.imread(tif)
            coords, details = m.predict(image, device=device, prob_thresh=threshold)
            _save_output_csv(
                coords,
                output_dir / f"{tif.stem}.csv",
                scores=getattr(details, "prob", None),
                scores_colname="probability",
            )

    elif tool == "piscis":
        import piscis.core
        import piscis.paths

        models_dir = Path(
            os.environ.get("PISCIS_MODELS_DIR", REPO_ROOT / "models" / "piscis")
        )
        models_dir.mkdir(parents=True, exist_ok=True)
        piscis.paths.MODELS_DIR = models_dir
        piscis.core.MODELS_DIR = models_dir

        from piscis import Piscis

        m = Piscis(model_name=model, device=device)
        for tif in images:
            image = tifffile.imread(tif)
            coords, maps = m.predict(image, threshold=threshold, intermediates=True)
            coords = np.asarray(coords).reshape(-1, 2)
            try:
                labels = np.asarray(maps)
                labels = labels[0] if labels.ndim == 3 else labels
                row_idx = np.clip(
                    coords[:, 0].round().astype(int), 0, labels.shape[0] - 1
                )
                col_idx = np.clip(
                    coords[:, 1].round().astype(int), 0, labels.shape[1] - 1
                )
                scores = labels[row_idx, col_idx] if len(coords) else np.array([])
            except Exception:
                scores = None
            _save_output_csv(
                coords,
                output_dir / f"{tif.stem}.csv",
                scores=scores,
                scores_colname="confidence",
            )

    elif tool == "ufish":
        from ufish.api import UFish

        uf = UFish()
        if str(model).endswith(".onnx"):
            # build the session ourselves: by default onnxruntime starts one thread
            # per core of the node instead of per core SLURM allocated, and crawls
            import onnxruntime

            opts = onnxruntime.SessionOptions()
            opts.intra_op_num_threads = (
                len(os.sched_getaffinity(0))
                if hasattr(os, "sched_getaffinity")
                else os.cpu_count()
            )
            opts.inter_op_num_threads = 1
            uf.ort_session = onnxruntime.InferenceSession(
                str(model), opts, providers=["CPUExecutionProvider"]
            )
            print(f"[{tool}] onnxruntime threads: {opts.intra_op_num_threads}")
        else:
            uf.load_weights_from_path(str(model))

        for tif in images:
            image = tifffile.imread(tif)
            spots_df, enhanced = uf.predict(image, intensity_threshold=threshold)
            coords = spots_df[["axis-0", "axis-1"]].to_numpy()
            intens = (
                enhanced[coords[:, 0].astype(int), coords[:, 1].astype(int)]
                if len(coords)
                else np.array([])
            )
            _save_output_csv(
                coords,
                output_dir / f"{tif.stem}.csv",
                scores=intens,
                scores_colname="intensity",
            )

    elif tool == "log":
        from log_detection import detect_spots_2d

        for tif in images:
            image = tifffile.imread(tif)
            # h is in grey levels of the image dtype (uint8 or uint16 here):
            # clip it to the dtype range, a larger value would overflow
            h = min(int(round(threshold)), np.iinfo(image.dtype).max)
            coords = detect_spots_2d(image, lateral_sigma=LOG_LATERAL_SIGMA, h=h)
            _save_output_csv(coords, output_dir / f"{tif.stem}.csv")

    else:
        raise ValueError(
            f"unknown tool: {tool}. Please choose from 'deepblink', 'spotiflow', 'piscis', 'ufish', or 'log'."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run inference using a specified tool on input images."
    )
    parser.add_argument(
        "--tool",
        type=str,
        choices=["deepblink", "spotiflow", "piscis", "ufish", "log"],
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
            "deepblink=0.5, spotiflow=model default, piscis=0.5, ufish=0.5, "
            "log=20 (h-maxima height in grey levels)."
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
