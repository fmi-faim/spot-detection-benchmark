import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

TOOLS = ["deepblink", "spotiflow", "piscis", "ufish"]

DEFAULT_PISCIS_MODEL_NAME = "piscis_customed_512"
DEFAULT_PISCIS_INPUT_SIZE = (512, 512)

# ufish has no config of its own, so these are the values it is trained with
DEFAULT_UFISH_HPARAMS = {"lr": 1e-4, "batch_size": 16, "num_epochs": 200}


def _get_values(dct):
    """Flatten deepBlink's {description, value} config to just the values."""
    return {
        k: v["value"] if isinstance(v, dict) and "value" in v else _get_values(v)
        for k, v in dct.items()
    }


def _log_gpu(tool):
    """Print whether training runs on GPU (deepblink: TensorFlow, others: PyTorch)."""
    if tool == "deepblink":
        import tensorflow as tf

        on_gpu = bool(tf.config.list_physical_devices("GPU"))
    else:
        import torch

        on_gpu = torch.cuda.is_available()

    print(f"[{tool}] training on {'GPU' if on_gpu else 'CPU'}")


def _overrides(**kwargs):
    """Keep only the hyperparameters that were actually given."""
    return {k: v for k, v in kwargs.items() if v is not None}


def train(
    tool,
    input_dir=None,
    output_dir=None,
    val_dir=None,
    config=None,
    model_name=None,
    epochs=None,
    lr=None,
    batch_size=None,
):
    """
    Train a model with the specified tool.

    Must be run in the tool's own pixi environment (pixi run -e <tool> ...).

    Args:
        tool (str): The name of the tool to train.
        input_dir (str): Training data.
            - deepblink: .npz dataset (optional, overrides dataset_args.name of the config).
            - spotiflow: folder containing train/ and val/ subfolders.
            - piscis: piscis dataset (.npz file or folder).
            - ufish: training data folder.
        output_dir (str): Where the trained model is saved.
            - deepblink: optional, overrides savedir of the config.
            - spotiflow: required.
            - piscis: optional, piscis models folder (default: $PISCIS_MODELS_DIR
              or models/.piscis/models).
            - ufish: required.
        val_dir (str): Validation data folder (ufish only).
        config (str): deepBlink configuration YAML file (deepblink only).
        model_name (str): Name of the trained model (piscis only,
            default: piscis_customed_512).
        epochs (int, optional): Number of epochs. If None, uses the tool's default.
        lr (float, optional): Learning rate. If None, uses the tool's default.
        batch_size (int, optional): Batch size. If None, uses the tool's default.
    """
    if sys.platform.startswith("win"):
        device = "cpu"
    else:
        device = "cuda"

    _log_gpu(tool)
    print(f"[{tool}] input: {input_dir}")
    print(f"[{tool}] output: {output_dir or '<tool default>'}")

    if tool == "deepblink":
        if config is None:
            raise ValueError("deepblink requires --config")

        import yaml
        from deepblink.training import run_experiment

        with open(config) as f:
            cfg = _get_values(yaml.safe_load(f))

        if input_dir is not None:
            cfg["dataset_args"]["name"] = str(input_dir)
        if output_dir is not None:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            cfg["savedir"] = str(output_dir)
        cfg["train_args"].update(
            _overrides(epochs=epochs, learning_rate=lr, batch_size=batch_size)
        )

        print(f"[{tool}] config: {config}")
        print(f"[{tool}] train_args: {cfg['train_args']}")
        run_experiment(cfg)

    elif tool == "spotiflow":
        if input_dir is None or output_dir is None:
            raise ValueError("spotiflow requires --input_dir and --output_dir")

        from spotiflow.model import Spotiflow
        from spotiflow.utils import get_data

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        train_imgs, train_spots, val_imgs, val_spots = get_data(input_dir)

        train_config = _overrides(num_epochs=epochs, lr=lr, batch_size=batch_size)
        print(f"[{tool}] train_config overrides: {train_config or '<none>'}")

        model = Spotiflow()
        model.fit(
            train_imgs,
            train_spots,
            val_imgs,
            val_spots,
            save_dir=output_dir,
            **({"train_config": train_config} if train_config else {}),
        )

    elif tool == "piscis":
        if input_dir is None:
            raise ValueError("piscis requires --input_dir")

        models_dir = Path(
            output_dir
            or os.environ.get(
                "PISCIS_MODELS_DIR", REPO_ROOT / "models" / ".piscis" / "models"
            )
        )
        checkpoints_dir = Path(
            os.environ.get(
                "PISCIS_CHECKPOINTS_DIR",
                REPO_ROOT / "models" / ".piscis" / "checkpoints",
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

        model_name = model_name or DEFAULT_PISCIS_MODEL_NAME
        hparams = _overrides(epochs=epochs, learning_rate=lr, batch_size=batch_size)
        print(f"[{tool}] model name: {model_name} -> {models_dir}")
        print(f"[{tool}] hyperparameter overrides: {hparams or '<none>'}")

        train_model(
            model_name=model_name,
            dataset_path=input_dir,
            input_size=DEFAULT_PISCIS_INPUT_SIZE,
            device=device,
            **hparams,
        )

    elif tool == "ufish":
        if input_dir is None or val_dir is None or output_dir is None:
            raise ValueError("ufish requires --input_dir, --val_dir and --output_dir")

        from ufish.api import UFish

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        hparams = {
            **DEFAULT_UFISH_HPARAMS,
            **_overrides(num_epochs=epochs, lr=lr, batch_size=batch_size),
        }
        print(f"[{tool}] validation: {val_dir}")
        print(f"[{tool}] hyperparameters: {hparams}")

        ufish = UFish()
        ufish.train(input_dir, val_dir, model_save_dir=output_dir, **hparams)

    else:
        raise ValueError(
            f"unknown tool: {tool}. Please choose from 'deepblink', 'spotiflow', 'piscis', or 'ufish'."
        )

    print(f"[{tool}] training complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a spot detection model with the specified tool."
    )
    parser.add_argument("--tool", required=True, choices=TOOLS)
    parser.add_argument(
        "--input_dir",
        default=None,
        help=(
            "Training data: deepblink=.npz (optional, overrides config), "
            "spotiflow=folder with train/ and val/, piscis=piscis dataset, "
            "ufish=training folder."
        ),
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help=(
            "Where the model is saved (required for spotiflow and ufish; "
            "optional for deepblink and piscis)."
        ),
    )
    parser.add_argument(
        "--val_dir", default=None, help="Validation data folder (ufish only)"
    )
    parser.add_argument(
        "--config", default=None, help="Configuration YAML file (deepblink only)"
    )
    parser.add_argument(
        "--model_name",
        default=None,
        help=f"Name of the trained model (piscis only, default: {DEFAULT_PISCIS_MODEL_NAME})",
    )
    parser.add_argument(
        "--epochs", type=int, default=None, help="Number of epochs (tool default if omitted)"
    )
    parser.add_argument(
        "--lr", type=float, default=None, help="Learning rate (tool default if omitted)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=None, help="Batch size (tool default if omitted)"
    )
    args = parser.parse_args()

    train(
        args.tool,
        args.input_dir,
        args.output_dir,
        args.val_dir,
        args.config,
        args.model_name,
        args.epochs,
        args.lr,
        args.batch_size,
    )
