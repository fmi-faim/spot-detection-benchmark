import argparse
import logging
import shutil
from pathlib import Path

from spot_detection_benchmark.utils import (
    get_device,
    given_options,
    set_piscis_dir,
    setup_logging,
)

log = logging.getLogger(__name__)

DEFAULT_PISCIS_MODEL_NAME = "piscis_customed_512"
DEFAULT_PISCIS_INPUT_SIZE = (512, 512)
DEFAULT_UFISH_HPARAMS = {"lr": 1e-4, "batch_size": 16, "num_epochs": 200}


def _log_gpu(tool):
    if tool == "deepblink":
        import tensorflow as tf

        on_gpu = bool(tf.config.list_physical_devices("GPU"))
    else:
        import torch

        on_gpu = torch.cuda.is_available()

    log.info(f"[{tool}] training on {'GPU' if on_gpu else 'CPU'}")


def _get_values(dct):
    return {
        k: v["value"] if isinstance(v, dict) and "value" in v else _get_values(v)
        for k, v in dct.items()
    }


def train_deepblink(
    config,
    input_dir=None,
    output_dir=None,
    epochs=None,
    lr=None,
    batch_size=None,
    base_model=None,
):
    if config is None:
        raise ValueError("deepblink requires --config")

    import yaml
    from deepblink.io import load_model
    from deepblink.training import run_experiment

    with open(config, encoding="utf-8") as f:
        cfg = _get_values(yaml.safe_load(f))

    if input_dir is not None:
        cfg["dataset_args"]["name"] = str(input_dir)
    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        cfg["savedir"] = str(output_dir)
    cfg["train_args"].update(
        given_options(epochs=epochs, learning_rate=lr, batch_size=batch_size)
    )
    log.info(f"[deepblink] config: {config}")
    log.info(f"[deepblink] train_args: {cfg['train_args']}")

    pre_model = load_model(str(base_model)) if base_model is not None else None
    run_experiment(cfg, pre_model=pre_model)


def train_spotiflow(
    input_dir, output_dir, epochs=None, lr=None, batch_size=None, base_model=None
):
    if input_dir is None or output_dir is None:
        raise ValueError("spotiflow requires --input_dir and --output_dir")

    from spotiflow.model import Spotiflow
    from spotiflow.utils import get_data

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    train_imgs, train_spots, val_imgs, val_spots = get_data(input_dir)

    train_config = given_options(num_epochs=epochs, lr=lr, batch_size=batch_size)
    log.info(f"[spotiflow] train_config overrides: {train_config or '<none>'}")

    if base_model is None:
        model = Spotiflow()
    else:
        # inference_mode=False: the loaded model is left trainable
        model = Spotiflow.from_folder(
            str(base_model), inference_mode=False, map_location=get_device()
        )
        train_config["finetuned_from"] = str(base_model)  # recorded with the model
    model.fit(
        train_imgs,
        train_spots,
        val_imgs,
        val_spots,
        save_dir=output_dir,
        train_config=train_config or None,
    )


def train_piscis(
    input_dir,
    output_dir=None,
    model_name=None,
    epochs=None,
    lr=None,
    batch_size=None,
    base_model=None,
):
    if input_dir is None:
        raise ValueError("piscis requires --input_dir")
    model_name = model_name or DEFAULT_PISCIS_MODEL_NAME
    if base_model is not None and model_name == str(base_model):
        raise ValueError(
            f"piscis: --model_name {model_name} would overwrite the base model; "
            "give the fine-tuned model another name"
        )

    from piscis.training import train_model

    # Piscis looks for the base model in its models folder: when it lives in the
    # default folder and output_dir is another one, copy it there first
    default_models_dir, _ = set_piscis_dir()
    models_dir, checkpoints_dir = set_piscis_dir(output_dir)
    if base_model is not None:
        base_file = f"{base_model}.pt"
        if (
            not (models_dir / base_file).exists()
            and (default_models_dir / base_file).exists()
        ):
            shutil.copy2(default_models_dir / base_file, models_dir / base_file)

    hparams = given_options(epochs=epochs, learning_rate=lr, batch_size=batch_size)
    log.info(f"[piscis] model: {models_dir / (model_name + '.pt')}")
    log.info(f"[piscis] checkpoints: {checkpoints_dir}")
    log.info(f"[piscis] hyperparameter overrides: {hparams or '<none>'}")

    train_model(
        model_name=model_name,
        dataset_path=input_dir,
        initial_model_name=base_model,
        input_size=DEFAULT_PISCIS_INPUT_SIZE,
        device=get_device(),
        **hparams,
    )


def train_ufish(
    input_dir,
    val_dir,
    output_dir,
    epochs=None,
    lr=None,
    batch_size=None,
    base_model=None,
):
    if input_dir is None or val_dir is None or output_dir is None:
        raise ValueError("ufish requires --input_dir, --val_dir and --output_dir")
    if base_model is not None and not str(base_model).endswith(".pth"):
        raise ValueError(
            f"ufish: {base_model} cannot be fine-tuned: it is not a PyTorch .pth model"
        )

    from ufish.api import UFish

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    hparams = {
        **DEFAULT_UFISH_HPARAMS,
        **given_options(num_epochs=epochs, lr=lr, batch_size=batch_size),
    }
    log.info(f"[ufish] validation: {val_dir}")
    log.info(f"[ufish] hyperparameters: {hparams}")

    ufish = UFish()
    if base_model is not None:
        ufish.load_weights_from_path(str(base_model))  # train() then starts from it
    ufish.train(
        input_dir,
        val_dir,
        model_save_dir=output_dir,
        summary_dir=str(Path(output_dir) / "runs"),
        **hparams,
    )


TRAINERS = {
    "deepblink": train_deepblink,
    "spotiflow": train_spotiflow,
    "piscis": train_piscis,
    "ufish": train_ufish,
}

TOOLS = list(TRAINERS)


def train(
    tool,
    input_dir=None,
    output_dir=None,
    epochs=None,
    lr=None,
    batch_size=None,
    base_model=None,
    **tool_kwargs,
):
    if tool not in TRAINERS:
        raise ValueError(f"Unknown tool: {tool}. Available tools: {TOOLS}")

    _log_gpu(tool)
    log.info(f"[{tool}] input: {input_dir}")
    log.info(f"[{tool}] output: {output_dir or '<tool default>'}")
    log.info(f"[{tool}] start: {base_model or 'from scratch'}")

    TRAINERS[tool](
        input_dir=input_dir,
        output_dir=output_dir,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        base_model=base_model,
        **tool_kwargs,
    )

    log.info(f"[{tool}] training complete")
    log.info(
        f"[{tool}] to use this model, add it to models.yaml (trained / 2d / {tool})"
    )


def tool_kwargs_from_args(tool, args):
    if tool == "deepblink":
        return {"config": args.config}
    if tool == "piscis":
        return {"model_name": args.model_name}
    if tool == "ufish":
        return {"val_dir": args.val_dir}
    return {}


def add_training_args(parser, require_dirs=False):
    parser.add_argument("--tool", required=True, choices=TOOLS)
    parser.add_argument(
        "--input_dir",
        required=require_dirs,
        default=None,
        help=(
            "Training data: deepblink=.npz (overrides the config's dataset), "
            "spotiflow=folder with train/ and val/, piscis=folder with train/, val/ "
            "and test/, each with x/ and y/, ufish=training folder."
        ),
    )
    parser.add_argument(
        "--output_dir",
        required=require_dirs,
        default=None,
        help=(
            "Where the model is saved. spotiflow, ufish: the model folder. "
            "deepblink: overrides the config's savedir. piscis: the Piscis folder "
            "holding models/ and checkpoints/ (default $PISCIS_DIR or models/piscis); "
            "keep models/piscis, the only folder inference loads Piscis models from."
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
        "--epochs",
        type=int,
        default=None,
        help="Number of epochs (tool default if omitted)",
    )
    parser.add_argument(
        "--lr", type=float, default=None, help="Learning rate (tool default if omitted)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size (tool default if omitted)",
    )
    return parser


def main():
    parser = argparse.ArgumentParser(
        description="Train a spot detection model from scratch with the specified tool."
    )
    add_training_args(parser)
    args = parser.parse_args()
    setup_logging()

    train(
        args.tool,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        **tool_kwargs_from_args(args.tool, args),
    )


if __name__ == "__main__":
    main()
