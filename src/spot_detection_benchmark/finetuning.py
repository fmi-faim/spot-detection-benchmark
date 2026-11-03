import argparse
from pathlib import Path

from spot_detection_benchmark.detectors import default_model
from spot_detection_benchmark.train import (
    add_training_args,
    tool_kwargs_from_args,
    train,
)
from spot_detection_benchmark.utils import setup_logging


def _check_not_overwritten(base_model, output_dir):
    base = Path(str(base_model))
    if not base.exists():
        return
    base = base.resolve()
    if Path(output_dir).resolve() in (base, base.parent):
        raise ValueError(
            f"--output_dir {output_dir} would overwrite the base model {base_model}: "
            "save the fine-tuned model in another folder"
        )


def finetune(
    tool,
    input_dir,
    output_dir,
    base_model=None,
    from_trained=False,
    epochs=None,
    lr=None,
    batch_size=None,
    **tool_kwargs,
):
    if input_dir is None or output_dir is None:
        raise ValueError("fine-tuning requires --input_dir and --output_dir")
    if base_model is None:
        base_model = default_model(tool, "2d", pretrained=not from_trained)
    _check_not_overwritten(base_model, output_dir)

    if tool == "piscis" and tool_kwargs.get("model_name") is None:
        tool_kwargs["model_name"] = f"{Path(str(base_model)).stem}_finetuned"

    train(
        tool,
        input_dir=input_dir,
        output_dir=output_dir,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        base_model=base_model,
        **tool_kwargs,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune an existing spot detection model on new data."
    )
    add_training_args(parser, require_dirs=True)
    parser.add_argument(
        "--base_model",
        default=None,
        help=(
            "Model to fine-tune: deepblink .h5, spotiflow folder, piscis name, "
            "ufish .pth. Omit to use the tool's pretrained 2D model in models.yaml."
        ),
    )
    parser.add_argument(
        "--from_trained",
        action="store_true",
        help="Without --base_model: start from the model trained here (models.yaml "
        "trained / 2d) instead of the pretrained one.",
    )
    args = parser.parse_args()
    setup_logging()

    finetune(
        args.tool,
        args.input_dir,
        args.output_dir,
        base_model=args.base_model,
        from_trained=args.from_trained,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        **tool_kwargs_from_args(args.tool, args),
    )


if __name__ == "__main__":
    main()
