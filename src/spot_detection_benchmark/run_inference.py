import argparse
import logging
from pathlib import Path

import tifffile

from spot_detection_benchmark.detectors import (
    TOOLS,
    detect,
    make_detector,
    resolve_model,
)
from spot_detection_benchmark.utils import (
    MODES,
    given_options,
    list_images,
    results_dirname,
    setup_logging,
    write_run_info,
)

log = logging.getLogger(__name__)


def predict_folder(detector, images, mode, threshold, output_dir, cache=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for tif in images:
        image = (
            cache[tif] if cache is not None and tif in cache else tifffile.imread(tif)
        )
        spots = detect(detector, image, mode, threshold)
        spots.to_csv(output_dir / f"{tif.stem}.csv", index=False)


def run_inference(
    tool,
    input_dir,
    output_root,
    mode,
    model_path=None,
    threshold=None,
    pretrained=False,
    detector_kwargs=None,
):
    output_dir = Path(output_root) / results_dirname(tool, pretrained)
    if any(output_dir.glob("*.csv")):
        raise FileExistsError(
            f"{output_dir} already holds results: choose another --output_root, "
            "or delete that folder to run again"
        )
    model_path = resolve_model(tool, mode, pretrained, model_path)
    detector = make_detector(
        tool,
        mode,
        model_path=model_path,
        pretrained=pretrained,
        **(detector_kwargs or {}),
    )
    images = list_images(input_dir)
    if not images:
        raise FileNotFoundError(f"No .tif images found in {input_dir}")
    used_threshold = threshold if threshold is not None else detector.default_threshold
    log.info(
        f"Running {tool} on {len(images)} images in {input_dir} -> {output_dir}. "
        f"Mode: {mode}, model: {model_path or 'none'}, "
        f"threshold: {used_threshold:.3g}, settings: {detector_kwargs or 'defaults'}."
    )
    predict_folder(detector, images, mode, threshold, output_dir)
    write_run_info(
        output_dir,
        script="run_inference",
        tool=tool,
        mode=mode,
        model=model_path,
        pretrained=pretrained,
        threshold=used_threshold,
        settings=detector_kwargs or {},
        input_dir=Path(input_dir).resolve(),
        n_images=len(images),
    )
    return output_dir


def add_detection_args(parser):
    g = parser.add_argument_group("detection mode and 3D parameters (see docs/modes_and_3d.md)")
    g.add_argument(
        "--mode",
        choices=MODES,
        default="2d",
        help="2d: (y, x) images; 2d_stack: (n, y, x) independent planes; "
        "3d: (z, y, x) volumes",
    )
    g.add_argument(
        "--voxel_size",
        type=float,
        nargs=3,
        default=None,
        metavar=("Z", "Y", "X"),
        help="3D: voxel size (any unit, only ratios matter), used by LoG and, in "
        "grid_search.py, to scale z when scoring. Default: 1 1 1",
    )
    g.add_argument(
        "--lateral_sigma",
        type=float,
        default=None,
        help="LoG: lateral spot sigma in px. Default: 1.5",
    )
    g.add_argument(
        "--axial_sigma",
        type=float,
        default=None,
        help="LoG 3D: axial spot sigma in planes. Default: lateral_sigma * X / Z",
    )
    g.add_argument(
        "--merge_search_range",
        type=float,
        default=None,
        help="deepBlink 3D z merge (trackpy.link search_range): max (y, x) distance in px between planes of a spot. Default: 2",
    )
    g.add_argument(
        "--merge_max_gap",
        type=int,
        default=None,
        help="deepBlink 3D z merge (trackpy.link memory): missing planes allowed inside a spot. Default: 0",
    )
    g.add_argument(
        "--merge_min_planes",
        type=int,
        default=None,
        help="deepBlink 3D z merge (trackpy.filter_stubs threshold): spots seen on at least this many planes are reduced to their brightest plane, shorter ones are kept as they are. Default: 2",
    )
    g.add_argument(
        "--mass_radius",
        type=int,
        default=None,
        help="deepBlink 3D z merge (trackpy.refine_com radius): disk radius in px used to measure spot brightness. Default: 3",
    )
    g.add_argument(
        "--no_ufish_blend_3d",
        action="store_true",
        help="U-FISH 3D: do not blend the z/y/x enhancements",
    )
    return parser


def detector_kwargs_from_args(tool, args):
    kwargs = {}
    if tool == "log":
        kwargs = given_options(
            lateral_sigma=args.lateral_sigma,
            axial_sigma=args.axial_sigma,
            voxel_size=args.voxel_size,
        )
    elif tool == "deepblink":
        kwargs["merge_params"] = given_options(
            search_range=args.merge_search_range,
            max_gap=args.merge_max_gap,
            min_planes=args.merge_min_planes,
            mass_radius=args.mass_radius,
        )
    elif tool == "ufish" and args.no_ufish_blend_3d:
        kwargs["blend_3d"] = False
    return kwargs


def main():
    parser = argparse.ArgumentParser(
        description="Run inference using a specified tool on input images."
    )
    parser.add_argument("--tool", required=True, choices=TOOLS)
    parser.add_argument("--input_dir", required=True, help="Folder of .tif images")
    parser.add_argument(
        "--output_root",
        required=True,
        help="Root folder; CSVs go to <output_root>/results_<tool>[_pretrained]",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model file, folder or name. Omit to use the tool's model in models.yaml.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Detection threshold. Omit to use the tool's default: 0.5 for deepblink, "
        "piscis and ufish, the model's thresholds.yaml for spotiflow, 20 for log "
        "(h-maxima height in grey levels).",
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Use the tool's pretrained model. Results go to "
        "results_<tool>_pretrained.",
    )
    add_detection_args(parser)

    args = parser.parse_args()
    setup_logging()
    run_inference(
        args.tool,
        args.input_dir,
        args.output_root,
        args.mode,
        model_path=args.model,
        threshold=args.threshold,
        pretrained=args.pretrained,
        detector_kwargs=detector_kwargs_from_args(args.tool, args),
    )


if __name__ == "__main__":
    main()
