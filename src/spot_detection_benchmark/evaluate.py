from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.spatial

from spot_detection_benchmark.evaluation_metrics import (
    EPS,
    f1_integral,
    linear_sum_assignment,
)
from spot_detection_benchmark.utils import MODES, load_coords, setup_logging, subset_of

log = logging.getLogger(__name__)

CUTOFF = 3.0
MDIST = 3.0
N_CUTOFFS = 50
MIN_TRUE = 1
STACK_PLANE_SCALE = 1e6

PER_IMAGE_COLUMNS = [
    "subset",
    "image",
    "TP",
    "FP",
    "FN",
    "n_true",
    "n_pred",
    "precision",
    "recall",
    "f1",
    "f1_integral",
    "distance_error",
]


def axis_scale(ndim, voxel_size=None, stack=False):
    scale = np.ones(ndim)
    if ndim == 3:
        if stack:
            scale[0] = STACK_PLANE_SCALE
        elif voxel_size is not None:
            vz, vy, vx = voxel_size
            scale[:] = (vz / vx, vy / vx, 1.0)
    return scale


def unmatched_stems(gt_dir, pred_dir):
    gt = {g.stem for g in Path(gt_dir).glob("*.csv")}
    pred = {p.stem for p in Path(pred_dir).glob("*.csv")}
    return sorted(gt - pred), sorted(pred - gt)


def score_image(
    pred,
    true,
    voxel_size=None,
    stack=False,
    cutoff=CUTOFF,
    mdist=MDIST,
    n_cutoffs=N_CUTOFFS,
):
    ndim = max(
        np.asarray(pred).shape[-1] if np.size(pred) else 0,
        np.asarray(true).shape[-1] if np.size(true) else 0,
        2,
    )
    scale = axis_scale(ndim, voxel_size, stack)
    pred = np.asarray(pred, float).reshape(-1, ndim) * scale
    true = np.asarray(true, float).reshape(-1, ndim) * scale

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f1i = float(f1_integral(pred, true, mdist=mdist, n_cutoffs=n_cutoffs))

    if pred.size == 0 or true.size == 0:
        tp, fp, fn = 0, len(pred), len(true)
        dist_err = np.nan
    else:
        matrix = scipy.spatial.distance.cdist(pred, true, "euclidean")
        rows, cols = linear_sum_assignment(matrix, cutoff)
        tp = len(rows)
        fp = len(pred) - tp
        fn = len(true) - tp
        dist_err = float(np.mean(matrix[rows, cols])) if tp else np.nan

    prec = tp / (tp + fp + EPS)
    rec = tp / (tp + fn + EPS)
    f1 = 2 * prec * rec / (prec + rec + EPS)

    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "f1_integral": f1i,
        "distance_error": dist_err,
        "n_pred": len(pred),
        "n_true": len(true),
    }


def evaluate_folder(gt_dir, pred_dir, voxel_size=None, stack=False, **kwargs):
    rows = []
    for pcsv in sorted(Path(pred_dir).glob("*.csv")):
        gcsv = Path(gt_dir) / pcsv.name
        if not gcsv.exists():
            log.warning(f"[evaluate] GT file {gcsv} does not exist.")
            continue
        m = score_image(
            load_coords(pcsv), load_coords(gcsv), voxel_size, stack, **kwargs
        )
        m["image"] = pcsv.stem
        m["subset"] = subset_of(pcsv.stem)
        rows.append(m)
    if not rows:
        return pd.DataFrame(columns=PER_IMAGE_COLUMNS)
    return pd.DataFrame(rows)[PER_IMAGE_COLUMNS]


def summarize(per_image, by=("subset",), min_true=MIN_TRUE, add_all=True):
    scored = per_image[per_image["n_true"] >= min_true]
    if scored.empty:
        return pd.DataFrame()
    agg = dict(
        n_images=("image", "nunique"),
        TP_total=("TP", "sum"),
        FP_total=("FP", "sum"),
        FN_total=("FN", "sum"),
        total_gt_spots=("n_true", "sum"),
        total_pred_spots=("n_pred", "sum"),
        precision_mean=("precision", "mean"),
        recall_mean=("recall", "mean"),
        f1_mean=("f1", "mean"),
        f1_integral_mean=("f1_integral", "mean"),
        f1_integral_std=("f1_integral", "std"),
        f1_integral_median=("f1_integral", "median"),
        distance_error_mean=("distance_error", "mean"),
    )
    by = list(by)
    frames = [scored.groupby(by).agg(**agg).reset_index()]
    if add_all and "subset" in by:
        rest = [b for b in by if b != "subset"]
        if rest:
            whole = scored.groupby(rest).agg(**agg).reset_index()
        else:
            whole = pd.DataFrame(
                [{name: scored[col].agg(func) for name, (col, func) in agg.items()}]
            )
        frames.append(whole.assign(subset="ALL"))
    out = pd.concat(frames, ignore_index=True)
    out["precision_pooled"] = out["TP_total"] / (
        out["TP_total"] + out["FP_total"] + EPS
    )
    out["recall_pooled"] = out["TP_total"] / (out["TP_total"] + out["FN_total"] + EPS)
    return out[by + [c for c in out.columns if c not in by]]


def evaluate(
    gt_dir,
    pred_dir,
    output_dir,
    voxel_size=None,
    stack=False,
    min_true=MIN_TRUE,
    **kwargs,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    per_image = evaluate_folder(gt_dir, pred_dir, voxel_size, stack, **kwargs)
    missing_pred, missing_gt = unmatched_stems(gt_dir, pred_dir)
    if missing_pred:
        log.warning(
            f"[evaluate] {len(missing_pred)} image(s) have a ground-truth CSV but no prediction CSV: "
            + ", ".join(missing_pred)
        )
    if missing_gt:
        log.warning(
            f"[evaluate] {len(missing_gt)} image(s) have a prediction CSV but no ground-truth CSV: "
            + ", ".join(missing_gt)
        )
    if per_image.empty:
        raise FileNotFoundError(
            f"no prediction CSV in {pred_dir} has a ground truth in {gt_dir}"
        )
    summary = summarize(per_image, min_true=min_true)
    per_image.to_csv(output_dir / "metrics_per_image.csv", index=False)
    summary.to_csv(output_dir / "metrics_per_subset.csv", index=False)
    return per_image, summary


def main():
    parser = argparse.ArgumentParser(
        description="Evaluation metrics of prediction CSVs against ground-truth CSVs"
    )
    parser.add_argument("--gt_dir", required=True, help="Ground-truth CSVs")
    parser.add_argument(
        "--pred_dir", required=True, help="Prediction CSVs (same stems)"
    )
    parser.add_argument(
        "--output_dir", required=True, help="Where the two CSVs are written"
    )
    parser.add_argument(
        "--mode",
        default="2d",
        choices=MODES,
        help="2d: (y, x) CSVs; 2d_stack: (plane, y, x) of independent 2D planes, "
        "never matched across planes; 3d: (z, y, x), z rescaled with --voxel_size. "
        "Default: 2d",
    )
    parser.add_argument(
        "--voxel_size",
        type=float,
        nargs=3,
        default=None,
        metavar=("Z", "Y", "X"),
        help="Voxel size, any unit; z is rescaled by Z/X. Used only if --mode 3d.",
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        default=CUTOFF,
        help=f"Matching distance in px for TP/FP/FN (default {CUTOFF})",
    )
    parser.add_argument(
        "--mdist",
        type=float,
        default=MDIST,
        help=f"Upper bound of the F1-integral in px (default {MDIST})",
    )
    parser.add_argument(
        "--min_true",
        type=int,
        default=MIN_TRUE,
        help="Images with fewer GT spots are left out of the summary",
    )

    args = parser.parse_args()
    setup_logging()
    voxel_size = args.voxel_size if args.mode == "3d" else None
    stack = args.mode == "2d_stack"
    per_image, summary = evaluate(
        args.gt_dir,
        args.pred_dir,
        args.output_dir,
        voxel_size=voxel_size,
        stack=stack,
        min_true=args.min_true,
        cutoff=args.cutoff,
        mdist=args.mdist,
    )
    columns = ["subset", "n_images", "f1_integral_mean", "f1_mean"]
    columns += ["precision_mean", "recall_mean"]
    log.info(f"{len(per_image)} images scored -> {args.output_dir}")
    log.info(summary[columns].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
