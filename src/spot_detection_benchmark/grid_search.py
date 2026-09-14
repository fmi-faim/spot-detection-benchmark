import argparse
import shutil
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.spatial
from evaluation_metrics import EPS, f1_integral, linear_sum_assignment, offset_euclidean
from run_inference import run_inference

CUTOFF = 3.0
MDIST = 3.0
N_CUTOFFS = 50
MIN_TRUE = 1

THRESHOLD_GRIDS = {
    "deepblink": np.round(np.arange(0.1, 1.0, 0.05), 3).tolist(),
    "spotiflow": np.round(np.arange(0.1, 1.0, 0.05), 3).tolist(),
    "piscis": np.round(np.arange(0.1, 1, 0.05), 3).tolist(),
    "ufish": np.round(np.arange(0.1, 1.0, 0.05), 3).tolist(),
}

# column order of the sweep CSV (one row per threshold x image type)
COLUMNS = [
    "threshold",
    "subset",
    "n_images",
    "TP_total",
    "FP_total",
    "FN_total",
    "total_gt_spots",
    "total_pred_spots",
    "precision_mean",
    "recall_mean",
    "f1_mean",
    "f1_integral_mean",
    "f1_integral_std",
    "distance_error_mean",
]


def _load(csv):
    df = pd.read_csv(csv)
    return df[["axis-0", "axis-1"]].to_numpy() if len(df) else np.empty((0, 2))


def score_image(pred, true):
    pred = np.asarray(pred, float).reshape(-1, 2)
    true = np.asarray(true, float).reshape(-1, 2)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f1i = float(f1_integral(pred, true, mdist=MDIST, n_cutoffs=N_CUTOFFS))

    if pred.size == 0 and true.size == 0:
        tp = fp = fn = 0
        dist_err = np.nan
    elif pred.size == 0:
        tp, fp, fn = 0, 0, len(true)
        dist_err = np.nan
    elif true.size == 0:
        tp, fp, fn = 0, len(pred), 0
        dist_err = np.nan
    else:
        matrix = scipy.spatial.distance.cdist(pred, true, "euclidean")
        rows, cols = linear_sum_assignment(matrix, CUTOFF)
        tp = len(rows)
        fp = len(pred) - tp
        fn = len(true) - tp
        if tp:
            offs = [
                (pred[r][0] - true[c][0], pred[r][1] - true[c][1])
                for r, c in zip(rows, cols)
            ]
            dist_err = float(np.mean(offset_euclidean(offs)))
        else:
            dist_err = np.nan

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


def score_folder(pred_dir, gt_dir):
    """Score every prediction CSV against its stem-matched ground truth."""
    rows = []
    for pcsv in sorted(Path(pred_dir).glob("*.csv")):
        gcsv = Path(gt_dir) / pcsv.name
        if not gcsv.exists():
            continue
        m = score_image(_load(pcsv), _load(gcsv))
        m["image"] = pcsv.stem
        m["subset"] = pcsv.stem.split("_")[0]  # data source = filename prefix
        rows.append(m)
    return pd.DataFrame(rows)


def aggregate(df):
    """Collapse per-image rows into a single summary row.

    Means exclude empty-GT images (MIN_TRUE); counts use every image.
    """
    scored = df[df["n_true"] >= MIN_TRUE]
    return pd.Series(
        {
            "n_images": len(df),
            "TP_total": int(df["TP"].sum()),
            "FP_total": int(df["FP"].sum()),
            "FN_total": int(df["FN"].sum()),
            "total_gt_spots": int(df["n_true"].sum()),
            "total_pred_spots": int(df["n_pred"].sum()),
            "precision_mean": scored["precision"].mean(),
            "recall_mean": scored["recall"].mean(),
            "f1_mean": scored["f1"].mean(),
            "f1_integral_mean": scored["f1_integral"].mean(),
            "f1_integral_std": scored["f1_integral"].std(),
            "distance_error_mean": scored["distance_error"].mean(),
        }
    )


def grid_search(tool, val_dir, gt_dir, output_root, model, pretrained=False):
    """Sweep the detection threshold on the validation set, per image type.

    Produces two outputs:
      1. a sweep CSV with one row per (threshold, image type), holding the
         aggregated metrics for that combination;
      2. a predictions folder holding, for every image, the prediction made at
         the optimal threshold *for that image's type*.

    Whole-dataset metrics can be recomputed from the sweep CSV: the count columns
    are sums, and the mean columns can be recombined with n_images as weights.

    Args:
        tool (str): Tool to sweep.
        val_dir (str): Validation images folder.
        gt_dir (str): Ground-truth CSVs folder (same stems as the images).
        output_root (str): Root folder for sweep results.
        model (str): Model file, folder or name. None uses the tool's default.
        pretrained (bool): Sweep the tool's pretrained model instead of the one
            trained in this project.

    Returns:
        (best_thresholds_by_subset, sweep_dataframe), or (None, None) if nothing scored.
    """
    val_dir = Path(val_dir).resolve()
    gt_dir = Path(gt_dir).resolve()
    output_root = Path(output_root).resolve()

    # run_inference writes into <output_root>/results_<tool>[_pretrained]
    results_subdir = f"results_{tool}_pretrained" if pretrained else f"results_{tool}"
    tag = f"{tool}_pretrained" if pretrained else tool

    tool_out = output_root / tag
    tool_out.mkdir(parents=True, exist_ok=True)

    # scratch folder, reused and wiped for every threshold
    tmp_dir = tool_out / "_tmp_predictions"

    thresholds = THRESHOLD_GRIDS[tool]
    sweep_rows = []

    print(
        f"[{tag}] sweeping {len(thresholds)} thresholds: "
        f"{thresholds[0]} -> {thresholds[-1]}"
    )

    for thr in thresholds:
        print(f"  threshold={thr} ...", end=" ", flush=True)

        # fresh scratch folder so stale CSVs can never be scored
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)

        try:
            run_inference(
                tool, val_dir, tmp_dir, model, threshold=thr, pretrained=pretrained
            )
        except Exception as e:
            print(f"FAILED: {e}")
            continue

        pred_dir = tmp_dir / results_subdir
        if not pred_dir.exists():
            print("no output dir")
            continue

        df = score_folder(pred_dir, gt_dir)
        if df.empty:
            print("no scored images")
            continue

        # one aggregated row per image type
        by_subset = df.groupby("subset").apply(aggregate).reset_index()
        by_subset.insert(0, "threshold", thr)
        sweep_rows.append(by_subset)

        overall = aggregate(df)
        print(
            f"F1-integral={overall['f1_integral_mean']:.4f}  "
            f"F1={overall['f1_mean']:.4f}  "
            f"P={overall['precision_mean']:.4f}  "
            f"R={overall['recall_mean']:.4f}  "
            f"({by_subset['subset'].nunique()} types)"
        )

    # scratch folder is no longer needed
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)

    if not sweep_rows:
        print(f"[{tag}] no valid results - check errors above")
        return None, None

    # ---------- output 1: sweep CSV, one row per (threshold, image type) ----------
    sweep = pd.concat(sweep_rows, ignore_index=True)
    sweep = sweep[COLUMNS].sort_values(["subset", "threshold"])
    sweep_csv = tool_out / f"{tag}_threshold_sweep.csv"
    sweep.to_csv(sweep_csv, index=False)
    print(f"\n[{tag}] sweep saved to {sweep_csv}")
    print(
        f"[{tag}] {sweep['subset'].nunique()} image types x "
        f"{sweep['threshold'].nunique()} thresholds = {len(sweep)} rows"
    )

    # ---------- optimal threshold per image type ----------
    best_idx = sweep.groupby("subset")["f1_integral_mean"].idxmax()
    best = sweep.loc[best_idx, ["subset", "threshold", "f1_integral_mean",
                                "precision_mean", "recall_mean"]]
    best = best.rename(columns={"threshold": "optimal_threshold"})
    best_csv = tool_out / f"{tag}_optimal_thresholds.csv"
    best.to_csv(best_csv, index=False)

    print(f"\n[{tag}] optimal threshold per image type:")
    print(best.to_string(index=False))
    print(f"[{tag}] saved to {best_csv}")

    best_by_subset = dict(zip(best["subset"], best["optimal_threshold"]))

    # ---------- output 2: predictions at each type's optimal threshold ----------
    optimal_out = tool_out / "optimal"
    if optimal_out.exists():
        shutil.rmtree(optimal_out, ignore_errors=True)
    final_dir = optimal_out / results_subdir
    final_dir.mkdir(parents=True, exist_ok=True)

    # group image types that share the same optimal threshold: one inference each
    subsets_by_threshold = {}
    for subset, thr in best_by_subset.items():
        subsets_by_threshold.setdefault(thr, []).append(subset)

    print(
        f"\n[{tag}] rebuilding predictions at {len(subsets_by_threshold)} "
        f"distinct optimal threshold(s) -> {final_dir}"
    )

    for thr, subsets in sorted(subsets_by_threshold.items()):
        print(f"  threshold={thr} for {subsets} ...", end=" ", flush=True)

        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        try:
            run_inference(
                tool, val_dir, tmp_dir, model, threshold=thr, pretrained=pretrained
            )
        except Exception as e:
            print(f"FAILED: {e}")
            continue

        # keep only the CSVs belonging to the image types this threshold is best for
        n_copied = 0
        for pcsv in sorted((tmp_dir / results_subdir).glob("*.csv")):
            if pcsv.stem.split("_")[0] in subsets:
                shutil.copy2(pcsv, final_dir / pcsv.name)
                n_copied += 1
        print(f"kept {n_copied} CSVs")

    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)

    # record which threshold produced each image, for traceability
    provenance = pd.DataFrame(
        [
            {"image": p.stem, "subset": p.stem.split("_")[0],
             "threshold": best_by_subset.get(p.stem.split("_")[0], np.nan)}
            for p in sorted(final_dir.glob("*.csv"))
        ]
    )
    provenance.to_csv(optimal_out / f"{tag}_threshold_per_image.csv", index=False)

    print(
        f"[{tag}] {len(provenance)} predictions saved to {final_dir} "
        f"(threshold per image recorded in {tag}_threshold_per_image.csv)"
    )

    return best_by_subset, sweep


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Grid search over detection thresholds on the validation set."
    )
    parser.add_argument(
        "--tool", required=True, choices=["deepblink", "spotiflow", "piscis", "ufish"]
    )
    parser.add_argument("--val_dir", required=True, help="Validation images folder")
    parser.add_argument("--gt_dir", required=True, help="Ground-truth CSVs folder")
    parser.add_argument(
        "--output_root", required=True, help="Root folder for sweep results"
    )
    parser.add_argument(
        "--model", default=None, help="Path to trained model (or name for piscis)"
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Sweep the tool's pretrained model instead of the one trained here.",
    )
    args = parser.parse_args()

    grid_search(
        args.tool,
        args.val_dir,
        args.gt_dir,
        args.output_root,
        args.model,
        args.pretrained,
    )
