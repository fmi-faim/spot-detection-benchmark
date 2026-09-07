import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.spatial

# deepBlink's exact metric implementation
from evaluation_metrics import EPS, f1_integral, linear_sum_assignment, offset_euclidean

# direct import — works because grid_search.py runs in the same env as run_inference.py
from run_inference import run_inference

CUTOFF = 3.0
MDIST = 3.0
N_CUTOFFS = 50
MIN_TRUE = 1

THRESHOLD_GRIDS = {
    "deepblink": np.round(np.arange(0.1, 1.0, 0.05), 3).tolist(),
    "spotiflow": np.round(np.arange(0.1, 1.0, 0.05), 3).tolist(),
    "piscis": np.round(np.arange(0.1, 9.0, 0.5), 3).tolist(),
    "ufish": np.round(np.arange(0.1, 1.0, 0.05), 3).tolist(),
}


def _load(csv):
    df = pd.read_csv(csv)
    return df[["axis-0", "axis-1"]].to_numpy() if len(df) else np.empty((0, 2))


def score_image(pred, true):
    pred = np.asarray(pred, float).reshape(-1, 2)
    true = np.asarray(true, float).reshape(-1, 2)

    import warnings

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
    rows = []
    for pcsv in sorted(Path(pred_dir).glob("*.csv")):
        gcsv = Path(gt_dir) / pcsv.name
        if not gcsv.exists():
            continue
        m = score_image(_load(pcsv), _load(gcsv))
        m["image"] = pcsv.stem
        rows.append(m)
    return pd.DataFrame(rows)


def aggregate(df):
    scored = df[df["n_true"] >= MIN_TRUE]
    return {
        "TP_total": int(df["TP"].sum()),
        "FP_total": int(df["FP"].sum()),
        "FN_total": int(df["FN"].sum()),
        "precision_mean": scored["precision"].mean(),
        "recall_mean": scored["recall"].mean(),
        "f1_mean": scored["f1"].mean(),
        "f1_integral_mean": scored["f1_integral"].mean(),
        "f1_integral_std": scored["f1_integral"].std(),
        "distance_error_mean": scored["distance_error"].mean(),
        "n_images": len(df),
    }


def grid_search(tool, val_dir, gt_dir, output_root, model):
    val_dir = Path(val_dir)
    gt_dir = Path(gt_dir)
    output_root = Path(output_root)

    tool_out = output_root / tool
    tool_out.mkdir(parents=True, exist_ok=True)

    thresholds = THRESHOLD_GRIDS[tool]
    sweep_rows = []
    print(
        f"[{tool}] sweeping {len(thresholds)} thresholds: "
        f"{thresholds[0]} -> {thresholds[-1]}"
    )

    for thr in thresholds:
        thr_dir = tool_out / f"threshold_{thr}"
        print(f"  threshold={thr} ...", end=" ", flush=True)

        try:
            run_inference(tool, val_dir, thr_dir, model, threshold=thr)
        except Exception as e:
            print(f"FAILED: {e}")
            continue

        pred_dir = thr_dir / f"results_{tool}"
        if not pred_dir.exists():
            print("no output dir")
            continue

        df = score_folder(pred_dir, gt_dir)
        if df.empty:
            print("no scored images")
            continue

        agg = aggregate(df)
        agg["threshold"] = thr
        sweep_rows.append(agg)
        print(
            f"F1-integral={agg['f1_integral_mean']:.4f}  "
            f"F1={agg['f1_mean']:.4f}  "
            f"P={agg['precision_mean']:.4f}  "
            f"R={agg['recall_mean']:.4f}"
        )

    if not sweep_rows:
        print(f"[{tool}] no valid results — check errors above")
        return None, None

    sweep_df = pd.DataFrame(sweep_rows).sort_values("threshold")
    sweep_csv = tool_out / f"{tool}_threshold_sweep.csv"
    sweep_df.to_csv(sweep_csv, index=False)
    print(f"\n[{tool}] sweep saved to {sweep_csv}")

    # optimal threshold = max mean F1-integral on validation
    best_row = sweep_df.loc[sweep_df["f1_integral_mean"].idxmax()]
    best_thr = best_row["threshold"]
    print(
        f"[{tool}] optimal threshold: {best_thr} "
        f"(val F1-integral = {best_row['f1_integral_mean']:.4f})"
    )

    # save optimal threshold for later reference
    (tool_out / f"{tool}_optimal_threshold.txt").write_text(str(best_thr))

    # re-run inference at optimal threshold — output named with threshold value
    optimal_out = tool_out / f"results_{tool}_thr{best_thr}"
    print(f"[{tool}] re-running at optimal threshold -> {optimal_out}")
    try:
        run_inference(tool, val_dir, optimal_out, model, threshold=best_thr)
        print(f"[{tool}] optimal predictions saved to {optimal_out}/results_{tool}/")
    except Exception as e:
        print(f"[{tool}] optimal re-run failed: {e}")

    return best_thr, sweep_df


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
    args = parser.parse_args()

    grid_search(args.tool, args.val_dir, args.gt_dir, args.output_root, args.model)
