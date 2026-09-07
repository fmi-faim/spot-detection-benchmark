#!/bin/bash
#SBATCH --account=faim-dev
#SBATCH --job-name=grid_search
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --partition=main,several
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
#SBATCH --output=logs/grid_search-%j.out
#SBATCH --error=logs/grid_search-%j.err

# Grid Search SLURM Batch Script
# Usage:
# sbatch scripts/submit_grid_search.sh <tool> <val_dir> <gt_dir> <output_root> [<model_path>]

set -eu

if [ $# -lt 4 ]; then
    echo "Usage: sbatch scripts/submit_grid_search.sh <tool> <val_dir> <gt_dir> <output_root> [<model_path>]"
    exit 1
fi

# Activate environment
WD="$(pwd)"
export PATH="$PATH:$WD/infrastructure/apps/pixi/bin"
export PIXI_CACHE_DIR="/tachyon/scratch/gmicro/soucemma/pixi-cache"
export UV_CACHE_DIR="/tachyon/scratch/gmicro/soucemma/uv-cache"
export TMPDIR="$WD/infrastructure/.tmp_$USER"
mkdir -p "$TMPDIR"

PIXIBIN="$WD/infrastructure/apps/pixi/bin/pixi"
if [[ ! -x "$PIXIBIN" ]]; then
    echo "[INFO] Pixi binary not found; running install.sh"
    bash "$WD/install.sh"
fi

TOOL=$1
VAL_DIR=$2
GT_DIR=$3
OUTPUT_ROOT=$4
MODEL_PATH=${5:-""}

echo "[INFO] Job ID: $SLURM_JOB_ID"
echo "[INFO] Node: $SLURMD_NODENAME"
echo "[INFO] Starting grid search job $SLURM_JOB_ID"
echo "[INFO] Tool: $TOOL"
echo "[INFO] Validation data: $VAL_DIR"
echo "[INFO] Ground truth: $GT_DIR"
echo "[INFO] Output root: $OUTPUT_ROOT"
echo "[INFO] Model: ${MODEL_PATH:-<tool default>}"

mkdir -p "$OUTPUT_ROOT"

COMMAND=(
    pixi run -e "$TOOL"
    python src/spot_detection_benchmark/grid_search.py
    --tool "$TOOL"
    --val_dir "$VAL_DIR"
    --gt_dir "$GT_DIR"
    --output_root "$OUTPUT_ROOT"
)

if [ -n "$MODEL_PATH" ]; then
    COMMAND+=(--model "$MODEL_PATH")
fi

"${COMMAND[@]}"

echo "[INFO] Grid search complete"
echo "[INFO] Sweep CSV: $OUTPUT_ROOT/$TOOL/${TOOL}_threshold_sweep.csv"
echo "[INFO] Optimal threshold: $OUTPUT_ROOT/$TOOL/${TOOL}_optimal_threshold.txt"