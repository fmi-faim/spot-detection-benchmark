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
# sbatch bash_scripts/submit_grid_search.sh <tool> --val_dir <dir> --gt_dir <dir> --output_root <dir> [grid_search.py options...]
#
# Every option of grid_search.py is passed through (python src/spot_detection_benchmark/grid_search.py -h):
#   --model <path|name>   --pretrained   --thresholds t1 t2 ...
#   --mode 2d|2d_stack|3d    --voxel_size Z Y X   --cache_gb 4   (and the detection options of run_inference.py)
#
# Choose the thresholds on the val split, report the benchmark on test.
# Examples:
#   sbatch bash_scripts/submit_grid_search.sh piscis --val_dir datasets/train_val_test_split/val --gt_dir datasets/train_val_test_split/val --output_root results/results_grid_search_ufish_dataset
#   sbatch bash_scripts/submit_grid_search.sh deepblink --val_dir datasets/train_val_test_split/val --gt_dir datasets/train_val_test_split/val --output_root results/results_grid_search_ufish_dataset --pretrained
#   sbatch bash_scripts/submit_grid_search.sh log --val_dir datasets/3d/image --gt_dir datasets/3d/csv --output_root results/results_grid_search_3d --mode 3d --voxel_size 300 100 100

set -eu

if [ $# -lt 1 ]; then
    echo "Usage: sbatch bash_scripts/submit_grid_search.sh <tool> --val_dir <dir> --gt_dir <dir> --output_root <dir> [options...]"
    exit 1
fi

# Activate environment
WD="$(pwd)"
export PATH="$PATH:$WD/infrastructure/apps/pixi/bin"
export PIXI_CACHE_DIR="$WD/infrastructure/apps/pixi/.pixi_cache"
export TMPDIR="$WD/infrastructure/.tmp_$USER"
mkdir -p "$TMPDIR"

PIXIBIN="$WD/infrastructure/apps/pixi/bin/pixi"
if [[ ! -x "$PIXIBIN" ]]; then
    echo "[INFO] Pixi binary not found; running install.sh"
    bash "$WD/bash_scripts/install.sh"
fi

TOOL=$1
shift

# the plain ufish environment has a CPU-only PyTorch (for Windows)
if [ "$TOOL" = "ufish" ]; then
    PIXI_ENV="ufish-gpu"
else
    PIXI_ENV="$TOOL"
fi

echo "[INFO] Job ID: $SLURM_JOB_ID"
echo "[INFO] Node: $SLURMD_NODENAME"
echo "[INFO] Starting grid search job $SLURM_JOB_ID"
echo "[INFO] Tool: $TOOL (pixi environment: $PIXI_ENV)"
echo "[INFO] Arguments: $*"

pixi run -e "$PIXI_ENV" python src/spot_detection_benchmark/grid_search.py \
    --tool "$TOOL" \
    "$@"

echo "[INFO] Grid search complete"
echo "[INFO]   <output_root>/<tag>/<tag>_threshold_sweep.csv    per threshold x image"
echo "[INFO]   <output_root>/<tag>/<tag>_threshold_summary.csv  per threshold x image type"
echo "[INFO]   <output_root>/<tag>/<tag>_optimal_thresholds.csv"
echo "[INFO]   <output_root>/<tag>/optimal/                     predictions at the optimum"
echo "[INFO]   <output_root>/<tag>/all_spots/                   every spot at the lowest threshold (tools swept by filter)"
echo "[INFO]   <output_root>/<tag>/run_info.json                settings of the sweep"
