#!/bin/bash
#SBATCH --account=faim-dev
#SBATCH --job-name=evaluate
#SBATCH --cpus-per-task=2
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --partition=main,several
#SBATCH --mem=8G
#SBATCH --time=2:00:00
#SBATCH --output=logs/evaluate-%j.out
#SBATCH --error=logs/evaluate-%j.err

# Evaluation SLURM Batch Script (CPU only)
# Usage:
# sbatch bash_scripts/submit_evaluate.sh --gt_dir <dir> --pred_dir <dir> --output_dir <dir> [evaluate.py options...]
#   options: --mode 2d|2d_stack|3d   --voxel_size Z Y X (3D)   --cutoff 3   --mdist 3   --min_true 1
#
# Example:
#   sbatch bash_scripts/submit_evaluate.sh --gt_dir datasets/train_val_test_split/test --pred_dir results/results_inference_ufish_dataset/results_piscis --output_dir results/metrics_inference_ufish_dataset/results_piscis

set -eu

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

echo "[INFO] Job ID: $SLURM_JOB_ID"
echo "[INFO] Arguments: $*"

pixi run -e default python src/spot_detection_benchmark/evaluate.py "$@"

echo "[INFO] Evaluation complete"
