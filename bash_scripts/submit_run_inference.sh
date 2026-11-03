#!/bin/bash
#SBATCH --account=faim-dev
#SBATCH --job-name=run_inference
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --partition=main,several
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --output=logs/inference-%j.out
#SBATCH --error=logs/inference-%j.err

# Inference SLURM Batch Script
# Usage:
# sbatch bash_scripts/submit_run_inference.sh <tool> --input_dir <dir> --output_root <dir> [run_inference.py options...]
#
# Every option of run_inference.py is passed through (python src/spot_detection_benchmark/run_inference.py -h):
#   --model <path|name>   --threshold <value>   --pretrained
#   --mode 2d|2d_stack|3d    --voxel_size Z Y X    --lateral_sigma / --axial_sigma (LoG)
#   --merge_search_range, --merge_min_planes, ... (deepBlink 3D z merge)   --no_ufish_blend_3d
#
# Examples:
#   sbatch bash_scripts/submit_run_inference.sh spotiflow --input_dir datasets/train_val_test_split/test --output_root results/results_inference_ufish_dataset
#   sbatch bash_scripts/submit_run_inference.sh ufish --input_dir datasets/train_val_test_split/test --output_root results/results_inference_ufish_dataset --pretrained --threshold 0.4
#   sbatch bash_scripts/submit_run_inference.sh log --input_dir datasets/3d/image --output_root results/inf_3d --mode 3d --voxel_size 300 100 100

set -eu

if [ $# -lt 1 ]; then
    echo "Usage: sbatch bash_scripts/submit_run_inference.sh <tool> --input_dir <dir> --output_root <dir> [options...]"
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
echo "[INFO] Starting inference job $SLURM_JOB_ID"
echo "[INFO] Tool: $TOOL (pixi environment: $PIXI_ENV)"
echo "[INFO] Arguments: $*"

pixi run -e "$PIXI_ENV" python src/spot_detection_benchmark/run_inference.py \
    --tool "$TOOL" \
    "$@"

echo "[INFO] Inference complete"
