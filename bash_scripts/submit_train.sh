#!/bin/bash
#SBATCH --account=faim-dev
#SBATCH --job-name=train
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --partition=main,several
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=56:00:00
#SBATCH --output=logs/train-%j.out
#SBATCH --error=logs/train-%j.err

# Training SLURM Batch Script
# Usage:
# sbatch bash_scripts/submit_train.sh <tool> [train.py arguments...]
#
# Examples:
#   sbatch --mem=128G bash_scripts/submit_train.sh deepblink --config <config_file>
#   sbatch bash_scripts/submit_train.sh spotiflow --input_dir <input_dir> --output_dir <output_dir>
#   sbatch bash_scripts/submit_train.sh piscis --input_dir <input_dir> [--output_dir <Piscis folder>] [--model_name <name>]
#   sbatch bash_scripts/submit_train.sh ufish --input_dir <train_dir> --val_dir <val_dir> --output_dir <output_dir>
#
# Optional for every tool: --epochs, --lr, --batch_size (tool default if omitted)

set -eu

if [ $# -lt 1 ]; then
    echo "Usage: sbatch bash_scripts/submit_train.sh <tool> [train.py arguments...]"
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
echo "[INFO] Starting training job $SLURM_JOB_ID"
echo "[INFO] Tool: $TOOL (pixi environment: $PIXI_ENV)"
echo "[INFO] Arguments: $*"

# Run training in the tool's own environment
pixi run -e "$PIXI_ENV" python src/spot_detection_benchmark/train.py \
    --tool "$TOOL" \
    "$@"

echo "[INFO] Training complete"
