#!/bin/bash
#SBATCH --account=faim-dev
#SBATCH --job-name=train_deepblink
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --partition=main,several
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=45:00:00
#SBATCH --output=logs/train-%j.out
#SBATCH --error=logs/train-%j.err

# DeepBlink Training SLURM Batch Script
# Usage:
# sbatch scripts/submit_train_deepblink.sh <config_file>

set -eu

if [ $# -lt 1 ]; then
    echo "Usage: sbatch scripts/submit_train_deepblink.sh <config_file>"
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
    bash "$WD/install.sh"
fi

# Ensure environment is properly installed
echo "[INFO] Ensuring DeepBlink environment is ready..."
echo "[INFO] Using Pixi DeepBlink environment"
echo "[INFO] Job ID: $SLURM_JOB_ID"
echo "[INFO] Node: $SLURMD_NODENAME"

CONFIG_FILE=$1

echo "[INFO] Starting training job $SLURM_JOB_ID"
echo "[INFO] Config: $CONFIG_FILE"

# Run training
pixi run -e deepblink python src/spot_detection_benchmark/train_deepblink.py \
    --config "$CONFIG_FILE"

echo "[INFO] Training complete"
