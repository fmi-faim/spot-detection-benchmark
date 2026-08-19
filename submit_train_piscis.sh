#!/bin/bash
#SBATCH --account=faim-dev
#SBATCH --job-name=train_piscis
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --partition=main,several
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=45:00:00
#SBATCH --output=logs/train-%j.out
#SBATCH --error=logs/train-%j.err

# Piscis Training SLURM Batch Script
# Usage:
# sbatch scripts/submit_train_piscis.sh <input_dir>

set -eu

if [ $# -lt 1 ]; then
    echo "Usage: sbatch scripts/submit_train_piscis.sh <input_dir>"
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
echo "[INFO] Ensuring Piscis environment is ready..."
echo "[INFO] Using Pixi Piscis environment"
echo "[INFO] Job ID: $SLURM_JOB_ID"
echo "[INFO] Node: $SLURMD_NODENAME"

INPUT_DIR=$1
OUTPUT_DIR=$2

echo "[INFO] Starting training job $SLURM_JOB_ID"
echo "[INFO] Input: $INPUT_DIR"
echo "[INFO] Output: $OUTPUT_DIR"

# Ensure output directory exists
if [ -n "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
fi

# Run training
pixi run -e piscis python src/spot_detection_benchmark/train_piscis.py \
    --input_dir "$INPUT_DIR" \

echo "[INFO] Training complete"
