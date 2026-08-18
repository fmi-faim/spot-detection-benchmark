#!/bin/bash
#SBATCH --account=dlthings
#SBATCH --job-name=train_spotiflow
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --partition=main,several
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=56:00:00
#SBATCH --output=logs/train-%j.out
#SBATCH --error=logs/train-%j.err

# Spotiflow Training SLURM Batch Script
# Usage:
# sbatch scripts/submit_train_spotiflow.sh <input_dir> <output_dir>

set -eu

if [ $# -lt 2 ]; then
    echo "Usage: sbatch scripts/submit_train_spotiflow.sh <input_dir> <output_dir>"
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
echo "[INFO] Ensuring Spotiflow environment is ready..."
echo "[INFO] Using Pixi Spotiflow environment"
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
pixi run -e spotiflow python src/spot_detection_benchmark/train_spotiflow.py \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \

echo "[INFO] Training complete"
