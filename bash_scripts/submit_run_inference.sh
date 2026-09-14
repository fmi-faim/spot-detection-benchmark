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
# sbatch scripts/submit_run_inference.sh <tool> <input_dir> <output_root> [<model_path>] [<threshold>] [<pretrained>]
#   <model_path>  optional, "" to use the tool default
#   <threshold>   optional, "" to use the tool default
#   <pretrained>  optional, pass "pretrained" to use the tool's pretrained model

set -eu

if [ $# -lt 3 ]; then
    echo "Usage: sbatch scripts/submit_run_inference.sh <tool> <input_dir> <output_root> [<model_path>] [<threshold>] [<pretrained>]"
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
INPUT_DIR=$2
OUTPUT_ROOT=$3
MODEL_POINTER=${4:-""}
THRESHOLD=${5:-""}
PRETRAINED=${6:-""}

# Ensure environment is properly installed
echo "[INFO] Job ID: $SLURM_JOB_ID"
echo "[INFO] Node: $SLURMD_NODENAME"
echo "[INFO] Starting inference job $SLURM_JOB_ID"
echo "[INFO] Tool: $TOOL"
echo "[INFO] Input data: $INPUT_DIR"
echo "[INFO] Output root: $OUTPUT_ROOT"
echo "[INFO] Model pointer: ${MODEL_POINTER:-<tool default>}"
echo "[INFO] Threshold: ${THRESHOLD:-<tool default>}"
echo "[INFO] Pretrained: ${PRETRAINED:-no}"

# Ensure output directory exists
if [ -n "$OUTPUT_ROOT" ]; then
    mkdir -p "$OUTPUT_ROOT"
fi

COMMAND=(
    pixi run -e "$TOOL"
    python src/spot_detection_benchmark/run_inference.py
    --tool "$TOOL"
    --input_dir "$INPUT_DIR"
    --output_root "$OUTPUT_ROOT"
)

if [ -n "$MODEL_POINTER" ]; then
    COMMAND+=(--model "$MODEL_POINTER")
fi

if [ -n "$THRESHOLD" ]; then
    COMMAND+=(--threshold "$THRESHOLD")
fi

if [ -n "$PRETRAINED" ]; then
    COMMAND+=(--pretrained)
fi

"${COMMAND[@]}"

echo "[INFO] Inference complete"
