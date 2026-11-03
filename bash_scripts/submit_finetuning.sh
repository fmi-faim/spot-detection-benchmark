#!/bin/bash
#SBATCH --account=faim-dev
#SBATCH --job-name=finetune
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --partition=main,several
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=56:00:00
#SBATCH --output=logs/finetune-%j.out
#SBATCH --error=logs/finetune-%j.err

# Fine-tuning SLURM Batch Script
# Usage:
# sbatch bash_scripts/submit_finetuning.sh <tool> --input_dir <dir> --output_dir <dir> [finetuning.py arguments...]
#
# Starts from the tool's pretrained 2D model in models.yaml, or --from_trained (trained one),
# or --base_model <model>. --input_dir and --output_dir are required.
#
# Examples:
#   sbatch bash_scripts/submit_finetuning.sh spotiflow --input_dir <data> --output_dir models/spotiflow/general_finetuned --lr 3e-5 --epochs 50
#   sbatch bash_scripts/submit_finetuning.sh ufish --input_dir <train> --val_dir <val> --output_dir models/ufish/finetuned --from_trained --lr 1e-5
#   sbatch bash_scripts/submit_finetuning.sh piscis --input_dir <piscis dataset> --output_dir models/piscis --model_name my_finetuned
#   sbatch --mem=128G bash_scripts/submit_finetuning.sh deepblink --config models/deepblink/config.yaml --input_dir <data.npz> --output_dir models/deepblink/finetuned
# Optional for every tool: --epochs, --lr, --batch_size (tool default if omitted)

set -eu

if [ $# -lt 1 ]; then
    echo "Usage: sbatch bash_scripts/submit_finetuning.sh <tool> --input_dir <dir> --output_dir <dir> [finetuning.py arguments...]"
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

# the plain ufish environment has a CPU-only PyTorch (kept for Windows)
if [ "$TOOL" = "ufish" ]; then
    PIXI_ENV="ufish-gpu"
else
    PIXI_ENV="$TOOL"
fi

echo "[INFO] Job ID: $SLURM_JOB_ID"
echo "[INFO] Node: $SLURMD_NODENAME"
echo "[INFO] Starting fine-tuning job $SLURM_JOB_ID"
echo "[INFO] Tool: $TOOL (pixi environment: $PIXI_ENV)"
echo "[INFO] Arguments: $*"

# Run fine-tuning in the tool's own environment
pixi run -e "$PIXI_ENV" python src/spot_detection_benchmark/finetuning.py \
    --tool "$TOOL" \
    "$@"

echo "[INFO] Fine-tuning complete"
