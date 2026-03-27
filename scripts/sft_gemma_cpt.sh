#!/bin/bash
#SBATCH --job-name=sft_gemma_cpt
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=48:00:00
#SBATCH --output=/mnt/fast/nobackup/scratch4weeks/%u/logs/%x.%N.%j.out
#SBATCH --error=/mnt/fast/nobackup/scratch4weeks/%u/logs/%x.%N.%j.err

set -euo pipefail

SCRATCH=/mnt/fast/nobackup/scratch4weeks/$USER
SIF="$SCRATCH/dialect-grpo-qwen-cu121.sif"
REPO="$SCRATCH/repos/dial_post_train"
CONFIG="$REPO/configs/gemma/sft-gemma-cpt.json"

mkdir -p \
  "$SCRATCH/logs" \
  "$SCRATCH/tmp" \
  "$SCRATCH/wandb" \
  "$SCRATCH/hf/hub" \
  "$SCRATCH/hf/datasets" \
  "$SCRATCH/hf/transformers" \
  "$SCRATCH/cache/sft-gemma-cpt" \
  "$SCRATCH/models/sft-gemma-cpt"

export HF_HOME="$SCRATCH/hf"
export HUGGINGFACE_HUB_CACHE="$SCRATCH/hf/hub"
export HF_DATASETS_CACHE="$SCRATCH/hf/datasets"
export TRANSFORMERS_CACHE="$SCRATCH/hf/transformers"
export HF_HUB_ENABLE_HF_TRANSFER=0
export WANDB_DIR="$SCRATCH/wandb"
export WANDB_ENTITY="${WANDB_ENTITY:-jordanpainter}"
export WANDB_PROJECT="sft-gemma-cpt"
export WANDB_NAME="sft-gemma-cpt-$(date +%Y%m%d-%H%M%S)"
export WANDB_MODE="online"
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export HF_TOKEN="${HF_TOKEN:-}"
export TMPDIR="$SCRATCH/tmp"
export TEMP="$SCRATCH/tmp"
export TMP="$SCRATCH/tmp"
export TOKENIZERS_PARALLELISM=false

echo "HOST=$(hostname)"
echo "SIF=$SIF"
echo "REPO=$REPO"
echo "CONFIG=$CONFIG"
echo "WANDB_ENTITY=$WANDB_ENTITY"
echo "WANDB_PROJECT=$WANDB_PROJECT"
echo "WANDB_NAME=$WANDB_NAME"
nvidia-smi

apptainer exec --nv "$SIF" bash -lc "
    set -euo pipefail
    export HF_HOME='$HF_HOME'
    export HUGGINGFACE_HUB_CACHE='$HUGGINGFACE_HUB_CACHE'
    export HF_DATASETS_CACHE='$HF_DATASETS_CACHE'
    export TRANSFORMERS_CACHE='$TRANSFORMERS_CACHE'
    export HF_HUB_ENABLE_HF_TRANSFER='$HF_HUB_ENABLE_HF_TRANSFER'
    export WANDB_DIR='$WANDB_DIR'
    export WANDB_ENTITY='$WANDB_ENTITY'
    export WANDB_PROJECT='$WANDB_PROJECT'
    export WANDB_NAME='$WANDB_NAME'
    export WANDB_MODE='$WANDB_MODE'
    export WANDB_API_KEY='$WANDB_API_KEY'
    export HF_TOKEN='$HF_TOKEN'
    export TMPDIR='$TMPDIR'
    export TEMP='$TEMP'
    export TMP='$TMP'
    export TOKENIZERS_PARALLELISM='$TOKENIZERS_PARALLELISM'
    export PYTHONPATH='$REPO':\${PYTHONPATH:-}
    cd '$REPO'
    echo 'PWD=' \$(pwd)
    echo 'PYTHONPATH=' \$PYTHONPATH
    python3 --version
    python3 -c \"import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'count', torch.cuda.device_count())\"
    python3 -c \"import os; print('CONFIG exists:', os.path.exists('$CONFIG'))\"
    python3 -c \"import os; print('WANDB_ENTITY=', os.environ.get('WANDB_ENTITY')); print('WANDB_PROJECT=', os.environ.get('WANDB_PROJECT')); print('WANDB_NAME=', os.environ.get('WANDB_NAME'))\"
    accelerate launch --num_processes 1 -m src.sft -c '$CONFIG'
"
