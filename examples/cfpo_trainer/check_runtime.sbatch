#!/bin/bash
#SBATCH --job-name=bao-cfpo-preflight
#SBATCH --partition=g078t
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:05:00
#SBATCH --comment=xitucheng213

# This file never calls run_grpo.sh --train.
set -euo pipefail
: "${CFPO_RUN_DIR:?Set CFPO_RUN_DIR to a fresh personal log directory}"
repo=/groups/xitucheng213/home/share/u2025210309/CFPO-Next
overlay=/groups/xitucheng213/home/share/u2025210309/tf-cc7ab9be-bYDrTa/site-packages
logging_overlay=/groups/xitucheng213/home/share/u2025210309/swanlab-env-ecn63f2h/site-packages
work=/groups/xitucheng213/home/share/u2025210309/grpo-qwen3vl2b
driver=/opt/app/nvidia/570.195.03/lib
printf 'JOB=%s NODE=%s RUN=%s\n' "$SLURM_JOB_ID" "$(hostname)" "$CFPO_RUN_DIR"
git -C "$repo" rev-parse HEAD > "$CFPO_RUN_DIR/base-commit.txt"
git -C "$repo" diff --binary > "$CFPO_RUN_DIR/tracked-changes.patch"
git -C "$repo" status --short > "$CFPO_RUN_DIR/worktree-status.txt"
git -C "$repo" ls-files --modified --others --exclude-standard -z |
    tar -C "$repo" --null -T - -cf "$CFPO_RUN_DIR/changed-source.tar"

/opt/app/singularity/bin/singularity exec --nv --cleanenv \
    -B /groups -B "$driver" --pwd "$repo" \
    /groups/xitucheng213/home/share/yzy/Counterfact-Projects/verl-vllm018.sif \
    env CFPO_REPO="$repo" CFPO_OVERLAY="$overlay" CFPO_RUN_DIR="$CFPO_RUN_DIR" \
        MODEL_PATH="$work/models/Qwen3-VL-2B-Instruct" \
        PYTHONPATH="$logging_overlay:$overlay:$repo" PYTHONNOUSERSITE=1 \
        PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 LD_LIBRARY_PATH="$driver" \
        HF_HOME="$work/hf-cache" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
        TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 \
        SWANLAB_MODE=offline SWANLAB_LOG_DIR="$CFPO_RUN_DIR/swanlog" \
        RAY_USAGE_STATS_ENABLED=0 \
    bash -euo pipefail <<'INNER'
bash -n examples/cfpo_trainer/run_grpo.sh
bash examples/cfpo_trainer/run_grpo.sh --check-config > "$CFPO_RUN_DIR/resolved-config.yaml"
echo GRPO_CONFIG_RESOLUTION_OK
python3 -s examples/cfpo_trainer/check_runtime.py
INNER
