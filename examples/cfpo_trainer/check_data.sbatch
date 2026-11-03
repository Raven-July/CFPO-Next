#!/bin/bash
#SBATCH --job-name=bao-cfpo-data
#SBATCH --partition=g078t
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#SBATCH --comment=xitucheng213

# Data/reward validation only: no model initialization, rollout, or training.
set -euo pipefail
: "${CFPO_CHECK_DIR:?Set CFPO_CHECK_DIR to the personal log directory}"
repo=/groups/xitucheng213/home/share/u2025210309/CFPO-Next
overlay=/groups/xitucheng213/home/share/u2025210309/tf-cc7ab9be-bYDrTa/site-packages
sif=/groups/xitucheng213/home/share/yzy/Counterfact-Projects/verl-vllm018.sif
printf 'JOB=%s NODE=%s REPO=%s\n' "$SLURM_JOB_ID" "$(hostname)" "$repo"
git -C "$repo" rev-parse HEAD

# No --nv: this check does not use CUDA. GPU allocation is a partition requirement.
/opt/app/singularity/bin/singularity exec --cleanenv -B /groups --pwd "$repo" "$sif" \
    env PYTHONPATH="$overlay:$repo" PYTHONNOUSERSITE=1 \
        PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
        HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
        CFPO_CHECK_DIR="$CFPO_CHECK_DIR" \
    python3 -s examples/cfpo_trainer/check_data.py
