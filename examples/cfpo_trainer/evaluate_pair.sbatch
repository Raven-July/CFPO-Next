#!/bin/bash
#SBATCH --job-name=bao-cfpo-paired-eval
#SBATCH --partition=g078t
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --comment=xitucheng213

set -euo pipefail
: "${CFPO_EVAL_RUN:?Set a fresh personal evaluation directory}"
repo=${CFPO_REPO:-/groups/xitucheng213/home/share/u2025210309/CFPO-Next}
train_run=${CFPO_TRAIN_RUN:-/groups/xitucheng213/home/share/u2025210309/grpo-20step-imD15m}
base_model=${CFPO_BASE_MODEL:-/groups/xitucheng213/home/share/u2025210309/grpo-qwen3vl2b/models/Qwen3-VL-2B-Instruct}
tf=${CFPO_TF_OVERLAY:-/groups/xitucheng213/home/share/u2025210309/tf-cc7ab9be-bYDrTa/site-packages}
log=${CFPO_LOGGING_OVERLAY:-/groups/xitucheng213/home/share/u2025210309/swanlab-env-ecn63f2h/site-packages}
driver=${CFPO_DRIVER_LIB:-/opt/app/nvidia/570.195.03/lib}
sif=${CFPO_SIF:-/groups/xitucheng213/home/share/yzy/Counterfact-Projects/verl-vllm018.sif}
step=${CFPO_EVAL_STEP:-20}
case "$step" in ''|*[!0-9]*) echo 'CFPO_EVAL_STEP must be a nonnegative integer' >&2; exit 2 ;; esac
printf 'JOB=%s NODE=%s RUN=%s\n' "$SLURM_JOB_ID" "$(hostname)" "$CFPO_EVAL_RUN"
trap 'rc=$?; printf "PAIRED_EVAL_EXIT=%s END=%s\n" "$rc" "$(date -Is)"' EXIT
git -C "$repo" rev-parse HEAD > "$CFPO_EVAL_RUN/base-commit.txt"
git -C "$repo" diff --binary > "$CFPO_EVAL_RUN/tracked-changes.patch"
git -C "$repo" ls-files --modified --others --exclude-standard -z |
    tar -C "$repo" --null -T - -cf "$CFPO_EVAL_RUN/changed-source.tar"
cp "${BASH_SOURCE[0]}" "$CFPO_EVAL_RUN/submitted.sbatch"

/opt/app/singularity/bin/singularity exec --nv --cleanenv \
    -B /groups -B "$driver" --pwd "$repo" \
    "$sif" \
    env CFPO_EVAL_RUN="$CFPO_EVAL_RUN" CFPO_TRAIN_RUN="$train_run" CFPO_BASE_MODEL="$base_model" \
        CFPO_EVAL_STEP="$step" \
        PYTHONPATH="$log:$tf:$repo" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
        PYTHONUNBUFFERED=1 LD_LIBRARY_PATH="$driver" \
        HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
        OMP_NUM_THREADS=1 VLLM_WORKER_MULTIPROC_METHOD=spawn \
    bash -euo pipefail <<'INNER'
actor="$CFPO_TRAIN_RUN/checkpoints/global_step_$CFPO_EVAL_STEP/actor"
merged="$CFPO_EVAL_RUN/merged-step$CFPO_EVAL_STEP"
test -s "$actor/model_world_size_1_rank_0.pt"
test -s "$actor/optim_world_size_1_rank_0.pt"
python3 -s examples/cfpo_trainer/evaluate_pair.py prepare \
    --data-root /groups/xitucheng213/home/share/yzy/Counterfact-Projects/Datasets \
    --train-run "$CFPO_TRAIN_RUN" --run "$CFPO_EVAL_RUN" --per-benchmark 64

python3 -m verl.model_merger merge --backend fsdp \
    --local_dir "$actor" --target_dir "$merged"
echo CHECKPOINT_HF_EXPORT_COMPLETE

# Separate processes release GPU memory between model evaluations.
python3 -s examples/cfpo_trainer/evaluate_pair.py infer \
    --run "$CFPO_EVAL_RUN" --model "$CFPO_BASE_MODEL" \
    --tokenizer "$CFPO_BASE_MODEL" --label before
python3 -s examples/cfpo_trainer/evaluate_pair.py infer \
    --run "$CFPO_EVAL_RUN" --model "$merged" \
    --tokenizer "$CFPO_BASE_MODEL" --label after
python3 -s examples/cfpo_trainer/evaluate_pair.py compare --run "$CFPO_EVAL_RUN"
INNER
