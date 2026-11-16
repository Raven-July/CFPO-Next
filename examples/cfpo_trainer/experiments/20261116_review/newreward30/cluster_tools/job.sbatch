#!/bin/bash
set -euo pipefail
PACK=${1:?}; CANDIDATE=${2:?}
SESSION=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["session"])' "$CANDIDATE/receipt.json")
source "$SESSION/session.env"
RUN="$CANDIDATE/run"; REPO="$CANDIDATE/source"
test ! -e "$RUN/exit-code.txt"
trap 'rc=$?; printf "%s\n" "$rc" > "$RUN/exit-code.txt"' EXIT
cp "${BASH_SOURCE[0]}" "$RUN/submitted.sbatch"
printf 'job=%s\ncpus=%s\nmemory_per_node_MiB=%s\ngpus=%s\n' "$SLURM_JOB_ID" "$SLURM_CPUS_PER_TASK" "$SLURM_MEM_PER_NODE" "${SLURM_JOB_GPUS:-unknown}" > "$RUN/slurm-resources.txt"
driver=/opt/app/nvidia/570.195.03/lib
tf=/groups/xitucheng213/home/share/u2025210309/tf-cc7ab9be-bYDrTa/site-packages
sif=/groups/xitucheng213/home/share/yzy/Counterfact-Projects/verl-vllm018.sif
/opt/app/singularity/bin/singularity exec --cleanenv --nv -B /groups -B "$driver" -B "$PACK" --pwd "$REPO" "$sif" \
 env CFPO_RUN_DIR="$RUN" CFPO_MANIFEST_DIR="$CFPO_MANIFEST_DIR" CFPO_DASH_OVERLAY="$CFPO_DASH_OVERLAY" \
 CFPO_VALIDATION30_AUTHORIZED=yes MODEL_PATH=/groups/xitucheng213/home/share/u2025210309/grpo-qwen3vl2b/models/Qwen3-VL-2B-Instruct \
 PYTHONPATH="$CFPO_DASH_OVERLAY:$tf:$REPO" CFPO_TF_OVERLAY="$tf" \
 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 LD_LIBRARY_PATH="$driver" \
 CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:?Slurm GPU visibility missing}" \
 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HOME=/groups/xitucheng213/home/share/u2025210309/grpo-qwen3vl2b/hf-cache \
 RAY_USAGE_STATS_ENABLED=0 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 \
 PACK="$PACK" CANDIDATE="$CANDIDATE" SESSION="$SESSION" \
 bash "$PACK/inside.sh"
