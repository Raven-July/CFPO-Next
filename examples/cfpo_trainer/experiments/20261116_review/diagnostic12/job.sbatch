#!/bin/bash
set -euo pipefail
PACK=${1:?};CANDIDATE=${2:?}
SESSION=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["session"])' "$CANDIDATE/receipt.json")
REPO=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["source"])' "$CANDIDATE/receipt.json")
source "$SESSION/session.env"
RUN="$CANDIDATE/run"
test ! -e "$RUN/exit-code.txt"
trap 'rc=$?;printf "%s\n" "$rc" > "$RUN/exit-code.txt"' EXIT
cp "${BASH_SOURCE[0]}" "$RUN/submitted.sbatch"
printf 'job=%s\ncpus=%s\nmemory_per_node_MiB=%s\ngpus=%s\n' "$SLURM_JOB_ID" "$SLURM_CPUS_PER_TASK" "$SLURM_MEM_PER_NODE" "${SLURM_JOB_GPUS:-unknown}" > "$RUN/slurm-resources.txt"
driver=/opt/app/nvidia/570.195.03/lib
tf=/groups/xitucheng213/home/share/u2025210309/tf-cc7ab9be-bYDrTa/site-packages
sif=/groups/xitucheng213/home/share/yzy/Counterfact-Projects/verl-vllm018.sif
/opt/app/singularity/bin/singularity exec --cleanenv --nv -B /groups -B "$driver" -B "$PACK" --pwd "$REPO" "$sif" \
 env CANDIDATE="$CANDIDATE" PACK="$PACK" PYTHONPATH="$CFPO_DASH_OVERLAY:$tf:$REPO" \
 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONUTF8=1 \
 LD_LIBRARY_PATH="$driver" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:?Slurm GPU missing}" \
 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HOME=/groups/xitucheng213/home/share/u2025210309/grpo-qwen3vl2b/hf-cache \
 RAY_USAGE_STATS_ENABLED=0 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 VLLM_WORKER_MULTIPROC_METHOD=spawn \
 bash -euo pipefail -c '
python3 "$PACK/run_probe.py" --backend hf
python3 "$PACK/run_probe.py" --backend vllm
python3 "$PACK/collect.py"
'
