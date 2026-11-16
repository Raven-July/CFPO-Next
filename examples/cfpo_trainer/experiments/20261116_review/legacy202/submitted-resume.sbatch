#!/bin/bash
#SBATCH --partition=g078t
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=3-00:00:00
#SBATCH --comment=xitucheng213
set -euo pipefail
PACK=${1:?}; CANDIDATE=${2:?}
SESSION=$(cat "$PACK/session-path.txt")
source "$SESSION/session.env"
RUN="$CANDIDATE/run"; REPO="$CANDIDATE/source"
test ! -e "$RUN/exit-code.txt"
trap 'rc=$?; printf "%s\n" "$rc" > "$RUN/exit-code.txt"' EXIT
cp "${BASH_SOURCE[0]}" "$RUN/submitted.sbatch"
printf 'job=%s\ncpus=%s\nmemory_per_node_MiB=%s\ngpus=%s\n' \
    "$SLURM_JOB_ID" "$SLURM_CPUS_PER_TASK" "$SLURM_MEM_PER_NODE" "${SLURM_JOB_GPUS:-unknown}" > "$RUN/slurm-resources.txt"
driver=/opt/app/nvidia/570.195.03/lib
tf=/groups/xitucheng213/home/share/u2025210309/tf-cc7ab9be-bYDrTa/site-packages
sif=/groups/xitucheng213/home/share/yzy/Counterfact-Projects/verl-vllm018.sif
/opt/app/singularity/bin/singularity exec --cleanenv --nv -B /groups -B "$driver" -B "$PACK" --pwd "$REPO" "$sif" \
 env CFPO_RUN_DIR="$RUN" CFPO_MANIFEST_DIR="$CFPO_MANIFEST_DIR" CFPO_DASH_OVERLAY="$CFPO_DASH_OVERLAY" \
 CFPO_REVIEWED=yes MODEL_PATH="${CFPO_BASE_MODEL:-/groups/xitucheng213/home/share/u2025210309/grpo-qwen3vl2b/models/Qwen3-VL-2B-Instruct}" \
 PYTHONPATH="$CFPO_DASH_OVERLAY:$tf:$REPO" CFPO_TF_OVERLAY="$tf" \
 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 LD_LIBRARY_PATH="$driver" \
 CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:?Slurm CUDA_VISIBLE_DEVICES missing}" \
 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HOME=/groups/xitucheng213/home/share/u2025210309/grpo-qwen3vl2b/hf-cache \
 RAY_USAGE_STATS_ENABLED=0 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 \
 PACK="$PACK" CANDIDATE="$CANDIDATE" SESSION="$SESSION" \
 bash -euo pipefail -c '
  python3 -c "import torch; assert torch.cuda.device_count()==2, torch.cuda.device_count(); [torch.ones(1,device=\"cuda:\"+str(i)).sum().item() for i in range(2)]; print(\"ALLOCATED_TWO_GPU_PROBE_OK\")"
  checkpoint="$SESSION/train/checkpoints/global_step_100"
  overrides=(trainer.resume_mode=resume_path "trainer.resume_from_path=$checkpoint" "trainer.experiment_name=$(basename "$CANDIDATE")")
  bash examples/cfpo_trainer/run_formal.sh --check-config "${overrides[@]}" > "$CFPO_RUN_DIR/resolved-config.yaml"
  python3 "$PACK/resume100_config_guard.py" "$CANDIDATE"
  python3 examples/cfpo_trainer/formal_contract.py --run "$CFPO_RUN_DIR"
  bash examples/cfpo_trainer/run_formal.sh --train "${overrides[@]}"
  python3 examples/cfpo_trainer/formal_tools.py check-training --run "$CFPO_RUN_DIR" --step 202
  echo RESUME100_TO202_COMPLETE
 '
