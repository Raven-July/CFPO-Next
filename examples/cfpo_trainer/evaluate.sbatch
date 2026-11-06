#!/bin/bash
#SBATCH --job-name=cfpo-eval
#SBATCH --partition=g078t
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --comment=xitucheng213
#SBATCH --output=/groups/xitucheng213/home/share/u2024213357/GRPO/logs/cfpo-eval-%j.out

set -euo pipefail
export GRPO_ROOT=${GRPO_ROOT:-/groups/xitucheng213/home/share/u2024213357/GRPO}
export EVAL_MODEL=${EVAL_MODEL:?Set EVAL_MODEL to a base or merged Hugging Face model directory}
export EVAL_LABEL=${EVAL_LABEL:?Set EVAL_LABEL to distinguish base and trained results}
export EVAL_N=${EVAL_N:-8}
export CHECKPOINT_ACTOR=${CHECKPOINT_ACTOR:-}
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" && -n "$CHECKPOINT_ACTOR" ]]; then
  echo 'Merge the checkpoint once before submitting an evaluation array' >&2
  exit 1
fi
export NVIDIA_LIB=${NVIDIA_LIB:-/opt/app/nvidia/570.195.03/lib}
export CFPO_SIF=${CFPO_SIF:-/groups/xitucheng213/home/share/yzy/Counterfact-Projects/verl-vllm018.sif}
/opt/app/singularity/bin/singularity exec --nv -B /groups -B "$NVIDIA_LIB" "$CFPO_SIF" bash <<'INNER'
set -euo pipefail
export LD_LIBRARY_PATH="$NVIDIA_LIB:${LD_LIBRARY_PATH:-}"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$GRPO_ROOT/cfpo-next:$GRPO_ROOT/python-packages"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
cd "$GRPO_ROOT/cfpo-next"
export EVAL_CONFIG="$GRPO_ROOT/data/cfpo/datasets_eval_full.json"
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  eval_config=$(mktemp)
  trap 'rm -f -- "$eval_config"' EXIT
  python - "$eval_config" <<'PY'
import json, os, sys
from pathlib import Path

specs = json.loads(Path(os.environ["EVAL_CONFIG"]).read_text(encoding="utf-8"))
index = int(os.environ["SLURM_ARRAY_TASK_ID"])
if len(specs) != 10 or not 0 <= index < len(specs):
    raise SystemExit(f"Expected ten evaluation datasets and index 0–9, got {len(specs)} and {index}")
Path(sys.argv[1]).write_text(json.dumps([specs[index]]), encoding="utf-8")
print("Evaluating", specs[index]["dataset_name"], flush=True)
PY
  export EVAL_CONFIG="$eval_config"
fi
if [ -n "$CHECKPOINT_ACTOR" ]; then
  python -m verl.model_merger merge --backend fsdp \
    --local_dir "$CHECKPOINT_ACTOR" --target_dir "$EVAL_MODEL" --use_cpu_initialization
fi
python Counterfactual-Eval/inference_eval.py \
  --model_name Qwen3-VL-2B-Instruct --model_prefix "$EVAL_LABEL" \
  --model_path "$EVAL_MODEL" --config_path "$EVAL_CONFIG" \
  --output_dir "$GRPO_ROOT/eval-results/$EVAL_LABEL" \
  --n "$EVAL_N" --cot --batch_size 16 --num_workers 4 --tensor_parallel_size 1
# The reference evaluator catches per-dataset exceptions, so verify completeness.
python - <<'PY'
import json, os
from pathlib import Path
root = Path(os.environ['GRPO_ROOT'])
specs = json.loads(Path(os.environ['EVAL_CONFIG']).read_text(encoding='utf-8'))
output = root/'eval-results'/os.environ['EVAL_LABEL']
n = int(os.environ['EVAL_N'])
for spec in specs:
    p = output / (f"results_Qwen3-VL-2B-Instruct-{os.environ['EVAL_LABEL']}_"
                  f"{spec['dataset_name']}_avg@{os.environ['EVAL_N']}_COT.json")
    result = json.loads(p.read_text())
    expected = len(json.loads(Path(spec['json_path']).read_text()))
    if len(result['details']) != expected or any(len(row['rollouts']) != n for row in result['details']):
        raise SystemExit(f"Incomplete evaluation: {p}")
    print(spec['dataset_name'], result['accuracies'][f'all_avg@{n}_acc'])
print(f'All {len(specs)} assigned evaluation datasets complete')
PY
INNER
