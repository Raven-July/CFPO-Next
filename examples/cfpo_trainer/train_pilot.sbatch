#!/bin/bash
#SBATCH --job-name=bao-cfpo-grpo-3step
#SBATCH --partition=g078t
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --comment=xitucheng213

# Defaults to three steps; Slurm export overrides also support the 20-step stability run.
set -euo pipefail
: "${CFPO_RUN_DIR:?Set CFPO_RUN_DIR to a fresh personal experiment directory}"
repo=/groups/xitucheng213/home/share/u2025210309/CFPO-Next
work=/groups/xitucheng213/home/share/u2025210309/grpo-qwen3vl2b
tf_overlay=/groups/xitucheng213/home/share/u2025210309/tf-cc7ab9be-bYDrTa/site-packages
logging_overlay=/groups/xitucheng213/home/share/u2025210309/swanlab-env-ecn63f2h/site-packages
driver=/opt/app/nvidia/570.195.03/lib
export CFPO_TRAIN_ROWS=${CFPO_TRAIN_ROWS:-32}
export CFPO_TRAIN_STEPS=${CFPO_TRAIN_STEPS:-3}
checkpoint_contents=${CFPO_CHECKPOINT_CONTENTS:-'["model","extra"]'}
if [ "${CFPO_SAVE_OPTIMIZER:-0}" = 1 ]; then
    checkpoint_contents='["model","optimizer","extra"]'
fi
printf 'JOB=%s NODE=%s RUN=%s START=%s\n' "$SLURM_JOB_ID" "$(hostname)" "$CFPO_RUN_DIR" "$(date -Is)"
trap 'rc=$?; printf "PILOT_EXIT=%s END=%s\n" "$rc" "$(date -Is)"' EXIT

# Save the actual dirty source tree as well as its base commit. Cluster HEAD may
# still be the transferred base even though the PR contains a newer commit.
git -C "$repo" rev-parse HEAD > "$CFPO_RUN_DIR/base-commit.txt"
git -C "$repo" diff --binary > "$CFPO_RUN_DIR/tracked-changes.patch"
git -C "$repo" status --short > "$CFPO_RUN_DIR/worktree-status.txt"
git -C "$repo" ls-files --modified --others --exclude-standard -z |
    tar -C "$repo" --null -T - -cf "$CFPO_RUN_DIR/changed-source.tar"
cp "${BASH_SOURCE[0]}" "$CFPO_RUN_DIR/submitted.sbatch"

# Normalize only the explicitly selected subset for this bounded experiment.
CFPO_RUN_DIR="$CFPO_RUN_DIR" python3 - <<'PY'
import hashlib
import json
import os
import shutil
from pathlib import Path

root = Path("/groups/xitucheng213/home/share/yzy/Counterfact-Projects/Datasets")
source = root / "V3-math/ViRL39K_train.json"
payload = source.read_bytes()
rows = json.loads(payload.decode("utf-8"))
count = int(os.environ["CFPO_TRAIN_ROWS"])
steps = int(os.environ["CFPO_TRAIN_STEPS"])
if count < 4 or steps < 1 or not isinstance(rows, list) or len(rows) < count:
    raise SystemExit("Invalid subset/step count or insufficient training samples")
if steps > count // 4:
    raise SystemExit("One epoch has fewer batches than the requested steps")
run = Path(os.environ["CFPO_RUN_DIR"])
if os.environ.get("CFPO_SAVE_OPTIMIZER") == "1" and shutil.disk_usage(str(run)).free < 80 * 1024**3:
    raise SystemExit("Less than 80 GiB free on this filesystem for model/optimizer checkpoint saving")
subset = run / "pilot_samples.json"
subset.write_text(json.dumps(rows[:count], ensure_ascii=False), encoding="utf-8")
(run / "pilot_train.json").write_text(json.dumps([
    {"dataset": str(subset), "image": str(root / "ViRL39K_images")}
]), encoding="utf-8")
(run / "data-selection.json").write_text(json.dumps({
    "source": str(source), "source_sha256": hashlib.sha256(payload).hexdigest(),
    "selection": "first {} rows; dataset shuffle seed 42".format(count), "source_rows": len(rows),
    "steps": steps, "selected_ids": [row.get("id") for row in rows[:count]],
    "scope": "training flow pilot; validation disabled; no accuracy claim",
}, indent=2), encoding="utf-8")
print("PILOT_DATA_READY: {} training rows; {} steps".format(count, steps), flush=True)
PY

/opt/app/singularity/bin/singularity exec --nv --cleanenv \
    -B /groups -B "$driver" --pwd "$repo" \
    /groups/xitucheng213/home/share/yzy/Counterfact-Projects/verl-vllm018.sif \
    env CFPO_RUN_DIR="$CFPO_RUN_DIR" \
        MODEL_PATH="$work/models/Qwen3-VL-2B-Instruct" \
        TRAIN_FILE="$CFPO_RUN_DIR/pilot_train.json" \
        TRAIN_MAX_SAMPLES="$CFPO_TRAIN_ROWS" TOTAL_STEPS="$CFPO_TRAIN_STEPS" \
        MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-1024}" \
        ENABLE_CHUNKED_PREFILL="${ENABLE_CHUNKED_PREFILL:-True}" \
        CHECKPOINT_SAVE_CONTENTS="$checkpoint_contents" \
        SAVE_FREQ="${SAVE_FREQ:-$CFPO_TRAIN_STEPS}" \
        VAL_FILE="$work/data/geo3k/test.parquet" \
        CFPO_TF_OVERLAY="$tf_overlay" CFPO_LOGGING_OVERLAY="$logging_overlay" \
        PYTHONPATH="$logging_overlay:$tf_overlay:$repo" \
        PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
        LD_LIBRARY_PATH="$driver" HF_HOME="$work/hf-cache" \
        HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
        TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 RAY_NUM_CPUS=8 \
        SWANLAB_MODE=offline SWANLAB_LOG_DIR="$CFPO_RUN_DIR/swanlog" \
        RAY_USAGE_STATS_ENABLED=0 \
    bash -euo pipefail <<'INNER'
# The known Geo3K path only satisfies the required validation-dataset interface.
# Neither pre-training nor periodic/final validation is executed in this pilot;
# no train/validation separation or validation scores are claimed.
bash examples/cfpo_trainer/run_grpo.sh --check-config > "$CFPO_RUN_DIR/resolved-config.yaml"
echo PILOT_TRAINING_START
bash examples/cfpo_trainer/run_grpo.sh --train
echo PILOT_TRAINING_PROCESS_OK
INNER
