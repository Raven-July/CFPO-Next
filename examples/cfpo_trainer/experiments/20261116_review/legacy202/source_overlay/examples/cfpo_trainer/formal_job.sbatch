#!/bin/bash
#SBATCH --partition=g078t
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --comment=xitucheng213
set -euo pipefail
stage=${1:?Use config, smoke, train, or eval}
: "${CFPO_RUN_DIR:?}"
: "${CFPO_MANIFEST_DIR:?}"
: "${CFPO_DASH_OVERLAY:?}"
repo=${CFPO_REPO:-/groups/xitucheng213/home/share/u2025210309/CFPO-Next}
driver=/opt/app/nvidia/570.195.03/lib
sif=/groups/xitucheng213/home/share/yzy/Counterfact-Projects/verl-vllm018.sif
tf=/groups/xitucheng213/home/share/u2025210309/tf-cc7ab9be-bYDrTa/site-packages
mkdir -p "$CFPO_RUN_DIR"
trap 'rc=$?; printf "%s\n" "$rc" > "$CFPO_RUN_DIR/exit-code.txt"' EXIT
cd "$repo"
git rev-parse HEAD > "$CFPO_RUN_DIR/base-commit.txt"
git diff --binary > "$CFPO_RUN_DIR/changes.patch"
git ls-files --modified --others --exclude-standard -z | tar --null -T - -cf "$CFPO_RUN_DIR/changed-source.tar"
cp "${BASH_SOURCE[0]}" "$CFPO_RUN_DIR/submitted.sbatch"
/opt/app/singularity/bin/singularity exec --cleanenv --nv \
    -B /groups -B "$driver" --pwd "$repo" "$sif" \
    env CFPO_RUN_DIR="$CFPO_RUN_DIR" CFPO_MANIFEST_DIR="$CFPO_MANIFEST_DIR" \
    CFPO_DASH_OVERLAY="$CFPO_DASH_OVERLAY" CFPO_REVIEWED="${CFPO_REVIEWED:-no}" \
    CFPO_BASE_MODEL="${CFPO_BASE_MODEL:-/groups/xitucheng213/home/share/u2025210309/grpo-qwen3vl2b/models/Qwen3-VL-2B-Instruct}" \
    MODEL_PATH="${CFPO_BASE_MODEL:-/groups/xitucheng213/home/share/u2025210309/grpo-qwen3vl2b/models/Qwen3-VL-2B-Instruct}" \
    CFPO_TRAIN_RUN="${CFPO_TRAIN_RUN:-}" CFPO_EVAL_N="${CFPO_EVAL_N:-}" \
    CFPO_AUDIT_SCOPE="${CFPO_AUDIT_SCOPE:-all}" \
    CFPO_MATHVERSE_IMAGES="${CFPO_MATHVERSE_IMAGES:-}" \
    PYTHONPATH="$CFPO_DASH_OVERLAY:$tf:$repo" CFPO_TF_OVERLAY="$tf" \
    PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    LD_LIBRARY_PATH="$driver" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
    HF_HOME=/groups/xitucheng213/home/share/u2025210309/grpo-qwen3vl2b/hf-cache \
    RAY_USAGE_STATS_ENABLED=0 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 \
    bash -euo pipefail -c '
case "$1" in
 config)
  bash examples/cfpo_trainer/run_formal.sh --check-config > "$CFPO_RUN_DIR/resolved-config.yaml"
  python3 examples/cfpo_trainer/formal_contract.py --run "$CFPO_RUN_DIR"
  extra=()
  if [ -n "$CFPO_MATHVERSE_IMAGES" ]; then extra+=(--mathverse-images "$CFPO_MATHVERSE_IMAGES"); fi
  python3 examples/cfpo_trainer/formal_tools.py prepare --output "$CFPO_MANIFEST_DIR" --verify-images --scope "$CFPO_AUDIT_SCOPE" "${extra[@]}"
  python3 examples/cfpo_trainer/formal_tools.py check-length --run "$CFPO_RUN_DIR"
  python3 examples/cfpo_trainer/test_reference_optimizer.py
  python3 examples/cfpo_trainer/formal_tools.py check-board --run "$CFPO_RUN_DIR"
  ;;
 smoke)
  bash examples/cfpo_trainer/run_formal.sh --check-config trainer.total_training_steps=1 data.val_max_samples=16 trainer.test_freq=1 > "$CFPO_RUN_DIR/resolved-config.yaml"
  bash examples/cfpo_trainer/run_formal.sh --train trainer.total_training_steps=1 data.val_max_samples=16 trainer.test_freq=1
  python3 examples/cfpo_trainer/formal_tools.py check-training --run "$CFPO_RUN_DIR" --step 1
  ;;
 train)
  bash examples/cfpo_trainer/run_formal.sh --check-config > "$CFPO_RUN_DIR/resolved-config.yaml"
  python3 examples/cfpo_trainer/formal_contract.py --run "$CFPO_RUN_DIR"
  python3 examples/cfpo_trainer/formal_tools.py check-length --run "$CFPO_RUN_DIR"
  bash examples/cfpo_trainer/run_formal.sh --train
  python3 examples/cfpo_trainer/formal_tools.py check-training --run "$CFPO_RUN_DIR" --step 202
  ;;
 eval)
  test "${CFPO_REVIEWED:-}" = yes
  test -n "$CFPO_TRAIN_RUN"
  test -n "$CFPO_EVAL_N"
  actor="$CFPO_TRAIN_RUN/checkpoints/global_step_202/actor"
  test -s "$actor/model_world_size_2_rank_0.pt"
  test -s "$actor/model_world_size_2_rank_1.pt"
  python3 -m verl.model_merger merge --backend fsdp --local_dir "$actor" --target_dir "$CFPO_RUN_DIR/merged-step202"
  python3 examples/cfpo_trainer/evaluate_full.py --manifest "$CFPO_MANIFEST_DIR/test.json" --model "$CFPO_BASE_MODEL" --output "$CFPO_RUN_DIR/before" --n "$CFPO_EVAL_N"
  python3 examples/cfpo_trainer/evaluate_full.py --manifest "$CFPO_MANIFEST_DIR/test.json" --model "$CFPO_RUN_DIR/merged-step202" --output "$CFPO_RUN_DIR/after" --n "$CFPO_EVAL_N"
  python3 examples/cfpo_trainer/formal_tools.py summarize --run "$CFPO_RUN_DIR"
  ;;
 *) exit 2 ;;
esac
' bash "$stage"
