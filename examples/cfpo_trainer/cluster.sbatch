#!/bin/bash
#SBATCH --job-name=cfpo-grpo
#SBATCH --partition=g078t
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --comment=xitucheng213
#SBATCH --output=/groups/xitucheng213/home/share/u2024213357/GRPO/logs/cfpo-%j.out

set -euo pipefail
export GRPO_ROOT=${GRPO_ROOT:-/groups/xitucheng213/home/share/u2024213357/GRPO}
export CFPO_REPO=${CFPO_REPO:-$GRPO_ROOT/cfpo-next}
export CFPO_SIF=${CFPO_SIF:-/groups/xitucheng213/home/share/yzy/Counterfact-Projects/verl-vllm018.sif}
export NVIDIA_LIB=${NVIDIA_LIB:-/opt/app/nvidia/570.195.03/lib}
export RUN_MODE=${RUN_MODE:-smoke}
# Keep this stable when resubmitting a full run to resume its checkpoints.
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-cfpo-${RUN_MODE}-${SLURM_JOB_ID}}
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES
/opt/app/singularity/bin/singularity exec --nv -B /groups -B "$NVIDIA_LIB" "$CFPO_SIF" bash <<'INNER'
set -euo pipefail
export LD_LIBRARY_PATH="$NVIDIA_LIB:${LD_LIBRARY_PATH:-}"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$CFPO_REPO:$GRPO_ROOT/python-packages"
export PATH="$GRPO_ROOT/python-packages/bin:$PATH"
export HF_HOME="$GRPO_ROOT/hf-cache"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HYDRA_FULL_ERROR=1 VERL_USE_UV=0 DEVICE=gpu
export MODEL_PATH="$GRPO_ROOT/models/Qwen3-VL-2B-Instruct"
export NDEVICES_PER_NODE=1 ROLLOUT_TP=1 ROLLOUT_N=4
export MAX_PROMPT_LENGTH=4096
export SWANLAB_LOG_DIR="$GRPO_ROOT/swanlog/$EXPERIMENT_NAME"
export GENERATIONS_DIR="$GRPO_ROOT/generations/$EXPERIMENT_NAME"
export RAY_TMPDIR="/tmp/cfpo-${SLURM_JOB_ID}"
cd "$CFPO_REPO"
python -u -c 'import swanlab,torch,verl; print("SwanLab ready", swanlab.__version__); print(torch.cuda.get_device_name(0)); print(verl.__file__)'
case "$RUN_MODE" in
  smoke)
    export TRAIN_FILE="$GRPO_ROOT/data/cfpo/datasets_train.json"
    export TEST_FILE="$GRPO_ROOT/data/cfpo/datasets_val.json"
    export TRAIN_BATCH_SIZE=4 PPO_MINI_BATCH_SIZE=4 MAX_RESPONSE_LENGTH=1024
    EXTRA=(data.train_max_samples=32 data.val_max_samples=4 trainer.total_training_steps=2
      trainer.val_before_train=False trainer.log_val_generations=4 trainer.save_freq=2 trainer.test_freq=2)
    ;;
  full)
    export TRAIN_FILE="$GRPO_ROOT/data/cfpo/datasets_train_full.json"
    export TEST_FILE="$GRPO_ROOT/data/cfpo/datasets_val_full.json"
    export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-16} PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-16}
    export MAX_RESPONSE_LENGTH=2048 TOTAL_EPOCHS=2
    export PPO_MAX_TOKEN_LEN_PER_GPU=8192
    EXTRA=(data.train_max_samples=-1 data.val_max_samples=-1 trainer.val_before_train=True
      trainer.save_freq=20 trainer.test_freq=100 trainer.max_actor_ckpt_to_keep=3)
    ;;
  pilot)
    export TRAIN_FILE="$GRPO_ROOT/data/cfpo/datasets_train.json"
    export TEST_FILE="$GRPO_ROOT/data/cfpo/datasets_val.json"
    export TRAIN_BATCH_SIZE=4 PPO_MINI_BATCH_SIZE=4 MAX_RESPONSE_LENGTH=2048
    export PPO_MAX_TOKEN_LEN_PER_GPU=8192
    EXTRA=(data.train_max_samples=128 data.val_max_samples=20 trainer.total_training_steps=20
      trainer.val_before_train=True trainer.log_val_generations=10 trainer.save_freq=10 trainer.test_freq=10)
    ;;
  *) echo "Unknown RUN_MODE=$RUN_MODE" >&2; exit 1 ;;
esac
bash examples/cfpo_trainer/run_qwen3_vl_2b_fsdp.sh \
  ++ray_kwargs.ray_init.num_cpus=8 \
  ++ray_kwargs.ray_init.object_store_memory=2147483648 \
  data.filter_overlong_prompts_workers=2 \
  actor_rollout_ref.rollout.max_model_len=8192 \
  actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=256 \
  "trainer.default_local_dir=\"$GRPO_ROOT/checkpoints/$EXPERIMENT_NAME\"" \
  "${EXTRA[@]}"
INNER
