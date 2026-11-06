#!/bin/bash
#SBATCH --job-name=cfpo-grpo
#SBATCH --partition=g078t
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --comment=xitucheng213
#SBATCH --output=/groups/xitucheng213/home/share/u2024213357/GRPO/logs/cfpo-%j.out

set -euo pipefail
export GRPO_ROOT=${GRPO_ROOT:-/groups/xitucheng213/home/share/u2024213357/GRPO}
export CFPO_REPO=${CFPO_REPO:-$GRPO_ROOT/cfpo-next}
export CFPO_SIF=${CFPO_SIF:-/groups/xitucheng213/home/share/yzy/Counterfact-Projects/verl-vllm018.sif}
export NVIDIA_LIB=${NVIDIA_LIB:-/opt/app/nvidia/570.195.03/lib}
export RUN_MODE=${RUN_MODE:-smoke}
export CFPO_ALGORITHM=${CFPO_ALGORITHM:-grpo}
# Keep this stable when resubmitting a full run to resume its checkpoints.
if [[ "$RUN_MODE" == full ]]; then
  export EXPERIMENT_NAME=${EXPERIMENT_NAME:-cfpo-${CFPO_ALGORITHM}-full-4gpu-bs384-${SLURM_JOB_ID}}
  if [[ "$EXPERIMENT_NAME" != *-4gpu-bs384* ]]; then
    echo 'Use a new full experiment name containing -4gpu-bs384 to avoid legacy checkpoints' >&2
    exit 1
  fi
elif [[ "$CFPO_ALGORITHM" == dapo ]]; then
  export EXPERIMENT_NAME=${EXPERIMENT_NAME:-cfpo-dapo-${RUN_MODE}-${SLURM_JOB_ID}}
else
  export EXPERIMENT_NAME=${EXPERIMENT_NAME:-cfpo-${RUN_MODE}-${SLURM_JOB_ID}}
fi
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
export ROLLOUT_TP=1 ROLLOUT_N=4
export MAX_PROMPT_LENGTH=4096
export SWANLAB_LOG_DIR="$GRPO_ROOT/swanlog/$EXPERIMENT_NAME"
export GENERATIONS_DIR="$GRPO_ROOT/generations/$EXPERIMENT_NAME"
export RAY_TMPDIR="/tmp/cfpo-${SLURM_JOB_ID}"
cd "$CFPO_REPO"
export NDEVICES_PER_NODE=1
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
    export NDEVICES_PER_NODE=4 TRAIN_BATCH_SIZE=384 PPO_MINI_BATCH_SIZE=128
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
case "$CFPO_ALGORITHM" in
  grpo) ;;
  dapo)
    if [[ "$EXPERIMENT_NAME" != *dapo* ]]; then
      echo 'Use a separate DAPO experiment name containing dapo' >&2
      exit 1
    fi
    # V1 implements DAPO through normalized group advantages and online group refill.
    EXTRA+=(algorithm.norm_adv_by_std_in_grpo=True
      algorithm.filter_groups.enable=True algorithm.filter_groups.metric=accuracy
      actor_rollout_ref.actor.clip_ratio_low=0.2 actor_rollout_ref.actor.clip_ratio_high=0.28
      actor_rollout_ref.actor.loss_agg_mode=token-mean
      actor_rollout_ref.actor.use_kl_loss=False actor_rollout_ref.actor.kl_loss_coef=0.0)
    ;;
  *) echo "Unknown CFPO_ALGORITHM=$CFPO_ALGORITHM" >&2; exit 1 ;;
esac
python -u -c '
import os
import swanlab, torch, verl
expected = int(os.environ["NDEVICES_PER_NODE"])
visible = torch.cuda.device_count()
if visible != expected:
    raise SystemExit(f"Expected {expected} visible GPUs, got {visible}. Submit full mode with sbatch --gres=gpu:4.")
print("SwanLab ready", swanlab.__version__)
print("Visible GPUs:", [torch.cuda.get_device_name(i) for i in range(visible)])
print(verl.__file__)
'
bash examples/cfpo_trainer/run_qwen3_vl_2b_fsdp.sh \
  "++ray_kwargs.ray_init.num_cpus=${SLURM_CPUS_PER_TASK:-8}" \
  ++ray_kwargs.ray_init.object_store_memory=2147483648 \
  data.filter_overlong_prompts_workers=2 \
  data.val_batch_size=16 \
  actor_rollout_ref.rollout.max_model_len=8192 \
  actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=256 \
  "trainer.default_local_dir=\"$GRPO_ROOT/checkpoints/$EXPERIMENT_NAME\"" \
  "${EXTRA[@]}"
INNER
