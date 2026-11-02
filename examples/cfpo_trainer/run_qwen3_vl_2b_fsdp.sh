#!/usr/bin/env bash
# CFPO GRPO baseline. Run inside the training environment from any directory.
set -euo pipefail
REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO_ROOT"

: "${TRAIN_FILE:?Set TRAIN_FILE to the CFPO training descriptor JSON}"
: "${TEST_FILE:?Set TEST_FILE to the CFPO held-out validation descriptor JSON}"
export TRAIN_FILE TEST_FILE
export MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-VL-2B-Instruct}
export PROJECT_NAME=${PROJECT_NAME:-cfpo_grpo_baseline}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_vl_2b_instruct_grpo_$(date +%Y%m%d_%H%M%S)}
export NDEVICES_PER_NODE=${NDEVICES_PER_NODE:-1}
export ROLLOUT_TP=${ROLLOUT_TP:-1}
export ROLLOUT_N=${ROLLOUT_N:-5}
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-384}
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-128}
export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-4096}
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-2048}
export TOTAL_EPOCHS=${TOTAL_EPOCHS:-2}
export SAVE_FREQ=${SAVE_FREQ:-50}
export TEST_FREQ=${TEST_FREQ:-5}
export SWANLAB_MODE=offline
export SWANLAB_LOG_DIR=${SWANLAB_LOG_DIR:-$REPO_ROOT/swanlog}
GENERATIONS_DIR=${GENERATIONS_DIR:-$REPO_ROOT/generations/$EXPERIMENT_NAME}

bash examples/grpo_trainer/run_qwen3_vl_8b_fsdp.sh \
    transfer_queue.backend.SimpleStorage.num_data_storage_units=1 \
    "data.system_prompt_path=\"$REPO_ROOT/examples/cfpo_trainer/system_prompt.txt\"" \
    data.dataloader_num_workers=0 \
    data.seed=42 \
    "reward.custom_reward_function.path=\"$REPO_ROOT/examples/cfpo_trainer/reward.py\"" \
    reward.custom_reward_function.name=compute_score \
    trainer.logger='["console","swanlab"]' \
    trainer.val_before_train=True \
    trainer.log_val_generations=30 \
    "trainer.rollout_data_dir=\"$GENERATIONS_DIR/train\"" \
    "trainer.validation_data_dir=\"$GENERATIONS_DIR/validation\"" \
    actor_rollout_ref.model.use_fused_kernels=False \
    "$@"
