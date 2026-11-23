#!/usr/bin/env bash

set -xeuo pipefail

export PYTHONUNBUFFERED=1

MODEL_PATH=/groups/xitucheng213/home/share/yzy/Pretrained/Qwen3-VL-2B-Thinking
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)

# verl reads these variables when initializing SwanLab. Offline mode writes a
# syncable run directory without starting the deprecated local SQLite backend.
export SWANLAB_LOG_DIR="${REPO_ROOT}/swanlog"
export SWANLAB_MODE=local

TRAIN_CONFIG=${SCRIPT_DIR}/datasets_math_train.json
VAL_CONFIG=${SCRIPT_DIR}/datasets_math_val.json
FORMAT_PROMPT=${SCRIPT_DIR}/format_prompt/math.jinja
REWARD_FUNCTION=${SCRIPT_DIR}/reward_function/math.py

train_files=$(python3 -c 'import json, sys; rows=json.load(open(sys.argv[1], encoding="utf-8")); print("[" + ",".join("{dataset:" + json.dumps(row["dataset"]) + ",image:" + json.dumps(row.get("image")) + "}" for row in rows) + "]")' "${TRAIN_CONFIG}")
val_files=$(python3 -c 'import json, sys; rows=json.load(open(sys.argv[1], encoding="utf-8")); print("[" + ",".join("{dataset:" + json.dumps(row["dataset"]) + ",image:" + json.dumps(row.get("image")) + "}" for row in rows) + "]")' "${VAL_CONFIG}")

cd "${REPO_ROOT}"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files="${train_files}" \
    data.val_files="${val_files}" \
    data.prompt_key=problem \
    data.format_prompt="${FORMAT_PROMPT}" \
    data.seed=42 \
    data.train_batch_size=384 \
    data.val_batch_size=512 \
    data.max_prompt_length=4096 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.filter_overlong_prompts_workers=16 \
    data.truncation=error \
    data.image_patch_size=16 \
    +data.mm_processor_kwargs='{min_pixels:200704,max_pixels:1003520}' \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=24576 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.05 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=0.99 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.max_model_len=6144 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=24576 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.99 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=False \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=24576 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward.custom_reward_function.path="${REWARD_FUNCTION}" \
    reward.custom_reward_function.name=compute_score \
    trainer.val_before_train=True \
    trainer.project_name=Counterfactual-R1 \
    trainer.experiment_name=qwen3_vl_2b_GRPO-math-bs384 \
    trainer.logger='["console","swanlab"]' \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.log_val_generations=30 \
    trainer.total_epochs=2 \
    trainer.test_freq=5 \
    trainer.save_freq=50 \
    trainer.max_actor_ckpt_to_keep=5 \
    trainer.max_critic_ckpt_to_keep=5 \
    trainer.default_local_dir=/groups/xitucheng213/home/share/yzy/Counterfact-Projects/CFPO-Next/checkpoints/qwen3_vl_2b_GRPO-math-bs384 \
    "$@"
