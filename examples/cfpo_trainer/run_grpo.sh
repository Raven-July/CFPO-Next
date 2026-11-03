#!/bin/bash
# Reviewable single-A800 Qwen3-VL-2B-Instruct GRPO pilot.
# --check-config resolves Hydra config only; --train is explicit and not used by preflight.
set -euo pipefail
mode=${1:---check-config}
case "$mode" in
    --check-config|--train) if [ "$#" -gt 0 ]; then shift; fi ;;
    *) echo 'Usage: bash run_grpo.sh [--check-config|--train] [Hydra overrides...]' >&2; exit 2 ;;
esac
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
work=/groups/xitucheng213/home/share/u2025210309/grpo-qwen3vl2b
: "${CFPO_RUN_DIR:?Set CFPO_RUN_DIR outside the repository}"
tf_overlay=${CFPO_TF_OVERLAY:-/groups/xitucheng213/home/share/u2025210309/tf-cc7ab9be-bYDrTa/site-packages}
logging_overlay=${CFPO_LOGGING_OVERLAY:-/groups/xitucheng213/home/share/u2025210309/swanlab-env-ecn63f2h/site-packages}
export PYTHONPATH="$logging_overlay:$tf_overlay:$repo"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 SWANLAB_MODE=offline
export SWANLAB_LOG_DIR="$CFPO_RUN_DIR/swanlog"
model=${MODEL_PATH:-$work/models/Qwen3-VL-2B-Instruct}
train_file=${TRAIN_FILE:-$repo/examples/cfpo_trainer/datasets_math_train.json}
# Geometry3k is only a known path for config inspection, not the team's final validation selection.
if [ "$mode" = --train ]; then
    : "${VAL_FILE:?Set the reviewed, disjoint validation file/manifest before training}"
fi
val_file=${VAL_FILE:-$work/data/geo3k/test.parquet}
checkpoint_contents=${CHECKPOINT_SAVE_CONTENTS:-'["model","extra"]'}
max_response_length=${MAX_RESPONSE_LENGTH:-1024}
max_model_len=$((2048 + max_response_length))
cd "$repo"
if [ "$mode" = --check-config ]; then
    set -- --cfg job --resolve "$@"
fi
exec python3 -u -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files="$train_file" \
    data.val_files="$val_file" \
    data.format_prompt="$repo/examples/cfpo_trainer/format_prompt/base.jinja" \
    data.train_batch_size=4 \
    data.train_max_samples="${TRAIN_MAX_SAMPLES:-128}" \
    data.val_max_samples=16 \
    data.max_prompt_length=2048 \
    data.max_response_length="$max_response_length" \
    data.filter_overlong_prompts=True \
    data.filter_overlong_prompts_workers=1 \
    data.dataloader_num_workers=0 \
    data.truncation=error data.image_key=images data.image_patch_size=16 \
    data.seed=42 \
    actor_rollout_ref.model.path="$model" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.use_fused_kernels=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.checkpoint.save_contents="$checkpoint_contents" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
    actor_rollout_ref.rollout.max_model_len="$max_model_len" \
    actor_rollout_ref.rollout.max_num_batched_tokens="$max_model_len" \
    actor_rollout_ref.rollout.max_num_seqs=8 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.enable_chunked_prefill="${ENABLE_CHUNKED_PREFILL:-True}" \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    reward.num_workers=1 \
    reward.custom_reward_function.path="$repo/examples/cfpo_trainer/reward_function/base.py" \
    reward.custom_reward_function.name=compute_score \
    trainer.use_v1=False \
    trainer.critic_warmup=0 \
    'trainer.logger=["console","swanlab"]' \
    trainer.project_name=bao_cfpo_next_grpo_pilot \
    trainer.experiment_name="$(basename "$CFPO_RUN_DIR")" \
    trainer.n_gpus_per_node=1 trainer.nnodes=1 \
    trainer.total_epochs=1 trainer.total_training_steps="${TOTAL_STEPS:-3}" \
    trainer.val_before_train=False trainer.test_freq=-1 \
    trainer.save_freq="${SAVE_FREQ:-${TOTAL_STEPS:-3}}" trainer.max_actor_ckpt_to_keep=1 \
    trainer.resume_mode=disable \
    trainer.default_local_dir="$CFPO_RUN_DIR/checkpoints" \
    trainer.rollout_data_dir="$CFPO_RUN_DIR/generations" \
    trainer.validation_data_dir="$CFPO_RUN_DIR/validation" \
    trainer.log_val_generations=4 \
    ray_kwargs.ray_init.num_cpus="${RAY_NUM_CPUS:-4}" \
    ray_kwargs.ray_init.runtime_env.py_executable=null \
    ++ray_kwargs.ray_init.include_dashboard=False \
    ++ray_kwargs.ray_init.object_store_memory=536870912 \
    ++ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH="$PYTHONPATH" \
    '++ray_kwargs.ray_init.runtime_env.env_vars.PYTHONNOUSERSITE="1"' \
    ++ray_kwargs.ray_init.runtime_env.env_vars.LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}" \
    '++ray_kwargs.ray_init.runtime_env.env_vars.HF_HUB_OFFLINE="1"' \
    '++ray_kwargs.ray_init.runtime_env.env_vars.HF_DATASETS_OFFLINE="1"' \
    ++ray_kwargs.ray_init.runtime_env.env_vars.SWANLAB_MODE=offline \
    ++ray_kwargs.ray_init.runtime_env.env_vars.SWANLAB_LOG_DIR="$CFPO_RUN_DIR/swanlog" \
    "$@"
