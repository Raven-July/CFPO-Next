#!/bin/bash
# Candidate GRPO migration for supervisor review. No implicit GPU submission.
set -euo pipefail
mode=${1:---check-config}
case "$mode" in --check-config|--train) shift ;; *) exit 2 ;; esac
: "${CFPO_RUN_DIR:?Set a fresh run directory}"
: "${CFPO_MANIFEST_DIR:?Run formal_tools.py prepare first}"
: "${CFPO_DASH_OVERLAY:?Set the private dashboard overlay directory}"
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export TRAIN_FILE="$CFPO_MANIFEST_DIR/train.json" VAL_FILE="$CFPO_MANIFEST_DIR/val.json"
export CFPO_LOGGING_OVERLAY="$CFPO_DASH_OVERLAY" CFPO_SWANLAB_MODE=local
export TRAIN_MAX_SAMPLES=-1 TOTAL_STEPS=202 MAX_RESPONSE_LENGTH=2048
export RAY_NUM_CPUS=8
export CHECKPOINT_SAVE_CONTENTS='["model","optimizer","extra"]'
if [ "$mode" = --train ]; then
    test "${CFPO_REVIEWED:-}" = yes || { echo 'Supervisor review and GPU authorization required' >&2; exit 2; }
fi
exec bash "$repo/examples/cfpo_trainer/run_grpo.sh" "$mode" \
    algorithm.norm_adv_by_std_in_grpo=True \
    data.train_batch_size=384 data.train_max_samples=-1 data.val_max_samples=-1 \
    data.val_batch_size=512 data.max_prompt_length=4096 data.seed=1 \
    data.format_prompt="$repo/examples/cfpo_trainer/format_prompt/reference_base.jinja" \
    '++data.mm_processor_kwargs={min_pixels:200704,max_pixels:1003520}' \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 actor_rollout_ref.actor.clip_ratio_high=0.3 \
    actor_rollout_ref.actor.clip_ratio_c=3.0 actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.optim.optimizer=AnyPrecisionAdamW \
    actor_rollout_ref.actor.optim.optimizer_impl=examples.cfpo_trainer.reference_optimizer \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.05 \
    actor_rollout_ref.actor.optim.weight_decay=0.01 'actor_rollout_ref.actor.optim.betas=[0.9,0.999]' \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.ref.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.n=5 actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=0.99 \
    actor_rollout_ref.rollout.max_model_len=6144 \
    actor_rollout_ref.rollout.max_num_batched_tokens=6144 \
    actor_rollout_ref.rollout.max_num_seqs=64 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.0 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=False \
    trainer.n_gpus_per_node=2 trainer.total_epochs=2 \
    trainer.total_training_steps=202 trainer.val_before_train=True trainer.test_freq=5 \
    trainer.log_val_generations=30 trainer.save_freq=50 trainer.max_actor_ckpt_to_keep=5 \
    trainer.project_name=bao_cfpo_formal trainer.experiment_name="$(basename "$CFPO_RUN_DIR")" \
    +trainer.cfpo_component_metrics=True "$@"
