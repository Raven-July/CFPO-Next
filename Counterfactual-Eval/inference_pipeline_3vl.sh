#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT=/groups/xitucheng213/home/share/yzy/Counterfact-Projects
REPO_ROOT=${PROJECT_ROOT}/CFPO-Next
EVAL_ROOT=${REPO_ROOT}/Counterfactual-Eval

NAME=qwen3_vl_2b_GRPO-math-bs384
MODEL_PATH=${REPO_ROOT}/checkpoints/qwen3_vl_2b_GRPO-math-bs384/global_step_202/actor
CONFIG_PATH=${EVAL_ROOT}/datasets_math_test.json
OUTPUT_DIR=${EVAL_ROOT}/results

cd "${REPO_ROOT}"

echo "Evaluating ${NAME} from ${MODEL_PATH}"
exec python3 "${EVAL_ROOT}/inference_eval_3vl.py" \
    --name "${NAME}" \
    --model_path "${MODEL_PATH}" \
    --config_path "${CONFIG_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --n 8 \
    --batch_size 512 \
    --num_workers 16 \
    --tensor_parallel_size 2 \
    --max_tokens 2048 \
    --max_model_len 6144 \
    --gpu_memory_utilization 0.6
