#!/bin/bash
#SBATCH --job-name=qwen3-vl-2b-grpo
#SBATCH --partition=g078t
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --time=1-00:00:00
#SBATCH --comment=xitucheng213
#SBATCH --output=slurm/slurm-%j.out
#SBATCH --error=slurm/slurm-%j.err

set -euo pipefail

PROJECT_ROOT=/groups/xitucheng213/home/share/yzy/Counterfact-Projects
REPO_ROOT=${PROJECT_ROOT}/CFPO-Next
IMAGE=${PROJECT_ROOT}/verl-vllm018.sif
SINGULARITY=/opt/app/singularity/bin/singularity
NVIDIA_LIB=/opt/app/nvidia/570.195.03/lib
VERL_PACKAGES=/groups/xitucheng213/home/share/yzy/singularity-envs/verl-vllm018/packages

echo "========== Job Info =========="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "=============================="

"${SINGULARITY}" exec --nv \
    -B /groups \
    -B "${NVIDIA_LIB}" \
    "${IMAGE}" \
    bash -c '
        set -euo pipefail

        repo_root=$1
        nvidia_lib=$2
        verl_packages=$3

        export LD_LIBRARY_PATH="${nvidia_lib}:${LD_LIBRARY_PATH:-}"
        export PYTHONNOUSERSITE=1
        export PYTHONPATH="${repo_root}:${verl_packages}"

        # Singularity --nv on this cluster also injects the ROCm selector. verl
        # rejects simultaneous CUDA and ROCm selectors, so retain only CUDA on
        # these NVIDIA nodes.
        unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES

        # The login shell proxy points at an SSH RemoteForward and disappears
        # when that SSH/VS Code session closes. This job is fully offline, so do
        # not inherit a stale localhost proxy into Ray or vLLM workers.
        unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY

        cd "${repo_root}"
        nvidia-smi
        python3 -m pip show verl
        echo "Runtime imports ready; verl loaded from:"
        python3 -c "import swanlab, torch, verl, vllm; print(verl.__file__)"
        exec bash examples/cfpo_trainer/qwen3_vl_2b_GRPO-math.sh
    ' bash "${REPO_ROOT}" "${NVIDIA_LIB}" "${VERL_PACKAGES}"

echo "Job finished: $(date)"
