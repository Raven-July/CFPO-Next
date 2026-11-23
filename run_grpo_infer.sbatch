#!/bin/bash
#SBATCH --job-name=run-grpo-infer
#SBATCH --partition=g078t
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --comment=xitucheng213
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -e

# 1. 项目目录和 SIF 镜像
PROJECT=/groups/xitucheng213/home/share/yzy/Counterfact-Projects
IMAGE=$PROJECT/verl-vllm018.sif

echo "========== Job Info =========="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "=============================="

# 2. 在分配的 GPU 节点运行 Singularity
/opt/app/singularity/bin/singularity exec --nv \
    -B /groups \
    -B /opt/app/nvidia/570.195.03/lib \
    "$IMAGE" \
    bash -c '
        set -e

        # 3. 设置 NVIDIA 驱动库
        export LD_LIBRARY_PATH=/opt/app/nvidia/570.195.03/lib:$LD_LIBRARY_PATH

        # 3. 设置 Python 扩展包并检查 verl 环境
        export PYTHONNOUSERSITE=1
        export PYTHONPATH=/groups/xitucheng213/home/share/yzy/singularity-envs/verl-vllm018/packages
        pip show verl

        # 5. 进入项目目录
        cd /groups/xitucheng213/home/share/yzy/Counterfact-Projects

        # 6. 验证 GPU
        nvidia-smi

        # 7. 执行实际推理脚本
        sh Counterfactual-R1/Counterfactual-Eval/inference_pipeline_math_grpo.sh
    '

echo "Job finished: $(date)"
