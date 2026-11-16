#!/bin/bash
#SBATCH --job-name=bao-cfpo-prepare
#SBATCH --partition=g078t
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --comment=xitucheng213
set -euo pipefail
PACK=${1:?Missing command pack path}
SESSION=${2:-$(cat "$PACK/session-path.txt")}
test "$(cat "$PACK/session-path.txt")" = "$SESSION" || { echo 'Active session mismatch' >&2; exit 2; }
trap 'rc=$?; printf "%s\n" "$rc" > "$SESSION/prep-batch/exit-code.txt"' EXIT
echo "PREP_JOB=$SLURM_JOB_ID HOST=$(hostname) START=$(date -Is)"
echo 'STAGE=dependencies; no training'
CFPO_SETUP_GPU=yes bash "$PACK/remote_steps.sh" deps
source "$SESSION/session.env"
export CFPO_REPO CFPO_MANIFEST_DIR CFPO_DASH_OVERLAY
export CFPO_MATHVERSE_IMAGES
echo 'STAGE=data_config_and_board_write; no training'
export CFPO_RUN_DIR="$SESSION/config"
export CFPO_AUDIT_SCOPE=training
test ! -e "$CFPO_RUN_DIR" || { echo 'Config directory already exists; inspect previous output instead of overwriting.' >&2; exit 2; }
mkdir "$CFPO_RUN_DIR"
bash "$CFPO_REPO/examples/cfpo_trainer/formal_job.sbatch" config
echo 'STAGE=runtime_imports_and_board_http; no training'
tf=/groups/xitucheng213/home/share/u2025210309/tf-cc7ab9be-bYDrTa/site-packages
sif=/groups/xitucheng213/home/share/yzy/Counterfact-Projects/verl-vllm018.sif
driver=/opt/app/nvidia/570.195.03/lib
/opt/app/singularity/bin/singularity exec --cleanenv --nv -B /groups -B "$PACK" -B "$driver" "$sif" \
    env PYTHONPATH="$CFPO_DASH_OVERLAY:$tf:$CFPO_REPO" PYTHONNOUSERSITE=1 \
    LD_LIBRARY_PATH="$driver" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
    python3 "$PACK/check_runtime_and_board.py" --run "$CFPO_RUN_DIR"
bash "$PACK/remote_steps.sh" inspect
echo 'PREPARATION_COMPLETE: ready for supervisor script review and native browser inspection; no smoke training or 202-step training submitted.'
