#!/bin/bash
# User-run stages only: no command here runs unless explicitly invoked.
set -euo pipefail
PACK=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
stage=${1:-help}
BASE=/groups/xitucheng213/home/share/u2025210309
SIF=/groups/xitucheng213/home/share/yzy/Counterfact-Projects/verl-vllm018.sif
SINGULARITY=/opt/app/singularity/bin/singularity
if [ "$stage" = init ]; then
    test ! -e "$PACK/session-path.txt" || { echo 'Session exists. Use status; do not initialize again.' >&2; exit 2; }
    SESSION=$(mktemp -d "$BASE/cfpo-formal-20261108-XXXXXX")
    git clone -q "$PACK/base.bundle" "$SESSION/repo"
    tar -xf "$PACK/changes.tar" -C "$SESSION/repo"
    python3 - "$PACK/sources.json" "$SESSION/repo" <<'PY'
import hashlib,json,sys
from pathlib import Path
for row in json.loads(Path(sys.argv[1]).read_text())['files']:
    assert hashlib.sha256((Path(sys.argv[2])/row['path']).read_bytes()).hexdigest()==row['sha256'],row['path']
print('TRANSFERRED_SOURCE_HASHES_OK')
PY
    CFPO_REPO="$SESSION/repo"
    CFPO_MANIFEST_DIR="$SESSION/manifests"
    CFPO_DASH_OVERLAY="$SESSION/dashboard-site"
    for name in SESSION CFPO_REPO CFPO_MANIFEST_DIR CFPO_DASH_OVERLAY; do
        printf '%s=%q\n' "$name" "${!name}"
    done > "$SESSION/session.env"
    printf '%s\n' "$SESSION" > "$PACK/session-path.txt"
    echo "SESSION=$SESSION"
    exit
fi
if [ "$stage" = help ]; then
    echo 'Stages: init, deps, config, inspect, export-view, watch-check, approve, smoke, watch-train, train, status, eval 1|8, results'
    exit
fi
SESSION=$(cat "$PACK/session-path.txt")
source "$SESSION/session.env"
export CFPO_REPO CFPO_MANIFEST_DIR CFPO_DASH_OVERLAY
export CFPO_MATHVERSE_IMAGES
complete() {
    test -f "$SESSION/$1/exit-code.txt" && test "$(cat "$SESSION/$1/exit-code.txt")" = 0 || {
        echo "$1 has not completed successfully. Inspect its job and train.out/err." >&2; exit 2;
    }
}
reviewed() {
    test -f "$SESSION/review-approved.sha256" || { echo 'Run approve only after supervisor review and matching GPU authorization.' >&2; exit 2; }
    (cd "$SESSION"; sha256sum -c review-approved.sha256)
    export CFPO_REVIEWED=yes
}
submit() {
    local kind=$1 wall=$2 gpus=$3
    local memory=128G
    if [ "$kind" = train ]; then memory=256G; fi
    local folder="$SESSION/$kind"
    test ! -e "$folder" || { echo "Already exists: $folder. Do not submit duplicate jobs." >&2; exit 2; }
    mkdir "$folder"
    printf 'stage=%s\nrequested_memory=%s\nrequested_gpus=%s\nrequested_walltime=%s\n' \
        "$kind" "$memory" "$gpus" "$wall" > "$folder/submission-resources.txt"
    export CFPO_RUN_DIR="$folder"
    local job
    job=$(sbatch --parsable --job-name="bao-formal-$kind" --gres="gpu:$gpus" \
        --mem="$memory" \
        --time="$wall" --output="$folder/train.out" --error="$folder/train.err" \
        --export=ALL "$CFPO_REPO/examples/cfpo_trainer/formal_job.sbatch" "$kind")
    printf '%s\n' "${job%%;*}" > "$folder/job-id.txt"
    echo "$kind JOB=${job%%;*} MEMORY=$memory RUN=$folder"
}
case "$stage" in
 deps)
    test "${CFPO_SETUP_GPU:-}" = yes || { echo 'This cluster requires a GPU allocation for setup. Invoke with CFPO_SETUP_GPU=yes only when authorizing 1 GPU for at most 15 minutes; no training.' >&2; exit 2; }
    test -d "$PACK/wheels"
    test -f "$PACK/dashboard-constraints.txt"
    constraint_hash=$(sha256sum "$PACK/dashboard-constraints.txt" | cut -d' ' -f1)
    if [ -f "$SESSION/deps-constraints.sha256" ] && [ "$(cat "$SESSION/deps-constraints.sha256")" = "$constraint_hash" ]; then
        echo 'Compatible dependency directory already installed; no allocation needed.'
        exit 0
    fi
    # Preserve the existing overlay. Switch the session only after install/import checks pass.
    install_target=$(mktemp -d "$SESSION/dashboard-site-compat-XXXXXX")
    allocation=()
    if [ -z "${SLURM_JOB_ID:-}" ]; then
        allocation=(srun --partition=g078t --nodes=1 --ntasks=1 --cpus-per-task=1
            --mem=4G --gres=gpu:1 --time=00:15:00 --comment=xitucheng213 --job-name=bao-cfpo-deps)
    fi
    "${allocation[@]}" "$SINGULARITY" exec --cleanenv -B /groups -B "$PACK" "$SIF" \
        env PYTHONNOUSERSITE=1 CFPO_PACK="$PACK" CFPO_INSTALL_TARGET="$install_target" \
        bash -euc '
          python3 -m pip install --no-index --ignore-installed \
            --find-links "$CFPO_PACK/wheels" --constraint "$CFPO_PACK/dashboard-constraints.txt" \
            --target "$CFPO_INSTALL_TARGET" "swanlab[dashboard]==0.10.3" "opentelemetry-api==1.40.0"
          PYTHONPATH="$CFPO_INSTALL_TARGET" python3 "$CFPO_PACK/verify_dashboard_runtime.py" "$CFPO_INSTALL_TARGET/compatibility-check.json"
        '
    CFPO_DASH_OVERLAY="$install_target"
    env_temp=$(mktemp "$SESSION/session.env.XXXXXX")
    for name in SESSION CFPO_REPO CFPO_MANIFEST_DIR CFPO_DASH_OVERLAY; do
        printf '%s=%q\n' "$name" "${!name}"
    done > "$env_temp"
    mv "$env_temp" "$SESSION/session.env"
    printf '%s\n' "$constraint_hash" > "$SESSION/deps-constraints.sha256"
    touch "$SESSION/deps-complete"
    echo 'DEPENDENCIES_INSTALLED: no training was started'
    ;;
 config)
    test "${CFPO_PREFLIGHT_GPU:-}" = yes || { echo 'Invoke with CFPO_PREFLIGHT_GPU=yes only after authorizing the 1-GPU configuration/data check and its walltime.' >&2; exit 2; }
    test -f "$SESSION/deps-complete"
    test -f "$SESSION/deps-constraints.sha256"
    test "$(cat "$SESSION/deps-constraints.sha256")" = "$(sha256sum "$PACK/dashboard-constraints.txt" | cut -d' ' -f1)"
    test -d "$CFPO_DASH_OVERLAY"
    submit config "${CONFIG_WALLTIME:-04:00:00}" 1
    ;;
 inspect)
    complete config
    cat "$SESSION/config/length-audit.json"
    cat "$CFPO_MANIFEST_DIR/data-audit.json"
    tar -czf "$PACK/review-materials.tar.gz" -C "$SESSION" \
        config/resolved-config.yaml config/length-audit.json manifests \
        repo/examples/cfpo_trainer/run_formal.sh repo/examples/cfpo_trainer/formal_job.sbatch \
        repo/examples/cfpo_trainer/formal_metrics.py repo/examples/cfpo_trainer/FORMAL_REVIEW.md \
        repo/examples/cfpo_trainer/formal_contract.py repo/examples/cfpo_trainer/reference_optimizer.py \
        repo/examples/cfpo_trainer/reference-optimizer-provenance.json repo/examples/cfpo_trainer/reference-inputs.json \
        repo/examples/cfpo_trainer/format_prompt/reference_base.jinja config/formal-config-check.json \
        config/changes.patch
    echo "REVIEW_PACKAGE=$PACK/review-materials.tar.gz"
    ;;
 export-view)
    if [ -d "$SESSION/train/swanlog" ]; then logs="$SESSION/train/swanlog"
    elif [ -d "$SESSION/smoke/swanlog" ]; then logs="$SESSION/smoke/swanlog"
    else logs="$SESSION/config/board-smoke"; fi
    test -d "$logs"
    tar -czf "$PACK/native-view-export.tar.gz" -C "$logs" .
    echo "NATIVE_LOG_SNAPSHOT=$PACK/native-view-export.tar.gz"
    echo "SOURCE=$logs"
    echo 'A local copy is historical log review, not a live monitor on the separate 4x4090 host.'
    ;;
 watch-check|watch-train)
    if [ "$(hostname)" = workstation ]; then
        echo 'Container startup on workstation is known to fail. Prepare a separate native SwanLab viewing environment; do not allocate a GPU just to keep a dashboard running.' >&2
        exit 2
    fi
    if [ "$stage" = watch-check ]; then
        complete config; logs="$SESSION/config/board-smoke"
    elif [ -d "$SESSION/train/swanlog" ]; then logs="$SESSION/train/swanlog"
    else logs="$SESSION/smoke/swanlog"; fi
    test -d "$logs"
    "$SINGULARITY" exec --cleanenv -B /groups "$SIF" \
        env PYTHONPATH="$CFPO_DASH_OVERLAY" PYTHONNOUSERSITE=1 python3 -m swanlab watch "$logs"
    ;;
 approve)
    complete config
    # This records Bao's acknowledgement; it does not obtain supervisor approval.
    (cd "$SESSION"; find repo/examples/cfpo_trainer -maxdepth 1 -type f \( -name '*.py' -o -name '*.sh' -o -name '*.sbatch' \) -print0 |
        sort -z | xargs -0 sha256sum
        sha256sum repo/verl/trainer/ppo/ray_trainer.py config/resolved-config.yaml manifests/*.json) > "$SESSION/review-approved.sha256"
    (cd "$SESSION"; sha256sum repo/examples/cfpo_trainer/reference-inputs.json \
        repo/examples/cfpo_trainer/reference-optimizer-provenance.json \
        repo/examples/cfpo_trainer/format_prompt/reference_base.jinja \
        repo/examples/cfpo_trainer/reward_function/base.py \
        repo/verl/utils/dataset/cfpo.py repo/verl/utils/dataset/rl_dataset.py) >> "$SESSION/review-approved.sha256"
    echo 'Recorded review acknowledgement. Invoke only after actual script approval, native board verification and GPU resource authorization.'
    ;;
 smoke)
    reviewed; submit smoke "${SMOKE_WALLTIME:-04:00:00}" 2
    ;;
 train)
    reviewed; complete smoke
    submit train "${TRAIN_WALLTIME:?Set TRAIN_WALLTIME within queue limit and authorized budget}" 2
    ;;
 eval)
    reviewed; complete train
    case "${2:-8}" in 8) export CFPO_EVAL_N=8 ;; *) echo 'Formal evaluation requires n=8 and temperature=1.0.'; exit 2 ;; esac
    export CFPO_TRAIN_RUN="$SESSION/train"
    submit eval "${EVAL_WALLTIME:?Set the authorized evaluation walltime}" 2
    ;;
 status)
    if [ -f "$SESSION/prep-batch/job-id.txt" ]; then
        echo "BACKGROUND PREPARATION RUN=$SESSION/prep-batch"
        sacct -j "$(cat "$SESSION/prep-batch/job-id.txt")" --format=JobID,State,Elapsed,ExitCode -X
        tail -n 15 "$SESSION/prep-batch/prepare.out" 2>/dev/null || true
        tail -n 10 "$SESSION/prep-batch/prepare.err" 2>/dev/null || true
    fi
    for kind in config smoke train eval; do
        folder="$SESSION/$kind"
        if [ -f "$folder/job-id.txt" ]; then
            echo "$kind RUN=$folder"
            sacct -j "$(cat "$folder/job-id.txt")" --format=JobID,State,Elapsed,ExitCode -X
            tail -n 12 "$folder/train.out" 2>/dev/null || true
            tail -n 8 "$folder/train.err" 2>/dev/null || true
        fi
    done
    ;;
 results)
    complete eval
    cat "$SESSION/eval/results-table.md"
    cp "$SESSION/eval/results-table.md" "$PACK/results-table.md"
    tar -czf "$PACK/result-summaries.tar.gz" -C "$SESSION" \
        eval/results-table.md eval/before/summary.json eval/after/summary.json \
        train/length-audit.json train/resolved-config.yaml
    echo "RESULTS_PACKAGE=$PACK/result-summaries.tar.gz"
    ;;
 *) echo 'Unknown stage' >&2; exit 2 ;;
esac
