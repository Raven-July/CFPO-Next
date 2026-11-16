#!/bin/bash
# Replace only the known pending setup job; never cancel a running job.
set -euo pipefail
PACK=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SESSION=$(cat "$PACK/session-path.txt")
old=${1:-221903}
case "$old" in ''|*[!0-9]*) echo 'Invalid prior job ID' >&2; exit 2 ;; esac
folder="$SESSION/prep-batch"
if [ -f "$folder/job-id.txt" ]; then
    echo "Already submitted: $(cat "$folder/job-id.txt"). Inspect it; do not duplicate."
    exit 0
fi
test ! -e "$SESSION/config" || { echo 'Config directory exists; inspect before scheduling replacement.' >&2; exit 2; }
test -f "$PACK/dashboard-constraints.txt"
test -f "$PACK/verify_dashboard_runtime.py"
test -f "$PACK/check_runtime_and_board.py"
old_info=$(squeue -h -j "$old" -o '%u|%T|%j' 2>/dev/null || true)
if [ -n "$old_info" ]; then
    IFS='|' read -r owner state name <<< "$old_info"
    test "$owner" = "$(id -un)" && test "$name" = bao-cfpo-deps || { echo 'Prior job identity mismatch; refusing replacement.' >&2; exit 2; }
    test "$state" = PENDING || { echo "Prior job is $state; leave it running. No replacement submitted." >&2; exit 2; }
fi
mkdir -p "$folder"
# Hold new job until we have safely resolved the old pending request.
job=$(sbatch --hold --parsable --output="$folder/prepare.out" --error="$folder/prepare.err" \
    "$PACK/prepare_checks.sbatch" "$PACK")
job=${job%%;*}
printf '%s\n' "$job" > "$folder/job-id.txt"
if [ -n "$old_info" ]; then
    if ! scancel --state=PENDING --user="$(id -un)" "$old"; then
        echo "Could not cancel prior pending request. New job $job remains HELD; inspect before release." >&2; exit 2
    fi
    remaining=$(squeue -h -j "$old" -o '%T' 2>/dev/null || true)
    if [ -n "$remaining" ]; then
        echo "Old job is now $remaining. New job $job remains HELD; no duplicate allocation started." >&2; exit 2
    fi
fi
scontrol release "$job"
echo "BACKGROUND_PREPARATION_JOB=$job"
echo "LOG=$folder/prepare.out"
echo '1 GPU, 8 CPUs, 128 GiB, at most 4 hours. Dependencies/data/config/imports/board backend only; no model training.'
echo 'You may disconnect after this successful submission. No later smoke training, 202-step training or evaluation is auto-submitted.'
squeue -j "$job" -o '%.18i %.2t %.10M %.30R'
