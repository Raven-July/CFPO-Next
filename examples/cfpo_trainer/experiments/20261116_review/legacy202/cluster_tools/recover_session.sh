#!/bin/bash
# Fresh private source tree, existing verified dependencies, no shared data edits.
set -euo pipefail
PACK=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BASE=/groups/xitucheng213/home/share/u2025210309
stage=${1:-help}
case "$stage" in
 sync)
    if [ -f "$PACK/recovery-session-path.txt" ]; then
        SESSION=$(cat "$PACK/recovery-session-path.txt")
        test "$(cat "$PACK/session-path.txt")" = "$SESSION" || { echo 'Active session changed; inspect before continuing.' >&2; exit 2; }
        echo "RECOVERY_SESSION_ALREADY_READY=$SESSION"
        exit 0
    fi
    old=$(cat "$PACK/session-path.txt")
    if [ -f "$old/prep-batch/job-id.txt" ]; then
        old_job=$(cat "$old/prep-batch/job-id.txt")
        active=$(squeue -h -j "$old_job" -o '%T' 2>/dev/null || true)
        test -z "$active" || { echo "Old preparation job is still $active; do not duplicate it." >&2; exit 2; }
    fi
    source "$old/session.env"
    test -d "$CFPO_DASH_OVERLAY"
    test -f "$old/deps-constraints.sha256"
    test "$(cat "$old/deps-constraints.sha256")" = "$(sha256sum "$PACK/dashboard-constraints.txt" | cut -d' ' -f1)"
    verified_overlay=$CFPO_DASH_OVERLAY
    SESSION=$(mktemp -d "$BASE/cfpo-formal-20261109-XXXXXX")
    git clone -q "$PACK/base.bundle" "$SESSION/repo"
    python3 - "$PACK" "$SESSION/repo" <<'PY'
import hashlib,json,sys,tarfile,subprocess
from pathlib import Path
pack,repo=map(Path,sys.argv[1:])
manifest=json.loads((pack/'sources.json').read_text())
actual_commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD']).decode().strip()
if actual_commit!=manifest['base_commit']: raise ValueError('Base commit mismatch')
expected=manifest['files']
names={row['path'] for row in expected}
with tarfile.open(str(pack/'changes.tar')) as archive:
    if set(archive.getnames())!=names: raise ValueError('Unexpected source archive entries')
    for entry in archive.getmembers():
        relative=Path(entry.name)
        if not entry.isfile() or relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Unsafe archive entry')
    archive.extractall(str(repo))
for row in expected:
    if hashlib.sha256((repo/row['path']).read_bytes()).hexdigest()!=row['sha256']:
        raise ValueError('Transferred source hash mismatch: '+row['path'])
print('UPDATED_SOURCE_HASHES_OK')
PY
    CFPO_REPO="$SESSION/repo"; CFPO_MANIFEST_DIR="$SESSION/manifests"; CFPO_DASH_OVERLAY="$verified_overlay"
    for name in SESSION CFPO_REPO CFPO_MANIFEST_DIR CFPO_DASH_OVERLAY; do
        printf '%s=%q\n' "$name" "${!name}"
    done > "$SESSION/session.env"
    printf '%s\n' "$old" > "$SESSION/previous-session.txt"
    cp "$old/deps-constraints.sha256" "$SESSION/deps-constraints.sha256"
    cp "$old/deps-complete" "$SESSION/deps-complete"
    printf '%s\n' "$SESSION" > "$PACK/recovery-session-path.txt"
    printf '%s\n' "$SESSION" > "$PACK/session-path.txt"
    echo "RECOVERY_SESSION_READY=$SESSION"
    echo "Previous failed run preserved: $old"
    echo 'No GPU job submitted.'
    ;;
 access)
    source "$(cat "$PACK/session-path.txt")/session.env"
    extra=()
    if [ -n "${CFPO_MATHVERSE_IMAGES:-}" ]; then extra+=(--mathverse-images "$CFPO_MATHVERSE_IMAGES"); fi
    python3 "$PACK/audit_validation_access.py" "${extra[@]}"
    ;;
 prepare)
    SESSION=$(cat "$PACK/recovery-session-path.txt")
    test "$(cat "$PACK/session-path.txt")" = "$SESSION"
    source "$SESSION/session.env"
    if [ -f "$SESSION/prep-batch/job-id.txt" ]; then
        echo "Already submitted: $(cat "$SESSION/prep-batch/job-id.txt"); use remote_steps.sh status."
        exit 0
    fi
    # Confirm access before requesting another allocation. Compute-side decode is checked again.
    extra=()
    if [ -n "${CFPO_MATHVERSE_IMAGES:-}" ]; then extra+=(--mathverse-images "$CFPO_MATHVERSE_IMAGES"); fi
    python3 "$PACK/audit_validation_access.py" "${extra[@]}"
    mkdir "$SESSION/prep-batch"
    job=$(sbatch --parsable --output="$SESSION/prep-batch/prepare.out" --error="$SESSION/prep-batch/prepare.err" \
        "$PACK/prepare_checks.sbatch" "$PACK" "$SESSION")
    job=${job%%;*}
    printf '%s\n' "$job" > "$SESSION/prep-batch/job-id.txt"
    echo "RECOVERY_PREPARATION_JOB=$job"
    echo '1 GPU, 8 CPUs, 128 GiB, at most 4 hours; preparation only. Safe to disconnect after submission.'
    ;;
 *) echo 'Use: recover_session.sh sync | access | prepare'; exit 2 ;;
esac
