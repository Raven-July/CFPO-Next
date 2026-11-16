#!/bin/bash
set -euo pipefail
PACK=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
stage=${1:-status}
if [ "$stage" = prepare ]; then exec python3 "$PACK/prepare.py" "${@:2}";fi
test -s "$PACK/candidate-path.txt" || { echo 'Run prepare first; no job submitted.' >&2;exit 2; }
CANDIDATE=$(cat "$PACK/candidate-path.txt");RUN="$CANDIDATE/run"
case "$stage" in
 submit)
  python3 - "$PACK" "$CANDIDATE" <<'PY'
import sys
sys.path.insert(0,sys.argv[1])
from protocol import verify_package,verify_prepared
verify_package(sys.argv[1]);verify_prepared(sys.argv[2])
PY
  test ! -e "$RUN/job-id.txt" || { echo 'Already submitted; use status.' >&2;exit 2; }
  # A failed query is not evidence of an empty queue. No automatic retry.
  active=$(squeue -h -u "$USER" -n bao-diagnostic12 -o '%A|%T')
  test -z "$active" || { echo "Existing diagnostic job: $active" >&2;exit 2; }
  mkdir "$CANDIDATE/submission-lock" || { echo 'Prior submission attempt; inspect first.' >&2;exit 2; }
  job=$(sbatch --hold --parsable --partition=g078t --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=8 \
    --mem=128G --time=01:00:00 --comment=xitucheng213 --job-name=bao-diagnostic12 \
    --output="$RUN/probe.out" --error="$RUN/probe.err" "$PACK/job.sbatch" "$PACK" "$CANDIDATE")
  job=${job%%;*}
  case "$job" in ''|*[!0-9]*) echo 'Unexpected job ID; inspect scheduler.' >&2;exit 2;;esac
  printf '%s\n' "$job" > "$RUN/job-id.txt"
  scontrol release "$job"
  echo "JOB=$job RUN=$RUN; 12 completions, no training"
  ;;
 status)
  echo "RUN=$RUN"
  if [ -f "$RUN/job-id.txt" ];then
    job=$(cat "$RUN/job-id.txt")
    squeue -h -j "$job" -o '%A|%T|%M|%R'
    sacct -X -j "$job" --format=JobID,State,Elapsed,ReqMem,ExitCode -P
  fi
  if [ -f "$RUN/exit-code.txt" ];then printf 'EXIT_CODE=';cat "$RUN/exit-code.txt";fi
  tail -n 14 "$RUN/probe.out" 2>/dev/null || true
  tail -n 8 "$RUN/probe.err" 2>/dev/null || true
  ;;
 export)
  test -f "$RUN/exit-code.txt" || { echo 'Job still running/unknown; preserve active outputs.' >&2;exit 2; }
  test ! -e "$PACK/diagnostic12-results.tar.gz" || { echo 'Prior export preserved.' >&2;exit 2; }
  tar -czf "$PACK/diagnostic12-results.tar.gz" -C "$CANDIDATE" receipt.json samples.json run
  sha256sum "$PACK/diagnostic12-results.tar.gz" > "$PACK/diagnostic12-results.tar.gz.sha256"
  ;;
 *) echo 'Use prepare | submit | status | export' >&2;exit 2;;
esac
