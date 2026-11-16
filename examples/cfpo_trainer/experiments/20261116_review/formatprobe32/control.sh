#!/bin/bash
set -euo pipefail
PACK=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
stage=${1:-status}
if [ "$stage" = prepare ]; then exec python3 "$PACK/prepare.py"; fi
test -s "$PACK/candidate-path.txt" || { echo 'Run prepare successfully first; no job submitted.' >&2;exit 2; }
CANDIDATE=$(cat "$PACK/candidate-path.txt");RUN="$CANDIDATE/run"
case "$stage" in
 submit)
  python3 - "$PACK" "$CANDIDATE" <<'PY'
import sys
sys.path.insert(0,sys.argv[1])
from core import verify_package,verify_prepared
verify_package(sys.argv[1]);verify_prepared(sys.argv[2])
PY
  test ! -e "$RUN/job-id.txt" || { echo 'Already submitted; use status.';exit 2; }
  mkdir "$CANDIDATE/submission-lock" || { echo 'Prior submission attempt exists; inspect first.';exit 2; }
  job=$(sbatch --hold --parsable --partition=g078t --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=8 \
      --mem=128G --time=04:00:00 --comment=xitucheng213 --job-name=bao-format32 \
      --output="$RUN/probe.out" --error="$RUN/probe.err" "$PACK/job.sbatch" "$PACK" "$CANDIDATE")
  job=${job%%;*}
  case "$job" in ''|*[!0-9]*) echo 'Unexpected job ID; inspect scheduler.' >&2;exit 2;; esac
  printf '%s\n' "$job" > "$RUN/job-id.txt"
  scontrol release "$job"
  echo "FORMAT_PROBE_JOB=$job RUN=$RUN;1GPU 8CPU 128GiB at most4h;480 outputs, no training"
  ;;
 status)
  echo "FORMAT_PROBE_RUN=$RUN"
  if [ -f "$RUN/job-id.txt" ];then
   job=$(cat "$RUN/job-id.txt")
   squeue -h -u "$USER" -o '%A|%T|%M|%R' | awk -F'|' -v j="$job" '$1==j'
   sacct -X -j "$job" --format=JobID,State,Elapsed,ReqMem,ExitCode -P
  fi
  if [ -f "$RUN/exit-code.txt" ];then printf 'EXIT_CODE=';cat "$RUN/exit-code.txt";fi
  if [ -f "$RUN/summary.json" ];then cat "$RUN/summary.json";echo;fi
  tail -n 10 "$RUN/probe.out" 2>/dev/null || true
  tail -n 8 "$RUN/probe.err" 2>/dev/null || true
  ;;
 export)
  test -f "$RUN/exit-code.txt" || { echo 'Run has not exited; export after it finishes.';exit 2; }
  test -s "$RUN/probe.out" || { echo 'No run log to export yet';exit 2; }
  test ! -e "$PACK/formatprobe32-results.tar.gz" || { echo 'Existing export preserved.';exit 2; }
  tar -czf "$PACK/formatprobe32-results.tar.gz" -C "$CANDIDATE" receipt.json samples.json run
  sha256sum "$PACK/formatprobe32-results.tar.gz" > "$PACK/formatprobe32-results.tar.gz.sha256"
  echo "FORMAT_PROBE_EXPORT=$PACK/formatprobe32-results.tar.gz"
  ;;
 *) echo 'Use prepare | submit | status | export' >&2;exit 2;;
esac
