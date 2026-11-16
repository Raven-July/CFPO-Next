#!/bin/bash
set -euo pipefail
PACK=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
stage=${1:-status}
if [ "$stage" = prepare ]; then exec python3 "$PACK/prepare.py"; fi
test -s "$PACK/candidate-path.txt" || { echo 'Preparation did not complete. Fix prepare first; no GPU job submitted.' >&2; exit 2; }
CANDIDATE=$(cat "$PACK/candidate-path.txt")
RUN="$CANDIDATE/run"
case "$stage" in
 submit)
  python3 - "$PACK" "$CANDIDATE" <<'PY'
import sys
sys.path.insert(0,sys.argv[1])
from guard30 import verify_receipt,verify_package
verify_package(sys.argv[1])
verify_receipt(sys.argv[2])
PY
  test ! -e "$RUN/job-id.txt" || { echo 'Already submitted; use status.'; exit 2; }
  mkdir "$CANDIDATE/submission-lock" || { echo 'Submission attempted; inspect before retry.'; exit 2; }
  job=$(sbatch --hold --parsable --job-name=bao-newreward30 --partition=g078t --nodes=1 --ntasks=1 \
      --gres=gpu:2 --cpus-per-task=8 --mem=256G --time=1-00:00:00 --comment=xitucheng213 \
      --output="$RUN/train.out" --error="$RUN/train.err" "$PACK/job.sbatch" "$PACK" "$CANDIDATE")
  job=${job%%;*}
  case "$job" in ''|*[!0-9]*) echo 'Unexpected job ID; inspect scheduler.' >&2; exit 2;; esac
  printf '%s\n' "$job" > "$RUN/job-id.txt"
  scontrol release "$job"
  echo "VALIDATION30_JOB=$job RUN=$RUN; 2GPU 8CPU 256GiB at most24h; original model, exactly30 updates"
  ;;
 status)
  echo "VALIDATION30_RUN=$RUN"
  if [ -f "$RUN/job-id.txt" ]; then
   job=$(cat "$RUN/job-id.txt")
   squeue -h -u "$USER" -o '%A|%T|%M|%R' | awk -F'|' -v j="$job" '$1==j'
   sacct -X -j "$job" --format=JobID,State,Elapsed,ReqMem,ExitCode -P
  fi
  if [ -f "$RUN/exit-code.txt" ]; then printf 'EXIT_CODE='; cat "$RUN/exit-code.txt"; fi
  for f in first-output-check.json completion-check.json; do if [ -f "$RUN/$f" ]; then cat "$RUN/$f"; echo; fi; done
  tail -n 8 "$RUN/train.out" 2>/dev/null || true
  tail -n 6 "$RUN/train.err" 2>/dev/null || true
  ;;
 export-view)
  test -d "$RUN/swanlog"
  tar -czf "$PACK/newreward30-native-view.tar.gz" -C "$RUN/swanlog" .
  echo "NATIVE_LOG_EXPORT=$PACK/newreward30-native-view.tar.gz"
  ;;
 *) echo 'Use control.sh prepare | submit | status | export-view' >&2;exit 2;;
esac
