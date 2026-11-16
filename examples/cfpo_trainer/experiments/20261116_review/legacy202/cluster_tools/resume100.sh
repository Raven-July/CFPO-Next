#!/bin/bash
set -euo pipefail
PACK=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
case "${1:-help}" in
 prepare) exec python3 "$PACK/prepare_resume100.py" ;;
 node-logs)
  # Raw tails preserve read errors and do not depend on grep matches.
  exec ssh -o BatchMode=yes -o ConnectTimeout=8 gpu5 'R=/tmp/ray/session_2026-11-10_08-40-13_692680_2201057/logs
hostname
ls -ld "$R"
tail -n 80 "$R/raylet.err"
tail -n 120 "$R/raylet.out"'
  ;;
 submit)
  CANDIDATE=$(cat "$PACK/resume100-path.txt")
  test -s "$CANDIDATE/resume-receipt.json"
  test ! -e "$CANDIDATE/run/job-id.txt" || { echo 'Resume already submitted; use status.' >&2; exit 2; }
  test ! -e "$CANDIDATE/run/submitted.sbatch" || { echo 'Previous attempt exists; inspect instead of duplicate submission.' >&2; exit 2; }
  mkdir "$CANDIDATE/submission-lock" || { echo 'Submission attempt already exists; inspect status, do not duplicate.' >&2; exit 2; }
  # Hold until the submission receipt is written, preventing a run without a recorded job ID.
  job=$(sbatch --hold --parsable --partition=g078t --nodes=1 --gres=gpu:2 --cpus-per-task=8 \
      --mem=256G --time=3-00:00:00 --comment=xitucheng213 --job-name=bao-resume100 --output="$CANDIDATE/run/train.out" \
      --error="$CANDIDATE/run/train.err" "$PACK/resume100.sbatch" "$PACK" "$CANDIDATE")
  job=${job%%;*}
  case "$job" in ''|*[!0-9]*) echo 'Unexpected job ID; inspect scheduler before retry.' >&2; exit 2 ;; esac
  printf '%s\n' "$job" > "$CANDIDATE/run/job-id.txt"
  scontrol release "$job"
  echo "RESUME100_JOB=$job RUN=$CANDIDATE/run; 2GPU, 8CPU, 256G, at most72h; target step202"
  ;;
 status)
  CANDIDATE=$(cat "$PACK/resume100-path.txt"); RUN="$CANDIDATE/run"
  echo "RESUME_RUN=$RUN"
  if [ -f "$RUN/job-id.txt" ]; then
    job=$(cat "$RUN/job-id.txt")
    sacct -j "$job" -X --format=JobID,State,Elapsed,ReqMem,ExitCode -P
  else echo 'Prepared only; no resume job submitted.'; fi
  if [ -f "$RUN/exit-code.txt" ]; then printf 'EXIT_CODE='; cat "$RUN/exit-code.txt"; fi
  tail -n 15 "$RUN/train.out" 2>/dev/null || true
  tail -n 15 "$RUN/train.err" 2>/dev/null || true
  ;;
 export-view)
  CANDIDATE=$(cat "$PACK/resume100-path.txt")
  test -d "$CANDIDATE/run/swanlog"
  tar -czf "$PACK/resume100-native-view.tar.gz" -C "$CANDIDATE/run/swanlog" .
  echo "RESUME_NATIVE_LOG_SNAPSHOT=$PACK/resume100-native-view.tar.gz"
  ;;
 *) echo 'Stages: node-logs | prepare (no GPU) | submit (2GPU) | status | export-view' ;;
esac
