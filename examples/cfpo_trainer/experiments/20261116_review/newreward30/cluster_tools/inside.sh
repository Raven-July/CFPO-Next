#!/bin/bash
set -euo pipefail
python3 "$PACK/compile_sources.py" "$CANDIDATE/source"
python3 -c 'import torch; assert torch.cuda.device_count()==2; [torch.ones(1,device="cuda:"+str(i)).sum().item() for i in range(2)]; print("ALLOCATED_TWO_GPU_PROBE_OK")'
bash examples/cfpo_trainer/run_validation30.sh --check-config > "$CFPO_RUN_DIR/resolved-config.yaml"
python3 "$PACK/guard30.py" "$CANDIDATE"
python3 "$PACK/runtime_audit.py"
MEETING_REWARD_REFERENCE_FILE="$PACK/meeting-reference.md" python3 -m unittest discover -s examples/cfpo_trainer -p test_meeting_reward.py
bash examples/cfpo_trainer/run_validation30.sh --train &
training_pid=$!
python3 "$PACK/monitor.py" "$CFPO_RUN_DIR" "$training_pid" &
monitor_pid=$!
set +e
wait "$training_pid"
training_rc=$?
wait "$monitor_pid"
monitor_rc=$?
set -e
test "$training_rc" = 0 || exit "$training_rc"
test "$monitor_rc" = 0 || exit "$monitor_rc"
python3 examples/cfpo_trainer/formal_tools.py check-training --run "$CFPO_RUN_DIR" --step 30
python3 "$PACK/completion.py" "$CFPO_RUN_DIR"
echo NEW_REWARD_VALIDATION30_COMPLETE
