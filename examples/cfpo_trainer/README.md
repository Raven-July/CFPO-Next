# CFPO-Next: Qwen3-VL GRPO adaptation and bounded validation

Data/reward/system-prompt adaptation, a completed 20-step GRPO run, and a paired
320-sample evaluation are delivered here. These establish the tested training →
checkpoint → HF export → inference path. They do not establish convergence or a
reliable accuracy improvement. This work and its evidence are separate from PR #1.
AI assistance was used; human review and understanding remain necessary.

[Requirements review](WEEKLY_REVIEW.md) maps every item in the exported weekly
task to evidence and gaps, including the Thinking reference table, parameter
differences, missed supervisor pre-review, untouched reference evaluation, and
unverified separate offline-board viewing. This is a bounded engineering delivery,
not a claim that every original-document item has been accepted by the supervisor.

## Data and reward interfaces

`data.train_files` accepts a JSON manifest or direct entries of the form
`[{"dataset": "...json", "image": "...directory"}]`. Manifest-relative paths use
the manifest directory; direct paths use the working directory. Multiple sources
keep their own image roots. Native JSON/JSONL/Parquet loading is preserved.

For V3-math fields `images/problem/answer/id/type/is_cf`, the problem and images
form user messages; the answer enters only `reward_model.ground_truth`, and
metadata enters `extra_info`. Missing images/answers or placeholder mismatches
fail explicitly. The 38,870-row source was used to select bounded training
candidates; 512 is the candidate pool, not verified exposure to all 512 problems.
It is not a full-training claim.

`data.format_prompt` injects the team system prompt before length filtering and
rollout, without mutating input or duplicating it; conflicting prompts fail.
The reference math grader, normalization and format regex are preserved:
`score = 0.9 * accuracy + 0.1 * format`. Format requires paired `<think>` tags
followed by `\boxed{}`. It has not been relaxed to hide zero format scores.

```text
data.format_prompt=examples/cfpo_trainer/format_prompt/base.jinja
reward.custom_reward_function.path=examples/cfpo_trainer/reward_function/base.py
reward.custom_reward_function.name=compute_score
```

## Tested environment

Transformers is an external `--no-deps` overlay fixed to
`cc7ab9be508ce6ed3637bba9e50367b29b742dc6` (`5.3.0.dev0`). This is below the
repository's declared range; compatibility is demonstrated for these runs only.
The existing SIF provides Python 3.12, torch 2.10.0+cu129 and vLLM 0.18.0.
SwanLab 0.10.3 is a separate overlay. `PYTHONPATH` is SwanLab → Transformers →
personal repository, including Ray workers. Hugging Face and SwanLab stay offline.
No shared SIF or dependency replacement is required.

Training uses `trainer.use_v1=False` (legacy synchronous trainer); this is not a
claim of V1 trainer support. The Slurm scripts specify partition `g078t`,
`--gres=gpu:1` and comment `xitucheng213`. The SIF and personal overlay/model paths
are explicit in the scripts. Login-node host preparation is Python 3.6 compatible;
the evaluator itself runs under container Python 3.12.

## Observed training

Evidence below is from user-supplied cluster outputs, not a local GPU rerun.

| Job | Scope | Outcome |
|---|---|---|
| 220231 | First 32 candidates, 3 steps, 1024 response tokens | COMPLETED 0:0; 6m34s; model/extra saved, no optimizer |
| 220236 | First 512 candidates, 20 steps, 2048 response tokens | COMPLETED 0:0; 32m07s; model/optimizer/extra/dataloader saved |

Both used one A800 and started from the original Qwen3-VL-2B-Instruct model.
The 20-step run used 8 CPUs/128 GiB, enabled chunked prefill, and saved every
10 steps with one retained model checkpoint. Final model and optimizer sizes
are 9,755,102,283 and 17,020,305,635 bytes. A step-10 dataloader remnant is not a
second complete checkpoint. Optimizer-state training resume has not been tested;
step-20 HF export and actual inference have passed separately below.

The 320 training generations cover steps 1–20: mean score 0.23625, accuracy
component 0.2625, format 0; 194 contain boxed answers and none paired think tags.
These are mixed-policy training outputs, not held-out accuracy. Real input
contains the team prompt and both think tokens have `special=False`; prompt
omission or special-token stripping does not explain the observed missing tags.
The cause and a remedy remain unresolved.

## Paired sampled evaluation: job 220325

COMPLETED 0:0 in 1h13m16s; `PAIRED_EVALUATION_COMPLETE` and `PAIRED_EVAL_EXIT=0`.
Both models produced 320 results with complete metadata and matching sample
hashes/settings. The native FSDP merger exported step20; the exported model then
completed all 320 inference/grading cases. This verifies HF export/load/inference,
not optimizer-state resume.

Frozen samples: five benchmarks × 64, seed 42, greedy n=1, max_tokens=2048,
max_model_len=8192, image area bounds 200704–1003520; identical original tokenizer,
team prompt and preprocessing on both sides. All 512 training candidates were
checked for exact ID or identical image bytes; two geometry3k candidates were
excluded for image-byte overlap. This does not exclude semantic/re-encoded
duplicates. Answers are used for grading only. Length-stopped cases stay in the
denominator. This is sampled validation, not full benchmark performance or Avg@8.

| Benchmark | Before correct | After correct | Delta (percentage points) | Length-stop before → after |
|---|---|---|---|---|
| geometry3k | 23/64 (35.9375%) | 20/64 (31.2500%) | -4.6875 | 56.2500% → 57.8125% |
| LogicVista | 16/64 (25.0000%) | 20/64 (31.2500%) | +6.2500 | 57.8125% → 53.1250% |
| MathVerse | 17/64 (26.5625%) | 18/64 (28.1250%) | +1.5625 | 53.1250% → 42.1875% |
| C-VQA-Real | 36/64 (56.2500%) | 38/64 (59.3750%) | +3.1250 | 7.8125% → 6.2500% |
| MARS_Bench | 22/64 (34.3750%) | 21/64 (32.8125%) | -1.5625 | 7.8125% → 9.3750% |
| All selected samples | 114/320 (35.6250%) | 117/320 (36.5625%) | +0.9375 | 36.5625% → 33.7500% |

Format is 0 on both sides for every benchmark. Three benchmark scores rose and
two fell; the net difference is only three correct responses. One small fixed
subset/run does not establish a stable benefit. Substantial length truncation,
especially on the first three benchmarks, limits conclusions under this cap.
No new training, cap expansion, model swap or reward relaxation was performed.

[Machine-readable evidence](results/step20-sampled-220325.json) preserves the
four small reported JSON artifacts and recomputed aggregates, with provenance.
Raw responses, model weights, datasets and large logs remain on the cluster.

## Reproduction and evidence paths

All commands below are **remote Bash**, supplied for reproduction; they do not
request additional runs. Existing successful runs should not be resubmitted.

```bash
cd /groups/xitucheng213/home/share/u2025210309/CFPO-Next
# Reproduce the bounded 20-step allocation/settings in a NEW directory.
run=$(mktemp -d /groups/xitucheng213/home/share/u2025210309/grpo-20step-XXXXXX)
train_job=$(sbatch --parsable --mem=128G --time=02:00:00 --job-name=bao-cfpo-grpo-20step \
  --output="$run/train.out" --error="$run/train.err" \
  --export=ALL,CFPO_RUN_DIR="$run",CFPO_TRAIN_ROWS=512,CFPO_TRAIN_STEPS=20,MAX_RESPONSE_LENGTH=2048,CFPO_SAVE_OPTIMIZER=1,SAVE_FREQ=10,ENABLE_CHUNKED_PREFILL=True \
  examples/cfpo_trainer/train_pilot.sbatch)

# Separate paired evaluation; select the training run/checkpoint explicitly.
eval_run=$(mktemp -d /groups/xitucheng213/home/share/u2025210309/paired-eval-XXXXXX)
sbatch --dependency=afterok:"${train_job%%;*}" \
  --output="$eval_run/eval.out" --error="$eval_run/eval.err" \
  --export=ALL,CFPO_EVAL_RUN="$eval_run",CFPO_TRAIN_RUN="$run",CFPO_EVAL_STEP=20 \
  examples/cfpo_trainer/evaluate_pair.sbatch
```

The evaluation launcher accepts `CFPO_TRAIN_RUN`, `CFPO_EVAL_STEP`,
`CFPO_BASE_MODEL`, `CFPO_REPO`, `CFPO_TF_OVERLAY`, `CFPO_LOGGING_OVERLAY`,
`CFPO_DRIVER_LIB` and `CFPO_SIF`; defaults retain the recorded step20 setup.
These overrides received local shell checks, not a new GPU execution. The original
job remains identified by its saved source snapshot/evaluator hash.

The training script defaults to 3 steps/32 candidates; `run_grpo.sh` alone defaults
to config inspection and 128 candidates. Training validation was disabled; its
Geo3K Parquet argument only satisfied dataset construction, with no reported
validation score or pilot split-isolation claim. Held-out results come solely
from the separate audited evaluation.

Under `/groups/xitucheng213/home/share/u2025210309/`:

- Training: `grpo-20step-imD15m/`, including `resolved-config.yaml`,
  `data-selection.json`, `pilot_samples.json`, `generations/`, and
  `checkpoints/global_step_20/actor/`.
- Successful evaluation: `paired-eval-retry-v5RHjX/`, including `comparison.json`,
  `data-audit.json`, both `meta.*.json`, both `results.*.jsonl`, and `eval.out/err`.
- Earlier pilot: `grpo-pilot-CLZTsx/` (job 220231).
- Failed evaluation: `paired-eval-mQSQ72/` (220239, FAILED 1:0, 1m56s): exported
  successfully but failed on the first before request with unsupported modality
  `images`; corrected to vLLM's `image`. Retry used identical frozen sample SHA-256
  `28bca7f621f3ee3c981843d581eb7199278f0ec20ef780b8d26b0dbe599c6ae9`.

The server may retain Git HEAD `4a9dc911` plus offline-delivered modifications.
Run-local `base-commit.txt`, `tracked-changes.patch`, `changed-source.tar` and
`submitted.sbatch` identify the executed source; HEAD alone does not. The retry
script's LF SHA-256 is recorded in the small evidence file. Consolidating the
PR history did not modify any cluster source, artifacts or job.

## Checks and scope remaining

```bash
python examples/cfpo_trainer/test_adapter.py
python examples/cfpo_trainer/test_evaluate_pair.py
python examples/cfpo_trainer/test_analyze_pair.py
```

Local checks passed 9/9 adapter tests, 7/7 evaluation contract tests (including a
mocked infer request that rejects plural modality keys), and 3/3 read-only output
analysis tests on 2026-11-03. These use synthetic fixtures, not new GPU results.
`analyze_pair.py --run <existing-eval-directory>` reads the saved pair to diagnose
answer transitions, format failures and truncation without model loading or
regrading. Actual CPU diagnostics of the retrieved 320-pair outputs are saved in
[diagnostic evidence](results/step20-diagnostics-220325.json): 20 wrong→correct,
17 correct→wrong, 97 correct→correct, 186 wrong→wrong. All 640 texts lack both
think tags, including natural stops. Every length-stopped output is graded wrong
(117 before / 108 after); 11 gains change length→stop and 8 losses stop→length,
while nine gains and nine losses both stop naturally. This association does not
establish a causal cap effect or a stable reasoning benefit. No new inference,
regrading or reward relaxation was performed. The original .swanlab archive was
also fully read locally (4,249 records; finish RUN_STATE_FINISHED), recovering
118 metric series and 20-step key curves without SDK init or cloud sync.
Selected summaries/config fields are in the same diagnostic evidence; a local
static replay page is separate from native SwanLab watch and is not a claim of
real-time monitoring. Recorded warmup is 0.0 and rollout top_p is 1.0, differing
from reference 0.05 and 0.99; the requirements review lists these deviations. Job 220202 passed real data,
image decoding and reward checks. Jobs 220204/220208 passed documented subchecks
but failed overall (stale module reference/missing SwanLab); both issues were
addressed, and job 220209 passed isolated offline logging with protected versions
unchanged. These environment checks are distinct from actual training and scores.

The requested adaptation, examples scripts, personal-branch PR and bounded real
GRPO loop have execution evidence. DAPO remains optional and unrun. No full-training
step count, convergence target or minimum runtime was confirmed. Format compliance,
length truncation and reliable generalization remain unresolved; they do not turn
optional large experiments into requirements. Bao's independent understanding of
the loop is not established by successful AI-assisted execution alone.
