# CFPO-Next data and reward adaptation

This directory adapts the team's existing Counterfactual-R1 data format and
reward to the current CFPO-Next dataset/reward interfaces. AI assistance was used;
Bao must review the changes and understand the data flow before submitting the PR.

## Data flow

`[{"dataset": "...json", "image": "...directory"}]` can be provided directly as
`data.train_files` or via a JSON manifest path. Relative entries are resolved
against the manifest directory; direct entries use the working directory.
Manifests use both `dataset` and `image` keys (image may be null for text-only data).
Ordinary sample JSON/JSONL and Parquet files retain the existing loading path.

The confirmed V3-math sample schema is `images`, `problem`, `answer`, `id`, `type`,
and `is_cf`. `problem` becomes the user message; each `<image>` resolves to the
corresponding absolute image path. `answer` becomes only
`reward_model.ground_truth`; `id/type/is_cf` remain in `extra_info`.
Missing images, answers, and mismatched placeholders fail explicitly.

`data.format_prompt` points to a plain-text system-prompt file. It is added
before both length filtering and rollout messages. Repeated calls do not mutate
input messages or add duplicates. A conflicting pre-existing system prompt fails
instead of silently changing the experiment. Null preserves the original behavior.

The example training manifest points to the observed V3-math file (38,870 rows).
This is an interface-check input; it is not evidence that the team selected this
dataset revision for final training. No evaluation split is guessed or generated.

## Reward

The reference batch reward was adapted to verl's per-sample signature:
`compute_score(data_source, solution_str, ground_truth, extra_info, ...)`.
It returns `score`, `format`, and `accuracy`. The original normalization,
mathruler grader, formatting regex, and 0.9 accuracy / 0.1 format weights are kept.
Configure the current framework with:

```text
data.format_prompt=examples/cfpo_trainer/format_prompt/base.jinja
reward.custom_reward_function.path=examples/cfpo_trainer/reward_function/base.py
reward.custom_reward_function.name=compute_score
```

These keys were checked against the current source, not the older training script.

## Validation and limits

```bash
python examples/cfpo_trainer/test_adapter.py
```

This dependency-free test covers labels, path resolution, manifest detection,
multiple sources, prompt conflicts, and message immutability. Its image fixture
checks file paths only, not decoding.

`check_data.py`, launched with `check_data.sbatch`, uses eight real training
samples to check PIL decoding, actual RLHFDataset loading and resume, direct and
manifest configs, native JSON/Parquet compatibility, and the real mathruler reward.
It deliberately skips tokenizer execution and prompt-length filtering and loads
no model. Passing it does not validate Ray workers, GPU kernels, rollout, or training.

The Transformers overlay is fixed to commit
`cc7ab9be508ce6ed3637bba9e50367b29b742dc6` (5.3.0.dev0), installed with no dependencies
outside the SIF. Main-process imports have passed in the existing image, but this
is below the repository's declared Transformers range. Keep that constraint
visible until model/worker checks establish the actual supported combination.

GRPO/DAPO training and evaluation results are still pending.

## Recorded validation and training draft

Job 220202 completed with exit code 0:0 in 44 seconds. The observed output included
`REAL_IMAGE_DECODE_OK`, `MANIFEST_MESSAGES_LABELS_AND_RESUME_OK`,
`DIRECT_CONFIG_AND_MULTIPLE_SOURCES_OK`, `NATIVE_JSON_AND_PARQUET_REGRESSION_OK`,
`REAL_MATHRULER_REWARD_CASES_OK`, and `CFPO_DATA_REWARD_CHECK_OK`.
The recorded base commit is 4a9dc911; adapter changes were uncommitted, so the
base hash alone does not identify the checked code. Preserve the delivery package
checksum and worktree diff alongside these logs.

`run_grpo.sh` is now a draft for review. Its default action is `--check-config`,
which asks Hydra to print the resolved configuration without entering training.
It uses the existing local Qwen3-VL-2B-Instruct model, the team prompt/reward,
single-A800 pilot parameters, three steps, and 128 training samples by default.
This is a pilot configuration, not a complete baseline or a claimed reproduction.
It explicitly selects the legacy synchronous trainer (`trainer.use_v1=False`)
to stay close to the previously tested training loop; it does not claim V1 support.
It uses the container's Python and forwards the overlay/repository paths to Ray;
it does not invoke uv or synchronize new dependencies.

`--train` requires an explicit `VAL_FILE`. The default Geometry3k path is only a
known placeholder for config inspection and is not a final evaluation selection.
The supplied old multi-benchmark validation manifest contains outdated container
paths and a `COCO_val2014_imagess` typo; no silently corrected evaluation manifest
is supplied. Resolve dataset versions and train/validation separation before training.
The draft disables periodic/pre-training validation for the initial pilot.
SwanLab is offline, and `trainer.rollout_data_dir` saves training generations;
the old framework's `log_generations` name is not a current config key.

`check_runtime.sbatch` resolves that script's config, exercises two real prompts
with the local tokenizer/processor and length filter, checks tiny GPU arithmetic,
checks Ray worker import provenance, and imports selected trainer/backend modules.
It never loads model weights, starts a vLLM engine, performs rollout, or updates
parameters. Runtime checks are pending until their output is inspected.
Submit the training script to the supervisor before actual training.

## Preflight evidence (2026-11-02)

- Job 220204: config resolution, real tokenizer/processor, length filtering on two
  training samples, A800 arithmetic, and main/Ray-worker import provenance passed.
  The job ultimately failed because the check referenced the removed module
  `verl.workers.fsdp_workers`. The checker now uses the actual entry point's
  `verl.workers.engine_workers`. Do not report 220204 as a fully passing job.
- Job 220208: engine_workers, FSDP engine, and vLLM async server imports passed.
  It failed only on the missing SwanLab package; it is not a fully passing job.
- Job 220209: minimal SwanLab 0.10.3 overlay installed and wrote a real offline log;
  completed 0:0 in 13 seconds with no stderr. It reused existing compatible
  dependencies and verified torch/vllm/transformers/tokenizers/huggingface-hub/
  pydantic/protobuf versions were unchanged.

The run/preflight scripts now prepend the independent logging overlay to
`PYTHONPATH`, before the Transformers overlay and this repository. Both launcher
and Ray workers use the same path order. The training shell accepts
`CFPO_TF_OVERLAY` and `CFPO_LOGGING_OVERLAY` overrides for another user's personal
installation paths. Logs and checkpoints are stored outside the repository;
`.gitignore` also excludes SwanLab log directories.

The preflight checks above do not by themselves validate model weights,
generation, updates, or evaluation. The separate real-training result below
provides evidence for this specific three-step configuration only.

## Actual three-step training pilot

`train_pilot.sbatch` starts real training on the first 32 V3-math training rows,
using the existing model, both personal overlays, one A800, eight CPUs, 64 GiB
host memory, and a 20-minute walltime limit. It runs three optimizer steps and
saves the final model/extra checkpoint plus generations and offline metrics.
It records the source JSON hash, selected IDs, resolved config, and actual source
snapshot. These are training attempts until update/checkpoint evidence is inspected.

Validation is disabled. A known Geo3K Parquet path is supplied only because the
trainer constructs a validation dataset even when validation is disabled; no
validation score or train/validation disjointness is claimed for this pilot.
This exception is limited to the flow pilot. Select and audit the team's intended
validation manifests before running an evaluated baseline.
The model/extra-only pilot checkpoint does not include optimizer state and is
not a full-state resume checkpoint for a longer baseline.

### Observed pilot result: job 220231

User-supplied Slurm accounting reports COMPLETED, exit 0:0, elapsed 6m34s on gpu1.
The output includes `PILOT_TRAINING_PROCESS_OK` and `PILOT_EXIT=0`.
Training steps 1/2/3 logged losses 0.06029538, -0.06679699, 0.00814930 and gradient
norms 1.02691317, 1.72622943, 6.59431410, with learning rate 1e-6 throughout.
Generation files 1.jsonl, 2.jsonl and 3.jsonl were reported written. The checkpoint
directory `grpo-pilot-CLZTsx/checkpoints/global_step_3` was logged; checkpoint
listing supplied by the user confirms 10 nonempty files, including
`actor/model_world_size_1_rank_0.pt` (9,755,102,283 bytes), actor extra state,
HF config/tokenizer files, and dataloader state. No optimizer file is present,
consistent with the pilot's model/extra-only save setting. This confirms file
existence and nonzero size; checkpoint reload has not been tested.

This is evidence of actual updates on the current CFPO-Next implementation with
the specified Transformers development overlay, not the older official-verl run.
It does not establish convergence, a complete baseline, improved accuracy, or
general compatibility with all current dependency combinations. No validation was run.
Reported mean rewards are training rewards, not held-out accuracy.

Response clipping ratios were 0.125, 0.1875 and 0.0 under the 1024-token limit.
The user's inspection of all 48 saved responses found 41 boxed answers, zero
paired think tags, and zero matches to the original formatting reward regex.
Inspect the actual input prompt and token special flags before attributing this
to model behavior or changing experiment settings. Do not relax the reference
reward to hide the zero formatting score. vLLM also
warned that this model does not officially support disabling chunked prefill;
the next training configuration should address that warning. Optional-engine,
deprecation and repeated SwanLab finish warnings were retained in the raw logs.

The supplied generation input contains the complete team system prompt. The
original tokenizer's added-token entries mark both `<think>` and `</think>` as
`special: false`. The saved outputs' missing tags therefore are not explained by
their removal as special tokens. Keep the zero-format result as observed behavior;
no prompt/reward relaxation or model replacement is applied.

## 20-step stability run

The launcher now supports `CFPO_TRAIN_ROWS`, `CFPO_TRAIN_STEPS`,
`MAX_RESPONSE_LENGTH`, `CFPO_CHECKPOINT_CONTENTS`, and `SAVE_FREQ` overrides.
`CFPO_SAVE_OPTIMIZER=1` selects model/optimizer/extra saving without passing a
comma-containing JSON value through Slurm's `--export` parser.
The next run selects 512 training rows, executes 20 steps, uses a 2048-token
response limit, enables chunked prefill, and saves model/optimizer/extra state.
SwanLab remains offline. It starts from the same fixed original model revision;
the preceding model/extra-only checkpoint is not used as a full-state resume.
Validation stays disabled and no validation score is claimed.

The intended allocation is one A800, eight CPUs, 128 GiB host memory, and a
two-hour walltime limit. Checkpoints are saved every ten steps with one retained
checkpoint; transient disk use can include the previous checkpoint during saving.
This is a stability/output-quality run, not an experiment isolating the effect
of the changed response limit or a complete evaluated baseline.

User-supplied Slurm accounting confirms job 220236 COMPLETED, exit 0:0,
elapsed 32m07s. The progress log reaches 20/20, with the progress-bar interval
covering 27m24s. Results are under `grpo-20step-imD15m`. These durations measure
different scopes; do not present progress-bar time as the complete job duration.
Checkpoint optimizer-file existence, final metric values, generation quality,
and checkpoint reload are pending inspection. No held-out evaluation was run.
