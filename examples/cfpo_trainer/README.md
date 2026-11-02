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

No model weights, generation engine, parameter update, or full evaluation has
been validated in this CFPO-Next checkout. Training and model-runtime compatibility
remain pending. These results are separate partial checks, not one successful
end-to-end training run.
