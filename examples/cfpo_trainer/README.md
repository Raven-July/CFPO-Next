# CFPO GRPO baseline

Implements the 2026-10-27 task on CFPO-Next. Reference behavior comes from
Raven-July/CFPO's `verl/utils/dataset.py`, `examples/reward_function/base.py`,
and `examples/format_prompt/base.jinja`.

## Data

Download `RavenInJuly/CFPO_Datasets` from Hugging Face. Keep the annotation JSON
and image directories separate. Copy the example train/validation descriptors
and replace their paths with your actual files. Do not use the training file as
validation. Each descriptor is a JSON array of `{ "dataset": "...json", "image":
"...images" }` objects. Relative paths in a descriptor file resolve relative to
that file. Inline descriptor lists are also accepted by `data.train_files` and
`data.val_files`; their paths resolve relative to the launch directory.

Rows require `problem`, `answer`, and an `images` list matching the `<image>`
placeholders. Image names resolve under the descriptor's `image` directory.
Metadata (`id`, `type`, `is_cf`, `solution`) is retained in `extra_info`.
Missing images or mismatched placeholders fail before training. Existing verl
chat-formatted JSON/JSONL and Parquet paths retain their original loading path.

## Prompt and reward

The original CFPO prompt is supplied as a separate system message at training
time with `data.system_prompt_path`; annotations contain only questions.
Alternatively set `data.system_prompt` directly, but not both. A configured
system prompt replaces dataset system messages. Filtering and rollout use the
same messages.

The original reward is preserved: `0.9 * accuracy + 0.1 * format`. Accuracy uses
mathruler's boxed-answer extraction and mathematical equivalence grader. Format
requires `<think>...</think>` followed by `\boxed{...}`. The adapter returns
verl's required `score`, plus `overall`, `accuracy` and `format` metrics.

## Launch and review

Review the resolved script/config before allocating GPUs. The default model is
**Qwen/Qwen3-VL-2B-Instruct**, not Qwen3.5 or a Thinking checkpoint.

```bash
export TRAIN_FILE=/absolute/path/datasets_train.json
export TEST_FILE=/absolute/path/datasets_val.json
export MODEL_PATH=/absolute/path/Qwen3-VL-2B-Instruct
bash examples/cfpo_trainer/run_qwen3_vl_2b_fsdp.sh
```

SwanLab runs offline (`SWANLAB_MODE=offline`); install swanlab in the runtime or
its external package directory. Both training and validation generations are
saved locally, and 30 validation generations are logged to SwanLab. Run the
offline dashboard separately from training. Logs, checkpoints and generation
files are ignored by Git.

For a two-step smoke run, retain at least two responses per question:

```bash
TRAIN_BATCH_SIZE=4 PPO_MINI_BATCH_SIZE=4 ROLLOUT_N=2 \
  bash examples/cfpo_trainer/run_qwen3_vl_2b_fsdp.sh \
  trainer.total_training_steps=2 trainer.save_freq=1 trainer.test_freq=1
```

The upstream launcher uses uv by default. Set `VERL_USE_UV=0` only when using an
already provisioned environment. This checkout's lockfile specifies newer
PyTorch/vLLM/Transformers than the historical `verl-vllm018.sif`; do not assume
that image is compatible. Check imports, resolved configuration, reward metrics,
nonzero advantages/gradients, checkpoint output and final Slurm exit status.
A successful process with zero gradients proves execution, not learning.

After training, keep the reference `Counterfactual-Eval` directory separate from
training changes and evaluate the saved checkpoint against the baseline.

## Verified smoke run (2026-11-02)

Slurm job 220052 completed two steps on one A800 80GB in 6m03s with exit code
0:0 and Slurm batch MaxRSS 28.20 GB. Runtime: Python 3.12.3, PyTorch 2.10.0+cu129,
Transformers 5.3.0 and vLLM 0.18.0 in the historical SIF, plus external packages.
The Transformers cc7ab9be overlay from the Qwen3.5 experiment was not used.

This run sampled 32 ViRL39K training rows and four Geo3K validation rows,
used batch/mini-batch size 4, four responses per prompt, max prompt length 4096,
max response length 1024, and disabled pre-training validation. Save/test
frequency was 2. The Ray driver advertised eight CPUs and a 2 GiB object store;
the rollout checkpoint update bucket was 256 MiB. SwanLab 0.10.3 ran offline
with its missing dependencies installed as external packages for Python 3.12.

TransferQueue's default eight storage actors plus its controller exhausted
the eight Ray CPUs and prevented initialization. This launcher now defaults to
one storage unit; larger runs can override this parameter at the command line.

Both steps had nonzero advantages and gradient norms (1.361 and 1.695).
The global_step_2 checkpoint includes model, optimizer and extra state;
training/validation generation JSONL and offline SwanLab logs were produced.
Four-sample validation accuracy was 0.25 and format reward was 0.0. This verifies
the training pipeline, not full benchmark performance. Inspect generated
answers and their think/boxed formatting before scaling up.

## Remaining experiment workflow

The repository now includes the cluster wrapper in `cluster.sbatch` (smoke,
pilot and full modes), preparation and diagnosis tools, and the original
`Counterfactual-Eval` directory copied byte-for-byte from Raven-July/CFPO commit
`98ca5a4123be2e85e94855fead46c9a385611136`. Its old shell scripts and hard-coded
example JSON are reference material; use `evaluate.sbatch` with generated paths.
These additions have local unit/config checks; only the earlier two-step smoke
run has been executed on the cluster so far.

After every SSH reconnect, restore the download interpreter before running
the standard-library tools below:

```bash
export GRPO_ROOT=/groups/xitucheng213/home/share/u2024213357/GRPO
source "$GRPO_ROOT/download-env311/bin/activate"
cd "$GRPO_ROOT/cfpo-next"
git pull --ff-only
python examples/cfpo_trainer/inspect_generations.py \
  "$GRPO_ROOT/generations/cfpo-smoke-220052/validation/2.jsonl"
python -u examples/cfpo_trainer/prepare_data.py \
  --raw "$GRPO_ROOT/data/cfpo-raw" --output "$GRPO_ROOT/data/cfpo"
```

Preparation extracts all downloaded image archives under `cfpo-raw/extracted`,
including another copy of the two archives already extracted for the smoke
test. It reuses matching extracted files on subsequent runs and rejects paths
outside the target directory. It generates full ViRL39K training, the original
five validation subsets, and all ten reference test subsets. Annotation-relative
image files must all exist. Validation descriptors carry individual data_source
labels so metrics are reported per benchmark. The validation subsets sampled
from test data are for monitoring, not an independent held-out claim.

First run a 20-step pilot at response length 2048 (128 training and 20 Geo3K
validation rows). The inspected smoke outputs contained the original system
instruction but omitted think tags and ended mid-answer at the 1024-token cap.
Keep the original reward/prompt while diagnosing compliance and truncation.

```bash
RUN_MODE=pilot sbatch --time=02:00:00 examples/cfpo_trainer/cluster.sbatch
```

Inspect gradients, truncation, format/accuracy metrics and generated answers
before starting full training. Full mode uses every training/validation row,
two epochs, a global batch of 384 prompts, a global PPO mini-batch of 128 prompts,
four completions per prompt, and response length 2048. Full mode requires two
GPUs on a single node. With 38,870 source rows, the number of complete batches
before length filtering is 101 per epoch. It checkpoints
every 20 steps, keeps three actor checkpoints and validates every 100 steps.
It is bounded by the Slurm 12-hour limit; completion is not guaranteed in one
allocation. Keep EXPERIMENT_NAME unchanged when resubmitting to resume the most
recent saved checkpoint. Never resubmit while the same experiment is running.
Each submission writes SwanLab records to its own directory under
`$GRPO_ROOT/swanlog-v2/$EXPERIMENT_NAME/job-$SLURM_JOB_ID`.

```bash
RUN_MODE=full EXPERIMENT_NAME=qwen3-vl-2b-grpo-full-2gpu-bs384 \
  sbatch --gres=gpu:2 --cpus-per-task=16 --mem=256G examples/cfpo_trainer/cluster.sbatch
```

Evaluate the initial base model and the final merged actor using identical n=8
sampling and reference CoT settings. Use separate fresh labels for each run.
The comparison is within this Instruct experiment; the paper's Thinking-model
table is context and is not a matched control. Do not label the smoke checkpoint
as the fully trained baseline.

```bash
EVAL_MODEL="$GRPO_ROOT/models/Qwen3-VL-2B-Instruct" EVAL_LABEL=base-n8 \
  sbatch examples/cfpo_trainer/evaluate.sbatch

# Replace NNN with the actual completed training step, after verifying accounting.
CHECKPOINT_ACTOR="$GRPO_ROOT/checkpoints/qwen3-vl-2b-grpo-full/global_step_NNN/actor" \
EVAL_MODEL="$GRPO_ROOT/models/qwen3-vl-2b-grpo-full-merged" EVAL_LABEL=trained-n8 \
  sbatch examples/cfpo_trainer/evaluate.sbatch

python examples/cfpo_trainer/compare_evaluations.py \
  --config "$GRPO_ROOT/data/cfpo/datasets_eval_full.json" \
  --base "$GRPO_ROOT/eval-results/base-n8" \
  --trained "$GRPO_ROOT/eval-results/trained-n8" --n 8
```

The evaluation wrapper merges FSDP model shards when CHECKPOINT_ACTOR is supplied.
It checks that every result has as many rows as its annotation file, since the
original evaluator catches per-dataset errors. Always check Slurm exit status.

## Full-run memory correction

Job 220059 completed initial validation on 2,837 filtered validation rows but
was killed by the Slurm cgroup OOM handler before the first training update.
The job ended OUT_OF_MEMORY after 26m42s. The default null val_batch_size
submitted the entire validation set as one batch, retaining large multimodal
payloads simultaneously. The cluster wrapper now sets data.val_batch_size=16
and requests 128 GiB of host RAM instead of 64 GiB. Validation still covers
every row and aggregates metrics across batches; neither the dataset nor the
training batch size is reduced. This correction requires another cluster run
to verify. Preserve the failed run's logs by using a fresh experiment name.

## Separate-server dashboard

SwanLab 0.10.3 distinguishes offline sync records from the legacy local dashboard
database. `offline` records alone must not be claimed as a verified `watch`
dashboard. Official instructions: https://docs.swanlab.cn/en/guide_cloud/offline-board.html

On the separate 4x4090 host, use an isolated environment with
`swanlab[dashboard]==0.10.3`. Copy the Slurm output and the replay tool to that
host using the approved account/file-transfer route (or access the shared path).
Then run:

```bash
python replay_dashboard.py /path/to/cfpo-smoke-220052.out \
  --logdir /path/to/board-logs --name smoke-220052-replayed
swanlab watch /path/to/board-logs --host 127.0.0.1 --port 5092
```

The replay is explicitly labelled and contains numeric metrics parsed from the
Slurm log, not an invented recreation of media/generation artifacts. For new
native local logs, install dashboard dependencies in the training overlay and
set `SWANLAB_MODE=local` before submission. The launcher preserves this override.
Use the existing SSH forwarding route to access port 5092; server/account and
log-transfer details must be supplied before this can be verified remotely.

## Matched DAPO baseline and experiment exports

Use `CFPO_ALGORITHM=dapo` with the same `RUN_MODE` as GRPO. Model, splits,
seed, prompt, reward, learning rate, batch sizes, rollout count, token limits,
validation and checkpoint cadence are inherited from that profile. DAPO starts
from the original Instruct model in a separate checkpoint directory.

The reference CFPO DAPO script disables KL and clips with lower/upper ratios
0.20/0.28. V1 uses the same normalized group advantages as GRPO and filters
groups using the `accuracy` returned by the shared reward function. With four
binary accuracy scores per group, rejecting identical groups corresponds to
the reference filter's 0.01–0.99 interval. V1 replenishes filtered groups, so
matching optimizer steps does not imply matching generation compute. Token-mean
loss is explicit. Hardware, batching and learning-rate warmup follow the current
GRPO profile for comparison rather than the reference script's older settings.

```bash
CFPO_ALGORITHM=dapo RUN_MODE=pilot EXPERIMENT_NAME=qwen3-vl-2b-dapo-pilot \
  sbatch --time=02:00:00 --account="$USER" --partition=g078t --qos=normal \
    --comment=xitucheng213 examples/cfpo_trainer/cluster.sbatch
```

After verifying filtering, gradient updates and checkpoint saves, submit the
two-GPU full command above with `CFPO_ALGORITHM=dapo` and experiment
`qwen3-vl-2b-dapo-full-2gpu-bs384`. Compare at matched saved steps using the same
evaluation configuration. Keep full and pilot results separate.

`export_results.py` reads actual Slurm metrics and optionally copies validation
generation JSONL with a CoT-format summary and three sampled outputs. Repeat
`--run` with the same label in submission order to merge resumed job segments.
Use `--tables-only` on the login node to export without matplotlib. With
matplotlib installed it also saves training and validation PNGs.

```bash
python examples/cfpo_trainer/export_results.py \
  --run "GRPO-pilot=$GRPO_ROOT/logs/cfpo-220056.out" \
  --generations "GRPO-pilot-20=$GRPO_ROOT/generations/cfpo-pilot-220056/validation/20.jsonl" \
  --output "$GRPO_ROOT/reports/grpo-pilot" --tables-only
```
