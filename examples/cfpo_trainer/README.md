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
