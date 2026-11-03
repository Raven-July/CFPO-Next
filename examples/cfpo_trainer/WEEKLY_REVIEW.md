# Weekly requirements review 2026-11-03

Source: [2026-10-27 weekly task](https://docs.qq.com/doc/DRXB1cEhtT1FZQXJ6),
checked against Bao's exported Word text and all three embedded images.
Export SHA-256: `7538987e1b7ab379a6c5811bb87995926cf015bf42530020133fdd0fcbbb29f3`.
The export itself and its screenshots are not included in this PR.
Bao's later explicit scope keeps DAPO optional and authorizes bounded training
and evaluation; the document does not authorize unbounded training or shared
environment changes. No numeric acceptance threshold or training duration is set.

## Requirement to evidence

| Requirement | Current evidence and remaining boundary |
|---|---|
| Qwen3-VL-2B-Instruct GRPO | Jobs 220231/220236 completed 3/20 steps; a bounded loop/stability run, not a full-convergence baseline |
| Verify existing vLLM image before training | Recorded preflight subchecks and later successful actual A800 training; failures are not reported as passing jobs |
| Detect dataset/image manifests without breaking native loading | RLHFDataset dispatch + cfpo.py; native JSON/JSONL/Parquet path retained; adapter tests and real-data check 220202 |
| Separate JSON and image paths, preserve labels | Schema matches document screenshot; answer is a reward label, not model input; checks reject missing images/answers/placeholders |
| Configure system prompt at training time | data.format_prompt + nonmutating insertion in message building and length filtering; real training input includes it |
| Migrate prompt and math reward | examples/cfpo_trainer/format_prompt/base.jinja and reward_function/base.py; original 0.9/0.1 weighting and format regex preserved |
| Training scripts in examples/cfpo_trainer and personal-branch PR | run_grpo.sh, train_pilot.sbatch, independent PR #2; no PR #1 results used |
| Send training script to supervisor before running | Bao confirms the scripts were not sent before training; this process requirement was missed. Current scripts are ready for review, but that does not retroactively satisfy pre-review |
| Keep original training parameters apart from model path | Not fully met by the bounded configuration; explicit differences below must be reviewed before calling this an equivalent baseline |
| Preserve Counterfactual-Eval under repository root | Six upstream source files copied byte-for-byte from pinned CFPO commit; hashes in reference-eval-provenance.json |
| Evaluate GRPO against reference expectations | Job 220325 completes a separate greedy sampled comparison; original Counterfactual-Eval pipeline has not been run on this checkpoint |
| Enable generation logging | trainer.rollout_data_dir replaces the old configuration name; 320 JSONL generations for steps 1–20 inspected. This is separate from SwanLab dashboard rendering |
| SwanLab offline recording | SwanLab 0.10.3 personal overlay and offline records passed; no shared dependency changes |
| Watch offline board on separate 4x4090 host | Bao identifies bupt-cluster as the known GPU/SSH host; a separate 4x4090 viewing host/session and local-dashboard replay are not verified. Offline records alone do not fulfill viewing |
| Ignore logs/weights/data and submit necessary code | .gitignore excludes swanlog/checkpoints/pt and reference result directories; only small aggregate evidence is versioned |

The document names an existing 8B example as a modification reference, while also
requiring completed scripts in cfpo_trainer. The delivered 2B entry point lives
in cfpo_trainer; the shared upstream 8B example is deliberately not repurposed.
The schema and locally observed dataset checks pass, but an HF dataset revision
linking the existing team data to the screenshot has not been independently
established; no duplicate dataset download was performed.

## Parameter migration differences

Reference: `Raven-July/CFPO` commit `98ca5a4123be2e85e94855fead46c9a385611136`,
`examples/qwen2_5_vl_3b_GRPO-math.sh` and its `examples/config.yaml`.
Current values are launcher settings; the saved resolved config remains the
source for the executed job and for any inherited defaults.

| Item | Original example | Bounded 20-step run |
|---|---|---|
| Base model | Qwen2.5-VL-3B-Instruct | Qwen3-VL-2B-Instruct, as requested |
| Prompt / response cap | 4096 / 2048 | 2048 / 2048 |
| Rollout prompts per batch | 384 | 4 |
| Answers per prompt | 5 | 4 |
| Rollout temperature / top_p | 1.0 / 0.99 | Recorded 1.0 / 1.0 |
| Dataset shuffle seed | Config 1 | Recorded 42 |
| Actor global / mini batch | 128 | 4 (current PPO mini batch interface) |
| Update micro batch | 4 | 1 |
| Learning rate | 1e-6 | 1e-6 |
| Warmup ratio | Explicit 0.05 | Recorded config 0.0; no warmup |
| Training data / duration | Full configured manifest, 2 epochs | First 512 candidates, capped at 20 steps |
| GPU count | 2 | 1 A800 |
| Validation | Before training and every 5 steps | Disabled in training; separate paired evaluation |
| Image area bounds | Explicit 200704–1003520 | Not explicitly migrated in training launcher; evaluation uses these bounds |
| Optimizer strategy | adamw_bf16 | Recorded torch.optim.AdamW |

Single-card batch reduction and bounded steps were chosen for the authorized
flow check. The verified warmup/top_p/optimizer differences and inherited image settings need attention before a
future equivalence claim. None of these settings is silently changed after the
fact, and current job results are not represented as an original-parameter run.
512 is the candidate pool, not a claim that all 512 problems were trained on:
320 responses at n=4 correspond to 80 prompt sampling positions, possibly with
repeated problems; unique exposure needs the saved input/ID records.

## What the performance reference means

The third image in the task document is Table 5 for **Qwen3-VL-2B-Thinking**.
Its GRPO row reports Geo3k 39.29, MathVerse 47.98, LogicVista 29.84 and the
nine-benchmark Average 46.84. These are document reference values, not acceptance
thresholds and not independently revalidated paper results in this handoff.

Our checkpoint is Instruct, trained only 20 steps, evaluated on five selected
validation sources with greedy n=1. The provided reference shell pipeline uses
n=8 (temperature 1.0), and its test-source manifest contains ten sources distinct
from both our five-source selection and the nine-column screenshot. Its grader
also has fallbacks absent from the preserved training reward (e.g. unboxed
answers, multiple targets and selected colon-prefix handling). Consequently,
36.5625% versus 46.84 is **not** a valid like-for-like comparison or improvement
estimate. A matched benchmark/split/grader/training budget is a later decision.

## Reference code and safe use

`../../Counterfactual-Eval/` contains the upstream files unchanged. One inherited trailing-whitespace line in inference_pipeline_math_grpo.sh is
retained for exact source preservation; it is not a newly introduced logic defect.
Source commit
and SHA-256 values are in [reference-eval-provenance.json](reference-eval-provenance.json).
The old shell scripts still contain Qwen2.5 checkpoints and machine-specific
paths. The evaluator can replace unreadable images with gray images and skip
template failures; importing the reference does not validate these behaviors.
Do not label its results complete unless candidate counts, missing images and
actual evaluated counts have been audited. None of these scripts was executed
or substituted for the successful job 220325 evidence.

For a future explicitly selected, audited config, the untouched entry point is
`python Counterfactual-Eval/inference_eval.py --model_name Qwen3-VL-2B-Instruct
--model_prefix step20 --model_path <exported-model> --config_path <audited-config>
--output_dir <new-personal-output> --n <agreed-n> --cot --batch_size 4
--tensor_parallel_size 1 --num_workers 4` inside the existing container.
This is a parameter template, not a job submission or an instruction to run a
full ten-source n=8 experiment. The candidate config and budget remain unchosen.

## Improve interpretation without further inference

`python3 examples/cfpo_trainer/analyze_pair.py --run <successful-eval-directory>`
works with login-node Python 3.6+, reads existing artifacts only and prints no
answer text. It verifies matching metadata/keys/grades, then reports:

- correct→wrong / wrong→correct / unchanged counts and deterministic example keys;
- missing/open-only/close-only/both think tags, boxed markers and reference format matches;
- correct counts split by finish reason and mean output token counts.

The six real result/sample/audit/meta files have now been retrieved and checked.
[CPU diagnostic evidence](results/step20-diagnostics-220325.json) contains only
aggregates and source-file hashes, with no raw questions, answers or images.
The aggregate/per-benchmark counts match the original comparison exactly.

| Correctness transition | Count |
|---|---|
| Correct → correct | 97 |
| Wrong → wrong | 186 |
| Wrong → correct | 20 |
| Correct → wrong | 17 |

All 640 generated texts contain neither opening nor closing think tags, including
203 before and 212 after responses that stopped naturally. Thus missing tags
are not confined to length-stopped outputs; the zero format score is verified
against the unchanged reference regex. Bao's read-only model-file inspection shows both chat_template.json and
tokenizer_config.json append only `<|im_start|>assistant` plus a newline for
add_generation_prompt, with no think-tag prefill. Missing output opening tags
therefore are not explained by a think prefix supplied by that template. The
remaining model/prompt/configuration cause is unresolved. Boxed markers increase from 194 to 209. Mean output tokens fall from
925.23125 to 880.640625; this is not an end-to-end speed measurement.

Before/after length-stopped responses are 117/108 and all have accuracy grade 0.
Eleven wrong→correct cases also change length→stop; eight correct→wrong cases
change stop→length. The remaining nine gains and nine losses both stop naturally.
This describes the observed association; it does not prove that raising the cap
would fix the errors or that mathematical/visual reasoning has improved.

Deterministically selected first change keys per benchmark were inspected
locally: some cases replace repetitive/truncated output with a finished boxed
answer, while others newly repeat until the cap or change a finished answer.
No source images were re-read and no independent visual/ground-truth adjudication
was performed. Raw texts and example answers remain outside Git. Synthetic tests
validate the tool separately and are not substituted for these real counts.
README reproduction commands now capture the training job ID and submit evaluation
with an `afterok` dependency, so copying the command block does not race the new
checkpoint. This command block passed local Bash syntax checking.
The new mocked infer test exercises the actual vLLM request construction and
rejects the earlier plural `images` modality regression without loading a model.

## Separate viewing and human review

SwanLab `offline` and dashboard-oriented `local` are different modes. The
[official init reference](https://docs.swanlab.cn/api/py-init.html) distinguishes
them, and the [offline board guide](https://docs.swanlab.cn/guide_cloud/offline-board.html)
requires the dashboard extra. The retained SwanLab 0.10.3 wheel also checks for
`swanboard==0.1.10b2` in local mode. No dashboard dependency is added to training.

On a separate viewing machine that already has a compatible local-mode log
directory and dashboard environment, the documented command is
`swanlab watch -l <local-mode-log-directory> -h 127.0.0.1 -p 5092`.
The actual training log inventory contains the nonempty binary run archive
`run-20261102_232310-hz03nu8f/run-hz03nu8f.swanlab` (358,950 bytes), config.yaml,
metadata and debug files. Presence alone does not demonstrate a readable board.
The archive has now been read completely on the local review machine using
SwanLab 0.10.3's official DataStoreReader.scan and generated protobuf definitions,
without SDK initialization or sync: 4,249 records, including 4,070 scalar records
across 118 metric series; finish state RUN_STATE_FINISHED. All scalar values are
finite. Key series cover exactly steps 1–20, and the recorded mean training reward
0.2362499926 agrees with the generation-derived 0.23625 to floating-point precision.
Config and archive hashes plus selected metric summaries are in the existing
diagnostic evidence file. This local reading does not constitute a native watch
session or a real-time monitor on a separate 4x4090 host.
A fully offline static curve-review page was generated outside Git for Bao;
it is explicitly labelled as historical log replay, not the SwanLab watch UI.
The existing offline log's native local-board compatibility has not been verified;
renaming its directory or changing the current environment variable does not
convert past logs. Select a supported replay/viewing route after inspecting the
actual files. Do not sync logs to the cloud or install in shared environments.

Bao can review the loop by tracing run_grpo.sh → main_ppo → rollout n=4 →
reward_function/base.py → group-relative advantages → actor update → checkpoint.
A small hand calculation for all-correct/all-wrong/mixed groups and explanation
of one changed evaluation case would provide personal understanding evidence;
successful AI-assisted execution alone cannot establish that understanding.
