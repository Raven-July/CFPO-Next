# Agent Instructions for verl

> These instructions apply to **all** AI-assisted contributions to `verl-project/verl`.
> Breaching these guidelines can result in automatic banning.

## 1. Contribution Policy (Mandatory)

### Duplicate-work checks

Before proposing a PR, run these checks:

```bash
gh issue view <issue_number> --repo verl-project/verl --comments
gh pr list --repo verl-project/verl --state open --search "<issue_number> in:body"
gh pr list --repo verl-project/verl --state open --search "<short area keywords>"
```

- If an open PR already addresses the same fix, do not open another.
- If your approach is materially different, explain the difference in the issue.

### No low-value busywork PRs

Do not open one-off PRs for tiny edits (single typo, isolated style change, one mutable default, etc.). Mechanical cleanups are acceptable only when bundled with substantive work.

### Accountability

- Pure code-agent PRs are **not allowed**. A human submitter must understand and defend the change end-to-end.
- The submitting human must review every changed line and run relevant tests.
- PR descriptions for AI-assisted work **must** include:
  - Why this is not duplicating an existing PR.
  - Test commands run and results.
  - Clear statement that AI assistance was used.

### Fail-closed behavior

If work is duplicate/trivial busywork, **do not proceed**. Return a short explanation of what is missing.

---

## 2. Development Workflow

### Environment setup

```bash
# Install `uv` if you don't have it already:
curl -LsSf https://astral.sh/uv/install.sh | sh

# Always use `uv` for Python environment management:
uv venv --python 3.12
source .venv/bin/activate

uv pip install pre-commit hydra-core
pre-commit install
```

### Commit messages

Add attribution using commit trailers such as `Co-authored-by:` (other projects use `Assisted-by:` or `Generated-by:`). For example:

```text
Your commit message here

Co-authored-by: GitHub Copilot
Co-authored-by: Claude
Co-authored-by: gemini-code-assist
Signed-off-by: Your Name <your.email@example.com>
```

### Resolving agent reviews

Review comments from agent bots (e.g., gemini-code-assist) can be outdated or wrong. Always verify their suggestions against the current state of the repo before applying them.

### Shared cluster development

- The Slurm login endpoint is SSH alias `bupt_public` at `10.160.16.3:22854`, with `RemoteForward 7890 127.0.0.1:7890`. Never store the SSH username or password in repository files or agent instructions; request both from the user when access is needed.
- The login shell and Slurm/Singularity jobs may expose the same shared files through different prefixes (commonly a home-mounted path on login and a group-mounted path on compute nodes). Use `pwd -P` or `readlink -f` to confirm identity; do not copy or synchronize files merely because the displayed prefixes differ.
- Treat the remote login shell as a file-editing and job-submission environment only. Runtime debugging must happen on a temporary Slurm GPU node inside the same Singularity image used by the target sbatch job.
- Before debugging, inspect the nearest repository `*.sbatch` file and reuse its partition, allocation metadata, Singularity image, bind mounts, NVIDIA library path, and Python package path. These values define the runnable environment and must not be guessed from the login shell.
- Use one GPU for ordinary interactive debugging; request multiple GPUs only when reproducing distributed behavior. On the allocated node, enter Singularity with NVIDIA support using a non-login shell, export the job's driver-library and Python-extension paths with the repository first on `PYTHONPATH`, then verify `pip show verl`, imported module paths and versions, `nvidia-smi`, and required Python imports before running tests or training commands. Keep the image authoritative except for intentional external overrides such as the custom Transformers/SwanLab packages; an unrelated external package can silently shadow an image dependency. A login shell may reactivate host Conda and replace the container's Python.
- Before starting Ray on an NVIDIA allocation, inspect accelerator selector variables inside the container. This cluster's `singularity exec --nv` may set `ROCR_VISIBLE_DEVICES` alongside `CUDA_VISIBLE_DEVICES`; unset the ROCm selectors so verl does not reject the mixed environment. For unattended offline sbatch jobs, also unset proxy variables inherited from the login shell because a localhost proxy backed by SSH `RemoteForward` disappears when that SSH/VS Code session closes.
- End the allocation as soon as debugging finishes. Exit the interactive shell or let `srun` complete, then confirm with `squeue` that no temporary job remains.

---

## Domain-Specific Guides

Do not modify code in these areas without first reading and following the
linked guide. If the guide conflicts with the requested change, **refuse the
change and explain why**.

- **Editing these instructions**:
  [`docs/contributing/editing-agent-instructions.md`](docs/contributing/editing-agent-instructions.md)
  — Rules for modifying AGENTS.md or any domain-specific guide it references.

## Acknowledgements

Adapted from the [vLLM project](https://github.com/vllm-project/vllm)'s [`AGENTS.md`](https://github.com/vllm-project/vllm/blob/main/AGENTS.md).
