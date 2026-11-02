"""Preflight only: processor/filtering, GPU arithmetic, Ray imports, no weights."""

import importlib
import json
import os
import tempfile
from pathlib import Path

import ray
import torch
from omegaconf import OmegaConf
from transformers import AutoConfig, AutoProcessor, AutoTokenizer

from verl.utils.dataset.rl_dataset import RLHFDataset


def check_sources():
    roots = {
        "transformers": Path(os.environ["CFPO_OVERLAY"]).resolve(),
        "verl": Path(os.environ["CFPO_REPO"]).resolve(),
    }
    result = {}
    for name, root in roots.items():
        module = importlib.import_module(name)
        source = Path(module.__file__).resolve()
        assert source.is_relative_to(root), (name, str(source))
        result[name] = str(source)
        if name == "transformers":
            assert module.__version__ == "5.3.0.dev0"
    return result


def main():
    repo = Path(os.environ["CFPO_REPO"])
    run = Path(os.environ["CFPO_RUN_DIR"])
    model = os.environ["MODEL_PATH"]
    print("MAIN_SOURCES:", check_sources(), flush=True)
    config = AutoConfig.from_pretrained(model, local_files_only=True)
    assert config.model_type == "qwen3_vl", config.model_type
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    processor = AutoProcessor.from_pretrained(model, local_files_only=True)
    print("MODEL_CONFIG_AND_PROCESSOR_OK:", config.model_type, flush=True)

    manifest = json.loads((repo / "examples/cfpo_trainer/datasets_math_train.json").read_text())
    with open(manifest[0]["dataset"], encoding="utf-8") as stream:
        rows = json.load(stream)[:2]
    with tempfile.TemporaryDirectory(dir=run) as temp:
        sample_path = Path(temp) / "samples.json"
        sample_path.write_text(json.dumps(rows), encoding="utf-8")
        data_config = OmegaConf.create({
            "cache_dir": str(Path(temp) / "cache"),
            "format_prompt": str(repo / "examples/cfpo_trainer/format_prompt/base.jinja"),
            "filter_overlong_prompts": True,
            "filter_overlong_prompts_workers": 1,
            "max_prompt_length": 2048,
            "image_patch_size": 16,
        })
        dataset = RLHFDataset(
            [{"dataset": str(sample_path), "image": manifest[0]["image"]}],
            tokenizer=tokenizer, processor=processor, config=data_config,
        )
        assert len(dataset) == len(rows), "One of the first two prompts failed processor/length filtering"
        print("PROCESSOR_AND_LENGTH_FILTER_OK:", len(dataset), flush=True)

    assert torch.cuda.is_available(), "CUDA is unavailable"
    value = torch.tensor([2.0], device="cuda") * 3
    torch.cuda.synchronize()
    assert value.item() == 6.0
    print("GPU_ARITHMETIC_OK:", torch.cuda.get_device_name(0), flush=True)
    del value
    torch.cuda.empty_cache()

    # Use an isolated Ray instance; no actor/rollout engine or model weights are created.
    environment = {
        name: os.environ[name] for name in (
            "PYTHONPATH", "PYTHONNOUSERSITE", "PYTHONDONTWRITEBYTECODE",
            "LD_LIBRARY_PATH", "CFPO_REPO", "CFPO_OVERLAY", "MODEL_PATH",
            "HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE",
        )
    }
    ray.init(
        address="local", num_cpus=2, num_gpus=0, include_dashboard=False,
        object_store_memory=536870912,
        runtime_env={"env_vars": environment},
    )
    try:
        @ray.remote(num_cpus=1)
        def worker_check():
            return check_sources()

        print("RAY_WORKER_SOURCES:", ray.get(worker_check.remote(), timeout=75), flush=True)
    finally:
        ray.shutdown()
    print("RAY_WORKER_IMPORT_OK", flush=True)

    # These are the modules needed by the selected legacy synchronous trainer.
    # Resolve config/imports only, without instantiating any worker or vLLM engine.
    for name in (
        "verl.trainer.main_ppo_v0",
        "verl.workers.engine_workers",
        "verl.workers.rollout.vllm_rollout.vllm_async_server",
        "swanlab",
    ):
        importlib.import_module(name)
        print("TRAINING_IMPORT_OK:", name, flush=True)
    print("CFPO_RUNTIME_PREFLIGHT_OK", flush=True)


if __name__ == "__main__":
    main()
