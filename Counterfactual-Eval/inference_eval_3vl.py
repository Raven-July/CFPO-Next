import argparse
import csv
import json
import logging
import os
import re
import subprocess
import sys
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional

from PIL import Image
from tqdm import tqdm


warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
logging.basicConfig(level=logging.ERROR)
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

MIN_PIXELS = 200704
MAX_PIXELS = 1003520


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a Qwen3-VL checkpoint on the configured datasets."
    )
    parser.add_argument("--name", required=True, help="Model name used in result paths")
    parser.add_argument(
        "--model_path",
        required=True,
        help="Actor checkpoint path or a merged Hugging Face model path",
    )
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--max_model_len", type=int, default=6144)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.6)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if Path(args.name).name != args.name or args.name in {".", ".."}:
        raise ValueError("--name must be a single directory-safe name")
    if args.n < 1:
        raise ValueError("--n must be at least 1")
    if args.batch_size < 1 or args.num_workers < 1:
        raise ValueError("--batch_size and --num_workers must be at least 1")
    if args.tensor_parallel_size < 1:
        raise ValueError("--tensor_parallel_size must be at least 1")
    if args.max_tokens < 1 or args.max_model_len < args.max_tokens:
        raise ValueError("--max_model_len must be at least --max_tokens")
    if not 0 < args.gpu_memory_utilization <= 1:
        raise ValueError("--gpu_memory_utilization must be in (0, 1]")


def locate_model_directories(model_path: str) -> tuple[Optional[Path], Path]:
    path = Path(model_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Model path does not exist: {path}")

    v2_hf_dir = path / "model" / "huggingface"
    v1_hf_dir = path / "huggingface"
    if v2_hf_dir.is_dir():
        return path, v2_hf_dir
    if v1_hf_dir.is_dir():
        return path, v1_hf_dir

    if path.name == "huggingface":
        checkpoint_dir = path.parent.parent if path.parent.name == "model" else path.parent
        return checkpoint_dir, path

    return None, path


def has_safetensors(hf_dir: Path) -> bool:
    return any(path.is_file() and path.stat().st_size > 0 for path in hf_dir.glob("*.safetensors"))


def prepare_hf_model(model_path: str, repo_root: Path) -> Path:
    checkpoint_dir, hf_dir = locate_model_directories(model_path)
    if has_safetensors(hf_dir):
        print(f"[INFO] Found Hugging Face safetensors in {hf_dir}")
        return hf_dir

    if checkpoint_dir is None:
        raise FileNotFoundError(
            f"No *.safetensors found in {hf_dir}, and {model_path} is not a verl actor checkpoint."
        )

    shard_pattern = "model_world_size_*_rank_*.pt"
    if not any(checkpoint_dir.glob(shard_pattern)):
        raise FileNotFoundError(
            f"No *.safetensors found in {hf_dir}, and no FSDP shards matching "
            f"{shard_pattern} were found in {checkpoint_dir}."
        )

    merger = repo_root / "scripts" / "legacy_model_merger.py"
    if not merger.is_file():
        raise FileNotFoundError(f"Model merger not found: {merger}")

    print(f"[INFO] No safetensors found in {hf_dir}; merging FSDP checkpoint first.")
    command = [
        sys.executable,
        str(merger),
        "merge",
        "--backend",
        "fsdp",
        "--local_dir",
        str(checkpoint_dir),
        "--target_dir",
        str(hf_dir),
    ]
    subprocess.run(command, cwd=repo_root, check=True)

    if not has_safetensors(hf_dir):
        raise RuntimeError(f"Model merger completed but produced no *.safetensors in {hf_dir}")
    print(f"[INFO] Merged Hugging Face model is ready in {hf_dir}")
    return hf_dir


def dataset_name_from_path(dataset_path: str) -> str:
    stem = Path(dataset_path).stem
    return re.sub(r"_(?:train|val|test)(?:_V\d+)?$", "", stem, flags=re.IGNORECASE)


def load_dataset_config(config_path: str) -> list[dict[str, str]]:
    with open(config_path, encoding="utf-8") as file:
        raw_config = json.load(file)

    if not isinstance(raw_config, list) or not raw_config:
        raise ValueError("Dataset config must be a non-empty JSON list")

    datasets = []
    seen_names = set()
    for index, row in enumerate(raw_config):
        if not isinstance(row, dict) or "dataset" not in row or "image" not in row:
            raise ValueError(
                f"Dataset config row {index} must contain the same 'dataset' and 'image' keys "
                "used by examples/cfpo_trainer/datasets_math_val.json"
            )
        dataset_name = dataset_name_from_path(row["dataset"])
        if not dataset_name or dataset_name in seen_names:
            raise ValueError(f"Invalid or duplicate dataset name derived from {row['dataset']}")
        seen_names.add(dataset_name)
        datasets.append(
            {
                "dataset_name": dataset_name,
                "dataset": row["dataset"],
                "image": row["image"],
            }
        )
    return datasets


def global_step_name(model_path: str) -> str:
    for part in reversed(Path(model_path).parts):
        if re.fullmatch(r"global_step_\d+", part):
            return part
    raise ValueError(f"Could not derive global_step_xxx from model path: {model_path}")


def load_format_prompt(repo_root: Path) -> Any:
    from jinja2 import Template

    format_prompt_path = repo_root / "examples" / "cfpo_trainer" / "format_prompt" / "math.jinja"
    if not format_prompt_path.is_file():
        raise FileNotFoundError(f"Training format prompt not found: {format_prompt_path}")
    source = format_prompt_path.read_text(encoding="utf-8").strip()
    return Template(source)


def process_image(image: Image.Image) -> Image.Image:
    if image.mode == "P":
        return image.convert("RGBA").convert("RGB")
    if image.mode != "RGB":
        return image.convert("RGB")
    return image


def load_and_preprocess_single_item(
    item: dict[str, Any],
    image_root: str,
    processor: Any,
    format_prompt: Any,
) -> dict[str, Any]:
    problem_text = item["problem"]
    image_files = item["images"]
    image_tag_count = problem_text.count("<image>")
    if len(image_files) != image_tag_count:
        raise ValueError(
            f"Item {item.get('id', '<unknown>')} has {len(image_files)} images but "
            f"{image_tag_count} <image> tags"
        )

    item_images = []
    image_paths = []
    for image_name in image_files:
        image_path = Path(image_root) / image_name
        image_paths.append(image_path)
        with Image.open(image_path) as image:
            item_images.append(process_image(image.copy()))

    formatted_problem = format_prompt.render(content=problem_text)
    content = []
    image_index = 0
    for segment in re.split("(<image>)", formatted_problem):
        if not segment:
            continue
        if segment == "<image>":
            content.append(
                {
                    "type": "image",
                    "image": str(image_paths[image_index]),
                }
            )
            image_index += 1
        else:
            content.append({"type": "text", "text": segment})
    assert image_index == len(item_images)

    messages = [{"role": "user", "content": content}]
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return {
        "prompt": prompt,
        "multi_modal_data": {"image": item_images},
    }


def grade_completion(raw_output: str, ground_truth: Any) -> tuple[str, bool]:
    from mathruler.grader import extract_boxed_content, grade_answer

    answer = extract_boxed_content(raw_output)
    ground_truths = ground_truth if isinstance(ground_truth, list) else [ground_truth]
    for expected in ground_truths:
        if grade_answer(answer, expected):
            return answer, True
    return answer, False


def run_evaluation(
    args: argparse.Namespace,
    llm: Any,
    processor: Any,
    format_prompt: Any,
    dataset_config: dict[str, str],
    output_dir: Path,
) -> float:
    from vllm import SamplingParams

    dataset_path = dataset_config["dataset"]
    image_root = dataset_config["image"]
    dataset_name = dataset_config["dataset_name"]
    print(f"\n[INFO] Starting evaluation for dataset: {dataset_name}")

    with open(dataset_path, encoding="utf-8") as file:
        data = json.load(file)
    if not data:
        raise ValueError(f"Dataset is empty: {dataset_path}")

    sampling_params = SamplingParams(
        n=args.n,
        temperature=1.0 if args.n > 1 else 0.0,
        top_p=0.99 if args.n > 1 else 1.0,
        top_k=-1,
        max_tokens=args.max_tokens,
        stop=["<|eot_id|>", "</s>", "<|im_end|>"],
    )

    final_results = []
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        for start in tqdm(
            range(0, len(data), args.batch_size),
            desc=f"Processing {dataset_name}",
        ):
            batch = data[start : start + args.batch_size]
            futures = [
                executor.submit(
                    load_and_preprocess_single_item,
                    item,
                    image_root,
                    processor,
                    format_prompt,
                )
                for item in batch
            ]
            vllm_inputs = [future.result() for future in futures]
            outputs = llm.generate(prompts=vllm_inputs, sampling_params=sampling_params)
            if len(outputs) != len(batch):
                raise RuntimeError(
                    f"vLLM returned {len(outputs)} outputs for a batch of {len(batch)}"
                )

            for item, output in zip(batch, outputs):
                sample_result = {
                    "images": item["images"],
                    "question": item["problem"],
                    "ground_truth": item["answer"],
                    "is_cf": item.get("is_cf", False),
                    "type": item.get("type", "unknown"),
                    "rollouts": [],
                }
                correct_count = 0
                for rollout_index, completion in enumerate(output.outputs, start=1):
                    answer, is_correct = grade_completion(completion.text, item["answer"])
                    correct_count += int(is_correct)
                    sample_result["rollouts"].append(
                        {
                            f"model_output_{rollout_index}": completion.text,
                            f"extracted_answer_{rollout_index}": answer,
                            f"score_{rollout_index}": int(is_correct),
                        }
                    )
                if len(output.outputs) != args.n:
                    raise RuntimeError(
                        f"vLLM returned {len(output.outputs)} rollouts; expected {args.n}"
                    )
                sample_result["avg_score"] = correct_count / args.n
                final_results.append(sample_result)

    totals: defaultdict[str, int] = defaultdict(int)
    score_sums: defaultdict[str, float] = defaultdict(float)
    for result in final_results:
        cf_type = "cf" if result["is_cf"] else "ncf"
        question_type = result["type"]
        for key in ["all", cf_type, question_type, f"{question_type}_{cf_type}"]:
            totals[key] += 1
            score_sums[key] += result["avg_score"]

    accuracies = {
        f"{key}_avg@{args.n}_acc": score_sums[key] / totals[key] * 100
        for key in sorted(totals)
    }
    all_accuracy = accuracies[f"all_avg@{args.n}_acc"]
    print(f"[INFO] {dataset_name} all_avg@{args.n}_acc: {all_accuracy:.2f}%")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{dataset_name}_avg@{args.n}.json"
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(
            {"accuracies": accuracies, "details": final_results},
            file,
            ensure_ascii=False,
            indent=2,
        )
    print(f"[INFO] Detailed results saved to {output_path}")
    return all_accuracy


def write_summary(
    output_dir: Path,
    model_name: str,
    dataset_names: list[str],
    scores: dict[str, float],
    n: int,
) -> Path:
    missing = [name for name in dataset_names if name not in scores]
    if missing:
        raise RuntimeError(f"Cannot write summary; missing dataset scores: {missing}")

    summary_path = output_dir / f"avg@{n}_acc.csv"
    mean_score = sum(scores[name] for name in dataset_names) / len(dataset_names)
    with open(summary_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["name", *dataset_names, "mean"])
        writer.writerow(
            [
                model_name,
                *(f"{scores[name]:.6f}" for name in dataset_names),
                f"{mean_score:.6f}",
            ]
        )
    return summary_path


def main() -> None:
    args = parse_args()
    validate_args(args)

    repo_root = Path(__file__).resolve().parent.parent
    datasets = load_dataset_config(args.config_path)
    format_prompt = load_format_prompt(repo_root)
    step_name = global_step_name(args.model_path)
    run_output_dir = Path(args.output_dir).expanduser().resolve() / args.name / step_name
    hf_model_dir = prepare_hf_model(args.model_path, repo_root)

    from transformers import AutoProcessor
    from vllm import LLM

    print(
        f"[INFO] Loading {args.name} from {hf_model_dir} "
        f"with tensor parallel size {args.tensor_parallel_size}"
    )
    llm = LLM(
        model=str(hf_model_dir),
        trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        enable_chunked_prefill=True,
        mm_processor_kwargs={
            "min_pixels": MIN_PIXELS,
            "max_pixels": MAX_PIXELS,
        },
        seed=42,
    )
    processor = AutoProcessor.from_pretrained(
        hf_model_dir,
        trust_remote_code=True,
    )

    scores = {}
    for dataset in datasets:
        scores[dataset["dataset_name"]] = run_evaluation(
            args,
            llm,
            processor,
            format_prompt,
            dataset,
            run_output_dir,
        )

    summary_path = write_summary(
        run_output_dir,
        args.name,
        [dataset["dataset_name"] for dataset in datasets],
        scores,
        args.n,
    )
    print(f"\n[INFO] Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
