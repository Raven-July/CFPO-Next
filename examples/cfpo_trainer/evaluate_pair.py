"""Freeze held-out validation samples, evaluate two models, compare complete results.

This is independently authored tooling for Bao's experiment. Image limits and
the math grader follow the provided original CFPO evaluation convention. Missing
images are errors, never replaced with placeholder images.
"""

import argparse
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

BENCHMARKS = [
    ("geometry3k", "V3-math/geometry3k_val_V3.json", "geometry3k_images"),
    ("LogicVista", "V3-math/LogicVista_val_V3.json", "LogicVista_images"),
    ("MathVerse", "V3-math/MathVerse_val_V3.json", "MathVerse_images"),
    ("C-VQA-Real", "V3/C-VQA-Real_val_V3.json", "COCO_val2014_images"),
    ("MARS_Bench", "V3/MARS_Bench_val_V3.json", "COCO_val2014_images"),
]
SETTINGS = {
    "n": 1, "temperature": 0.0, "top_p": 1.0, "seed": 42,
    "max_tokens": 2048, "max_model_len": 8192, "batch_size": 4,
    "min_pixels": 200704, "max_pixels": 1003520,
    "skip_special_tokens": True,
}


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_rows(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, list) or not value:
        raise ValueError(f"Expected a nonempty annotation list: {path}")
    return value


def row_images(row, image_root):
    images = row.get("images") or []
    if not isinstance(images, list) or not all(isinstance(image, str) for image in images):
        raise ValueError("Expected image filename list")
    if not isinstance(row.get("problem"), str) or row["problem"].count("<image>") != len(images):
        raise ValueError(f"Problem/image mismatch: {row.get('id')}")
    result = [(Path(image_root) / image).resolve() for image in images]
    for path in result:
        if not path.is_file():
            raise FileNotFoundError(path)
    return result


def prepare(data_root, train_run, output, per_benchmark=64, seed=42):
    """Select samples using no answer values, conservatively excluding shared images."""
    if per_benchmark < 1:
        raise ValueError("per_benchmark must be positive")
    output.mkdir(parents=True, exist_ok=True)
    frozen = output / "samples.json"
    if frozen.exists():
        raise FileExistsError("Refuse to replace an existing evaluation sample set")
    train_file = train_run / "pilot_samples.json"
    train_rows = read_rows(train_file)
    hash_cache = {}

    def image_hash(path):
        name = str(path)
        if name not in hash_cache:
            hash_cache[name] = digest(path)
        return hash_cache[name]

    train_ids = {str(row["id"]) for row in train_rows if row.get("id") is not None}
    train_images = set()
    for row in train_rows:
        train_images.update(image_hash(p) for p in row_images(row, data_root / "ViRL39K_images"))

    selected, audit = [], []
    for benchmark, annotation, images in BENCHMARKS:
        source = data_root / annotation
        rows = read_rows(source)
        order = list(range(len(rows)))
        random.Random(seed).shuffle(order)
        rejected = Counter()
        accepted, seen, scanned = [], set(), 0
        for index in order:
            scanned += 1
            row = rows[index]
            if row.get("id") is not None and str(row["id"]) in train_ids:
                rejected["shared_id"] += 1
                continue
            paths = row_images(row, data_root / images)
            hashes = [image_hash(p) for p in paths]
            if set(hashes) & train_images:
                rejected["shared_image_bytes"] += 1
                continue
            normalized_problem = re.sub(r"\s+", " ", row["problem"]).strip()
            key = (normalized_problem, tuple(hashes))
            if key in seen:
                rejected["duplicate_within_benchmark"] += 1
                continue
            seen.add(key)
            if row.get("answer") is None:
                raise ValueError(f"Missing evaluation answer: {benchmark}:{index}")
            accepted.append({
                "sample_key": f"{benchmark}:{index}", "benchmark": benchmark,
                "id": row.get("id"), "source_index": index,
                "problem": row["problem"], "answer": str(row["answer"]),
                "images": [str(path) for path in paths], "image_sha256": hashes,
            })
            if len(accepted) == per_benchmark:
                break
        if not accepted:
            raise ValueError(f"No eligible held-out rows: {benchmark}")
        selected.extend(accepted)
        audit.append({
            "benchmark": benchmark, "annotation": str(source), "source_sha256": digest(source),
            "source_count": len(rows), "scanned": scanned, "selected": len(accepted),
            "rejections_among_scanned": dict(rejected),
        })
    frozen.write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "data-audit.json").write_text(json.dumps({
        "train_file": str(train_file), "train_file_sha256": digest(train_file),
        "train_candidate_count": len(train_rows), "seed": seed,
        "samples_sha256": digest(frozen), "benchmarks": audit,
        "overlap_scope": "exact IDs and identical image file bytes against all selected training candidates",
        "limits": "Does not prove absence of re-encoded images or semantic/paraphrase duplicates; sampled validation, not full benchmark",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("FROZEN_EVAL_SAMPLES:", len(selected), flush=True)
    for entry in audit:
        print("BENCHMARK:", entry["benchmark"], "selected", entry["selected"],
              "scanned", entry["scanned"], "rejected", entry["rejections_among_scanned"], flush=True)
    print("EVAL_SAMPLES_SHA256:", digest(frozen), flush=True)


def infer(run, model, tokenizer_path, label):
    # Heavy imports occur only in the inference subcommand, not preparation/comparison.
    import importlib.util
    import signal

    from PIL import Image
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    frozen = run / "samples.json"
    rows = read_rows(frozen)
    audit = json.loads((run / "data-audit.json").read_text())
    assert digest(frozen) == audit["samples_sha256"], "Evaluation sample file changed"
    result_path = run / f"results.{label}.jsonl"
    if result_path.exists():
        raise FileExistsError(result_path)
    base = Path(__file__).resolve().parent
    prompt_file = base / "format_prompt/base.jinja"
    system_prompt = prompt_file.read_text(encoding="utf-8").strip()
    reward_spec = importlib.util.spec_from_file_location("cfpo_math_reward", base / "reward_function/base.py")
    reward = importlib.util.module_from_spec(reward_spec)
    reward_spec.loader.exec_module(reward)
    processor = AutoProcessor.from_pretrained(str(tokenizer_path), local_files_only=True)
    settings = dict(SETTINGS, tokenizer=str(tokenizer_path.resolve()), prompt_sha256=digest(prompt_file))
    llm = LLM(
        model=str(model), tokenizer=str(tokenizer_path), tensor_parallel_size=1,
        dtype="bfloat16", gpu_memory_utilization=0.6, max_model_len=SETTINGS["max_model_len"],
        max_num_seqs=4, max_num_batched_tokens=8192, enable_chunked_prefill=True,
        enforce_eager=True, enable_prefix_caching=True, seed=SETTINGS["seed"],
        limit_mm_per_prompt={"image": max(1, max(len(row["images"]) for row in rows))},
        mm_processor_kwargs={"min_pixels": SETTINGS["min_pixels"], "max_pixels": SETTINGS["max_pixels"]},
    )
    sampling = SamplingParams(
        n=1, temperature=0.0, top_p=1.0, max_tokens=SETTINGS["max_tokens"],
        seed=SETTINGS["seed"], skip_special_tokens=True,
    )

    def timeout_handler(signum, frame):
        raise TimeoutError("Math grading exceeded 10 seconds")

    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    try:
        with result_path.open("x", encoding="utf-8") as stream:
            for start in range(0, len(rows), SETTINGS["batch_size"]):
                batch = rows[start:start + SETTINGS["batch_size"]]
                requests, opened = [], []
                try:
                    for row in batch:
                        images = []
                        for path, expected in zip(row["images"], row["image_sha256"], strict=True):
                            assert digest(path) == expected, f"Image changed: {path}"
                            with Image.open(path) as original:
                                image = original.convert("RGB")
                            area = image.width * image.height
                            bound = min(max(area, SETTINGS["min_pixels"]), SETTINGS["max_pixels"])
                            if bound != area:
                                scale = math.sqrt(bound / area)
                                resized = image.resize((max(1, int(image.width * scale)), max(1, int(image.height * scale))))
                                image.close()
                                image = resized
                            images.append(image)
                            opened.append(image)
                        content = []
                        for index, part in enumerate(row["problem"].split("<image>")):
                            if index:
                                content.append({"type": "image"})
                            if part:
                                content.append({"type": "text", "text": part})
                        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": content}]
                        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                        request = {"prompt": prompt}
                        if images:
                            request["multi_modal_data"] = {"image": images}
                        requests.append(request)
                    outputs = llm.generate(requests, sampling, use_tqdm=False)
                    if len(outputs) != len(batch):
                        raise RuntimeError("Missing inference responses")
                    for row, output in zip(batch, outputs, strict=True):
                        if len(output.outputs) != 1:
                            raise RuntimeError("Expected exactly one answer per sample")
                        answer = output.outputs[0]
                        if answer.finish_reason not in ("stop", "length"):
                            raise RuntimeError(f"Unfinished/aborted evaluation response: {answer.finish_reason}")
                        signal.alarm(10)
                        try:
                            scores = reward.compute_score(row["benchmark"], answer.text, row["answer"])
                        finally:
                            signal.alarm(0)
                        record = {
                            "sample_key": row["sample_key"], "benchmark": row["benchmark"],
                            "ground_truth": row["answer"], "output": answer.text,
                            "finish_reason": answer.finish_reason, "output_tokens": len(answer.token_ids),
                            **scores,
                        }
                        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                    stream.flush()
                    print("EVAL_PROGRESS:", label, min(start + len(batch), len(rows)), "/", len(rows), flush=True)
                finally:
                    for image in opened:
                        image.close()
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
    assert digest(frozen) == audit["samples_sha256"]
    (run / f"meta.{label}.json").write_text(json.dumps({
        "complete": True, "label": label, "model_path": str(model.resolve()),
        "samples_sha256": digest(frozen), "expected_count": len(rows), "settings": settings,
    }, indent=2), encoding="utf-8")
    print("MODEL_EVALUATION_COMPLETE:", label, flush=True)


def compare(run):
    rows = read_rows(run / "samples.json")
    expected = {row["sample_key"] for row in rows}
    if len(expected) != len(rows):
        raise ValueError("Duplicate sample keys in frozen dataset")
    sample_hash = digest(run / "samples.json")
    reports, settings = {}, None
    for label in ("before", "after"):
        meta = json.loads((run / f"meta.{label}.json").read_text())
        if not meta.get("complete") or meta["samples_sha256"] != sample_hash or meta["expected_count"] != len(rows):
            raise ValueError(f"Incomplete/different evaluation metadata: {label}")
        if settings is None:
            settings = meta["settings"]
        elif settings != meta["settings"]:
            raise ValueError("Before/after decoding or prompt settings differ")
        with (run / f"results.{label}.jsonl").open(encoding="utf-8") as stream:
            results = [json.loads(line) for line in stream if line.strip()]
        if len(results) != len(expected) or {row["sample_key"] for row in results} != expected:
            raise ValueError(f"Incomplete/duplicate evaluation responses: {label}")
        groups = defaultdict(list)
        sample_lookup = {row["sample_key"]: row for row in rows}
        for result in results:
            source = sample_lookup[result["sample_key"]]
            if result["benchmark"] != source["benchmark"] or result["ground_truth"] != source["answer"]:
                raise ValueError("Result labels disagree with frozen sample list")
            if not all(result[key] in (0.0, 1.0) for key in ("accuracy", "format")):
                raise ValueError("Invalid binary grading result")
            groups[result["benchmark"]].append(result)
        reports[label] = {
            name: {
                "n": len(items), "accuracy": sum(item["accuracy"] for item in items) / len(items),
                "format": sum(item["format"] for item in items) / len(items),
                "length_stop": sum(item["finish_reason"] == "length" for item in items) / len(items),
            } for name, items in groups.items()
        }
    output = []
    for name in reports["before"]:
        before, after = reports["before"][name], reports["after"][name]
        output.append({"benchmark": name, "before": before, "after": after,
                       "accuracy_delta_points": 100 * (after["accuracy"] - before["accuracy"])})
    (run / "comparison.json").write_text(json.dumps({
        "scope": "sampled validation, greedy n=1; not paper Avg@8 or a complete benchmark",
        "samples_sha256": sample_hash, "settings": settings, "results": output,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    for result in output:
        print("COMPARISON:", json.dumps(result, ensure_ascii=False), flush=True)
    print("PAIRED_EVALUATION_COMPLETE", flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--data-root", type=Path, required=True)
    prep.add_argument("--train-run", type=Path, required=True)
    prep.add_argument("--run", type=Path, required=True)
    prep.add_argument("--per-benchmark", type=int, default=64)
    evaluation = sub.add_parser("infer")
    evaluation.add_argument("--run", type=Path, required=True)
    evaluation.add_argument("--model", type=Path, required=True)
    evaluation.add_argument("--tokenizer", type=Path, required=True)
    evaluation.add_argument("--label", choices=["before", "after"], required=True)
    comparison = sub.add_parser("compare")
    comparison.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.data_root, args.train_run, args.run, args.per_benchmark)
    elif args.command == "infer":
        infer(args.run, args.model, args.tokenizer, args.label)
    else:
        compare(args.run)


if __name__ == "__main__":
    main()
