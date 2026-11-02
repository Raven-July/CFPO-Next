# Copyright 2026 CFPO-Next contributors
# Licensed under the Apache License, Version 2.0.
"""CFPO JSON/image adapters; no model or accelerator dependencies."""

import copy
import json
import os
from collections.abc import Mapping
from pathlib import Path


def resolve_spec(spec: Mapping, base_dir: str) -> dict:
    """Resolve a dataset/image entry relative to its manifest (or the cwd)."""
    if not isinstance(spec.get("dataset"), str) or not spec["dataset"]:
        raise ValueError("CFPO dataset must be a nonempty local file path")
    image = spec.get("image")
    if image is not None and (not isinstance(image, str) or not image):
        raise ValueError("CFPO image must be a nonempty directory path or null")
    unknown = set(spec) - {"dataset", "image"}
    if unknown:
        raise ValueError(f"Unknown CFPO dataset configuration keys: {sorted(unknown)}")

    def absolute(value):
        if value is None:
            return None
        path = Path(os.path.expanduser(value))
        return str(path if path.is_absolute() else Path(base_dir) / path)

    return {"dataset": absolute(spec["dataset"]), "image": absolute(image)}


def read_manifest(path: str, base_dir: str) -> list[dict] | None:
    """Recognize a local JSON list of dataset/image entries, not a sample JSON."""
    if not path.endswith(".json"):
        return None
    try:
        with open(path, encoding="utf-8") as stream:
            payload = json.load(stream)
    except (ValueError, UnicodeError):
        return None  # Leave invalid/other JSON handling to the original loader.
    if not isinstance(payload, list) or not payload:
        return None
    if not isinstance(payload[0], dict) or not {"dataset", "image"}.issubset(payload[0]):
        return None
    if not all(isinstance(item, dict) for item in payload):
        raise ValueError("All CFPO manifest entries must be objects")
    return [resolve_spec(item, base_dir) for item in payload]


def normalize_row(
    row: dict, index: int, *, image_root: str | None, source: str,
    prompt_key: str = "prompt", image_key: str = "images", answer_key: str = "answer",
) -> dict:
    """Translate a CFPO problem into verl input and separate reward labels."""
    problem = row.get("problem", row.get(prompt_key))
    if not isinstance(problem, str):
        raise ValueError(f"CFPO sample {index} needs a string problem or {prompt_key}")
    if answer_key not in row or row[answer_key] is None:
        raise ValueError(f"CFPO sample {index} has no {answer_key}")
    images = row.get(image_key) or []
    if not isinstance(images, list) or not all(isinstance(image, str) for image in images):
        raise ValueError(f"CFPO sample {index} images must be a list of file paths")
    if problem.count("<image>") != len(images):
        raise ValueError(f"CFPO sample {index} has different image and placeholder counts")
    paths = []
    for image in images:
        path = Path(os.path.expanduser(image))
        if not path.is_absolute():
            if image_root is None:
                raise ValueError(f"CFPO sample {index} needs an image directory for {image}")
            path = Path(image_root) / path
        if not path.is_file():
            raise FileNotFoundError(f"CFPO sample {index}: {path}")
        paths.append(str(path))
    extra = copy.deepcopy(row.get("extra_info") or {})
    extra["index"] = row.get("id", index)
    for field in ("id", "type", "is_cf"):
        if field in row:
            extra[field] = row[field]
    return {
        prompt_key: [{"role": "user", "content": problem}],
        image_key: paths,
        "data_source": row.get("data_source") or source,
        "reward_model": {"style": "rule", "ground_truth": str(row[answer_key])},
        "extra_info": extra,
    }


def add_system_prompt(messages: list, system_prompt: str | None) -> list:
    """Copy messages and add the configured prompt once; reject conflicts."""
    result = copy.deepcopy(messages)
    if not system_prompt:
        return result
    existing = [message for message in result if message.get("role") == "system"]
    if existing:
        if len(existing) != 1 or result[0].get("role") != "system":
            raise ValueError("Expected at most one system message, in first position")
        if existing[0].get("content") != system_prompt:
            raise ValueError("Dataset system prompt conflicts with data.format_prompt")
        return result
    return [{"role": "system", "content": system_prompt}, *result]
