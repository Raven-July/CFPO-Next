# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Adapters for CFPO's separate annotation JSON and image directories."""

import copy
import json
import os
from collections.abc import Mapping


def read_dataset_specs(value):
    """Recognize a CFPO descriptor, leaving ordinary JSON/JSONL data untouched."""
    if isinstance(value, Mapping):
        if "dataset" not in value or not isinstance(value["dataset"], str):
            raise ValueError("CFPO dataset descriptors require a string 'dataset' path")
        return [dict(value)]
    if isinstance(value, str) and value.endswith(".json"):
        with open(value, encoding="utf-8") as stream:
            try:
                specs = json.load(stream)
            except json.JSONDecodeError:
                # Hugging Face also accepts JSON Lines in files named .json.
                # Leave their parsing and validation on the existing data path.
                return None
        if isinstance(specs, dict) and "dataset" in specs:
            specs = [specs]
        if isinstance(specs, list) and specs and all(isinstance(item, dict) and "dataset" in item for item in specs):
            base = os.path.dirname(os.path.abspath(value))
            result = []
            for spec in specs:
                spec = read_dataset_specs(spec)[0]
                for key in ("dataset", "image"):
                    if spec.get(key):
                        path = os.path.expandvars(os.path.expanduser(spec[key]))
                        spec[key] = path if os.path.isabs(path) else os.path.join(base, path)
                result.append(spec)
            return result
    return None


def normalize_cfpo_sample(example, index, image_dir=None, prompt_key="prompt", image_key="images"):
    """Translate a problem/answer row into verl's reward and chat schema."""
    problem = example.get("problem")
    if not isinstance(problem, str) or "answer" not in example:
        raise ValueError("CFPO rows require a string 'problem' and an 'answer'")
    images = example.get(image_key) or []
    if not isinstance(images, list):
        raise ValueError("CFPO images must be a list")
    if problem.count("<image>") != len(images):
        raise ValueError(f"CFPO sample {example.get('id', index)} has mismatched image placeholders")
    resolved = []
    for image in images:
        if isinstance(image, str):
            image = os.path.expandvars(os.path.expanduser(image))
            if image_dir and not os.path.isabs(image) and "://" not in image:
                image = os.path.join(image_dir, image)
            if "://" not in image and not os.path.isfile(image):
                raise FileNotFoundError(image)
        resolved.append(image)
    extra_info = dict(example.get("extra_info") or {})
    extra_info.update({"index": index})
    for key in ("id", "type", "is_cf", "solution"):
        if key in example:
            extra_info[key] = example[key]
    return {
        prompt_key: [{"role": "user", "content": problem}],
        image_key: resolved,
        "reward_model": {"style": "rule", "ground_truth": str(example["answer"])},
        "data_source": example.get("data_source") or "cfpo",
        "extra_info": extra_info,
    }


def with_system_prompt(messages, system_prompt=None):
    """Apply the training-time system prompt without modifying dataset rows."""
    messages = copy.deepcopy(messages)
    if system_prompt:
        messages = [message for message in messages if message.get("role") != "system"]
        messages.insert(0, {"role": "system", "content": system_prompt})
    return messages
