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

import copy
import json

import datasets
import pytest
from omegaconf import OmegaConf
from PIL import Image

from examples.cfpo_trainer.reward import compute_score
from verl.utils.dataset.rl_dataset import RLHFDataset


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return list(range(sum(len(message["content"]) for message in messages)))


def config(**kwargs):
    return OmegaConf.create({"filter_overlong_prompts": False, "filter_overlong_prompts_workers": 1, **kwargs})


def write_rows(tmp_path, name="train.json", **overrides):
    row = {"problem": "What is 1+1?", "answer": "2", "images": [], "id": name, "type": "math", "is_cf": False}
    row.update(overrides)
    path = tmp_path / name
    path.write_text(json.dumps([row]), encoding="utf-8")
    return path


def test_descriptor_file_multiple_sources_and_images(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    Image.new("RGB", (28, 28)).save(image_dir / "one.png")
    write_rows(tmp_path, problem="<image>Count", images=["one.png"])
    write_rows(tmp_path, "other.json")
    descriptor = tmp_path / "datasets.json"
    descriptor.write_text(
        json.dumps([{"dataset": "train.json", "image": "images"}, {"dataset": "other.json", "image": "images"}])
    )
    ds = RLHFDataset(str(descriptor), Tokenizer(), config(system_prompt="Solve"), processor=object())
    assert len(ds) == 2
    row = ds[0]
    assert row["raw_prompt"][0] == {"role": "system", "content": [{"type": "text", "text": "Solve"}]}
    assert row["raw_prompt"][1]["content"][0]["image"] == str(image_dir / "one.png")
    assert row["reward_model"]["ground_truth"] == "2"
    assert row["extra_info"]["id"] == "train.json"
    ds.resume_dataset_state()
    assert ds[0]["raw_prompt"] == row["raw_prompt"]


def test_inline_omegaconf_descriptor(tmp_path):
    path = write_rows(tmp_path)
    sources = OmegaConf.create([{"dataset": str(path)}])
    ds = RLHFDataset(sources, Tokenizer(), config())
    assert ds[0]["raw_prompt"] == [{"role": "user", "content": "What is 1+1?"}]


def test_single_descriptor_object_file(tmp_path):
    write_rows(tmp_path)
    descriptor = tmp_path / "datasets.json"
    descriptor.write_text(json.dumps({"dataset": "train.json"}))
    ds = RLHFDataset(str(descriptor), Tokenizer(), config())
    assert ds[0]["reward_model"]["ground_truth"] == "2"


def test_json_lines_with_json_extension_remains_supported(tmp_path):
    row = {"prompt": [{"role": "user", "content": "Hi"}], "data_source": "original"}
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
    ds = RLHFDataset(str(path), Tokenizer(), config())
    assert len(ds) == 2
    assert ds[0]["raw_prompt"] == row["prompt"]


@pytest.mark.parametrize("suffix", ["json", "jsonl", "parquet"])
def test_existing_verl_formats_unchanged(tmp_path, suffix):
    rows = [
        {
            "prompt": [{"role": "system", "content": "Original"}, {"role": "user", "content": "Hi"}],
            "reward_model": {"ground_truth": "2"},
            "data_source": "original",
        }
    ]
    path = tmp_path / f"legacy.{suffix}"
    if suffix == "parquet":
        datasets.Dataset.from_list(rows).to_parquet(str(path))
    elif suffix == "jsonl":
        path.write_text(json.dumps(rows[0]) + "\n")
    else:
        path.write_text(json.dumps(rows))
    ds = RLHFDataset(str(path), Tokenizer(), config())
    assert ds[0]["raw_prompt"] == rows[0]["prompt"]
    assert ds[0]["data_source"] == "original"


def test_prompt_path_and_text_length_filter(tmp_path):
    path = write_rows(tmp_path)
    system = tmp_path / "system.txt"
    system.write_text("A long system instruction")
    source = [{"dataset": str(path)}]
    ds = RLHFDataset(source, Tokenizer(), config(system_prompt_path=str(system)))
    assert ds[0]["raw_prompt"][0]["content"] == system.read_text()
    filtered = RLHFDataset(
        source, Tokenizer(), config(system_prompt_path=str(system), filter_overlong_prompts=True, max_prompt_length=15)
    )
    assert len(filtered) == 0
    with pytest.raises(ValueError, match="only one"):
        RLHFDataset(source, Tokenizer(), config(system_prompt="x", system_prompt_path=str(system)))


def test_build_messages_does_not_mutate_rows(tmp_path):
    ds = RLHFDataset(
        [{"dataset": str(write_rows(tmp_path))}], Tokenizer(), config(system_prompt="New"), processor=object()
    )
    row = {
        "prompt": [{"role": "system", "content": "Old"}, {"role": "user", "content": "<image>Hi"}],
        "images": ["a.png"],
    }
    original = copy.deepcopy(row)
    first = ds._build_messages(row, "prompt")
    assert ds._build_messages(row, "prompt") == first
    assert row == original
    assert first[0]["content"] == [{"type": "text", "text": "New"}]


@pytest.mark.parametrize(
    "problem,images,error",
    [
        ("<image>Hi", [], ValueError),
        ("Hi", ["missing.png"], ValueError),
        ("<image>Hi", ["missing.png"], FileNotFoundError),
    ],
)
def test_invalid_images_fail_early(tmp_path, problem, images, error):
    path = write_rows(tmp_path, problem=problem, images=images)
    with pytest.raises(error):
        RLHFDataset([{"dataset": str(path), "image": str(tmp_path)}], Tokenizer(), config())


@pytest.mark.parametrize(
    "response,truth,score,accuracy,format_score",
    [
        (r"<think>Compute</think>\boxed{2}", "2", 1.0, 1.0, 1.0),
        (r"<think>Compute</think>\boxed{3}", "2", 0.1, 0.0, 1.0),
        (r"\boxed{2}", "2", 0.9, 1.0, 0.0),
        ("No answer", "2", 0.0, 0.0, 0.0),
        (r"<think>Compute</think>\boxed{\frac{1}{2}}", "0.5", 1.0, 1.0, 1.0),
    ],
)
def test_original_reward_semantics(response, truth, score, accuracy, format_score):
    result = compute_score("cfpo", response, truth)
    assert result == {"score": score, "overall": score, "accuracy": accuracy, "format": format_score}


def test_invalid_reward_weight():
    with pytest.raises(ValueError):
        compute_score("cfpo", "", "2", format_weight=2)
