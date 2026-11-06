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

import json
import zipfile
from pathlib import Path

from examples.cfpo_trainer.compare_evaluations import compare
from examples.cfpo_trainer.inspect_generations import inspect
from examples.cfpo_trainer.prepare_data import BENCHMARKS, extract, prepare
from examples.cfpo_trainer.replay_dashboard import parse_metrics


def test_prepare_all_configs_and_archive_layouts(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    row = {"problem": "<image>Count", "answer": "1", "images": ["one.png"]}
    (raw / "ViRL39K_train.json").write_text(json.dumps([row]))
    archives = {"ViRL39K"} | {b[2] for b in BENCHMARKS}
    for i, name in enumerate(sorted(archives)):
        with zipfile.ZipFile(raw / f"{name}_images.zip", "w") as z:
            z.writestr("one.png" if i % 2 else f"{name}/one.png", b"test image")
    for i, (val, test, _) in enumerate(BENCHMARKS):
        (raw / f"{test}_test.json").write_text(json.dumps([row]))
        if i < 5:
            (raw / f"{val}_val_V3.json").write_text(json.dumps([row]))
    output = tmp_path / "configs"
    prepare(raw, output)
    validation = json.loads((output / "datasets_val_full.json").read_text())
    evaluation = json.loads((output / "datasets_eval_full.json").read_text())
    assert len(validation) == 5
    assert len(evaluation) == 10
    assert {x["data_source"] for x in validation} == {b[0] for b in BENCHMARKS[:5]}
    assert all((Path(x["image_root"]) / "one.png").is_file() for x in evaluation)
    prepare(raw, output)  # Repeating preparation must preserve extracted files/configs.


def test_unsafe_archive_rejected(tmp_path):
    import pytest

    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("../outside.txt", "unsafe")
    with pytest.raises(ValueError, match="Unsafe"):
        extract(archive, tmp_path / "target")


def test_inspect_format_failures(tmp_path):
    path = tmp_path / "generations.jsonl"
    rows = [
        {"input": "The user asks a question", "output": r"<think>Compute</think>\boxed{2}"},
        {"input": "question", "output": r"The answer is \boxed{2}"},
    ]
    path.write_text("\n".join(json.dumps(x) for x in rows))
    result = inspect(path, show=0)
    assert result["samples"] == 2
    assert result["format"] == 1
    assert result["boxed"] == 2
    assert result["system_in_input"] == 1


def test_replay_metrics_handles_prefix_and_duplicates():
    text = "(TaskRunner pid=1) step:2 - actor/grad_norm:1.5 - val/accuracy:0.25 - bad:text\n"
    assert parse_metrics(text + text) == {2: {"actor/grad_norm": 1.5, "val/accuracy": 0.25}}


def test_comparison_requires_complete_results(tmp_path):
    import pytest

    annotation = tmp_path / "test.json"
    annotation.write_text(json.dumps([{"answer": "A"}]))
    config = tmp_path / "config.json"
    config.write_text(json.dumps([{"dataset_name": "test", "json_path": str(annotation)}]))
    for label, accuracy in [("base", 25), ("trained", 50)]:
        directory = tmp_path / label
        directory.mkdir()
        result = {"details": [{"answer": "A"}], "accuracies": {"all_avg@8_acc": accuracy}}
        (directory / "results_test_avg@8_COT.json").write_text(json.dumps(result))
    rows = compare(config, tmp_path / "base", tmp_path / "trained")
    assert rows[0]["delta_points"] == 25
    (tmp_path / "trained/results_test_avg@8_COT.json").write_text(json.dumps({"details": []}))
    with pytest.raises(ValueError, match="Incomplete"):
        compare(config, tmp_path / "base", tmp_path / "trained")
