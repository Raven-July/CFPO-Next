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

"""Extract CFPO images safely and build the original five-way validation and ten-way evaluation configs."""

import argparse
import json
import zipfile
from pathlib import Path

BENCHMARKS = [
    ("geometry3k", "geometry3k", "geometry3k"),
    ("LogicVista", "LogicVista", "LogicVista"),
    ("MathVerse", "MathVerse-closed", "MathVerse"),
    ("C-VQA-Real", "C-VQA-Real", "COCO_val2014"),
    ("MARS_Bench", "MARS_Bench", "COCO_val2014"),
    ("WeMath", "WeMath", "WeMath"),
    ("MMK12", "MMK12", "MMK12"),
    ("MMMUPro-Vision", "MMMUPro-Vision", "MMMUPro-Vision"),
    ("POPE", "POPE", "COCO_val2014"),
    ("Textvqa", "Textvqa", "Textvqa"),
]


def read(path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def extract(archive, target):
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        base = target.resolve()
        for i, item in enumerate(z.infolist()):
            output = (base / item.filename).resolve()
            if not output.is_relative_to(base):
                raise ValueError(f"Unsafe archive path: {item.filename}")
            if not item.is_dir() and (not output.is_file() or output.stat().st_size != item.file_size):
                z.extract(item, target)
            if i % 5000 == 0:
                print(f"{archive.name}: {i}/{len(z.infolist())}", flush=True)


def find_image_root(extracted, rows):
    names = [name for row in rows for name in (row.get("images") or [])]
    if not names:
        return extracted
    # Archives may contain a top-level directory or bare image files.
    candidates = [extracted] + [p for p in extracted.iterdir() if p.is_dir()]
    for candidate in candidates:
        if all((candidate / name).is_file() for name in names):
            return candidate
    raise FileNotFoundError(f"Image paths do not match {extracted}; sample: {names[:3]}")


def prepare(raw, output):
    raw, output = Path(raw).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    specs = []
    eval_specs = []
    image_dirs = {}
    train_rows = read(raw / "ViRL39K_train.json")
    for archive in sorted(raw.glob("*_images.zip")):
        destination = raw / "extracted" / archive.stem
        print(f"Extracting {archive.name}", flush=True)
        extract(archive, destination)
        image_dirs[archive.stem.removesuffix("_images")] = destination
    train_root = find_image_root(image_dirs["ViRL39K"], train_rows)
    train = [{"dataset": str(raw / "ViRL39K_train.json"), "image": str(train_root), "data_source": "ViRL39K"}]
    for val_name, test_name, archive in BENCHMARKS:
        val_path = raw / f"{val_name}_val_V3.json"
        test_path = raw / f"{test_name}_test.json"
        test_rows = read(test_path)
        for row in test_rows:
            if row["problem"].count("<image>") != len(row.get("images") or []):
                raise ValueError(f"Image placeholder mismatch: {test_path}: {row.get('id')}")
        image_root = find_image_root(image_dirs[archive], test_rows)
        if val_path.is_file():
            find_image_root(image_dirs[archive], read(val_path))
            specs.append({"dataset": str(val_path), "image": str(image_root), "data_source": val_name})
        eval_specs.append({"dataset_name": val_name, "json_path": str(test_path), "image_root": str(image_root)})
        print(f"Verified {val_name}: {len(test_rows)} test rows", flush=True)
    for filename, value in (
        ("datasets_train_full.json", train),
        ("datasets_val_full.json", specs),
        ("datasets_eval_full.json", eval_specs),
    ):
        path = output / filename
        path.write_text(json.dumps(value, indent=2), encoding="utf-8")
        print(f"Saved {path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    prepare(args.raw, args.output)
