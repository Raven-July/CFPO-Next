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

"""Inspect saved verl generations without changing prompts or reward rules."""

import argparse
import json
import re
from collections import Counter
from pathlib import Path


def inspect(path, show=2):
    counts = Counter()
    for i, line in enumerate(Path(path).open(encoding="utf-8")):
        row = json.loads(line)
        response = row["output"]
        normalized = re.sub(r"\s*(<|>|/)\s*", r"\1", response)
        flags = {
            "think_open": "<think>" in response,
            "think_close": "</think>" in response,
            "boxed": r"\boxed{" in response,
            "format": re.fullmatch(r"<think>.*</think>.*\\boxed\{.*\}.*", normalized, re.DOTALL) is not None,
            "system_in_input": "The user asks a question" in row.get("input", ""),
        }
        counts["samples"] += 1
        counts.update({key: int(value) for key, value in flags.items()})
        if i < show:
            print(
                json.dumps(
                    {
                        "sample": i,
                        "flags": flags,
                        "input": row.get("input"),
                        "output": response,
                        "gts": row.get("gts"),
                        "score": row.get("score"),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
    print(json.dumps(dict(counts), ensure_ascii=False, indent=2))
    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path")
    parser.add_argument("--show", type=int, default=2)
    args = parser.parse_args()
    inspect(args.path, args.show)
