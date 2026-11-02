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

"""Compare complete reference-evaluator results under identical sampling settings."""

import argparse
import csv
import json
import sys
from pathlib import Path


def compare(config, base, trained, n=8):
    rows = []
    for spec in json.loads(Path(config).read_text()):
        name = spec["dataset_name"]
        results = []
        for directory in (base, trained):
            paths = list(Path(directory).glob(f"*_{name}_avg@{n}_COT.json"))
            if len(paths) != 1:
                raise ValueError(f"Expected exactly one {name} result in {directory}, found {len(paths)}")
            result = json.loads(paths[0].read_text())
            expected = len(json.loads(Path(spec["json_path"]).read_text()))
            if len(result["details"]) != expected:
                raise ValueError(f"Incomplete result: {paths[0]}")
            results.append(result)
        before, after = [r["accuracies"][f"all_avg@{n}_acc"] for r in results]
        rows.append(
            {
                "dataset": name,
                "samples": expected,
                "base_percent": before,
                "trained_percent": after,
                "delta_points": after - before,
            }
        )
    writer = csv.DictWriter(sys.stdout, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    print(
        f"# macro base={sum(r['base_percent'] for r in rows) / len(rows):.4f}, "
        f"trained={sum(r['trained_percent'] for r in rows) / len(rows):.4f}"
    )
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--trained", required=True)
    parser.add_argument("--n", type=int, default=8)
    args = parser.parse_args()
    compare(args.config, args.base, args.trained, args.n)
