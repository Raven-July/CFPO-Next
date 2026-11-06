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

"""Replay numeric training metrics into a local SwanLab dashboard on a separate host."""

import argparse
import math
import re
from pathlib import Path


def parse_metrics(text):
    steps = {}
    for line in text.splitlines():
        match = re.search(r"\bstep:(\d+)\s+-\s+", line)
        if not match:
            continue
        metrics = {}
        for entry in line[match.end() :].split(" - "):
            key, sep, value = entry.partition(":")
            if not sep:
                continue
            try:
                number = float(value)
            except ValueError:
                continue
            if math.isfinite(number):
                metrics[key] = number
        if metrics:
            steps.setdefault(int(match.group(1)), {}).update(metrics)
    return steps


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", help="Copied Slurm output, not the offline binary run file")
    parser.add_argument("--logdir", required=True)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    steps = parse_metrics(Path(args.log).read_text(encoding="utf-8", errors="replace"))
    if not steps:
        raise SystemExit("No step metrics found; dashboard was not created")
    import swanlab

    run = swanlab.init(
        project="CFPO-local-replay",
        experiment_name=args.name,
        mode="local",
        logdir=args.logdir,
        config={"source_log": str(Path(args.log).resolve()), "replayed_metrics": True},
    )
    for step, metrics in sorted(steps.items()):
        run.log(metrics, step=step)
    run.finish()
    print(f"Replayed {len(steps)} steps. Start: swanlab watch {args.logdir}")
