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

"""Export recorded metrics and saved model outputs for experiment review."""

import argparse
import csv
import re
from collections import defaultdict
from contextlib import redirect_stdout
from pathlib import Path
from shutil import copyfile

from inspect_generations import inspect
from replay_dashboard import parse_metrics


def source(value):
    label, sep, path = value.partition("=")
    if not sep or not path or not re.fullmatch(r"[A-Za-z0-9_-]+", label):
        raise argparse.ArgumentTypeError("Expected LABEL=PATH with an alphanumeric label")
    return label, Path(path)


def plot_curves(runs, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    metrics = (
        ("critic/score/mean", "Mean training score", 1),
        ("actor/grad_norm", "Actor gradient norm", 1),
        ("response_length/clip_ratio", "Responses at token limit (%)", 100),
        ("val-aux/cfpo/accuracy/mean@1", "Pilot / smoke validation accuracy (%)", 100),
    )
    for ax, (key, title, scale) in zip(axes.flat, metrics, strict=True):
        for label, steps in runs.items():
            points = [(step, scale * row[key]) for step, row in sorted(steps.items()) if key in row]
            if points:
                x, y = zip(*points, strict=True)
                ax.plot(x, y, marker="o", markersize=2.5, linewidth=1.2, label=label)
        ax.set(title=title, xlabel="Global step")
        ax.grid(alpha=0.25)
        if ax.lines:
            ax.legend(fontsize=8)
    fig.savefig(output / "training_curves.png", dpi=180)
    plt.close(fig)


def validation_table(runs, output, plots):
    keys = sorted(
        {
            key
            for steps in runs.values()
            for row in steps.values()
            for key in row
            if re.fullmatch(r"val-aux/[^/]+/accuracy/mean@1", key)
        }
    )
    if not keys:
        return
    headers = ["Run", "Step", *(key.split("/")[1] for key in keys), "Macro avg"]
    rows = []
    for label, steps in runs.items():
        for step, metrics in sorted(steps.items()):
            values = [metrics.get(key) for key in keys]
            present = [x for x in values if x is not None]
            if present:
                benchmarks = [metrics.get(key) for key in keys if key.split("/")[1] != "cfpo"]
                mean = (
                    f"{100 * sum(benchmarks) / len(benchmarks):.2f}"
                    if benchmarks and all(x is not None for x in benchmarks)
                    else "—"
                )
                rows.append([label, step, *[f"{100 * x:.2f}" if x is not None else "—" for x in values], mean])
    with (output / "validation_accuracy.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        csv.writer(stream).writerows([headers, *rows])
    if plots:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(max(11, 1.6 * len(headers)), 1.1 + 0.43 * min(len(rows), 16)))
        ax.axis("off")
        ax.set_title("Recorded validation accuracy (%) | missing values left blank", loc="left", pad=12)
        table = ax.table(
            cellText=rows[-16:],
            colLabels=headers,
            cellLoc="center",
            loc="center",
            colWidths=[0.23, 0.06, *([0.71 / (len(headers) - 2)] * (len(headers) - 2))],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.55)
        for (r, _), cell in table.get_celld().items():
            cell.set_edgecolor("white")
            cell.set_facecolor("#244369" if r == 0 else "#edf4fa" if r % 2 else "#f6f8fa")
            if r == 0:
                cell.get_text().set_color("white")
        fig.savefig(output / "validation_accuracy.png", dpi=180, bbox_inches="tight", pad_inches=0.25)
        plt.close(fig)


def export(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    runs = defaultdict(dict)
    for label, path in args.run:
        for step, metrics in parse_metrics(path.read_text(encoding="utf-8", errors="replace")).items():
            runs[label].setdefault(step, {}).update(metrics)
    if not runs:
        raise SystemExit("No recorded step metrics found")
    with (output / "metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["run", "step", "metric", "value"])
        for label, steps in runs.items():
            for step, metrics in sorted(steps.items()):
                writer.writerows((label, step, key, value) for key, value in sorted(metrics.items()))
    if not args.tables_only:
        plot_curves(runs, output)
    validation_table(runs, output, not args.tables_only)
    for label, path in args.generations:
        copyfile(path, output / f"generations_{label}.jsonl")
        with (output / f"cot_{label}.txt").open("w", encoding="utf-8") as stream, redirect_stdout(stream):
            inspect(path, show=3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", type=source, required=True, help="LABEL=Slurm log; repeat in order")
    parser.add_argument("--generations", action="append", type=source, default=[], help="LABEL=validation JSONL")
    parser.add_argument("--output", required=True)
    parser.add_argument("--tables-only", action="store_true", help="Skip PNG plots and matplotlib dependency")
    export(parser.parse_args())
