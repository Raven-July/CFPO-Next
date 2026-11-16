"""Compare format rules on saved JSONL outputs; no model, GPU or regrading.

Python 3.6+. Only aggregate counts and input hashes are printed.
"""
import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import re


def load_format(path):
    # Load only the format function, avoiding mathruler and model dependencies.
    tree = ast.parse(path.read_text(encoding="utf-8"))
    tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "format_reward"]
    if len(tree.body) != 1:
        raise ValueError("Expected exactly one format_reward: " + str(path))
    namespace = {"re": re}
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace["format_reward"]


def audit(path):
    base = Path(__file__).resolve().parent / "reward_function"
    old = load_format(base / "legacy_base.py")
    new = load_format(base / "base.py")
    counts = Counter()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row.get("output"), str):
                raise ValueError("Missing string output at line " + str(line_number))
            text = re.sub(r"\s*(<|>|/)\s*", r"\1", row["output"])
            opening, closing = "<think>" in text, "</think>" in text
            category = "both" if opening and closing else "opening_only" if opening else "closing_only" if closing else "neither"
            counts[category] += 1
            counts["n"] += 1
            before, after = old(text), new(text)
            counts["legacy_format"] += int(before)
            counts["current_format"] += int(after)
            counts["zero_to_one"] += int(after > before)
            counts["one_to_zero"] += int(after < before)
            if "format" in row:
                counts["stored_vs_legacy_mismatch"] += int(row["format"] != before)
    if not counts["n"]:
        raise ValueError("No output rows: " + str(path))
    return {"input": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "scope": "Format-only replay; stored grades unchanged; no accuracy regrading",
            "counts": dict(counts)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", type=Path)
    args = parser.parse_args()
    print(json.dumps([audit(path) for path in args.files], ensure_ascii=False, indent=2))
