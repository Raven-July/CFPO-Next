"""Read-only diagnostics for existing paired results; Python 3.6+, no GPU/imports.

Prints aggregate counts and deterministic example keys, never answer text/images.
Does not change samples, grades, rewards, or any existing run artifacts.
"""

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analyze(run):
    samples = json.loads((run / "samples.json").read_text(encoding="utf-8"))
    expected = {row["sample_key"]: row for row in samples}
    if not samples or len(expected) != len(samples):
        raise ValueError("Empty/duplicate frozen samples")
    sample_hash = sha256(run / "samples.json")
    audit = json.loads((run / "data-audit.json").read_text(encoding="utf-8"))
    if audit["samples_sha256"] != sample_hash:
        raise ValueError("Sample hash differs from data audit")
    pair, metadata = {}, {}
    for label in ("before", "after"):
        meta = json.loads((run / ("meta." + label + ".json")).read_text(encoding="utf-8"))
        if (not meta.get("complete") or meta["expected_count"] != len(samples)
                or meta["samples_sha256"] != sample_hash or meta.get("label") != label):
            raise ValueError("Incomplete/mismatched metadata: " + label)
        metadata[label] = meta
        rows = [json.loads(line) for line in
                (run / ("results." + label + ".jsonl")).read_text(encoding="utf-8").splitlines()
                if line.strip()]
        by_key = {row["sample_key"]: row for row in rows}
        if len(rows) != len(expected) or len(by_key) != len(rows) or set(by_key) != set(expected):
            raise ValueError("Incomplete/duplicate result keys: " + label)
        for key, row in by_key.items():
            source = expected[key]
            if row["benchmark"] != source["benchmark"] or row["ground_truth"] != source["answer"]:
                raise ValueError("Frozen/result labels disagree: " + key)
            if any(row[name] not in (0.0, 1.0) for name in ("accuracy", "format")):
                raise ValueError("Nonbinary grade: " + key)
            if row["finish_reason"] not in ("stop", "length"):
                raise ValueError("Invalid finish reason: " + key)
            if not isinstance(row["output"], str) or not isinstance(row["output_tokens"], int):
                raise ValueError("Missing output/token count: " + key)
        pair[label] = by_key
    if metadata["before"]["settings"] != metadata["after"]["settings"]:
        raise ValueError("Before/after settings differ")
    reference_format = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
    groups = sorted({row["benchmark"] for row in samples})
    report = {"samples_sha256": sample_hash, "scope": "Existing outputs only; no regrading or inference",
              "settings": metadata["before"]["settings"], "groups": {}}
    for group in [None] + groups:
        keys = sorted(key for key, row in expected.items() if group is None or row["benchmark"] == group)
        transitions, finish_changes, examples = Counter(), Counter(), {}
        for key in keys:
            before, after = pair["before"][key], pair["after"][key]
            transition = ("correct" if before["accuracy"] else "wrong") + "_to_" + (
                "correct" if after["accuracy"] else "wrong")
            transitions[transition] += 1
            finish_changes[transition + "/" + before["finish_reason"] + "_to_" + after["finish_reason"]] += 1
            examples.setdefault(transition, [])
            if len(examples[transition]) < 3:
                examples[transition].append(key)
        item = {"n": len(keys), "transitions": dict(transitions),
                "transitions_by_finish_reason": dict(finish_changes), "example_keys_first_3": examples}
        for label in ("before", "after"):
            counts = Counter()
            buckets = {"stop": [], "length": []}
            for key in keys:
                row = pair[label][key]
                normalized = re.sub(r"\s*(<|>|/)\s*", r"\1", row["output"])
                opening, closing = "<think>" in normalized, "</think>" in normalized
                category = ("both_tags" if opening and closing else "opening_only" if opening
                            else "closing_only" if closing else "neither_tag")
                counts[category] += 1
                counts["boxed_marker"] += int("\\boxed{" in normalized)
                matched = int(reference_format.fullmatch(normalized) is not None)
                counts["reference_format_match"] += matched
                if matched != row["format"]:
                    raise ValueError("Stored/reference format disagree: " + key)
                counts["correct"] += int(row["accuracy"])
                counts["output_tokens_total"] += row["output_tokens"]
                buckets[row["finish_reason"]].append(row)
            counts["mean_output_tokens"] = counts["output_tokens_total"] / len(keys)
            item[label] = dict(counts)
            item[label]["by_finish_reason"] = {
                reason: {"n": len(rows), "correct": sum(int(row["accuracy"]) for row in rows)}
                for reason, rows in buckets.items()}
        report["groups"][group if group is not None else "ALL"] = item
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.run), ensure_ascii=False, indent=2))
