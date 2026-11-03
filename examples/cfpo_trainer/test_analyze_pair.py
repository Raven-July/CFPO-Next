"""Synthetic read-only diagnostic tests; no model dependencies or GPU evidence."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location("pair_analysis", Path(__file__).with_name("analyze_pair.py"))
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.run = Path(self.temp.name)
        samples = [{"sample_key": "Toy:" + str(i), "benchmark": "Toy", "answer": "A"} for i in range(2)]
        (self.run / "samples.json").write_text(json.dumps(samples))
        sha = analysis.sha256(self.run / "samples.json")
        (self.run / "data-audit.json").write_text(json.dumps({"samples_sha256": sha}))
        for label in ("before", "after"):
            meta = dict(complete=True, expected_count=2, samples_sha256=sha, label=label, settings={"n": 1})
            (self.run / ("meta." + label + ".json")).write_text(json.dumps(meta))
            rows = [dict(sample_key=s["sample_key"], benchmark="Toy", ground_truth="A", format=0.0,
                         accuracy=float(i == int(label == "after")), output="fixture-only \\boxed{A}",
                         output_tokens=5, finish_reason="length" if i == 0 else "stop")
                    for i, s in enumerate(samples)]
            (self.run / ("results." + label + ".jsonl")).write_text(
                "\n".join(json.dumps(row) for row in rows))

    def test_detects_offsetting_changes_and_retains_truncation(self):
        report = analysis.analyze(self.run)
        group = report["groups"]["ALL"]
        self.assertEqual(group["transitions"], {"correct_to_wrong": 1, "wrong_to_correct": 1})
        self.assertEqual(group["transitions_by_finish_reason"],
                         {"correct_to_wrong/length_to_length": 1, "wrong_to_correct/stop_to_stop": 1})
        self.assertEqual(group["before"]["neither_tag"], 2)
        self.assertEqual(group["before"]["by_finish_reason"]["length"], {"n": 1, "correct": 1})
        self.assertNotIn("fixture-only", json.dumps(report))

    def test_duplicate_result_is_rejected(self):
        p = self.run / "results.after.jsonl"
        first = p.read_text().splitlines()[0]
        p.write_text(first + "\n" + first)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            analysis.analyze(self.run)

    def test_format_grades_are_checked_without_rewriting(self):
        p = self.run / "results.after.jsonl"
        p.write_text(p.read_text().replace('"format": 0.0', '"format": 1.0'))
        before_bytes = p.read_bytes()
        with self.assertRaisesRegex(ValueError, "format disagree"):
            analysis.analyze(self.run)
        self.assertEqual(p.read_bytes(), before_bytes)


if __name__ == "__main__":
    unittest.main(verbosity=2)
