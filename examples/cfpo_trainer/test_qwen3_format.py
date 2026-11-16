"""Format regression tests; math grader is stubbed only for weight checks."""
import importlib.util
import io
import json
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import MagicMock, patch

from audit_format import audit, load_format

BASE = Path(__file__).resolve().parent
legacy = load_format(BASE / "reward_function/legacy_base.py")
candidate = load_format(BASE / "reward_function/qwen3_format.py")


class FormatTests(unittest.TestCase):
    def test_screenshot_and_paired_forms(self):
        for text in [r"先分析题目条件…</think>最终答案是\boxed{42}",
                     "reasoning\n</think>\nfinal \\boxed{\\frac{1}{2}}",
                     r"<think>reasoning</think>answer \boxed{42}"]:
            with self.subTest(text=text):
                self.assertEqual(candidate(text), 1)
        self.assertEqual(legacy(r"reasoning</think>\boxed{42}"), 0)

    def test_missing_boundary_and_malformed_tags_rejected(self):
        for text in [r"\boxed{42}", r"reasoning \boxed{42}",
                     r"<think>reasoning \boxed{42}", r"reasoning</think>42",
                     r"\boxed{42}</think>final", r"x</think>y</think>\boxed{42}",
                     r"x</think><think>y\boxed{42}",
                     r"<think><think>x</think>\boxed{42}",
                     r"<think>x</think>\boxed{42}<think>"]:
            with self.subTest(text=text):
                self.assertEqual(candidate(text), 0)

    def test_weights_normalization_and_grader_interface(self):
        grader = ModuleType("mathruler.grader")
        grader.extract_boxed_content = lambda text: "42" if r"\boxed{42}" in text else "wrong"
        grader.grade_answer = lambda answer, truth: answer == truth
        with patch.dict(sys.modules, {"mathruler.grader": grader}):
            spec = importlib.util.spec_from_file_location("candidate_reward", BASE / "reward_function/qwen3_format.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        for text, truth, expected in [(r"x</think>\boxed{42}", "42", 1.0),
                                      (r"x</think>\boxed{42}", "0", .1),
                                      (r"\boxed{42}", "42", .9),
                                      (r"\boxed{42}", "0", 0.0),
                                      (r"x< / think >\boxed{42}", "42", 1.0)]:
            self.assertAlmostEqual(module.compute_score("test", text, truth)["score"], expected)
        with self.assertRaises(ValueError):
            module.compute_score("test", "x", "42", format_weight=1.1)

    def test_read_only_replay(self):
        path = MagicMock()
        original = json.dumps({"output": r"reasoning</think>\boxed{42}", "format": 0}) + "\n"
        path.open.return_value = io.StringIO(original)
        path.read_bytes.return_value = original.encode()
        counts = audit(path)["counts"]
        self.assertEqual(counts["zero_to_one"], 1)
        self.assertEqual(counts["stored_vs_legacy_mismatch"], 0)
        path.open.assert_called_once_with(encoding="utf-8")
        path.write_text.assert_not_called()
        path.write_bytes.assert_not_called()


if __name__ == "__main__":
    unittest.main()
