"""Dependency-free contract tests: python examples/cfpo_trainer/test_adapter.py."""

import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("cfpo_adapter", ROOT / "verl/utils/dataset/cfpo.py")
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "images").mkdir()
        (self.root / "images/a.jpg").write_bytes(b"path-only fixture; no image decoding in these tests")
        self.row = {
            "problem": "<image>\nWhich option?",
            "images": ["a.jpg"],
            "answer": "ANSWER_MUST_NOT_LEAK",
            "id": "sample-1",
            "type": "geometry",
            "is_cf": False,
        }

    def convert(self, row=None):
        return adapter.normalize_row(
            self.row if row is None else row, 0,
            image_root=str(self.root / "images"), source="train",
        )

    def test_labels_stay_out_of_prompt_and_input_unchanged(self):
        before = copy.deepcopy(self.row)
        result = self.convert()
        self.assertNotIn(self.row["answer"], json.dumps(result["prompt"]))
        self.assertNotIn("answer", result)
        self.assertNotIn("problem", result)
        self.assertEqual(result["reward_model"]["ground_truth"], self.row["answer"])
        self.assertEqual(result["extra_info"]["index"], "sample-1")
        self.assertFalse(result["extra_info"]["is_cf"])
        self.assertTrue(Path(result["images"][0]).is_absolute())
        self.assertEqual(self.row, before)

    def test_missing_image_fails(self):
        row = dict(self.row, images=["missing.jpg"])
        with self.assertRaises(FileNotFoundError):
            self.convert(row)

    def test_placeholder_mismatch_fails(self):
        with self.assertRaises(ValueError):
            self.convert(dict(self.row, problem="No image placeholder"))

    def test_missing_answer_fails(self):
        row = dict(self.row)
        del row["answer"]
        with self.assertRaises(ValueError):
            self.convert(row)

    def test_empty_images_and_numeric_zero_answer(self):
        result = self.convert(dict(self.row, problem="Compute 1-1", images=[], answer=0))
        self.assertEqual(result["reward_model"]["ground_truth"], "0")
        self.assertEqual(result["images"], [])

    def test_manifest_and_native_json_are_distinguished(self):
        path = self.root / "config.json"
        path.write_text(json.dumps([{"dataset": "train.json", "image": "images"}]))
        result = adapter.read_manifest(str(path), str(self.root))
        self.assertEqual(result[0]["dataset"], str(self.root / "train.json"))
        self.assertEqual(result[0]["image"], str(self.root / "images"))
        path.write_text(json.dumps([self.row]))
        self.assertIsNone(adapter.read_manifest(str(path), str(self.root)))
        self.assertIsNone(adapter.read_manifest(str(self.root / "x.parquet"), str(self.root)))

    def test_multiple_image_roots(self):
        path = self.root / "config.json"
        path.write_text(json.dumps([
            {"dataset": "first.json", "image": "images"},
            {"dataset": "second.json", "image": "second-images"},
        ]))
        result = adapter.read_manifest(str(path), str(self.root))
        self.assertNotEqual(result[0]["image"], result[1]["image"])

    def test_prompt_is_added_once_and_does_not_mutate(self):
        messages = self.convert()["prompt"]
        before = copy.deepcopy(messages)
        first = adapter.add_system_prompt(messages, "Team prompt")
        second = adapter.add_system_prompt(first, "Team prompt")
        self.assertEqual(first, second)
        self.assertEqual(messages, before)
        self.assertEqual([item["role"] for item in first], ["system", "user"])
        second[1]["content"] = "changed"
        self.assertNotEqual(second, first)

    def test_conflicting_prompt_fails(self):
        with self.assertRaises(ValueError):
            adapter.add_system_prompt([{"role": "system", "content": "Other"}], "Team prompt")


if __name__ == "__main__":
    unittest.main(verbosity=2)
