"""Standard-library tests for sample isolation and completeness; no inference claims."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

spec = importlib.util.spec_from_file_location("paired_eval", Path(__file__).with_name("evaluate_pair.py"))
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


class EvaluationContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.train_run = self.root / "train"
        self.train_run.mkdir()
        self.output = self.root / "evaluation"
        self.data = self.root / "data"
        for folder in ("ViRL39K_images", "toy_images"):
            (self.data / folder).mkdir(parents=True)
        # Hash/path fixtures only. Actual PIL image decoding happens in inference.
        (self.data / "ViRL39K_images/train.jpg").write_bytes(b"training-image")
        for name, payload in (("copy.jpg", b"training-image"), ("a.jpg", b"validation-a"), ("b.jpg", b"validation-b")):
            (self.data / "toy_images" / name).write_bytes(payload)
        self.train = [{"id": "shared-id", "problem": "<image>train", "answer": "A", "images": ["train.jpg"]}]
        (self.train_run / "pilot_samples.json").write_text(json.dumps(self.train))
        self.validation = [
            {"id": "shared-id", "problem": "<image>question", "answer": "A", "images": ["a.jpg"]},
            {"id": "different-id", "problem": "<image>question", "answer": "A", "images": ["copy.jpg"]},
            {"id": "a", "problem": "<image>question a", "answer": "A", "images": ["a.jpg"]},
            {"id": "b", "problem": "<image>question b", "answer": "B", "images": ["b.jpg"]},
            {"id": "duplicate-a", "problem": "<image>question a", "answer": "A", "images": ["a.jpg"]},
        ]
        (self.data / "toy.json").write_text(json.dumps(self.validation))
        original = evaluation.BENCHMARKS
        evaluation.BENCHMARKS = [("Toy", "toy.json", "toy_images")]
        self.addCleanup(setattr, evaluation, "BENCHMARKS", original)

    def prepare(self):
        evaluation.prepare(self.data, self.train_run, self.output, per_benchmark=16)
        return evaluation.read_rows(self.output / "samples.json")

    def results(self):
        samples = self.prepare()
        for label in ("before", "after"):
            meta = {"complete": True, "expected_count": len(samples),
                    "samples_sha256": evaluation.digest(self.output / "samples.json"),
                    "settings": {"n": 1, "temperature": 0.0}}
            (self.output / f"meta.{label}.json").write_text(json.dumps(meta))
            rows = [{"sample_key": item["sample_key"], "benchmark": "Toy", "ground_truth": item["answer"],
                     "accuracy": float(label == "after"), "format": 0.0, "finish_reason": "stop"}
                    for item in samples]
            (self.output / f"results.{label}.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
        return samples

    def test_excludes_ids_image_bytes_and_duplicates(self):
        samples = self.prepare()
        self.assertEqual(len(samples), 2)
        self.assertTrue(all(item["id"] != "shared-id" for item in samples))
        self.assertTrue(all("copy.jpg" not in item["images"][0] for item in samples))
        audit = json.loads((self.output / "data-audit.json").read_text())
        self.assertEqual(audit["benchmarks"][0]["rejections_among_scanned"],
                         {"duplicate_within_benchmark": 1, "shared_image_bytes": 1, "shared_id": 1})

    def test_frozen_samples_cannot_be_overwritten(self):
        self.prepare()
        with self.assertRaises(FileExistsError):
            self.prepare()

    def test_missing_image_fails_without_placeholder(self):
        (self.data / "toy_images/a.jpg").unlink()
        with self.assertRaises(FileNotFoundError):
            self.prepare()

    def test_complete_pair_compares(self):
        self.results()
        result = evaluation.compare(self.output)
        self.assertEqual(result[0]["accuracy_delta_points"], 100.0)

    def test_missing_result_fails(self):
        self.results()
        (self.output / "results.after.jsonl").write_text("")
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            evaluation.compare(self.output)

    def test_different_decode_settings_fail(self):
        self.results()
        path = self.output / "meta.after.json"
        meta = json.loads(path.read_text())
        meta["settings"]["n"] = 8
        path.write_text(json.dumps(meta))
        with self.assertRaisesRegex(ValueError, "settings differ"):
            evaluation.compare(self.output)

    def test_infer_sends_vllm_image_modality_and_writes_complete_metadata(self):
        # Exercise the actual request-building/infer path with fake dependencies.
        # This catches the images/image regression; it is not real GPU inference.
        import signal
        import sys

        samples = self.prepare()
        image = MagicMock(width=512, height=512)
        image.convert.return_value = image
        image.__enter__.return_value = image
        processor = MagicMock()
        processor.apply_chat_template.return_value = "rendered-prompt"
        auto_processor = MagicMock()
        auto_processor.from_pretrained.return_value = processor
        test = self

        class FakeLLM:
            def __init__(self, **kwargs):
                pass

            def generate(self, requests, sampling, use_tqdm):
                for request in requests:
                    test.assertEqual(set(request["multi_modal_data"]), {"image"})
                    test.assertEqual(len(request["multi_modal_data"]["image"]), 1)
                return [SimpleNamespace(outputs=[SimpleNamespace(
                    text="<think>fixture</think>\\boxed{A}", token_ids=[1, 2], finish_reason="stop")])
                    for request in requests]

        fake_grader = SimpleNamespace(extract_boxed_content=lambda value: "A",
                                      grade_answer=lambda answer, truth: answer == truth)
        modules = {"PIL": SimpleNamespace(Image=SimpleNamespace(open=lambda path: image)),
                   "transformers": SimpleNamespace(AutoProcessor=auto_processor),
                   "vllm": SimpleNamespace(LLM=FakeLLM, SamplingParams=lambda **kwargs: kwargs),
                   "mathruler": SimpleNamespace(), "mathruler.grader": fake_grader}
        with patch.dict(sys.modules, modules), patch.object(signal, "SIGALRM", 14, create=True), \
                patch.object(signal, "signal"), patch.object(signal, "alarm", create=True):
            evaluation.infer(self.output, self.root / "model", self.root / "tokenizer", "before")
        meta = json.loads((self.output / "meta.before.json").read_text())
        self.assertTrue(meta["complete"])
        self.assertEqual(meta["expected_count"], len(samples))
        self.assertEqual(len((self.output / "results.before.jsonl").read_text().splitlines()), len(samples))


if __name__ == "__main__":
    unittest.main(verbosity=2)
