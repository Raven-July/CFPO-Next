"""Small real-data integration check, with no model load or training.

Run in the configured container: python examples/cfpo_trainer/check_data.py
"""

import importlib.util
import json
import os
import tempfile
from pathlib import Path

import datasets
from omegaconf import OmegaConf
from PIL import Image

from verl.utils.dataset.rl_dataset import RLHFDataset


def main():
    example_dir = Path(__file__).resolve().parent
    entries = json.loads((example_dir / "datasets_math_train.json").read_text())
    source = entries[0]
    with open(source["dataset"], encoding="utf-8") as stream:
        rows = json.load(stream)[:8]
    assert len(rows) == 8, "Expected at least eight training samples"
    for row in rows:
        for image in row["images"]:
            with Image.open(Path(source["image"]) / image) as opened:
                opened.verify()
    print("REAL_IMAGE_DECODE_OK: first 8 training samples", flush=True)

    with tempfile.TemporaryDirectory(dir=os.environ.get("CFPO_CHECK_DIR")) as temp:
        root = Path(temp)
        sample_file = root / "samples.json"
        sample_file.write_text(json.dumps(rows), encoding="utf-8")
        spec = {"dataset": str(sample_file), "image": source["image"]}
        manifest = root / "manifest.json"
        manifest.write_text(json.dumps([spec]), encoding="utf-8")
        config = OmegaConf.create({
            "cache_dir": str(root / "cache"),
            "format_prompt": str(example_dir / "format_prompt/base.jinja"),
            "filter_overlong_prompts": False,
            "filter_overlong_prompts_workers": 1,
        })

        def load(files):
            # A sentinel suffices for constructing raw messages. Tokenizer and
            # image-processor execution are deliberately not claimed here.
            return RLHFDataset(files, tokenizer=None, config=config, processor=object(), max_samples=3)

        dataset = load(str(manifest))
        first = dataset[0]
        assert len(dataset) == 3
        assert first["reward_model"]["ground_truth"] == str(rows[0]["answer"])
        assert "answer" not in first and "problem" not in first
        assert [message["role"] for message in first["raw_prompt"]] == ["system", "user"]
        user_content = first["raw_prompt"][1]["content"]
        image_items = [item for item in user_content if item["type"] == "image"]
        assert len(image_items) == len(rows[0]["images"])
        text = "".join(item["text"] for item in user_content if item["type"] == "text")
        assert text == rows[0]["problem"].replace("<image>", "")
        assert dataset[0]["raw_prompt"] == first["raw_prompt"]
        dataset.resume_dataset_state()
        assert dataset[0]["raw_prompt"] == first["raw_prompt"]
        print("MANIFEST_MESSAGES_LABELS_AND_RESUME_OK", flush=True)

        direct = load(OmegaConf.create([spec]))
        assert direct[0]["raw_prompt"] == first["raw_prompt"]
        multi = load([spec, spec])
        assert len(multi) == 3
        print("DIRECT_CONFIG_AND_MULTIPLE_SOURCES_OK", flush=True)

        # Existing file-based loading must still accept already-normalized rows.
        native_rows = [dataset.dataframe[0]]
        native_json = root / "native.json"
        native_json.write_text(json.dumps(native_rows), encoding="utf-8")
        native_parquet = root / "native.parquet"
        datasets.Dataset.from_list(native_rows).to_parquet(str(native_parquet))
        for native_file in (native_json, native_parquet):
            native = load(str(native_file))
            assert native[0]["raw_prompt"] == first["raw_prompt"]
        print("NATIVE_JSON_AND_PARQUET_REGRESSION_OK", flush=True)

    reward_path = example_dir / "reward_function/base.py"
    reward_spec = importlib.util.spec_from_file_location("cfpo_reward", reward_path)
    reward_module = importlib.util.module_from_spec(reward_spec)
    reward_spec.loader.exec_module(reward_module)
    cases = [
        (r"<think>1+1=2</think>\boxed{2}", "2", 1.0),
        (r"\boxed{2}", "2", 0.9),
        (r"<think>incorrect</think>\boxed{3}", "2", 0.1),
        (r"\boxed{3}", "2", 0.0),
    ]
    for response, answer, expected in cases:
        result = reward_module.compute_score(
            data_source="check", solution_str=response, ground_truth=answer, extra_info={}
        )
        assert abs(result["score"] - expected) < 1e-8, result
    print("REAL_MATHRULER_REWARD_CASES_OK", flush=True)
    print("CFPO_DATA_REWARD_CHECK_OK", flush=True)


if __name__ == "__main__":
    main()
