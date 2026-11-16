# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CFPO reward aligned with the senior's 2026-11-11 meeting snippet.

The opening think tag may belong to the prompt, not the completion.
The historical 202-step reward is preserved in legacy_base.py; scores stay unchanged.
"""

import re
from typing import Any

from mathruler.grader import extract_boxed_content, grade_answer


def format_reward(response: str) -> float:
    return float(re.search(r"</think>.*\\boxed\{.*\}", response, re.DOTALL) is not None)


def accuracy_reward(response: str, ground_truth: str) -> float:
    answer = extract_boxed_content(response)
    return float(bool(answer) and grade_answer(answer, str(ground_truth)))


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
    format_weight: float = 0.1,
    **kwargs: Any,
) -> dict[str, float]:
    """Called by verl's naive reward manager for a single model response."""
    if not 0 <= format_weight <= 1:
        raise ValueError("format_weight must be between 0 and 1")
    response = re.sub(r"\s*(<|>|/)\s*", r"\1", solution_str)
    format_score = format_reward(response)
    accuracy_score = accuracy_reward(response, ground_truth)
    score = (1 - format_weight) * accuracy_score + format_weight * format_score
    return {
        "score": score,
        "overall": score,
        "format": format_score,
        "accuracy": accuracy_score,
    }
