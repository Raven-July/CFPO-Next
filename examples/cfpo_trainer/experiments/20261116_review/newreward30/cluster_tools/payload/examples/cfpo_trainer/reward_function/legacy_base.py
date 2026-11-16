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

"""Adapt the team's CFPO math reward to verl's per-sample reward API.

Source: Counterfactual-R1/examples/reward_function/base.py supplied by Bao.
Keep its normalization, grading, and 0.9 accuracy + 0.1 format weighting.
Only the calling convention and aggregate result key differ from the source.
"""

import re
from typing import Any

from mathruler.grader import extract_boxed_content, grade_answer


def format_reward(response: str) -> float:
    pattern = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
    return 1.0 if re.fullmatch(pattern, response) else 0.0


def accuracy_reward(response: str, ground_truth: str) -> float:
    answer = extract_boxed_content(response)
    return 1.0 if grade_answer(answer, ground_truth) else 0.0


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
    format_weight: float = 0.1,
    **kwargs: Any,
) -> dict[str, float]:
    """Called by verl's naive reward manager for a single model response."""
    if not 0.0 <= format_weight <= 1.0:
        raise ValueError("format_weight must be between 0 and 1")
    response = re.sub(r"\s*(<|>|/)\s*", r"\1", solution_str)
    format_score = format_reward(response)
    accuracy_score = accuracy_reward(response, ground_truth)
    return {
        "score": (1 - format_weight) * accuracy_score + format_weight * format_score,
        "format": format_score,
        "accuracy": accuracy_score,
    }
