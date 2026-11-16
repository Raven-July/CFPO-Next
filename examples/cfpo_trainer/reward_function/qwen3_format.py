"""Candidate format adapter for responses with an implicit opening think tag.

Not selected by existing training scripts. Historical reward is in legacy_base.py.
Accuracy grading, normalization and reward weights follow that adapter.
"""

import re
from typing import Any

from mathruler.grader import extract_boxed_content, grade_answer


def format_reward(response: str) -> float:
    # An opening tag may be part of the generation prefix. Require exactly one
    # closing boundary before the boxed answer; never reward boxed-only text.
    pattern = r"(?:<think>)?(?:(?!</?think>).)*</think>(?:(?!</?think>).)*\\boxed\{.*\}.*"
    if response.count("</think>") != 1 or response.count("<think>") > 1:
        return 0.0
    return float(re.fullmatch(pattern, response, re.DOTALL) is not None)


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
    format_weight: float = 0.1,
    **kwargs: Any,
) -> dict[str, float]:
    if not 0.0 <= format_weight <= 1.0:
        raise ValueError("format_weight must be between 0 and 1")
    response = re.sub(r"\s*(<|>|/)\s*", r"\1", solution_str)
    accuracy = float(grade_answer(extract_boxed_content(response), ground_truth))
    format_score = format_reward(response)
    return {
        "score": (1 - format_weight) * accuracy + format_weight * format_score,
        "format": format_score,
        "accuracy": accuracy,
    }
