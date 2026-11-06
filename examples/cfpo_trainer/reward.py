# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Original CFPO math reward adapted to verl's per-response callback API."""

import re

from mathruler.grader import extract_boxed_content, grade_answer


def compute_score(data_source, solution_str, ground_truth, extra_info=None, format_weight=0.1, **kwargs):
    """Keep CFPO's 90% correctness + 10% think/boxed format reward."""
    if not 0 <= format_weight <= 1:
        raise ValueError("format_weight must be between zero and one")
    response = re.sub(r"\s*(<|>|/)\s*", r"\1", solution_str)
    # Thinking templates can put the opening tag in the prompt instead of the completion.
    format_score = float(re.search(r"</think>.*\\boxed\{.*\}", response, re.DOTALL) is not None)
    answer = extract_boxed_content(response)
    accuracy_score = float(bool(answer) and grade_answer(answer, str(ground_truth)))
    score = (1 - format_weight) * accuracy_score + format_weight * format_score
    return {"score": score, "overall": score, "format": format_score, "accuracy": accuracy_score}
