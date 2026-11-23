# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

import os
import sys
import types
from unittest.mock import MagicMock, call, patch

from verl.utils.tracking import (
    DapoFilteredRewardTableLogger,
    Tracking,
    ValidationGenerationsLogger,
    _SwanlabLoggingAdapter,
)


def test_tracking_finish_finalizes_wandb_once():
    tracking = Tracking.__new__(Tracking)
    tracking.logger = {"wandb": MagicMock()}
    tracking._finished = False

    tracking.finish(exit_code=1)
    tracking.finish(exit_code=0)

    tracking.logger["wandb"].finish.assert_called_once_with(exit_code=1)


def test_tracking_finish_finalizes_mlflow_once():
    tracking = Tracking.__new__(Tracking)
    tracking.logger = {"mlflow": MagicMock()}
    tracking._finished = False

    tracking.finish()
    tracking.finish()

    tracking.logger["mlflow"].finish.assert_called_once()


def test_dapo_filtered_reward_table_logs_incremental_rows():
    mock_wandb = MagicMock()
    mock_wandb.run = object()
    mock_wandb.__version__ = "0.20.0"
    table = mock_wandb.Table.return_value
    logger = DapoFilteredRewardTableLogger()

    with patch.dict(sys.modules, {"wandb": mock_wandb}):
        logger.log(["wandb"], {1.0: 2, 0.0: 3}, step=7)
        logger.log(["wandb"], {0.0: 4}, step=8)

    mock_wandb.Table.assert_called_once_with(columns=["step", "reward_counts"], log_mode="INCREMENTAL")
    assert table.add_data.call_args_list == [call(7, "0:3, 1:2"), call(8, "0:4")]
    assert mock_wandb.log.call_args_list == [
        call({"training/filter_groups/filtered_reward_counts": table}, step=7),
        call({"training/filter_groups/filtered_reward_counts": table}, step=8),
    ]


def test_dapo_filtered_reward_table_falls_back_to_full_history():
    mock_wandb = MagicMock()
    mock_wandb.run = object()
    mock_wandb.__version__ = "0.19.11"
    first_table, second_table = MagicMock(), MagicMock()
    mock_wandb.Table.side_effect = [first_table, second_table]
    logger = DapoFilteredRewardTableLogger()

    with patch.dict(sys.modules, {"wandb": mock_wandb}):
        logger.log(["wandb"], {1.0: 2, 0.0: 3}, step=7)
        logger.log(["wandb"], {0.0: 4}, step=8)

    assert mock_wandb.Table.call_args_list == [
        call(columns=["step", "reward_counts"], data=[[7, "0:3, 1:2"]]),
        call(columns=["step", "reward_counts"], data=[[7, "0:3, 1:2"], [8, "0:4"]]),
    ]
    assert mock_wandb.log.call_args_list == [
        call({"training/filter_groups/filtered_reward_counts": first_table}, step=7),
        call({"training/filter_groups/filtered_reward_counts": second_table}, step=8),
    ]


def test_validation_generations_logger_logs_trackio_traces():
    mock_trackio = MagicMock()
    mock_trackio.context_vars = types.SimpleNamespace(current_run=MagicMock())
    mock_trackio.context_vars.current_run.get.return_value = None
    mock_trackio.Trace.side_effect = lambda messages, metadata=None: {
        "_type": "trackio.trace",
        "messages": messages,
        "metadata": metadata or {},
    }

    with patch.dict(sys.modules, {"trackio": mock_trackio}):
        ValidationGenerationsLogger().log(
            ["trackio"],
            samples=[["question", "answer", 0.5]],
            step=7,
        )

    mock_trackio.Trace.assert_called_once()
    trace_kwargs = mock_trackio.Trace.call_args.kwargs
    assert trace_kwargs["messages"] == [
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "answer"},
    ]
    assert trace_kwargs["metadata"]["source"] == "validation_generations"
    assert trace_kwargs["metadata"]["score"] == 0.5
    mock_trackio.log.assert_called_once()
    assert mock_trackio.log.call_args.kwargs["step"] == 7


def test_swanlab_adapter_groups_metrics_by_first_path_component():
    mock_swanlab = MagicMock()
    mock_run = MagicMock()
    adapter = _SwanlabLoggingAdapter(mock_swanlab, mock_run)

    adapter.log({"actor/loss": 1.0, "val-aux/math/accuracy/mean@1": 0.5, "epoch": 2}, step=3)
    adapter.log({"actor/loss": 0.8}, step=4)

    assert mock_run.define_metric.call_args_list == [
        call("actor/loss", section_name="actor"),
        call("val-aux/math/accuracy/mean@1", section_name="val-aux"),
        call("epoch", section_name="metrics"),
    ]
    assert mock_swanlab.log.call_args_list == [
        call(data={"actor/loss": 1.0, "val-aux/math/accuracy/mean@1": 0.5, "epoch": 2}, step=3),
        call(data={"actor/loss": 0.8}, step=4),
    ]


def test_validation_generations_logger_uses_text_in_swanlab_local_mode():
    mock_swanlab = MagicMock()
    first_text, second_text = MagicMock(), MagicMock()
    mock_swanlab.Text.side_effect = [first_text, second_text]
    samples = [["question 1", "answer 1", 1.0], ["question 2", "answer 2", 0.0]]

    with (
        patch.dict(sys.modules, {"swanlab": mock_swanlab}),
        patch.dict(os.environ, {"SWANLAB_MODE": "local"}),
    ):
        ValidationGenerationsLogger().log_generations_to_swanlab(samples, step=7)

    assert mock_swanlab.Text.call_args_list == [
        call("input: question 1\n\n---\n\noutput: answer 1\n\n---\n\nscore: 1.0", caption="sample 1"),
        call("input: question 2\n\n---\n\noutput: answer 2\n\n---\n\nscore: 0.0", caption="sample 2"),
    ]
    mock_swanlab.log.assert_called_once_with({"val/generations": [first_text, second_text]}, step=7)
    mock_swanlab.echarts.Table.assert_not_called()
