# -*- coding: utf-8 -*-

import os
from typing import Union, List

import pandas as pd

from ts_benchmark.common.constant import ROOT_PATH
from ts_benchmark.evaluation.strategy.constants import FieldNames
from ts_benchmark.recording import load_record_data
from ts_benchmark.report.utils.leaderboard import get_leaderboard

# currently we do not support showing or processing artifact columns
# these columns are dropped as soon as data is loaded in order to save memory
ARTIFACT_COLUMNS = [
    FieldNames.ACTUAL_DATA,
    FieldNames.INFERENCE_DATA,
    FieldNames.LOG_INFO,
]


# TODO: update the docstring to match the common format in OTB.
def report(report_config: dict) -> None:
    """
    Generate a report based on specified configuration parameters.

    Parameters:
    - report_config (dict): A dictionary containing the following keys and their respective values:
        - log_files_list (List[str]): A list of file paths for log files.
        - leaderboard_file_name (str): The name for the saved report file.
        - aggregate_type (str): The aggregation type used when reporting the final results of evaluation metrics.
        - report_metrics (Union[str, List[str]]): The metrics for the report, can be a string or a list of strings.
        - fill_type (str): The type of fill for missing values.
        - null_value_threshold (float): The threshold value for null metrics.

    Raises:
    - ValueError: If all metrics have too many null values, making performance comparison impossible.

    Returns:
    - None: The function does not return a value, but generates and saves a report to a CSV file.
    """
    log_files: Union[List[str], pd.DataFrame] = report_config.get("log_files_list")
    if not log_files:
        raise ValueError("No log files to report")

    log_data = (
        log_files
        if isinstance(log_files, pd.DataFrame)
        else load_record_data(log_files, drop_columns=ARTIFACT_COLUMNS)
    )

    leaderboard_df = get_leaderboard(
        log_data,
        report_config["report_metrics"],
        report_config.get("aggregate_type", "mean"),
        report_config.get("fill_type", "mean_value"),
        report_config.get("null_value_threshold", 0.3),
    )

    num_rows = leaderboard_df.shape[0]
    leaderboard_df.insert(0, "strategy_args", [log_data.iloc[0, 1]] * num_rows)

    # Create final DataFrame and save to CSV
    if report_config.get("save_path", None) is not None:
        save_path = report_config.get("save_path", None)
        leaderboard_df.to_csv(
            os.path.join(
                ROOT_PATH, "result", save_path, report_config["leaderboard_file_name"]
            ),
            index=False,
        )
    else:
        leaderboard_df.to_csv(
            os.path.join(ROOT_PATH, "result", report_config["leaderboard_file_name"]),
            index=False,
        )
