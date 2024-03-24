# -*- coding: utf-8 -*-
import copy
import logging
import os
from typing import Tuple, List, Union

import numpy as np
import pandas as pd

from ts_benchmark.common.constant import ROOT_PATH


def fill_null_value(result_df: pd.DataFrame, fill_type: str) -> pd.DataFrame:
    """
    填充 DataFrame 中的空值。

    根据填充类型，将 DataFrame 中的空值进行填充。

    :param result_df: 要填充的 DataFrame。
    :param fill_type: 填充类型，可选值为 "mean_value"。

    :return: 填充后的 DataFrame。
    """
    has_no_null_result_df = copy.deepcopy(result_df)
    if fill_type == "mean_value":
        numeric_columns = has_no_null_result_df.select_dtypes(
            include=[int, float]
        ).columns

        for column in numeric_columns:
            mean_val = has_no_null_result_df[column].mean(skipna=True)
            column_data = has_no_null_result_df[
                column
            ].copy()  # Explicitly copy the column data
            column_data.fillna(mean_val, inplace=True)
            has_no_null_result_df[
                column
            ] = column_data  # Assign the modified column data back to the DataFrame

    return has_no_null_result_df


def extract_info_and_merge(
    log_files_list: pd.DataFrame,
    results_of_evaluated_algorithm: pd.DataFrame,
) -> Tuple[dict, list, pd.DataFrame]:
    """
    提取信息并合并数据。

    根据提供的报告模型、基准结果和评估算法的结果，提取信息并合并数据。

    :param report_model: 要报告的模型名称或模型名称列表，或 "single" 表示单个模型报告。
    :param baseline_result: 基准结果的 DataFrame。
    :param results_of_evaluated_algorithm: 评估算法的结果的 DataFrame。

    :return: 模型评估配置、数据文件名列表和合并后的 DataFrame。
    """

    # Load baseline data
    baseline_result = (
        pd.concat(
            [
                pd.read_csv(os.path.join(ROOT_PATH, log_file))
                for log_file in log_files_list[1:]
            ],
            axis=0,
            ignore_index=True,
        )
        if len(log_files_list) > 1
        else pd.DataFrame()
    )

    # Combine and process data
    model_config_of_evaluated_algorithm = results_of_evaluated_algorithm[
        "strategy_args"
    ][0]
    data_filename_list_of_evaluated_algorithm = results_of_evaluated_algorithm[
        "file_name"
    ]
    adjusted_df = pd.concat(
        [results_of_evaluated_algorithm, baseline_result], ignore_index=True
    )
    return (
        # TODO: change the name in accordance with the new header name
        # TODO: the type of this variable seems to be str rather than dict???
        model_config_of_evaluated_algorithm,
        data_filename_list_of_evaluated_algorithm,
        adjusted_df,
    )


def filter_data_and_calculate_result(
    metric_name: str,
    report_type: str,
    threshold_value: float,
    fill_type: str,
    adjusted_df: pd.DataFrame,
    model_config_of_evaluated_algorithm: dict,
    data_filename_list_of_evaluated_algorithm: list,
) -> Tuple[List[float], List[str]]:
    """
    过滤数据并计算结果。

    根据提供的度量名称、报告类型、阈值、填充类型、调整后的数据框架和评估算法的配置，过滤数据并计算结果。

    :param metric_name: 要计算的度量名称。
    :param report_type: 报告类型，可选值为 "mean"、"median"、"max" 等。
    :param threshold_value: 阈值值。
    :param fill_type: 填充类型，可选值为 "mean_value" 等。
    :param adjusted_df: 调整后的数据。
    :param model_config_of_evaluated_algorithm: 评估算法的配置。
    :param data_filename_list_of_evaluated_algorithm: 评估算法的数据文件名列表。

    :return: 计算得到的度量值列表和模型参数列表。
    """
    # Filter data and calculate means
    grouped_df = adjusted_df[
        adjusted_df["strategy_args"] == model_config_of_evaluated_algorithm
    ]

    # 首先对 data_filename_list_of_evaluated_algorithm 进行去重操作
    unique_data_filenames = list(set(data_filename_list_of_evaluated_algorithm))

    selected_df = grouped_df[grouped_df["file_name"].isin(unique_data_filenames)].copy()
    selected_df["model_and_params"] = (
        selected_df["model_name"] + ";" + selected_df["model_params"]
    )
    # todo:inf,-inf应该变成null
    metric_df = selected_df[[metric_name, "model_and_params", "file_name"]].pivot_table(
        values=metric_name,
        index="file_name",
        columns="model_and_params",
        aggfunc="mean",
    )
    threshold_count = float(threshold_value) * len(unique_data_filenames)

    nan_count = metric_df.isna().sum(axis=0)
    metric_values = fill_null_value(metric_df, fill_type).aggregate(report_type, axis=0)
    metric_values[nan_count > threshold_count] = np.nan
    metric_values = [metric_name] + metric_values.tolist()
    return metric_values, metric_df.columns.tolist()

def get_leaderboard(
    log_files_list: List[str],
    log_data: pd.DataFrame,
    aggregate_type: str,
    report_metrics: Union[str, List[str]],
    fill_type: str,
    null_value_threshold: float,
) -> pd.DataFrame:
    """
    Generate a report based on specified configuration parameters.

    Parameters:
    - log_files_list (List[str]): A list of file paths for log files.
    - log_data (pd.DataFrame): merged dataframe of log files
    - aggregate_type (str): The aggregation type used when reporting the final results of evaluation metrics.
    - report_metrics (Union[str, List[str]]): The metrics for the report, can be a string or a list of strings.
    - fill_type (str): The type of fill for missing values.
    - null_value_threshold (float): The threshold value for null metrics.

    Raises:
    - ValueError: If all metrics have too many null values, making performance comparison impossible.

    Returns:
    - None: The function does not return a value, but generates and saves a report to a CSV file.
    """

    # Load data
    results_of_evaluated_algorithm = pd.read_csv(
        os.path.join(ROOT_PATH, log_files_list[0])
    )


    if isinstance(report_metrics, str):
        report_metrics = [report_metrics]

    final_result = []
    many_null_metrics_nums = 0
    cols = []
    report_metrics_set = set(report_metrics)
    results_columns_set = set(results_of_evaluated_algorithm.columns.values)
    prefixes_set = set()
    for metric in results_columns_set:
        prefix = metric.split(";", 1)[0]
        prefixes_set.add(prefix)

    mapping = {metric.split(";", 1)[0]: metric for metric in results_columns_set}

    missing_metrics = report_metrics_set - prefixes_set

    if missing_metrics:
        raise ValueError(
            "Metrics in report_config but not in results_of_evaluated_algorithm.columns:",
            list(missing_metrics),
        )

    for metric_name in report_metrics:
        metric_name = mapping[metric_name]
        # 填补缺失值
        nan_count = results_of_evaluated_algorithm[metric_name].isna().sum()
        if nan_count > float(null_value_threshold) * len(
            results_of_evaluated_algorithm
        ):
            logging.warning(
                "metric %s has too many null values, we drop this metric in this report",
                metric_name,
            )
            many_null_metrics_nums = many_null_metrics_nums + 1
            continue

        model_config_of_evaluated_algorithm = results_of_evaluated_algorithm[
            "strategy_args"
        ][0]
        data_filename_list_of_evaluated_algorithm = results_of_evaluated_algorithm[
            "file_name"
        ]
        adjusted_df = log_data

        single_metric_value, cols = filter_data_and_calculate_result(
            metric_name,
            aggregate_type,
            null_value_threshold,
            fill_type,
            adjusted_df,
            model_config_of_evaluated_algorithm,
            data_filename_list_of_evaluated_algorithm,
        )
        final_result.append(single_metric_value)

    if many_null_metrics_nums == len(report_metrics):
        raise ValueError(
            "all metric has too many null values, we cannot obtain a performance "
            "comparison between this algorithm and the baseline algorithm"
        )
    else:
        result_df = pd.DataFrame(final_result, columns=["metric_name"] + cols)
        return result_df
