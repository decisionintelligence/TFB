# -*- coding: utf-8 -*-
import logging
from typing import List, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _fill_null_value(result_df: pd.DataFrame, fill_type: str) -> pd.DataFrame:
    """
    Fills NaN values in the benchmarking records.

    :param result_df: The records to be filled.
    :param fill_type: The type of the filling method, the allowed values are:

        - mean_value: Fill with the mean value of the non-NaN elements;

    :return: The filled DataFrame。
    """
    if fill_type == "mean_value":
        numeric_columns = result_df.select_dtypes(include=[np.number]).columns

        mean_val = result_df[numeric_columns].mean(axis=0, skipna=True)
        df_no_na = result_df.fillna(mean_val)
    else:
        raise ValueError(f"Unknown fill_type {fill_type}")

    return df_no_na


def _calculate_single_metric_result(
    full_metric_df: pd.DataFrame,
    metric_name: str,
    agg_type: str,
    nan_threshold: float,
    fill_type: str,
) -> pd.Series:
    """
    Calculates the leaderboard values for a single metric.

    :param full_metric_df: The full record data.
    :param metric_name: The name of the target metric.
    :param agg_type: Aggregation method, optional values include "mean", "median", "max".
    :param nan_threshold: The metric for any algorithm will be set to NaN if the ratio
        of NaN values from that algorithm exceeds this threshold.
    :param fill_type: Fill method, optional values include "mean_value".
    :return: The leaderboard values for a single metric.
    """
    metric_df = full_metric_df.copy()
    metric_df["model_and_params"] = (
        metric_df["model_name"] + ";" + metric_df["model_params"]
    )
    # todo:inf,-inf应该变成null
    metric_df = metric_df[[metric_name, "model_and_params", "file_name"]].pivot_table(
        values=metric_name,
        index="file_name",
        columns="model_and_params",
        aggfunc=np.nanmean,
        dropna=False,
    )
    threshold_count = float(nan_threshold) * len(metric_df)
    nan_count = metric_df.isna().sum(axis=0)
    metric_values = _fill_null_value(metric_df, fill_type).aggregate(agg_type, axis=0)
    metric_values[nan_count > threshold_count] = np.nan
    return metric_values


def _get_report_metrics(
    record_metrics: np.ndarray, report_metrics: np.ndarray
) -> np.ndarray:
    """
    Get the metrics to be included in the leaderboard.

    This function tries to find metrics specified in `report_metrics`.
    If any of the `report_metrics` does not exist in the records, a warning is
    logged and the metric is ignored.

    :param record_metrics: The list of metric names in the benchmarking records.
    :param report_metrics: The list of metrics that should be included in the leaderboard,
        each item in this list can be in either format:

        - exact names: When there exists ";" symbols in name (i.e. parametrized metrics),
          the name is compared with `record_metrics` using exact match;
        - stems: When there's no ";" symbol in the name, the name is considered as a
          stem name (i.e. metric name without parameters), and is compared with stem names
          in the `record_metrics`;

    :return: An ndarray of metric names that should be included in the leaderboard.
    """
    # a specified report metric may select multiple metrics with the same prefix
    log_metric_prefix = np.array([metric.split(";", 1)[0] for metric in record_metrics])
    matching_matrix = []
    for metric in report_metrics:
        if ";" in metric:
            # metric with parameters, use exact match
            matching_matrix.append(record_metrics == metric)
        else:
            # metric prefix, use prefix match
            matching_matrix.append(log_metric_prefix == metric)
    matching_matrix = np.stack(matching_matrix, axis=0)
    not_matching = ~matching_matrix.any(axis=1)
    if not_matching.any():
        logger.warning(
            "Report metrics %s not found in record files, ignoring.",
            list(report_metrics[not_matching]),
        )
    actual_report_metrics = record_metrics[matching_matrix.any(axis=0)]
    actual_report_metrics = [metric for metric in report_metrics if metric in actual_report_metrics]
    return np.array(actual_report_metrics)


def get_leaderboard(
    log_data: pd.DataFrame,
    report_metrics: Union[str, List[str]],
    aggregate_type: str,
    fill_type: str,
    nan_threshold: float,
) -> pd.DataFrame:
    """
    Generate a leaderboard from benchmarking records.

    :param log_data: Benchmarking records.
    :param report_metrics: The (list of) metrics that should be included in the leaderboard,
        each item can be in either format:

        - exact names: When there exists ";" symbols in name (i.e. parametrized metrics),
          the name is compared with `record_metrics` using exact match;
        - stems: When there's no ";" symbol in the name, the name is considered as a
          stem name (i.e. metric name without parameters), and is compared with stem names
          in the `record_metrics`;

    :param aggregate_type: Aggregation method, optional values include "mean", "median", "max".
    :param fill_type: Fill method, optional values include "mean_value".
    :param nan_threshold: The metric for any algorithm will be set to NaN if the ratio
        of NaN values from that algorithm exceeds this threshold.
    :return: The leaderboard in DataFrame format.
    """
    if isinstance(report_metrics, str):
        report_metrics = [report_metrics]

    actual_report_metrics = _get_report_metrics(
        log_data.columns.values, np.array(report_metrics)
    )

    final_result = []
    for metric_name in actual_report_metrics:
        if log_data["strategy_args"].nunique() != 1:
            raise ValueError("strategy_args are inconsistent in the log file.")

        single_metric_result = _calculate_single_metric_result(
            log_data, metric_name, aggregate_type, nan_threshold, fill_type
        )
        final_result.append(single_metric_result)

    result_df = pd.concat(final_result, axis=1).T.reset_index(drop=True)
    result_df.insert(0, "metric_name", actual_report_metrics)

    result_nan_count = result_df.isna().values.sum()
    if result_nan_count > 0:
        logger.info(
            "There are %d NaN values in the leaderboard due to a higher-than-threshold NaN ratio "
            "in the corresponding model+algorithm pairs.",
            result_nan_count,
        )

    return result_df
