# -*- coding: utf-8 -*-
import functools
import traceback
from typing import List, Tuple, Any

import numpy as np


from ts_benchmark.evaluation.metrics import METRICS


def encode_params(params):
    encoded_pairs = []
    for key, value in sorted(params.items()):
        if isinstance(value, (np.floating, float)):
            value = round(value, 3)
        encoded_pairs.append(f"{key}:{repr(value)}")
    return ";".join(encoded_pairs)


class Evaluator:
    """
    Evaluator class, used to calculate the evaluation metrics of the model.
    """

    def __init__(self, metric: List[dict]):
        """
        Initialize the evaluator object.

        :param metric: A list containing information on evaluation metrics.
        """
        self.metric = metric
        self.metric_funcs = []
        self.metric_names = []

        # Create a list of evaluation indicator functions and names
        for metric_info in self.metric:
            metric_info_copy = metric_info.copy()
            metric_name = metric_info_copy.pop("name")
            if metric_info_copy:
                metric_name += ";" + encode_params(metric_info_copy)
            self.metric_names.append(metric_name)
            metric_name_copy = metric_info.copy()
            name = metric_name_copy.pop("name")
            fun = METRICS[name]
            if metric_name_copy:
                self.metric_funcs.append(functools.partial(fun, **metric_name_copy))
            else:
                self.metric_funcs.append(fun)

    def evaluate(
        self,
        actual: np.ndarray,
        predicted: np.ndarray,
        scaler: object = None,
        hist_data: np.ndarray = None,
        **kwargs,
    ) -> list:
        """
        Calculate the evaluation index values of the model.

        :param actual: Actual observation data.
        :param predicted: Model predicted data.
        :param scaler: Normalization.
        :param hist_data:  Historical data (optional).
        :return: Indicator evaluation result.
        """
        return [
            m(actual, predicted, scaler=scaler, hist_data=hist_data)
            for m in self.metric_funcs
        ]

    def evaluate_with_log(
        self,
        actual: np.ndarray,
        predicted: np.ndarray,
        scaler: object = None,
        hist_data: np.ndarray = None,
        **kwargs,
    ) -> Tuple[List[Any], str]:
        """
        Calculate the evaluation index values of the model.

        :param actual: Actual observation data.
        :param predicted: Model predicted data.
        :param scaler: Normalization.
        :param hist_data:  Historical data (optional).
        :return: Indicator evaluation results and log information.
        """
        evaluate_result = []
        log_info = ""
        for m in self.metric_funcs:
            try:
                evaluate_result.append(
                    m(actual, predicted, scaler=scaler, hist_data=hist_data)
                )
            except Exception as e:
                evaluate_result.append(np.nan)
                log_info += f"Error in calculating {m.__name__}: {traceback.format_exc()}\n{e}\n"
        return evaluate_result, log_info

    def default_result(self):
        """
        Return the default evaluation metric results.

        :return: Default evaluation metric result.
        """
        return len(self.metric_names) * [np.nan]
