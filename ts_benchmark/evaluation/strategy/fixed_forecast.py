# -*- coding: utf-8 -*-
import time
from typing import List, Optional

import numpy as np
import pandas as pd

from ts_benchmark.evaluation.metrics import regression_metrics
from ts_benchmark.evaluation.strategy.constants import FieldNames
from ts_benchmark.evaluation.strategy.forecasting import ForecastingStrategy
from ts_benchmark.models import ModelFactory
from ts_benchmark.utils.data_processing import split_time
from ts_benchmark.utils.data_processing import split_channel


class FixedForecast(ForecastingStrategy):
    """
    Fixed forecast strategy class

    This strategy defines a forecasting task with fixed prediction length.

    The required strategy configs include:

    - horizon (int): The length to predict, i.e. the length of the test series;
    - train_ratio_in_tv (float): The ratio of the training series when performing train-validation split.

    The accepted metrics include all regression metrics.

    The return fields other than the specified metrics are (in order):

    - FieldNames.FILE_NAME: The name of the series;
    - FieldNames.FIT_TIME: The training time;
    - FieldNames.INFERENCE_TIME: The inference time;
    - FieldNames.ACTUAL_DATA: The true test data, encoded as a string.
    - FieldNames.INFERENCE_DATA: The predicted data, encoded as a string.
    - FieldNames.LOG_INFO: Any log returned by the evaluator.
    """

    REQUIRED_CONFIGS = [
        "horizon",
        "train_ratio_in_tv",
        "save_true_pred",
        "target_channel",
    ]

    def _execute(
        self,
        series: pd.DataFrame,
        meta_info: Optional[pd.Series],
        model_factory: ModelFactory,
        series_name: str,
    ) -> List:
        model = model_factory()

        target_channel = self._get_scalar_config_value("target_channel", series_name)
        horizon = self._get_scalar_config_value("horizon", series_name)
        train_ratio_in_tv = self._get_scalar_config_value(
            "train_ratio_in_tv", series_name
        )

        data_len = int(self._get_meta_info(meta_info, "length", len(series)))
        train_length = data_len - horizon
        if train_length <= 0:
            raise ValueError("The prediction step exceeds the data length")

        train_valid_data, test_data = split_time(series, train_length)
        target_train_valid_data, exog_train_valid_data = split_channel(
            train_valid_data, target_channel
        )
        target_test_data, _ = split_channel(test_data, target_channel)
        covariates = {}
        covariates["exog"] = exog_train_valid_data

        start_fit_time = time.time()
        fit_method = model.forecast_fit if hasattr(model, "forecast_fit") else model.fit
        fit_method(
            target_train_valid_data, covariates=covariates, train_ratio_in_tv=train_ratio_in_tv
        )
        end_fit_time = time.time()
        predicted = model.forecast(horizon, target_train_valid_data, covariates=covariates)
        end_inference_time = time.time()

        single_series_results, log_info = self.evaluator.evaluate_with_log(
            target_test_data.to_numpy(),
            predicted,
            # TODO: add configs to control scaling behavior
            self._get_eval_scaler(target_train_valid_data, train_ratio_in_tv),
            target_train_valid_data.values,
        )
        inference_data = pd.DataFrame(
            predicted, columns=target_test_data.columns, index=target_test_data.index
        )

        save_true_pred = self._get_scalar_config_value("save_true_pred", series_name)
        actual_data_encoded = (
            self._encode_data(target_test_data) if save_true_pred else np.nan
        )
        inference_data_encoded = (
            self._encode_data(inference_data) if save_true_pred else np.nan
        )

        single_series_results += [
            series_name,
            end_fit_time - start_fit_time,
            end_inference_time - end_fit_time,
            actual_data_encoded,
            inference_data_encoded,
            log_info,
        ]

        return single_series_results

    @staticmethod
    def accepted_metrics():
        return regression_metrics.__all__

    @property
    def field_names(self) -> List[str]:
        return self.evaluator.metric_names + [
            FieldNames.FILE_NAME,
            FieldNames.FIT_TIME,
            FieldNames.INFERENCE_TIME,
            FieldNames.ACTUAL_DATA,
            FieldNames.INFERENCE_DATA,
            FieldNames.LOG_INFO,
        ]
