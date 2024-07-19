# -*- coding: utf-8 -*-
import itertools
import time
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from ts_benchmark.evaluation.metrics import regression_metrics
from ts_benchmark.evaluation.strategy.constants import FieldNames
from ts_benchmark.evaluation.strategy.forecasting import ForecastingStrategy
from ts_benchmark.models import ModelFactory
from ts_benchmark.models.model_base import BatchMaker, ModelBase
from ts_benchmark.utils.data_processing import split_before


class RollingForecastEvalBatchMaker:

    def __init__(
        self,
        series: pd.DataFrame,
        index_list: List[int],
    ):
        self.series = series
        self.index_list = index_list
        self.current_sample_count = 0

    def make_batch_predict(self, batch_size: int, win_size: int) -> dict:
        """
        Return a batch of data with index and column to be used for batch prediction.

        :param batch_size: The size of batch.
        :param win_size: The length of data used for prediction.
        :return: a batch of data and its time stamps.
        """
        index_list = self.index_list[
            self.current_sample_count : self.current_sample_count + batch_size
        ]
        series = self.series.values
        windows = sliding_window_view(series, window_shape=(win_size, series.shape[1]))
        predict_batch = windows[np.array(index_list) - win_size]
        predict_batch = np.squeeze(predict_batch, axis=1)

        indexes = self.series.index
        windows_time_stamps = sliding_window_view(indexes, window_shape=win_size)
        time_stamps_batch = windows_time_stamps[np.array(index_list) - win_size]
        self.current_sample_count += len(index_list)
        return {"input": predict_batch, "time_stamps": time_stamps_batch}

    def make_batch_eval(self, horizon: int) -> dict:
        """
        Return all data to be used for batch evaluation.

        :param horizon: The size of horizon.
        :return: All data to be used for batch evaluation.
        """
        series = self.series.values
        horizons = sliding_window_view(series, window_shape=(horizon, series.shape[1]))
        test_batch = horizons[np.array(self.index_list)]
        return {
            "target": np.squeeze(test_batch, axis=1),
        }

    def has_more_batches(self) -> bool:
        """
        Check if there are more batches to process.

        :return: True if there are more batches, False otherwise.
        """
        return self.current_sample_count < len(self.index_list)


class RollingForecastPredictBatchMaker(BatchMaker):

    def __init__(self, batch_maker: RollingForecastEvalBatchMaker):
        self._batch_maker = batch_maker

    def make_batch(self, batch_size: int, win_size: int) -> dict:
        """
        Return a batch of data to be used for batch prediction.

        :param batch_size: The size of batch.
        :param win_size: The length of data used for prediction.
        :return: A batch of data.
        """
        return self._batch_maker.make_batch_predict(batch_size, win_size)

    def has_more_batches(self) -> bool:
        """
        Check if there are more batches to process.

        :return: True if there are more batches, False otherwise.
        """
        return self._batch_maker.has_more_batches()


class RollingForecast(ForecastingStrategy):
    """
    Rolling forecast strategy class

    This strategy defines a forecasting task that fits once on the training set and
    forecasts on the testing set in a rolling window style.

    The required strategy configs include:

    - horizon (int): The length of each prediction;
    - tv_ratio (float): The ratio of the train-validation series when performing
      train-test split;
    - train_ratio_in_tv (float): The ratio of the training series when performing
      train-validation split;
    - stride (int): Rolling stride, i.e. the interval between two windows;
    - num_rollings (int): The maximum number of steps to forecast;

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
        "tv_ratio",
        "train_ratio_in_tv",
        "stride",
        "num_rollings",
    ]

    @staticmethod
    def _get_index(
        train_length: int, test_length: int, horizon: int, stride: int
    ) -> List[int]:
        """
        Get the index list of the rolling windows.

        :param train_length: Training data length.
        :param test_length: Test data length.
        :param horizon: Prediction length.
        :param stride: Rolling stride.
        :return: Index list of the rolling windows.
        """
        data_len = train_length + test_length
        index_list = list(range(train_length, data_len - horizon + 1, stride)) + (
            [data_len - horizon] if (test_length - horizon) % stride != 0 else []
        )
        return index_list

    def _get_split_lens(
        self,
        series: pd.DataFrame,
        meta_info: Optional[pd.Series],
        tv_ratio: float,
    ) -> Tuple[int, int]:
        """
        Gets the size of the train-validation series and the test series

        :param series: Target series.
        :param meta_info: Meta-information of the target series.
        :param tv_ratio: The ratio of the train-validation series when performing
            train-test split;
        :return: The length of the train-validation series, and the length of the test series.
        """
        data_len = int(self._get_meta_info(meta_info, "length", len(series)))
        train_length = int(tv_ratio * data_len)
        test_length = data_len - train_length
        if train_length <= 0 or test_length <= 0:
            raise ValueError(
                "The length of training or testing data is less than or equal to 0"
            )
        return train_length, test_length

    def _execute(
        self,
        series: pd.DataFrame,
        meta_info: Optional[pd.Series],
        model_factory: ModelFactory,
        series_name: str,
    ) -> List:
        """
        The entry function of execution pipeline of forecasting tasks

        :param series: Target series to evaluate.
        :param meta_info: The corresponding meta-info.
        :param model_factory: The factory to create models.
        :param series_name: the name of the target series.
        :return: The evaluation results.
        """
        model = model_factory()
        if model.batch_forecast.__annotations__.get("not_implemented_batch"):
            return self._eval_sample(series, meta_info, model, series_name)
        else:
            return self._eval_batch(series, meta_info, model, series_name)

    def _eval_sample(
        self,
        series: pd.DataFrame,
        meta_info: Optional[pd.Series],
        model: ModelBase,
        series_name: str,
    ) -> List:
        """
        The sample execution pipeline of forecasting tasks.

        :param series: Target series to evaluate.
        :param meta_info: The corresponding meta-info.
        :param model: The model used for prediction.
        :param series_name: the name of the target series.
        :return: The evaluation results.
        """
        stride = self._get_scalar_config_value("stride", series_name)
        horizon = self._get_scalar_config_value("horizon", series_name)
        num_rollings = self._get_scalar_config_value("num_rollings", series_name)
        train_ratio_in_tv = self._get_scalar_config_value(
            "train_ratio_in_tv", series_name
        )
        tv_ratio = self._get_scalar_config_value("tv_ratio", series_name)

        train_length, test_length = self._get_split_lens(series, meta_info, tv_ratio)
        train_valid_data, test_data = split_before(series, train_length)

        start_fit_time = time.time()
        fit_method = model.forecast_fit if hasattr(model, "forecast_fit") else model.fit
        fit_method(train_valid_data, train_ratio_in_tv=train_ratio_in_tv)
        end_fit_time = time.time()

        eval_scaler = self._get_eval_scaler(train_valid_data, train_ratio_in_tv)

        index_list = self._get_index(train_length, test_length, horizon, stride)
        total_inference_time = 0
        all_test_results = []
        all_rolling_actual = []
        all_rolling_predict = []
        for i, index in itertools.islice(enumerate(index_list), num_rollings):
            train, rest = split_before(series, index)
            test, _ = split_before(rest, horizon)

            start_inference_time = time.time()
            predict = model.forecast(horizon, train)
            end_inference_time = time.time()
            total_inference_time += end_inference_time - start_inference_time

            single_series_result = self.evaluator.evaluate(
                test.to_numpy(), predict, eval_scaler, train_valid_data.values
            )
            inference_data = pd.DataFrame(
                predict, columns=test.columns, index=test.index
            )

            all_rolling_actual.append(test)
            all_rolling_predict.append(inference_data)
            all_test_results.append(single_series_result)

        average_inference_time = float(total_inference_time) / min(
            len(index_list), num_rollings
        )
        single_series_results = np.mean(np.stack(all_test_results), axis=0).tolist()
        # we do not save rolling results by default because it is often too large
        single_series_results += [
            series_name,
            end_fit_time - start_fit_time,
            average_inference_time,
            np.nan,
            np.nan,
            "",
        ]
        return single_series_results

    def _eval_batch(
        self,
        series: pd.DataFrame,
        meta_info: Optional[pd.Series],
        model: ModelBase,
        series_name: str,
    ) -> List:
        """
        The batch execution pipeline of forecasting tasks.

        :param series: Target series to evaluate.
        :param meta_info: The corresponding meta-info.
        :param model: The model used for prediction.
        :param series_name: The name of the target series.
        :return: The evaluation results.
        """
        stride = self._get_scalar_config_value("stride", series_name)
        horizon = self._get_scalar_config_value("horizon", series_name)
        num_rollings = self._get_scalar_config_value("num_rollings", series_name)

        train_ratio_in_tv = self._get_scalar_config_value(
            "train_ratio_in_tv", series_name
        )
        tv_ratio = self._get_scalar_config_value("tv_ratio", series_name)

        train_length, test_length = self._get_split_lens(series, meta_info, tv_ratio)
        train_valid_data, test_data = split_before(series, train_length)

        start_fit_time = time.time()
        fit_method = model.forecast_fit if hasattr(model, "forecast_fit") else model.fit
        fit_method(train_valid_data, train_ratio_in_tv=train_ratio_in_tv)
        end_fit_time = time.time()

        eval_scaler = self._get_eval_scaler(train_valid_data, train_ratio_in_tv)

        index_list = self._get_index(train_length, test_length, horizon, stride)
        index_list = index_list[:num_rollings]

        batch_maker = RollingForecastEvalBatchMaker(
            series,
            index_list,
        )

        all_predicts = []
        total_inference_time = 0
        predict_batch_maker = RollingForecastPredictBatchMaker(batch_maker)
        while predict_batch_maker.has_more_batches():
            start_inference_time = time.time()
            predicts = model.batch_forecast(horizon, predict_batch_maker)
            end_inference_time = time.time()
            total_inference_time += end_inference_time - start_inference_time
            all_predicts.append(predicts)

        all_predicts = np.concatenate(all_predicts, axis=0)
        targets = batch_maker.make_batch_eval(horizon)["target"]
        if len(targets) != len(all_predicts):
            raise RuntimeError("Predictions' len don't equal targets' len!")

        all_test_results = []
        for predicts, target in zip(all_predicts, targets):
            single_series_results = self.evaluator.evaluate(
                target,
                predicts,
                eval_scaler,
                train_valid_data.values,
            )
            all_test_results.append(single_series_results)
        single_series_results = np.mean(np.stack(all_test_results), axis=0).tolist()

        average_inference_time = float(total_inference_time) / min(
            len(index_list), num_rollings
        )
        single_series_results += [
            series_name,
            end_fit_time - start_fit_time,
            average_inference_time,
            np.nan,
            np.nan,
            "",
        ]
        return single_series_results

    @staticmethod
    def accepted_metrics() -> List[str]:
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
