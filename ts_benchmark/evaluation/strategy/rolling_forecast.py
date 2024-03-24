# -*- coding: utf-8 -*-
import base64
import pickle
import time
import traceback
from typing import List, Any

import numpy as np
import pandas as pd

from ts_benchmark.data_loader.data_pool import DataPool
from ts_benchmark.evaluation.evaluator import Evaluator
from ts_benchmark.evaluation.metrics import regression_metrics
from ts_benchmark.evaluation.strategy.constants import FieldNames
from ts_benchmark.evaluation.strategy.strategy import Strategy
from ts_benchmark.models.get_model import ModelFactory
from ts_benchmark.utils.data_processing import split_before
from ts_benchmark.utils.random_utils import fix_random_seed

SPLIT_DICT = {
    "ETTh1.csv": 0.75,
    "ETTh2.csv": 0.75,
    "ETTm1.csv": 0.75,
    "ETTm2.csv": 0.75,
    "PEMS03.csv": 0.75,
    "PEMS04.csv": 0.75,
    "PEMS07.csv": 0.75,
    "PEMS08.csv": 0.75,
    "Solar.csv": 0.75,
    "AQShunyi.csv": 0.75,
    "AQWan.csv": 0.75,
}


class RollingForecast(Strategy):
    REQUIRED_FIELDS = ["pred_len", "train_test_split", "stride", "num_rollings"]

    """
    Rolling forecast strategy class, used to perform rolling prediction on time series data.

    """

    def __init__(self, strategy_config: dict, evaluator: Evaluator):
        """
        Initialize the rolling forecast strategy object.


        :param strategy_config: Model evaluation configuration.
        """
        super().__init__(strategy_config, evaluator)
        self.data_lens = None
        self.pred_len = self.strategy_config["pred_len"]
        self.num_rollings = self.strategy_config["num_rollings"]

    def _get_index(self, test_length: int, train_length: int) -> List[int]:
        """
        Get the index list of the scrolling window.

        :param test_length: Test data length.
        :param train_length: Training data length.
        :return: Scroll through the index list of the window.
        """
        stride = self.strategy_config["stride"]
        index_list = list(
            range(train_length, self.data_lens - self.pred_len + 1, stride)
        ) + (
            [self.data_lens - self.pred_len]
            if (test_length - self.pred_len) % stride != 0
            else []
        )
        return index_list

    def execute(self, series_name: str, model_factory: ModelFactory) -> Any:
        """
        Execute rolling prediction strategy.

        :param series_name: The name of the sequence to be predicted.
        :param model_factory: Construction of model objects/factory functions.
        :return: The average of the evaluation results.
        """
        fix_random_seed()
        model = model_factory()

        data = DataPool().get_series(series_name)
        self.data_lens = int(
            DataPool().get_series_meta_info(series_name)["length"].item()
        )
        try:
            all_test_results = []

            train_length = int(
                self.strategy_config["train_test_split"] * self.data_lens
            )
            test_length = self.data_lens - train_length
            if train_length <= 0 or test_length <= 0:
                raise ValueError(
                    "The length of training or testing data is less than or equal to 0"
                )
            train_valid_data, test_data = split_before(data, train_length)

            train_data, rest = split_before(train_valid_data, int(train_length * SPLIT_DICT.get(series_name, self.strategy_config["train_valid_split"])))
            print(SPLIT_DICT.get(series_name, self.strategy_config["train_valid_split"]))
            self.scaler.fit(train_data.values)

            start_fit_time = time.time()
            if hasattr(model, "forecast_fit"):
                model.forecast_fit(train_valid_data, SPLIT_DICT.get(series_name, self.strategy_config["train_valid_split"]))
            else:
                model.fit(train_valid_data, SPLIT_DICT.get(series_name, self.strategy_config["train_valid_split"]))
            end_fit_time = time.time()
            index_list = self._get_index(test_length, train_length)
            total_inference_time = 0
            all_rolling_actual = []
            all_rolling_predict = []
            for i in range(min(len(index_list), self.num_rollings)):
                index = index_list[i]
                train, other = split_before(data, index)
                test, rest = split_before(other, self.pred_len)
                start_inference_time = time.time()
                predict = model.forecast(self.pred_len, train)
                end_inference_time = time.time()
                total_inference_time += end_inference_time - start_inference_time

                actual = test.to_numpy()
                single_series_result = self.evaluator.evaluate(
                    actual, predict, self.scaler, train_valid_data.values
                )

                inference_data = pd.DataFrame(
                    predict, columns=test.columns, index=test.index
                )

                all_rolling_actual.append(test)
                all_rolling_predict.append(inference_data)
                all_test_results.append(single_series_result)
            average_inference_time = float(total_inference_time) / min(
                len(index_list), self.num_rollings
            )
            single_series_results = np.mean(
                np.stack(all_test_results), axis=0
            ).tolist()
            print(single_series_results)
            all_rolling_actual_pickle = pickle.dumps(all_rolling_actual)
            # Encoding using base64
            all_rolling_actual_pickle = base64.b64encode(
                all_rolling_actual_pickle
            ).decode("utf-8")

            all_rolling_predict_pickle = pickle.dumps(all_rolling_predict)
            # Encoding using base64
            all_rolling_predict_pickle = base64.b64encode(
                all_rolling_predict_pickle
            ).decode("utf-8")

            # single_series_results += [
            #     series_name,
            #     end_fit_time - start_fit_time,
            #     average_inference_time,
            #     all_rolling_actual_pickle,
            #     all_rolling_predict_pickle,
            #     "",
            # ]
            single_series_results += [
                series_name,
                end_fit_time - start_fit_time,
                average_inference_time,
                np.nan,
                np.nan,
                "",
            ]
        except Exception as e:
            log = f"{traceback.format_exc()}\n{e}"
            single_series_results = self.get_default_result(
                **{FieldNames.LOG_INFO: log}
            )

        return single_series_results

    @staticmethod
    def accepted_metrics() -> List[str]:
        """
        Obtain a list of evaluation metrics accepted by the rolling prediction strategy.

        :return: List of evaluation metrics.
        """
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
