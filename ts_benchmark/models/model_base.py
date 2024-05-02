# -*- coding: utf-8 -*-
import abc

import numpy as np
import pandas as pd


class ModelBase(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def forecast_fit(
        self, train_data: pd.DataFrame, *, train_ratio_in_tv: float = 1.0, **kwargs
    ) -> "ModelBase":
        """
        Fit a model on time series data

        :param train_data: Time series data.
        :param train_ratio_in_tv: Represents the splitting ratio of the training set validation set.
            If it is equal to 1, it means that the validation set is not partitioned.
        :return: The fitted model object.
        """

    @abc.abstractmethod
    def forecast(self, horizon: int, series: pd.DataFrame, **kwargs) -> np.ndarray:
        """
        Forecasting with the model

        TODO: support returning DataFrames

        :param horizon: Forecast length.
        :param series: Time series data to make inferences on.
        :return: Forecast result.
        """
