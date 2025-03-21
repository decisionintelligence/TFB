# -*- coding: utf-8 -*-
import abc

import numpy as np
import pandas as pd
from typing import List, Optional, Tuple, Dict


def annotate(**kwargs):
    """
    Decorate a function to add or update its annotations.

    :param kwargs: Keyword arguments representing the annotations to be added or updated.
    :return: A wrapper function that updates the annotations of the original function.
    """

    def wrapper(func):
        func.__annotations__.update(kwargs)
        return func

    return wrapper


class BatchMaker(metaclass=abc.ABCMeta):
    """
    The standard interface of batch maker.

    """

    @abc.abstractmethod
    def make_batch(self, batch_size: int, win_size: int) -> dict:
        """
        Provide a batch of data to be used for batch prediction.

        :param batch_size: The length of one batch.
        :param win_size: The length of data for one prediction.
        :return: A batch of data for prediction.
        """


class ModelBase(metaclass=abc.ABCMeta):
    """
    The standard interface of benchmark-compatible models.

    Users are recommended to inherit this class to implement or adapt their own models.
    """

    @abc.abstractmethod
    def forecast_fit(
        self,
        train_valid_data: pd.DataFrame,
        *,
        covariates: Optional[dict] = None,
        train_ratio_in_tv: float = 1.0,
        **kwargs,
    ) -> "ModelBase":
        """
        Fit a model on time series data

        :param train_valid_data: Time series data used for training and validation.
        :param covariates: Additional external variables
            key "exog" : the corresponding exogenous data aligned with `train_data`.
        :param train_ratio_in_tv: Represents the splitting ratio of the training set validation set.
            If it is equal to 1, it means that the validation set is not partitioned.
        :return: The fitted model object.
        """

    @abc.abstractmethod
    def forecast(
        self,
        horizon: int,
        series: pd.DataFrame,
        *,
        covariates: Optional[dict] = None,
    ) -> np.ndarray:
        """
        Forecasting with the model

        TODO: support returning DataFrames

        :param horizon: Forecast length.
        :param series: Time series data to make inferences on.
        :param covariates: Additional external variables.
            key "exog" : the corresponding exogenous data aligned with `series`.
        :return: Forecast result.
        """

    @annotate(not_implemented_batch=True)
    def batch_forecast(
        self, horizon: int, batch_maker: BatchMaker, **kwargs
    ) -> np.ndarray:
        """
        Perform batch forecasting with the model.

        :param horizon: The length of each prediction.
        :param batch_maker: Make batch data used for prediction.
        :return: The prediction result.
        """
        raise NotImplementedError("Not implemented batch forecasting!")

    @property
    @abc.abstractmethod
    def model_name(self):
        """
        Returns the name of the model.
        """

        pass

    def __repr__(self):
        return self.model_name
