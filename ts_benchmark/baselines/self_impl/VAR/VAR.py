import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.api import VAR as VARModel

from ts_benchmark.models.model_base import ModelBase


class VAR(ModelBase):
    """
    VAR class.

    This class encapsulates a process of using VAR models for time series prediction.
    """

    def __init__(self, lags=13):
        self.scaler = StandardScaler()
        self.lags = lags
        self.results = None

    @property
    def model_name(self):
        """
        Returns the name of the model.
        """
        return "VAR"

    @staticmethod
    def required_hyper_params() -> dict:
        """
        Return the hyperparameters required by VAR.

        :return: An empty dictionary indicating that VAR does not require additional hyperparameters.
        """
        return {}

    def forecast_fit(
        self, train_data: pd.DataFrame, *, train_ratio_in_tv: float = 1.0, **kwargs
    ) -> "ModelBase":
        """
        Train the model.

        :param train_data: Time series data used for training.
        :param train_ratio_in_tv: Represents the splitting ratio of the training set validation set. If it is equal to 1, it means that the validation set is not partitioned.
        :return: The fitted model object.
        """

        self.scaler.fit(train_data.values)
        train_data_value = pd.DataFrame(
            self.scaler.transform(train_data.values),
            columns=train_data.columns,
            index=train_data.index,
        )
        model = VARModel(train_data_value)
        self.results = model.fit(self.lags)

        return self

    def forecast(self, horizon: int, series: pd.DataFrame, **kwargs) -> np.ndarray:
        """
        Make predictions.

        :param horizon: The predicted length.
        :param series: Time series data used for prediction.
        :return: An array of predicted results.
        """
        train = pd.DataFrame(
            self.scaler.transform(series.values),
            columns=series.columns,
            index=series.index,
        )
        z = self.results.forecast(train.values, steps=horizon)

        predict = self.scaler.inverse_transform(z)

        return predict


# TODO: VAR_model is kept for backward compatibility, remove all references to VAR_model in the future
VAR_model = VAR
