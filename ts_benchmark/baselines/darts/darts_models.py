# -*- coding: utf-8 -*-
import contextlib
import functools
import logging
import os
from typing import Dict, Optional, Any, Tuple, NoReturn, ContextManager

import darts
import darts.models as darts_models
import numpy as np
import pandas as pd
from darts import TimeSeries
from sklearn.preprocessing import StandardScaler

from ts_benchmark.baselines.utils import train_val_split
from ts_benchmark.common.constant import ROOT_PATH
from ts_benchmark.models import ModelBase

if darts.__version__ >= "0.25.0":
    from darts.models.utils import NotImportedModule

logger = logging.getLogger(__name__)

TAG = pd.read_csv(
    os.path.join(ROOT_PATH, "ts_benchmark", "baselines", "tag_csv", "darts_tag.csv")
)


class DartsConfig:
    def __init__(self, **kwargs):
        self.params = {
            **kwargs,
        }

    def __getattr__(self, key: str) -> Any:
        return self.get(key)

    def __getitem__(self, key: str) -> Any:
        return self.get(key)

    def get(self, key: str, default: Any = None) -> Any:
        return self.params.get(key, default)

    def get_darts_class_params(self) -> dict:
        ret = self.params.copy()
        ret.pop("norm")
        self._fix_multi_gpu(ret)
        return ret

    def _fix_multi_gpu(self, args_dict: Dict) -> NoReturn:
        """
        Check and disable using multi-gpu per task

        training and inferencing on multiple gpus with 'ddp' strategy (default in lightning)
        is error-prone in complicated work flow, the problems include but not limited to:

        - do heavy initialization in all processes (e.g. full data loading)
        - hangs when the program is interrupted (e.g. exceptions that are caught elsewhere)
        - not compatible with the parallel paradigm of ray

        As a result, we disallow a single worker to work on multiple gpus by changing
        gpu settings in the the input argument dictionary.

        TODO: what if we want to evaluate huge-sized models in the future?

        :param args_dict: argument dictionary to be passed to Darts models.
        """
        # CUDA_VISIBLE_DEVICES should be set by the parallel backend
        gpu_devices = list(
            filter(None, os.environ.get("CUDA_VISIBLE_DEVICES", "").split(","))
        )
        if len(gpu_devices) > 1:
            pl_args = args_dict.get("pl_trainer_kwargs", {})
            device_args = pl_args.get("devices", None)
            if (
                device_args is None
                or (isinstance(device_args, list) and len(device_args) > 1)
                or (isinstance(device_args, int) and device_args > 1)
            ):
                args_dict.setdefault("pl_trainer_kwargs", {})
                args_dict["pl_trainer_kwargs"]["devices"] = [0]
                logger.warning(
                    "Multi-gpu training is not supported, using only gpu %s",
                    gpu_devices[0],
                )


class DartsModelAdapter(ModelBase):
    """
    Darts model adapter class

    Adapts Darts models to OTB forecasting interface.
    """

    def __init__(
        self,
        model_class: type,
        model_args: dict,
        model_name: Optional[str] = None,
        allow_fit_on_eval: bool = False,
        supports_validation: bool = False,
        **kwargs
    ):
        """
        Initialize the Darts model adapter object.

        :param model_class: Darts model class.
        :param model_args: Model initialization parameters.
        :param model_name: Model name.
        :param allow_fit_on_eval: Is it allowed to fit the model during the prediction phase.
        :param supports_validation: Whether the model supports inputting a validation series.
        :param kwargs: other arguments added to model_args.
        """
        self.model = None
        self.model_class = model_class
        self.config = DartsConfig(**{**model_args, **kwargs})
        self._model_name = model_name
        self.allow_fit_on_eval = allow_fit_on_eval
        self.supports_validation = supports_validation
        self.scaler = StandardScaler()
        self.train_ratio_in_tv = 1

    @property
    def model_name(self):
        """
        Returns the name of the model.
        """

        return self._model_name

    def forecast_fit(
        self, train_data: pd.DataFrame, *, train_ratio_in_tv: float = 1.0, **kwargs
    ) -> "ModelBase":
        """
        Fit a suitable Darts model on time series data.

        :param train_data: Time series data.
        :param train_ratio_in_tv: Represents the splitting ratio of the training set validation set.
            If it is equal to 1, it means that the validation set is not partitioned.
        :return: The fitted model object.
        """
        self.train_ratio_in_tv = train_ratio_in_tv
        if self.allow_fit_on_eval or self.model_name == "RegressionModel":
            # If it is true, it means that statistical learning methods use retraining to predict
            # future values, because statistical learning does not require partitioning the validation set.
            # Therefore, the segmentation ratio is set to 1, which means that the validation set
            # is not segmented
            valid_data = None
        else:
            train_data, valid_data = train_val_split(
                train_data,
                self.train_ratio_in_tv,
                self.config.get("input_chunk_length", 0),
            )

        self.model = self.model_class(**self.config.get_darts_class_params())
        if self.config.norm:
            self.scaler.fit(train_data.values)
            train_data = pd.DataFrame(
                self.scaler.transform(train_data.values),
                columns=train_data.columns,
                index=train_data.index,
            )
            if self.supports_validation and valid_data is not None:
                valid_data = pd.DataFrame(
                    self.scaler.transform(valid_data.values),
                    columns=valid_data.columns,
                    index=valid_data.index,
                )

        with self._suppress_lightning_logs():
            train_data = TimeSeries.from_dataframe(train_data)
            if self.supports_validation and valid_data is not None:
                valid_data = TimeSeries.from_dataframe(valid_data)
                self.model.fit(train_data, val_series=valid_data)
            else:
                self.model.fit(train_data)
        return self

    def forecast(self, horizon: int, series: pd.DataFrame, **kwargs) -> np.ndarray:
        """
        Use the adapted Darts model for prediction.

        :param horizon: Forecast length.
        :param series: Time series data to make inferences on.
        :return: Forecast result.
        """
        if self.config.norm:
            series = pd.DataFrame(
                self.scaler.transform(series.values),
                columns=series.columns,
                index=series.index,
            )

        with self._suppress_lightning_logs():
            if self.allow_fit_on_eval:
                self.forecast_fit(series, train_ratio_in_tv=self.train_ratio_in_tv)
                fsct_result = self.model.predict(horizon)
            else:
                series = TimeSeries.from_dataframe(series)
                fsct_result = self.model.predict(horizon, series)
        predict = fsct_result.values()

        if self.config.norm:
            predict = self.scaler.inverse_transform(predict)

        return predict

    @contextlib.contextmanager
    def _suppress_lightning_logs(self) -> ContextManager:
        pl_logger = logging.getLogger("pytorch_lightning")
        old_level = pl_logger.level
        pl_logger.setLevel(logging.CRITICAL)
        try:
            yield
        finally:
            pl_logger.setLevel(old_level)


def _generate_model_factory(
    model_class: type,
    model_args: dict,
    model_name: str,
    required_args: dict,
    allow_fit_on_eval: bool,
    supports_validation: bool,
) -> Dict:
    """
    Generate model factory information for creating Darts model adapters.

    :param model_name: Model name.
    :param model_class: Darts model class.
    :param model_args: Predefined model hyperparameters that can be overwritten by the hyperparameters of the input factory function.
    :param required_args: Requires hyperparameters recommended by benchmark.
    :param allow_fit_on_eval: Is it allowed to fit the model during the prediction phase.
    :param supports_validation: Whether the model supports inputting a validation series.
    :return: A dictionary containing the model factory and required parameters.
    """
    model_factory = functools.partial(
        DartsModelAdapter,
        model_class=model_class,
        model_args=model_args,
        model_name=model_name,
        allow_fit_on_eval=allow_fit_on_eval,
        supports_validation=supports_validation,
    )

    return {"model_factory": model_factory, "required_hyper_params": required_args}


# predefined model_args and required_args for darts models
DEEP_MODEL_REQUIRED_ARGS = {
    "input_chunk_length": "input_chunk_length",
    "output_chunk_length": "output_chunk_length",
    "norm": "norm",
}
REGRESSION_MODEL_REQUIRED_ARGS = {
    "lags": "input_chunk_length",
    "output_chunk_length": "output_chunk_length",
    "norm": "norm",
}
STAT_MODEL_REQUIRED_ARGS = {
    "norm": "norm",
}
DEEP_MODEL_ARGS = {
    "pl_trainer_kwargs": {
        "enable_progress_bar": False,
    }
}


def _get_model_info(model_name: str, required_args: Dict, model_args: Dict) -> Tuple:
    """
    Helper function to retrieve darts model information by name

    :param model_name: name of the model.
    :param required_args: arguments that the model requires from the pipeline.
    :param model_args: specified model arguments.
    :return: a tuple including model name, model_class, required args and model args.
    """
    model_class = getattr(darts_models, model_name, None)
    if darts.__version__ >= "0.25.0" and isinstance(model_class, NotImportedModule):
        model_class = None
    return model_name, model_class, required_args, model_args


# deep models implemented by darts
DARTS_DEEP_MODELS = [
    _get_model_info("TCNModel", DEEP_MODEL_REQUIRED_ARGS, DEEP_MODEL_ARGS),
    _get_model_info(
        "TFTModel",
        DEEP_MODEL_REQUIRED_ARGS,
        DEEP_MODEL_ARGS,
    ),
    _get_model_info("TransformerModel", DEEP_MODEL_REQUIRED_ARGS, DEEP_MODEL_ARGS),
    _get_model_info("NHiTSModel", DEEP_MODEL_REQUIRED_ARGS, DEEP_MODEL_ARGS),
    _get_model_info("TiDEModel", DEEP_MODEL_REQUIRED_ARGS, DEEP_MODEL_ARGS),
    _get_model_info("BlockRNNModel", DEEP_MODEL_REQUIRED_ARGS, DEEP_MODEL_ARGS),
    _get_model_info("RNNModel", DEEP_MODEL_REQUIRED_ARGS, DEEP_MODEL_ARGS),
    _get_model_info("DLinearModel", DEEP_MODEL_REQUIRED_ARGS, DEEP_MODEL_ARGS),
    _get_model_info("NBEATSModel", DEEP_MODEL_REQUIRED_ARGS, DEEP_MODEL_ARGS),
    _get_model_info("NLinearModel", DEEP_MODEL_REQUIRED_ARGS, DEEP_MODEL_ARGS),
]

# regression models implemented by darts
DARTS_REGRESSION_MODELS = [
    _get_model_info("RandomForest", REGRESSION_MODEL_REQUIRED_ARGS, {}),
    _get_model_info("XGBModel", REGRESSION_MODEL_REQUIRED_ARGS, {}),
    _get_model_info("CatBoostModel", REGRESSION_MODEL_REQUIRED_ARGS, {}),
    _get_model_info("LightGBMModel", REGRESSION_MODEL_REQUIRED_ARGS, {}),
    _get_model_info("LinearRegressionModel", REGRESSION_MODEL_REQUIRED_ARGS, {}),
    _get_model_info("RegressionModel", REGRESSION_MODEL_REQUIRED_ARGS, {}),
]

# statistical models implemented by darts,
# these models are specially allowed to retrain during inference
DARTS_STAT_MODELS = [
    _get_model_info("KalmanForecaster", STAT_MODEL_REQUIRED_ARGS, {}),
    _get_model_info("ARIMA", STAT_MODEL_REQUIRED_ARGS, {}),
    _get_model_info("VARIMA", STAT_MODEL_REQUIRED_ARGS, {}),
    _get_model_info("AutoARIMA", STAT_MODEL_REQUIRED_ARGS, {}),
    _get_model_info("StatsForecastAutoCES", STAT_MODEL_REQUIRED_ARGS, {}),
    _get_model_info("StatsForecastAutoTheta", STAT_MODEL_REQUIRED_ARGS, {}),
    _get_model_info("StatsForecastAutoETS", STAT_MODEL_REQUIRED_ARGS, {}),
    _get_model_info("ExponentialSmoothing", STAT_MODEL_REQUIRED_ARGS, {}),
    _get_model_info("StatsForecastAutoARIMA", STAT_MODEL_REQUIRED_ARGS, {}),
    _get_model_info("FFT", STAT_MODEL_REQUIRED_ARGS, {}),
    _get_model_info("FourTheta", STAT_MODEL_REQUIRED_ARGS, {}),
    _get_model_info("Croston", STAT_MODEL_REQUIRED_ARGS, {}),
    _get_model_info("NaiveDrift", STAT_MODEL_REQUIRED_ARGS, {}),
    _get_model_info("NaiveMean", STAT_MODEL_REQUIRED_ARGS, {}),
    _get_model_info("NaiveSeasonal", STAT_MODEL_REQUIRED_ARGS, {}),
    _get_model_info("NaiveMovingAverage", STAT_MODEL_REQUIRED_ARGS, {}),
]

# Generate model factories for each model class and required parameters in DARTS_DEEP_MODELS
# and add them to global variables
for _model_name, _model_class, _required_args, _model_args in DARTS_DEEP_MODELS:
    if _model_class is None:
        logger.warning(
            "Model %s is not available, skipping model registration", _model_name
        )
        globals()[_model_name] = None
        continue
    globals()[_model_name] = _generate_model_factory(
        model_class=_model_class,
        model_args=_model_args,
        model_name=_model_name,
        required_args=_required_args,
        allow_fit_on_eval=False,
        supports_validation=True,
    )

# Generate model factories for each model class and required parameters in DARTS_REGRESSION_MODELS
# and add them to global variables
for _model_name, _model_class, _required_args, _model_args in DARTS_REGRESSION_MODELS:
    if _model_class is None:
        logger.warning(
            "Model %s is not available, skipping model registration", _model_name
        )
        globals()[_model_name] = None
        continue
    globals()[_model_name] = _generate_model_factory(
        model_class=_model_class,
        model_args=_model_args,
        model_name=_model_name,
        required_args=_required_args,
        allow_fit_on_eval=False,
        supports_validation=False,
    )

# Generate model factories for each model class and required parameters in DARTS_STAT_MODELS
# and add them to global variables
for _model_name, _model_class, _required_args, _model_args in DARTS_STAT_MODELS:
    if _model_class is None:
        logger.warning(
            "Model %s is not available, skipping model registration", _model_name
        )
        globals()[_model_name] = None
        continue
    globals()[_model_name] = _generate_model_factory(
        model_class=_model_class,
        model_args=_model_args,
        model_name=_model_class.__name__,
        required_args=_required_args,
        allow_fit_on_eval=True,
        supports_validation=False,
    )


# Adapters for general darts models


def darts_deep_model_adapter(model_class: type) -> Dict:
    """
    Adapts a Darts deep model class to OTB protocol

    :param model_class: a class of deep forecasting model from Darts library.
    :return: model factory that follows the OTB protocol.
    """
    return _generate_model_factory(
        model_class,
        DEEP_MODEL_ARGS,
        model_class.__name__,
        DEEP_MODEL_REQUIRED_ARGS,
        allow_fit_on_eval=False,
        supports_validation=True,
    )


def darts_statistical_model_adapter(model_class: type) -> Dict:
    """
    Adapts a Darts statistical model class to OTB protocol

    :param model_class: a class of statistical forecasting model from Darts library.
    :return: model factory that follows the OTB protocol.
    """
    return _generate_model_factory(
        model_class,
        {},
        model_class.__name__,
        STAT_MODEL_REQUIRED_ARGS,
        allow_fit_on_eval=True,
        supports_validation=False,
    )


def darts_regression_model_adapter(model_class: type) -> Dict:
    """
    Adapts a Darts regression model class to OTB protocol

    :param model_class: a class of regression forecasting model from Darts library.
    :return: model factory that follows the OTB protocol.
    """
    return _generate_model_factory(
        model_class,
        {},
        model_class.__name__,
        REGRESSION_MODEL_REQUIRED_ARGS,
        allow_fit_on_eval=True,
        supports_validation=False,
    )