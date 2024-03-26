import logging
import os
from typing import Dict, Optional, Any

import darts
import numpy as np
import pandas as pd
from darts import TimeSeries
import darts.models as model
from sklearn.preprocessing import StandardScaler

from ts_benchmark.baselines.utils import train_val_split
from ts_benchmark.common.constant import ROOT_PATH

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
        return self.params.get(key, 0)

    def get_param_dict(self) -> dict:
        ret = self.params.copy()
        ret.pop("normalization")
        return ret


class DartsModelAdapter:
    """
    Darts model adapter class, used to adapt models in the Darts framework to meet the requirements of prediction strategies.
    """

    def __init__(
        self,
        model_class: type,
        model_args: dict,
        model_name: Optional[str] = None,
        allow_fit_on_eval: bool = False,
    ):
        """
        Initialize the Darts model adapter object.

        :param model_class: Darts model class.
        :param model_args: Model initialization parameters.
        :param model_name: Model name.
        :param allow_fit_on_eval: Is it allowed to fit the model during the prediction phase.
        """
        self.model = None
        self.model_class = model_class
        self.config = DartsConfig(**model_args)
        self.model_name = model_name
        self.allow_fit_on_eval = allow_fit_on_eval
        self.scaler = StandardScaler()
        self.train_val_ratio = 1

    def forecast_fit(
        self, train_valid_data: pd.DataFrame, train_val_ratio: float
    ) -> object:
        """
        Fit a suitable Darts model on time series data.

        :param series: Time series data.
        :param train_val_ratio: Represents the splitting ratio of the training set validation set. If it is equal to 1, it means that the validation set is not partitioned.
        :return: The fitted model object.
        """
        # TODO: training and inferencing on multiple gpus with 'ddp' strategy is error prone
        #  in complicated work flow, the problems include but not limited to:
        #  - do heavy initialization in all processes (e.g. full data loading)
        #  - hangs when the program is interrupted (e.g. exceptions that are caught
        #  elsewhere)
        #  - not compatible with the parallel paradigm of ray
        #  As a result, we disallow a single worker to work on multiple gpus by now, but what if
        #  evaluating large-scale models is required in the future?
        # gpu_devices = list(
        #     filter(None, os.environ.get("CUDA_VISIBLE_DEVICES", "").split(","))
        # )
        # if gpu_devices:
        #     pl_args = self.model_args.get("pl_trainer_kwargs", {})
        #     device_args = pl_args.get("devices", None)
        #     if (
        #             device_args is None
        #             or (isinstance(device_args, list) and len(device_args) > 1)
        #             or (isinstance(device_args, int) and device_args > 1)
        #     ):
        #         self.model_args.setdefault("pl_trainer_kwargs", {})
        #         # self.model_args["pl_trainer_kwargs"]["devices"] = [int(gpu_devices[0])]
        #         self.model_args["pl_trainer_kwargs"]["devices"] = [0]
        #         logger.warning(
        #             "Multi-gpu training is not supported, using only gpu %s",
        #             self.model_args["pl_trainer_kwargs"]["devices"],
        #         )
        self.train_val_ratio = train_val_ratio
        if self.allow_fit_on_eval or self.model_name == "RegressionModel":
            # If it is true, it means that statistical learning methods use retraining to predict future values, because statistical learning does not require partitioning the validation set.
            # Therefore, the segmentation ratio is set to 1, which means that the validation set is not segmented
            self.train_val_ratio = 1
        train_data, valid_data = train_val_split(
            train_valid_data,
            self.train_val_ratio,
            self.config.__getattr__("input_chunk_length"),
        )

        config_copy = self.config.get_param_dict()
        self.model = self.model_class(**config_copy)
        self.scaler.fit(train_data.values)
        if self.config.normalization:
            train_data = pd.DataFrame(
                self.scaler.transform(train_data.values),
                columns=train_data.columns,
                index=train_data.index,
            )
        train_data = TimeSeries.from_dataframe(train_data)

        if self.train_val_ratio != 1:
            if self.config.normalization:
                valid_data = pd.DataFrame(
                    self.scaler.transform(valid_data.values),
                    columns=valid_data.columns,
                    index=valid_data.index,
                )
            valid_data = TimeSeries.from_dataframe(valid_data)
            return self.model.fit(train_data, val_series=valid_data)
        else:
            return self.model.fit(train_data)

    def forecast(self, pred_len: int, train: pd.DataFrame) -> np.ndarray:
        """
        Use the adapted Darts model for prediction.

        :param pred_len: Predict length.
        :param train: Used to fit the training data of the model.
        :return: Predicted result.
        """
        if self.config.normalization:
            train = pd.DataFrame(
                self.scaler.transform(train.values),
                columns=train.columns,
                index=train.index,
            )
        if self.allow_fit_on_eval:
            self.forecast_fit(train, self.train_val_ratio)
            fsct_result = self.model.predict(pred_len)
        else:
            train = TimeSeries.from_dataframe(train)
            fsct_result = self.model.predict(pred_len, train)
        predict = fsct_result.values()

        if self.config.normalization:
            predict = self.scaler.inverse_transform(predict)
        return predict

    def __repr__(self):
        """
        Returns a string representation of the model name.
        """
        return self.model_name


def generate_model_factory(
    model_class: type,
    model_args: dict,
    model_name: str,
    required_args: dict,
    allow_fit_on_eval: bool,
) -> Dict:
    """
    Generate model factory information for creating Darts model adapters.

    :param model_name: Model name.
    :param model_class: Darts model class.
    :param model_args: Predefined model hyperparameters that can be overwritten by the hyperparameters of the input factory function.
    :param required_args: Requires hyperparameters recommended by benchmark.
    :param allow_fit_on_eval: Is it allowed to fit the model during the prediction phase.
    :return: A dictionary containing the model factory and required parameters.
    """

    def model_factory(**kwargs) -> DartsModelAdapter:
        """
        Model factory, used to create Darts model adapter objects.

        :param kwargs: Model initialization parameters.
        :return: Darts Darts model adapter object.
        """
        return DartsModelAdapter(
            model_class=model_class,
            model_args={**model_args, **kwargs},
            model_name=model_name,
            allow_fit_on_eval=allow_fit_on_eval,
        )

    return {"model_factory": model_factory, "required_hyper_params": required_args}


DARTS_DEEP_MODEL_REQUIRED_ARGS1 = {
    "input_chunk_length": "input_chunk_length",
    "output_chunk_length": "output_chunk_length",
    "normalization": "norm",
}
DARTS_DEEP_MODEL_REQUIRED_ARGS2 = {
    "lags": "input_chunk_length",
    "normalization": "norm",
}
DARTS_DEEP_MODEL_REQUIRED_ARGS3 = {
    "lags": "input_chunk_length",
    "output_chunk_length": "output_chunk_length",
    "normalization": "norm",
}
DARTS_STAT_MODEL_REQUIRED_ARGS1 = {
    "normalization": "norm",
}
DARTS_DEEP_MODEL_ARGS = {
    "pl_trainer_kwargs": {
        "enable_progress_bar": False,
    }
}

DARTS_MODELS = [
    (model.KalmanForecaster, {}, {}),
    (model.TCNModel, DARTS_DEEP_MODEL_REQUIRED_ARGS1, DARTS_DEEP_MODEL_ARGS),
    (
        model.TFTModel,
        DARTS_DEEP_MODEL_REQUIRED_ARGS1,
        DARTS_DEEP_MODEL_ARGS,
    ),
    (model.TransformerModel, DARTS_DEEP_MODEL_REQUIRED_ARGS1, DARTS_DEEP_MODEL_ARGS),
    (model.NHiTSModel, DARTS_DEEP_MODEL_REQUIRED_ARGS1, DARTS_DEEP_MODEL_ARGS),
    (model.TiDEModel, DARTS_DEEP_MODEL_REQUIRED_ARGS1, DARTS_DEEP_MODEL_ARGS),
    (model.BlockRNNModel, DARTS_DEEP_MODEL_REQUIRED_ARGS1, DARTS_DEEP_MODEL_ARGS),
    (model.RNNModel, DARTS_DEEP_MODEL_REQUIRED_ARGS1, DARTS_DEEP_MODEL_ARGS),
    (model.DLinearModel, DARTS_DEEP_MODEL_REQUIRED_ARGS1, DARTS_DEEP_MODEL_ARGS),
    (model.NBEATSModel, DARTS_DEEP_MODEL_REQUIRED_ARGS1, DARTS_DEEP_MODEL_ARGS),
    (model.NLinearModel, DARTS_DEEP_MODEL_REQUIRED_ARGS1, DARTS_DEEP_MODEL_ARGS),
    (model.RandomForest, DARTS_DEEP_MODEL_REQUIRED_ARGS2, {}),
    (model.XGBModel, DARTS_DEEP_MODEL_REQUIRED_ARGS2, DARTS_DEEP_MODEL_ARGS),
    (model.CatBoostModel, DARTS_DEEP_MODEL_REQUIRED_ARGS2, {}),
    (model.LightGBMModel, DARTS_DEEP_MODEL_REQUIRED_ARGS3, {}),
    (model.LinearRegressionModel, DARTS_DEEP_MODEL_REQUIRED_ARGS2, {}),
    (model.RegressionModel, DARTS_DEEP_MODEL_REQUIRED_ARGS2, {}),
    (model.TiDEModel, DARTS_DEEP_MODEL_REQUIRED_ARGS1, DARTS_DEEP_MODEL_ARGS),
]

# The following models specifically allow for retraining during inference
DARTS_STAT_MODELS = [
    (model.ARIMA, DARTS_STAT_MODEL_REQUIRED_ARGS1, {}),
    (model.VARIMA, DARTS_STAT_MODEL_REQUIRED_ARGS1, {}),
    (model.AutoARIMA, DARTS_STAT_MODEL_REQUIRED_ARGS1, {}),
    (model.StatsForecastAutoCES, DARTS_STAT_MODEL_REQUIRED_ARGS1, {}),
    (model.StatsForecastAutoTheta, DARTS_STAT_MODEL_REQUIRED_ARGS1, {}),
    (model.StatsForecastAutoETS, DARTS_STAT_MODEL_REQUIRED_ARGS1, {}),
    (model.ExponentialSmoothing, DARTS_STAT_MODEL_REQUIRED_ARGS1, {}),
    (model.StatsForecastAutoARIMA, DARTS_STAT_MODEL_REQUIRED_ARGS1, {}),
    (model.FFT, DARTS_STAT_MODEL_REQUIRED_ARGS1, {}),
    (model.FourTheta, DARTS_STAT_MODEL_REQUIRED_ARGS1, {}),
    (model.Croston, DARTS_STAT_MODEL_REQUIRED_ARGS1, {}),
    (model.NaiveDrift, DARTS_STAT_MODEL_REQUIRED_ARGS1, {}),
    (model.NaiveMean, DARTS_STAT_MODEL_REQUIRED_ARGS1, {}),
    (model.NaiveSeasonal, DARTS_STAT_MODEL_REQUIRED_ARGS1, {}),
    (model.NaiveMovingAverage, DARTS_STAT_MODEL_REQUIRED_ARGS1, {}),
]

# Generate model factories for each model class and required parameters in DARTS-MODELS and add them to global variables
for model_class, required_args, model_args in DARTS_MODELS:
    if darts.__version__ >= "0.25.0" and isinstance(model_class, NotImportedModule):
        logger.warning("NotImportedModule encountered, skipping")
        continue
    globals()[model_class.__name__] = generate_model_factory(
        model_class=model_class,
        model_args=model_args,
        model_name=model_class.__name__,
        required_args=required_args,
        allow_fit_on_eval=False,
    )

# Generate model factories for each model class and required parameters in DARTS-STAT-MODELS and add them to global variables
for model_class, required_args, model_args in DARTS_STAT_MODELS:
    if darts.__version__ >= "0.25.0" and isinstance(model_class, NotImportedModule):
        logger.warning("NotImportedModule encountered, skipping")
        continue
    globals()[model_class.__name__] = generate_model_factory(
        model_class=model_class,
        model_args=model_args,
        model_name=model_class.__name__,
        required_args=required_args,
        allow_fit_on_eval=True,
    )


# TODO：darts 应该不止这两个 adapter，例如有些应该输入 DARTS_DEEP_MODEL_REQUIRED_ARGS2
#   而非 DARTS_DEEP_MODEL_REQUIRED_ARGS1。
#   因此暂时注释这两个 adapter，后续看是去掉这些 adapter 还是通过 inspect 来分析模型参数
#   还是预先定义好模型与 adapter 之间的映射关系。
# def deep_darts_model_adapter(model_info: Type[object]) -> object:
#     """
#     适配深度 DARTS 模型。
#
#     :param model_info: 要适配的深度 DARTS 模型类。必须是一个类或类型对象。
#     :return: 生成的模型工厂，用于创建适配的 DARTS 模型。
#     """
#     if not isinstance(model_info, type):
#         raise ValueError()
#
#     return generate_model_factory(
#         model_info.__name__,
#         model_info,
#         DARTS_DEEP_MODEL_REQUIRED_ARGS1,
#         allow_fit_on_eval=False,
#     )
#
#
# def statistics_darts_model_adapter(model_info: Type[object]) -> object:
#     """
#     适配统计学 DARTS 模型。
#
#     :param model_info: 要适配的统计学 DARTS 模型类。必须是一个类或类型对象。
#     :return: 生成的模型工厂，用于创建适配的 DARTS 模型。
#     """
#     if not isinstance(model_info, type):
#         raise ValueError()
#
#     return generate_model_factory(
#         model_info.__name__, model_info, {}, allow_fit_on_eval=True
#     )
