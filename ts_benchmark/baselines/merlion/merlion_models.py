import numpy as np
import pandas as pd
from merlion.utils import TimeSeries
from merlion.models.anomaly.isolation_forest import (
    IsolationForest,
    IsolationForestConfig,
)
from merlion.models.anomaly.vae import VAE, VAEConfig
from merlion.models.anomaly.windstats import WindStats, WindStatsConfig
from merlion.models.anomaly.autoencoder import AutoEncoder, AutoEncoderConfig
from merlion.models.anomaly.dagmm import DAGMM, DAGMMConfig
from merlion.models.anomaly.dbl import DynamicBaseline, DynamicBaselineConfig
from merlion.models.anomaly.deep_point_anomaly_detector import (
    DeepPointAnomalyDetector,
    DeepPointAnomalyDetectorConfig,
)
from merlion.models.anomaly.lstm_ed import LSTMED, LSTMEDConfig
from merlion.models.anomaly.random_cut_forest import (
    RandomCutForest,
    RandomCutForestConfig,
)
from merlion.models.anomaly.spectral_residual import (
    SpectralResidual,
    SpectralResidualConfig,
)
from merlion.models.anomaly.stat_threshold import StatThreshold, StatThresholdConfig
from merlion.models.anomaly.zms import ZMS, ZMSConfig
from merlion.models.anomaly.change_point.bocpd import BOCPD, BOCPDConfig
from merlion.models.anomaly.forecast_based.arima import (
    ArimaDetector,
    ArimaDetectorConfig,
)
from merlion.models.anomaly.forecast_based.sarima import (
    SarimaDetector,
    SarimaDetectorConfig,
)
from merlion.models.anomaly.forecast_based.ets import ETSDetector, ETSDetectorConfig
from merlion.models.anomaly.forecast_based.mses import MSESDetector, MSESDetectorConfig

from sklearn.preprocessing import StandardScaler


class MerlionModelAdapter:
    """
    Merlion model adapter class, used to adapt models in the Merlion framework to meet the requirements of prediction strategies.
    """

    def __init__(
        self,
        model_name: str,
        model_class: type,
        config_class: type,
        model_args: dict,
        allow_label_on_train: bool,
    ):
        """
        Initialize the Merlion model adapter object.

        :param model_name: Model name.
        :param model_class: Merlion model class.
        :param config_class: Merlion configuration class.
        :param model_args: Model initialization parameters.
        :param allow_label_on_train: Whether to use labels during training.
        """
        self.model = None
        self.model_class = model_class
        self.config_class = config_class
        self.model_args = model_args
        self.model_name = model_name
        self.scaler = StandardScaler()
        self.allow_label_on_train = allow_label_on_train

    def detect_fit(self, series: pd.DataFrame, label: pd.DataFrame) -> object:
        """
        Fit a suitable Merlion model on time series data.

        :param series: Time series data.
        :param label: Label data.
        :return: The fitted model object.
        """

        config_obj = self.config_class(**self.model_args)
        self.model = self.model_class(config_obj)
        series = TimeSeries.from_pd(series)
        label = TimeSeries.from_pd(label)

        return self.model.train(series)

    def detect_score(self, train: pd.DataFrame) -> np.ndarray:
        """
        Calculate anomaly scores using the adapted Merlion model.

        :param train: Training data used to calculate scores.
        :return: Anomaly score array.
        """
        train = TimeSeries.from_pd(train)
        fsct_result = self.model.get_anomaly_score(train)

        fsct_result = (fsct_result.to_pd()).reindex((train.to_pd()).index, fill_value=0)

        fsct_result = fsct_result.values.flatten()

        return fsct_result, fsct_result

    def detect_label(self, train: pd.DataFrame) -> np.ndarray:
        """
        Use the adapted Merlion model for anomaly detection and generate labels.

        :param train: Training data used for anomaly detection.
        :return: Anomaly label array.
        """
        train = TimeSeries.from_pd(train)
        fsct_result = self.model.get_anomaly_label(train)

        fsct_result = (fsct_result.to_pd()).reindex((train.to_pd()).index, fill_value=0)

        fsct_result = fsct_result.applymap(lambda x: 1 if x != 0 else 0)

        fsct_result = fsct_result.values.flatten()

        return fsct_result, fsct_result

    def __repr__(self):
        """
        Returns a string representation of the model name.
        """
        return self.model_name


def generate_model_factory(
    model_name: str,
    model_class: object,
    config_class: object,
    required_args: dict,
    allow_label_on_train: bool,
) -> object:
    """
    Generate model factory information for creating Merlion model adapters.

    :param model_name: Model name.
    :param model_class: Merlion model class.
    :param config_class: Merlion configuration class.
    :param required_args: Required parameters for model initialization.
    :param allow_label_on_train: Whether to use labels during training.
    :return: A dictionary containing the model factory and required parameters.
    """

    def model_factory(**kwargs) -> object:
        """
        Model factory, used to create Merlion model adapter objects.

        :param kwargs: Model initialization parameters.
        :return: Merlion model adapter object
        """
        return MerlionModelAdapter(
            model_name,
            model_class,
            config_class,
            kwargs,
            allow_label_on_train,
        )

    return {"model_factory": model_factory, "required_hyper_params": required_args}


MERLION_MODELS = [
    (IsolationForest, IsolationForestConfig, {}),
    (WindStats, WindStatsConfig, {}),
    (VAE, VAEConfig, {}),
    (AutoEncoder, AutoEncoderConfig, {}),
    (DAGMM, DAGMMConfig, {}),
    (DynamicBaseline, DynamicBaselineConfig, {}),
    (DeepPointAnomalyDetector, DeepPointAnomalyDetectorConfig, {}),
    (LSTMED, LSTMEDConfig, {}),
    (RandomCutForest, RandomCutForestConfig, {}),
    (SpectralResidual, SpectralResidualConfig, {}),
    (StatThreshold, StatThresholdConfig, {}),
    (ZMS, ZMSConfig, {}),
    (BOCPD, BOCPDConfig, {}),
]

MERLION_STAT_MODELS = [  # The training set does not require labels
    (ArimaDetector, ArimaDetectorConfig, {"max_forecast_steps": "max_forecast_steps"}),
    (
        SarimaDetector,
        SarimaDetectorConfig,
        {"max_forecast_steps": "max_forecast_steps"},
    ),
    (ETSDetector, ETSDetectorConfig, {"max_forecast_steps": "max_forecast_steps"}),
    (MSESDetector, MSESDetectorConfig, {"max_forecast_steps": "max_forecast_steps"}),
]

# Generate model factories for each model class, configuration class, and required parameters in MERLION-MODELS and add them to global variables
for model_class, config_class, required_args in MERLION_MODELS:
    globals()[model_class.__name__] = generate_model_factory(
        model_class.__name__,
        model_class,
        config_class,
        required_args,
        allow_label_on_train=True,
    )
# The model name is dynamically pointed to our model
# Generate model factories for each model class, configuration class, and required parameters in MERLION-STAT-MODELS and add them to global variables
for model_class, config_class, required_args in MERLION_STAT_MODELS:
    globals()[model_class.__name__] = generate_model_factory(
        model_class.__name__,
        model_class,
        config_class,
        required_args,
        allow_label_on_train=False,
    )
