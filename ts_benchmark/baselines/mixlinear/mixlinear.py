import math
import torch
import torch.nn as nn
from torch import optim

from ts_benchmark.baselines.mixlinear.models.MixLinear import Model
from ts_benchmark.baselines.deep_forecasting_model_base import DeepForecastingModelBase

# Default hyper-parameters consistent with the paper's authoritative scripts (etth1_best.sh, etc.).
# `seq_len` is 720 by default in the original repository. `period_len`, `lpf`, `alpha` are MixLinear's
# core hyperparameters. The model is extremely small so a high learning rate (0.03) and large batch
# size (256) are used by the authors.
MODEL_HYPER_PARAMS = {
    "seq_len": 720,
    "period_len": 24,
    "lpf": 1,
    "alpha": 0.95,
    "batch_size": 256,
    "lr": 0.03,
    "lradj": "type3",
    "patience": 10,
    "num_epochs": 30,
    "loss": "MSE",
    "use_gpu": True,
    "parallel_strategy": "DP",
}


class MixLinear(DeepForecastingModelBase):
    """
    MixLinear adapter class.

    Wraps the official MixLinear model from "MixLinear: Extreme Low Resource Multivariate Time
    Series Forecasting with 0.1K Parameters" (arXiv:2410.02081) so it can be used inside TFB.
    """

    def __init__(self, **kwargs):
        super(MixLinear, self).__init__(MODEL_HYPER_PARAMS, **kwargs)

    @property
    def model_name(self):
        return "MixLinear"

    def _init_model(self):
        return Model(self.config)

    def _process(self, input, target, input_mark, target_mark):
        output = self.model(input)
        return {"output": output}
