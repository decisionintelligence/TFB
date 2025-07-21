from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from sklearn.preprocessing import StandardScaler
from torch import optim
from torch.utils.data import DataLoader

from ts_benchmark.baselines.cov.models.cov_model import COVModel
from ts_benchmark.baselines.cov.utils.tools import EarlyStopping, adjust_learning_rate
from ts_benchmark.baselines.utils import (
    forecasting_data_provider,
    train_val_split, DBLoss,
)
from ..deep_forecasting_model_base import DeepForecastingModelBase
from ...models.model_base import ModelBase, BatchMaker

MODEL_HYPER_PARAMS = {

    "d_model": 512,
    "d_ff": 2048,
    "n_heads": 8,
    "factor": 1,
    "patch_len": 16,
    "stride": 8,
    "activation": "gelu",
    "batch_size": 256,
    "lradj": "type3",
    "lr": 0.02,
    "num_epochs": 100,
    "num_workers": 0,
    "loss": "MAE",
    "dbloss_alpha": 0.2,
    "dbloss_beta": 0.5,
    "patience": 10,
    "alpha": 0.2,
    "beta": 0.1,
    "use_c_exog": True,
    "use_t_exog": True,
    "use_c": True,
    "use_t": True,

}


class COV(DeepForecastingModelBase):
    """
    COV adapter class.

    Attributes:
        model_name (str): Name of the model for identification purposes.
        _init_model: Initializes an instance of the COVModel.
        _adjust_lr：Adjusts the learning rate of the optimizer based on the current epoch and configuration.
        _process: Executes the model's forward pass and returns the output.
        _init_criterion_and_optimizer: Defines the loss function and optimizer.
    """

    def __init__(self, **kwargs):
        super(COV, self).__init__(MODEL_HYPER_PARAMS, **kwargs)

    @property
    def model_name(self):
        return "COV"

    def _init_criterion(self):
        if self.config.loss == "MSE":
            criterion = nn.MSELoss()
        elif self.config.loss == "MAE":
            criterion = nn.L1Loss()
        elif self.config.loss == "DBLoss":
            criterion = DBLoss(self.config.dbloss_alpha, self.config.dbloss_beta)
        else:
            criterion = nn.HuberLoss(delta=0.5)
        self.config.criterion = criterion
        return criterion

    def _init_model(self):
        return COVModel(self.config)

    def _process(self, input, target, input_mark, target_mark, exog_future=None):
        output, causality_loss = self.model(input, exog_future)
        out_loss = {"output": output}
        if self.model.training:
            out_loss["additional_loss"] = causality_loss
        return out_loss
