# -*- coding: utf-8 -*-
import os

import torch.nn as nn
from torch import optim

from ts_benchmark.baselines.deep_forecasting_model_base import DeepForecastingModelBase

from .models.olinear_model import Model as OLinearModel

MODEL_HYPER_PARAMS = {
    "d_model": 256,
    "d_ff": 512,
    "e_layers": 2,
    "dropout": 0.1,
    "embed_size": 1,
    "temp_patch_len": 1,
    "temp_stride": 1,
    "activation": "gelu",
    "CKA_flag": False,
    "Q_chan_indep": False,
    "q_mat_file": "dataset/ILI_Q.npy",
    "q_out_mat_file": "dataset/ILI_Q_out.npy",
    "root_path": "./",
    "lr": 0.001,
}


class OLinear(DeepForecastingModelBase):
    def __init__(self, **kwargs):
        if "root_path" not in kwargs:
            kwargs["root_path"] = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..", "..")
            )
        super(OLinear, self).__init__(MODEL_HYPER_PARAMS, **kwargs)

    @property
    def model_name(self):
        return "OLinear"

    def _init_criterion_and_optimizer(self):
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr)
        return criterion, optimizer

    def _init_model(self):
        return OLinearModel(self.config)

    def _process(self, input, target, input_mark, target_mark):
        outputs = self.model(input)
        return {"output": outputs}