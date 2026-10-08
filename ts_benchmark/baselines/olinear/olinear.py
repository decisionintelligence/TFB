import os

import numpy as np
import torch
import torch.nn as nn
from torch import optim

from ts_benchmark.baselines.deep_forecasting_model_base import DeepForecastingModelBase
from ts_benchmark.common.constant import ROOT_PATH
from ts_benchmark.baselines.olinear.model.olinear_model import OLinear as OLinearNet


MODEL_HYPER_PARAMS = {
    "Q_chan_indep": False,
    "q_mat_file": None,
    "q_out_mat_file": None,
    "Q_MAT_file": None,
    "Q_OUT_MAT_file": None,
    "root_path": ROOT_PATH,
    "embed_size": 1,
    "d_model": 256,
    "d_ff": 512,
    "e_layers": 2,
    "dropout": 0.1,
    "activation": "gelu",
    "temp_patch_len": 16,
    "temp_stride": 8,
    "use_amp": 0,
    "batch_size": 32,
    "lradj": "type1",
    "lr": 0.0001,
    "num_epochs": 10,
    "num_workers": 0,
    "loss": "MSE",
    "patience": 3,
    "parallel_strategy": "DP",
}


class OLinear(DeepForecastingModelBase):
    def __init__(self, **kwargs):
        super(OLinear, self).__init__(MODEL_HYPER_PARAMS, **kwargs)

        if (
            self.config.q_mat_file is None
            and self.config.Q_MAT_file is None
            and self.config.q_out_mat_file is None
            and self.config.Q_OUT_MAT_file is None
        ):
            raise ValueError(
                "Missing Q matrix file paths. Please set q_mat_file and q_out_mat_file (or Q_MAT_file and Q_OUT_MAT_file) in model-hyper-params."
            )

    @property
    def model_name(self):
        return "OLinear"

    def _init_criterion_and_optimizer(self):
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr)
        return criterion, optimizer

    def _init_model(self):
        return OLinearNet(self.config)

    def _process(self, input, target, input_mark, target_mark):
        if self.config.use_amp == 1:
            with torch.cuda.amp.autocast():
                outputs = self.model(input)
        else:
            outputs = self.model(input)

        return {"output": outputs}
