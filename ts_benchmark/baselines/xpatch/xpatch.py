import math

import numpy as np
import torch
from ts_benchmark.baselines.xpatch.utils.tools import adjust_learning_rate
from ts_benchmark.baselines.xpatch.models.xPatch import xPatchModel
from ts_benchmark.baselines.deep_forecasting_model_base import DeepForecastingModelBase

# model hyper params
MODEL_HYPER_PARAMS = {
    "enc_in": 1,
    "patch_len": 16,
    "stride": 8,
    "padding_patch": "end",
    "ma_type": "ema",
    "alpha": 0.3,
    "beta": 0.3,
    "num_workers": 10,
    "num_epochs": 100,
    "batch_size": 32,
    "patience": 10,
    "lr": 0.0001,
    "loss": "MAE",
    "lradj": "type1",
    "revin": 1,
}


class xPatch(DeepForecastingModelBase):
    """
    xPatch adapter class.

    Attributes:
        model_name (str): Name of the model for identification purposes.
        _init_model: Initializes an instance of the AmplifierModel.
        _adjust_lr：Adjusts the learning rate of the optimizer based on the current epoch and configuration.
        _post_process: Performs post-processing on the output and target data.
        _process: Executes the model's forward pass and returns the output.
    """

    def __init__(self, **kwargs):
        super(xPatch, self).__init__(MODEL_HYPER_PARAMS, **kwargs)

    @property
    def model_name(self):
        return "xPatch"

    def _init_model(self):
        return xPatchModel(self.config)

    def _adjust_lr(self, optimizer, epoch, config):
        adjust_learning_rate(optimizer, epoch, config)

    def _post_process(self, output, target):
        self.ratio = np.array(
            [
                -1 * math.atan(i + 1) + math.pi / 4 + 1
                for i in range(self.config.horizon)
            ]
        )
        self.ratio = torch.tensor(self.ratio).unsqueeze(-1).to("cuda")

        output = output * self.ratio
        target = target * self.ratio
        return output, target

    def _process(self, input, target, input_mark, target_mark):
        output = self.model(input)

        return {"output": output}
