import math

import numpy as np
import torch

from ts_benchmark.baselines.hdmixer.models.HDMixer import HDMixerModel
from ts_benchmark.baselines.hdmixer.utils.tools import adjust_learning_rate
from ts_benchmark.models.deep_forecasting_model_base import DeepForecastingModelBase

# model hyper params
MODEL_HYPER_PARAMS = {
    "enc_in": 1,
    "mix_time": 1,
    "mix_variable": 1,
    "mix_channel": 1,
    "deform_patch": 1,
    "deform_range": 0.25,
    "lambda_": 1e-1,
    "r": 1e-2,
    "mlp_ratio": 1,
    "window_size": 6,
    "shift_size": 3,
    "weight_decay": 1e-3,
    "num_workers": 10,
    "num_epochs": 100,
    "batch_size": 32,
    "patience": 10,
    "lr": 0.0001,
    "loss": "MAE",
    "lradj": "type3",
    "pct_start": 0.3,
    "e_layers": 1,
    "d_model": 16,
    "d_ff": 32,
    "n_heads": 4,
    "fc_dropout": 0.3,
    "head_dropout": 0,
    "dropout": 0.8,
    "individual": 0,
    "patch_len": 16,
    "stride": 8,
    "padding_patch": "end",
    "revin": 1,
    "affine": 0,
    "subtract_last": 0,
    "decomposition": 0,
    "kernel_size": 25,
}

class HDMixer(DeepForecastingModelBase):
    """
    HDMixer adapter class.
    Attributes:
        model_name (str): Name of the model for identification purposes.
        _init_model: Initializes an instance of the AmplifierModel.
        _adjust_lr：Adjusts the learning rate of the optimizer based on the current epoch and configuration.
        _post_process: Performs post-processing on the output and target data.
        _process: Executes the model's forward pass and returns the output.
    """
    def __init__(self, **kwargs):
        super(HDMixer, self).__init__(MODEL_HYPER_PARAMS, **kwargs)

    @property
    def model_name(self):
        return "HDMixer"

    def _init_model(self):
        return HDMixerModel(self.config)

    def _adjust_lr(self, optimizer, epoch, config):
        adjust_learning_rate(
            optimizer, epoch + 1, config
        )

    def _post_process(self, output, target):
        ratio = np.array(
            [
                -1 * math.atan(i + 1) + math.pi / 4 + 1
                for i in range(self.config.horizon)
            ]
        )
        ratio = torch.tensor(ratio).unsqueeze(-1).to("cuda")

        output = output * ratio
        target = target * ratio
        return output, target

    def _process(self, input, target, input_mark, target_mark):
        output, PaEN_Loss = self.model(input)
        out_loss = {"output": output}
        if self.model.training:
            out_loss['additional_loss'] = PaEN_Loss
        return out_loss
