import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler

from ts_benchmark.baselines.timebridge.models.TimeBridge import Model as TimeBridge_model
from ts_benchmark.baselines.pdf.utils.tools import adjust_learning_rate
from ts_benchmark.baselines.deep_forecasting_model_base import DeepForecastingModelBase


# model hyper params
MODEL_HYPER_PARAMS = {
    "seq_len": 720,
    "horizon": 96,
    "period": 24,
    "num_p": 2,
    "ia_layers": 2,
    "pd_layers": 1,
    "ca_layers": 2,
    "stable_len": 3,
    "max_seq_len": 1024,
    "d_model": 16,
    "n_heads": 4,
    "d_k": None,
    "d_v": None,
    "d_ff": 128,
    "norm": "BatchNorm",
    "attn_dropout": 0.15,
    "dropout": 0,
    "activation": "gelu",
    "key_padding_mask": "auto",
    "padding_var": None,
    "attn_mask": None,
    "res_attention": True,
    "pre_norm": False,
    "store_attn": False,
    "pe": "zeros",
    "learn_pe": True,
    "head_dropout": 0,
    "fc_dropout": 0.15,
    "padding_patch": "end",
    "pretrain_head": False,
    "head_type": "flatten",
    "individual": False,
    "revin": 1,
    "affine": 0,
    "subtract_last": 0,
    "verbose": False,
    "pct_start": 0.3,
    "num_epochs": 100,
    "patience": 10,
    "batch_size": 128,
    "num_workers": 0,
    "loss": "MSE",
    "lr": 0.0001,
    "lradj": "type3",
    "use_amp": 0,
    "task_name": "short_term_forecast",
}


class TimeBridge(DeepForecastingModelBase):
    """
    PDF adapter class.

    Attributes:
        model_name (str): Name of the model for identification purposes.
        _adjust_lr: Adjusts the learning rate of the optimizer based on the current epoch and configuration.
        _init_model: Initializes an instance of the AmplifierModel.
        _process: Executes the model's forward pass and returns the output.
    """

    def __init__(self, **kwargs):
        super(TimeBridge, self).__init__(MODEL_HYPER_PARAMS, **kwargs)

    @property
    def model_name(self):
        return "TimeBridge"

    def _adjust_lr(self, optimizer, epoch, config):
        adjust_learning_rate(optimizer, self.scheduler, epoch, config, printout=False)
        if self.config.lradj == "TST":
            self.scheduler.step()

    def _init_criterion_and_optimizer(self):
        if self.config.loss == "MSE":
            criterion = nn.MSELoss()
        elif self.config.loss == "MAE":
            criterion = nn.L1Loss()
        else:
            criterion = nn.HuberLoss(delta=0.5)

        optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr)
        train_steps = len(self.train_data_loader)
        self.scheduler = lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            steps_per_epoch=train_steps,
            pct_start=self.config.pct_start,
            epochs=self.config.num_epochs,
            max_lr=self.config.lr,
        )
        return criterion, optimizer

    def _init_model(self):
        return TimeBridge_model(self.config)

    def _process(self, input, target, input_mark, target_mark):
        output = self.model(input)

        return {"output": output}
