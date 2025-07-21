import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler

from ts_benchmark.baselines.pdf.models.PDF import Model as PDF_model
from ts_benchmark.baselines.pdf.utils.tools import adjust_learning_rate
from ts_benchmark.baselines.deep_forecasting_model_base import DeepForecastingModelBase

# model hyper params
MODEL_HYPER_PARAMS = {
    "seq_len": 720,
    "horizon": 96,
    "wo_conv": False,
    "serial_conv": False,
    "add": True,
    "patch_len": [1],
    "kernel_list": [3, 7, 9, 11],
    "period": [24],
    "stride": [1],
    "max_seq_len": 1024,
    "e_layers": 3,
    "d_model": 16,
    "n_heads": 4,
    "d_k": None,
    "d_v": None,
    "d_ff": 128,
    "norm": "BatchNorm",
    "attn_dropout": 0.05,
    "dropout": 0.25,
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


class PDF(DeepForecastingModelBase):
    """
    PDF adapter class.

    Attributes:
        model_name (str): Name of the model for identification purposes.
        _adjust_lr: Adjusts the learning rate of the optimizer based on the current epoch and configuration.
        _init_model: Initializes an instance of the AmplifierModel.
        _process: Executes the model's forward pass and returns the output.
    """

    def __init__(self, **kwargs):
        super(PDF, self).__init__(MODEL_HYPER_PARAMS, **kwargs)

    @property
    def model_name(self):
        return "PDF"

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
        return criterion

    def _init_optimizer(self, CovariateFusion=None):
        if CovariateFusion is not None:
            optimizer = optim.Adam(
                [
                    {"params": self.model.parameters(), "lr": self.config.lr},
                    {"params": CovariateFusion.parameters(), "lr": self.config.lr},
                ]
            )
        else:
            optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr)
        # optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr)
        train_steps = len(self.train_data_loader)
        self.scheduler = lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            steps_per_epoch=train_steps,
            pct_start=self.config.pct_start,
            epochs=self.config.num_epochs,
            max_lr=self.config.lr,
        )
        return optimizer

    def _init_model(self):
        return PDF_model(self.config)

    def _process(self, input, target, input_mark, target_mark, exog_future=None):
        output = self.model(input)

        return {"output": output}
