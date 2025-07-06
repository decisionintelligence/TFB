import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler

from ts_benchmark.baselines.pathformer.models.pathformer_model import PathformerModel
from ts_benchmark.baselines.pathformer.utils.tools import adjust_learning_rate
from ts_benchmark.baselines.deep_forecasting_model_base import DeepForecastingModelBase

# model hyper params
MODEL_HYPER_PARAMS = {
    "k": 2,
    "enc_in": 1,
    "dec_in": 1,
    "c_out": 1,
    "e_layers": 1,
    "d_layers": 1,
    "d_model": 4,
    "d_ff": 64,
    "embed": "timeF",
    "freq": "h",
    "lradj": "TST",
    "moving_avg": 25,
    "num_kernels": 6,
    "factor": 1,
    "n_heads": 8,
    "seg_len": 6,
    "win_size": 2,
    "activation": "gelu",
    "output_attention": 0,
    "patch_len": 16,
    "stride": 8,
    "dropout": 0.1,
    "batch_size": 512,
    "lr": 0.0001,
    "num_epochs": 30,
    "num_workers": 0,
    "loss": "MAE",
    "itr": 1,
    "distil": True,
    "patience": 5,
    "p_hidden_dims": [128, 128],
    "p_hidden_layers": 2,
    "mem_dim": 32,
    "conv_kernel": [12, 16],
    "individual": False,
    "num_nodes": 21,
    "layer_nums": 3,
    "num_experts_list": [4, 4, 4],
    "patch_size_list": [[56, 28, 12, 24], [42, 28, 16, 21], [56, 16, 28, 42]],
    "revin": 1,
    "drop": 0.1,
    "pct_start": 0.4,
    "residual_connection": 0,
    "gpu": 0,
    "seq_len": 336,
    "batch_norm": 0,
}


class Pathformer(DeepForecastingModelBase):
    """
    Pathformer adapter class.

    Attributes:
        model_name (str): Name of the model for identification purposes.
        _adjust_lr：Adjusts the learning rate of the optimizer based on the current epoch and configuration.
        _init_model: Initializes an instance of the AmplifierModel.
        _process: Executes the model's forward pass and returns the output.
    """

    def __init__(self, **kwargs):
        super(Pathformer, self).__init__(MODEL_HYPER_PARAMS, **kwargs)

    @property
    def model_name(self):
        return "Pathformer"

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

    def _adjust_lr(self, optimizer, epoch, config):
        adjust_learning_rate(optimizer, self.scheduler, epoch, config, printout=False)
        if config.lradj == "TST":
            self.scheduler.step()

    def _init_model(self):
        return PathformerModel(self.config)

    def _process(self, input, target, input_mark, target_mark):
        output, balance_loss = self.model(input)

        return {"output": output}
