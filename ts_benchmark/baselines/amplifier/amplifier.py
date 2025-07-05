import torch
import torch.nn as nn
from torch import optim

from ts_benchmark.baselines.amplifier.models.amplifier_model import AmplifierModel
from ts_benchmark.baselines.deep_forecasting_model_base import DeepForecastingModelBase

# model hyper params
MODEL_HYPER_PARAMS = {
    "use_amp": 0,
    "label_len": 48,
    "embed": "timeF",
    "SCI": 0,
    "enc_in": 1,
    "hidden_size": 256,
    "output_attention": 0,
    "batch_size": 32,
    "lradj": "type1",
    "lr": 0.02,
    "num_epochs": 10,
    "num_workers": 0,
    "patience": 3,
}


class Amplifier(DeepForecastingModelBase):
    """
    Amplifier adapter class.

    Attributes:
        model_name (str): Name of the model for identification purposes.
        _init_model: Initializes an instance of the AmplifierModel.
        _adjust_lr：Adjusts the learning rate of the optimizer based on the current epoch and configuration.
        _process: Executes the model's forward pass and returns the output.
        _init_criterion_and_optimizer: Defines the loss function and optimizer.
    """

    def __init__(self, **kwargs):
        super(Amplifier, self).__init__(MODEL_HYPER_PARAMS, **kwargs)

    @property
    def model_name(self):
        return "Amplifier"

    def _init_criterion_and_optimizer(self):
        criterion = nn.MSELoss()

        optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr)
        return criterion, optimizer

    def _init_model(self):
        return AmplifierModel(self.config)

    def _process(self, input, target, input_mark, target_mark):
        dec_inp = torch.zeros_like(target[:, -self.config.pred_len :, :]).float()
        dec_inp = (
            torch.cat([target[:, : self.config.label_len, :], dec_inp], dim=1)
            .float()
            .to(input.device)
        )
        # encoder - decoder
        if self.config.use_amp == 1:
            with torch.cuda.amp.autocast():
                if self.config.output_attention == 1:
                    outputs = self.model(input, input_mark, dec_inp, target_mark)[0]
                else:
                    outputs = self.model(input, input_mark, dec_inp, target_mark)
        else:
            if self.config.output_attention == 1:
                outputs = self.model(input, input_mark, dec_inp, target_mark)[0]
            else:
                outputs = self.model(input, input_mark, dec_inp, target_mark)

        return {"output": outputs}
