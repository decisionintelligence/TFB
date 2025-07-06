from ts_benchmark.baselines.duet.models.duet_model import DUETModel
from ts_benchmark.baselines.deep_forecasting_model_base import DeepForecastingModelBase

# model hyper params
MODEL_HYPER_PARAMS = {
    "enc_in": 1,
    "dec_in": 1,
    "c_out": 1,
    "e_layers": 2,
    "d_layers": 1,
    "d_model": 512,
    "d_ff": 2048,
    "hidden_size": 256,
    "freq": "h",
    "factor": 1,
    "n_heads": 8,
    "seg_len": 6,
    "win_size": 2,
    "activation": "gelu",
    "output_attention": 0,
    "patch_len": 16,
    "stride": 8,
    "period_len": 4,
    "dropout": 0.2,
    "fc_dropout": 0.2,
    "moving_avg": 25,
    "batch_size": 256,
    "lradj": "type3",
    "lr": 0.02,
    "num_epochs": 100,
    "num_workers": 0,
    "loss": "huber",
    "patience": 10,
    "num_experts": 4,
    "noisy_gating": True,
    "k": 1,
    "CI": True,
    "parallel_strategy": "DP"
}


class DUET(DeepForecastingModelBase):
    """
    DUET adapter class.

    Attributes:
        model_name (str): Name of the model for identification purposes.
        _init_model: Initializes an instance of the DUETModel.
        _adjust_lr：Adjusts the learning rate of the optimizer based on the current epoch and configuration.
        _process: Executes the model's forward pass and returns the output.
    """

    def __init__(self, **kwargs):
        super(DUET, self).__init__(MODEL_HYPER_PARAMS, **kwargs)

    @property
    def model_name(self):
        return "DUET"

    def _init_model(self):
        return DUETModel(self.config)

    def _process(self, input, target, input_mark, target_mark):
        output, loss_importance = self.model(input)
        out_loss = {"output": output}
        if self.model.training:
            out_loss["additional_loss"] = loss_importance
        return out_loss
