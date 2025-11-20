from ts_benchmark.baselines.timebase.model.timebase import TimeBaseModel
from ts_benchmark.baselines.deep_forecasting_model_base import DeepForecastingModelBase

MODEL_HYPER_PARAMS = {
    "enc_in": 1,
    "dec_in": 1,
    "c_out": 1,
    "e_layers": 1,
    "d_layers": 2,
    "d_model": 32,
    "d_ff": 2048,
    "embed": "timeF",
    "freq": "h",
    "lradj": "type1",
    "moving_avg": 25,
    "num_kernels": 6,
    "factor": 1,
    "n_heads": 2,
    "heads": 2,
    "seg_len": 6,
    "win_size": 2,
    "activation": "gelu",
    "output_attention": 0,
    "patch_len": 16,
    "stride": 8,
    "dropout": 0.1,
    "batch_size": 32,
    "lr": 0.0001,
    "num_epochs": 100,
    "num_workers": 0,
    "loss": "MSE",
    "itr": 1,
    "distil": True,
    "patience": 3,
    "period_len": 24,
    "basis_num": 6,
    "individual": 0,
    "orthogonal_weight": 0.16,
    "use_orthogonal": 1,
    "use_period_norm": 1,
}


class TimeBase(DeepForecastingModelBase):
    """
    TimeBase adapter class.

    Attributes:
        model: The TimeBase model instance.
    """

    def __init__(self, **kwargs):
        """
        Initialize the TimeBase adapter.

        Args:
            configs: Configuration object containing model parameters.
        """
        super(TimeBase, self).__init__(MODEL_HYPER_PARAMS, **kwargs)

    def _init_model(self):
        return TimeBaseModel(self.config)

    @property
    def model_name(self):
        return "TimeBase"

    def _process(self, input, target, input_mark, target_mark):
        if self.config.use_orthogonal:
            output, orthogonal_loss = self.model(input)
        else:
            output = self.model(input)
            orthogonal_loss = 0
        addional_loss = self.config.orthogonal_weight * orthogonal_loss
        return {"output": output, "additional_loss": addional_loss}
