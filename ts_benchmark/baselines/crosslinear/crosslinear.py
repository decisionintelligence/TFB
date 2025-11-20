from ts_benchmark.baselines.crosslinear.model.cross_linear import CrossLinearModel
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
    "alpha": 1,
    "beta": 0.5,
    
}
class CrossLinear(DeepForecastingModelBase):
    """
    CrossLinear adapter class.

    Attributes:
        model: The CrossLinear  model instance.
    """

    def __init__(self, **kwargs):
        """
        Initialize the CrossLinear adapter.

        Args:
            configs: Configuration object containing model parameters.
        """
        super(CrossLinear, self).__init__(MODEL_HYPER_PARAMS, **kwargs)

    def _init_model(self):
        return CrossLinearModel(self.config)

    @property
    def model_name(self):
        return "CrossLinear"
    def _process(self, input, target, input_mark, target_mark):
        output = self.model(input)

        return {"output": output}