from ts_benchmark.baselines.deep_forecasting_model_base import DeepForecastingModelBase
from ts_benchmark.baselines.mixlinear.mixlinear_model import MixlinearModel

MODEL_HYPER_PARAMS = {
    "alpha": 0.5,
    "lpf": 1,
    "kernel": 24,
    "freq": "h",
    "embed": "learned",
    "lradj": "type3",
    "factor": 1,
    "activation": "gelu",
    "dropout": 0.05,
    "batch_size": 32,
    "lr": 0.0001,
    "num_epochs": 100,
    "num_workers": 10,
    "loss": "MSE",
    "itr": 2,
    "distil": True,
    "patience": 3,
    "period_len": 24,
}

class MIXLINEAR(DeepForecastingModelBase):

    def __init__(self, **kwargs):
        super(MIXLINEAR, self).__init__(MODEL_HYPER_PARAMS, **kwargs)

    @property
    def model_name(self):
        return "MIXLINEAR"

    def _init_model(self):
        return MixlinearModel(self.config)

    def _process(self, input, target, input_mark, target_mark):
        """
        input: shape [batch, seq_len, enc_in]
        output: shape [batch, pred_len, enc_in]
        """
        output = self.model(input)
        return {"output": output}

