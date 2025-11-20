from ts_benchmark.baselines.sparsetsf.models.sparsetsf_model import SparseTSFMoedl
from ts_benchmark.baselines.deep_forecasting_model_base import DeepForecastingModelBase

# model hyper params
MODEL_HYPER_PARAMS = {
    "embed": "learned",
    "lradj": "type3",
    "factor": 1,
    "batch_size": 256,
    "lr": 0.0001,
    "num_epochs": 100,
    "loss": "MSE",
    "d_model": 128,
    "itr": 1,
    "patience": 3,
    "period_len": 24,
    "model_type": "linear",
}


class SparseTSF(DeepForecastingModelBase):
    """
    FITS adapter class.

    Attributes:
        model_name (str): Name of the model for identification purposes.
        _init_model: Initializes an instance of the AmplifierModel.
        _adjust_lr：Adjusts the learning rate of the optimizer based on the current epoch and configuration.
        _process: Executes the model's forward pass and returns the output.
    """

    def __init__(self, **kwargs):
        super(SparseTSF, self).__init__(MODEL_HYPER_PARAMS, **kwargs)


    @property
    def model_name(self):
        return "SparseTSF"

    def _init_model(self):
        return SparseTSFMoedl(self.config)

    def _process(self, input, target, input_mark, target_mark):
        output = self.model(input)

        return {"output": output}
