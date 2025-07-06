from ts_benchmark.baselines.fits.fits_model import FITSModel
from ts_benchmark.baselines.deep_forecasting_model_base import DeepForecastingModelBase

# model hyper params
MODEL_HYPER_PARAMS = {
    "embed": "timeF",
    "freq": "h",
    "lradj": "type1",
    "factor": 1,
    "activation": "gelu",
    "dropout": 0.1,
    "batch_size": 32,
    "lr": 0.0001,
    "num_epochs": 100,
    "num_workers": 0,
    "loss": "MSE",
    "itr": 1,
    "distil": True,
    "patience": 3,
    "cut_freq": 0,
    "train_mode": 1,
    "base_T": 24,
    "H_order": 2,
    "individual": False,
}


class FITS(DeepForecastingModelBase):
    """
    FITS adapter class.

    Attributes:
        model_name (str): Name of the model for identification purposes.
        _init_model: Initializes an instance of the AmplifierModel.
        _adjust_lr：Adjusts the learning rate of the optimizer based on the current epoch and configuration.
        _process: Executes the model's forward pass and returns the output.
    """

    def __init__(self, **kwargs):
        super(FITS, self).__init__(MODEL_HYPER_PARAMS, **kwargs)
        self.config.cut_freq = (
            int(self.seq_len // self.config.base_T + 1) * self.config.H_order + 10
        )

    @property
    def model_name(self):
        return "FITS"

    def _init_model(self):
        return FITSModel(self.config)

    def _process(self, input, target, input_mark, target_mark):
        output, low = self.model(input)

        return {"output": output}
