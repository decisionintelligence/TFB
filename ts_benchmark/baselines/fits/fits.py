from ts_benchmark.models.advanced_model_base import Advanced_Model_Base
from .fits_model import FITSModel

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

class FITS(Advanced_Model_Base):
    """
    FITS adapter class.

    Attributes:
        model_name (str): Name of the model for identification purposes.
        _init_model: Initializes an instance of the AmplifierModel.
        _process: Executes the model's forward pass and returns the output.
    """
    def __init__(self, **kwargs):
        super(FITS, self).__init__(MODEL_HYPER_PARAMS, **kwargs)

    @property
    def model_name(self):
        return "FITS"

    def _init_model(self):
        return FITSModel(self.config)

    def _process(self, input, target, input_mark, target_mark):
        output, low = self.model(input)

        return {"output": output}
