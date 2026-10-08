from ts_benchmark.baselines.deep_forecasting_model_base import DeepForecastingModelBase
from ts_benchmark.baselines.olinear.models.olinear_model import Model
from ts_benchmark.common.constant import FORECASTING_DATASET_PATH

MODEL_HYPER_PARAMS = {
    "pred_len": 96,
    "seq_len": 96,
    "enc_in": 7,
    "d_model": 512,
    "d_ff": 2048,
    "Q_chan_indep": 0,
    "Q_MAT_file": None,
    "q_mat_file": None,
    "q_out_mat_file": None,
    "Q_OUT_MAT_file": None,
    "root_path": FORECASTING_DATASET_PATH,
    "data_path": "Covid-19.csv",
    "temp_patch_len": 16,
    "temp_stride": 8,
    "embed_size": 8,
    "dropout": 0.1,
    "d_model": 512,
    "d_ff": 2048,
    "activation": "gelu",
    "e_layers": 2,
    "loss": "MSE",
}


class OLinear(DeepForecastingModelBase):
    """
    OLinear adapter class for TFB baseline.

    Attributes:
        model_name (str): Name of the model for identification purposes.
        _init_model: Initializes an instance of the OLinear.
        _adjust_lr：Adjusts the learning rate of the optimizer based on the current epoch and configuration.
        _process: Executes the model's forward pass and returns the output.
    """

    def __init__(self, **kwargs):
        super(OLinear, self).__init__(MODEL_HYPER_PARAMS, **kwargs)
        if self.config.Q_chan_indep:
            if self.config.Q_MAT_file is None or self.config.Q_OUT_MAT_file is None:
                raise ValueError("Please set Q_MAT_file and Q_OUT_MAT_file")
        else:
            if self.config.q_mat_file is None or self.config.q_out_mat_file is None:
                raise ValueError("Please set q_mat_file and q_out_mat_file")

    def _init_model(self):
        return Model(self.config)

    @property
    def model_name(self):
        return "OLinear"

    def _process(self, input, target, input_mark, target_mark):
        output = self.model(input)

        return {"output": output}
