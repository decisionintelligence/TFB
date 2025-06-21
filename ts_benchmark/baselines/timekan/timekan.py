from ts_benchmark.models.advanced_model_base import Advanced_Model_Base
from .models.timekan_model import TimeKANModeL

# model hyper params
MODEL_HYPER_PARAMS = {
    "lradj": "type1",
    "data": "custom",
    "label_len": 48,
    "freq": "h",
    "seq_len": 96,
    "top_k": 5,
    "num_kernels": 6,
    "enc_in": 7,
    "dec_in": 7,
    "c_out": 7,
    "d_model": 16,
    "n_heads": 4,
    "e_layers": 2,
    "d_layers": 1,
    "d_ff": 32,
    "moving_avg": 25,
    "factor": 1,
    "distil": True,
    "dropout": 0.1,
    "embed": "timeF",
    "activation": "gelu",
    "output_attention": False,
    "channel_independence": 1,
    "decomp_method": "moving_avg",
    "use_norm": 1,
    "down_sampling_layers": 2,
    "down_sampling_window": 2,
    "use_future_temporal_feature": 0,
    "begin_order": 1,
    "mask_rate": 0.25,
    "anomaly_ratio": 0.25,
    "num_workers": 10,
    "itr": 1,
    "num_epochs": 10,
    "batch_size": 16,
    "patience": 10,
    "lr": 0.001,
    "des": "test",
    "loss": "MSE",
    "pct_start": 0.2,
    "use_amp": False,
    "comment": "none",
    "use_gpu": True,
    "gpu": 0,
    "use_multi_gpu": False,
    "devices": "0,1",
    "p_hidden_dims": [128, 128],
    "p_hidden_layers": 2,
    "task_name": "long_term_forecast"
}

class TimeKAN(Advanced_Model_Base):
    """
    TimeKAN adapter class.

    Attributes:
        model_name (str): Name of the model for identification purposes.
        _init_model: Initializes an instance of the AmplifierModel.
        _process: Executes the model's forward pass and returns the output.
    """
    def __init__(self, **kwargs):
        super(TimeKAN, self).__init__(MODEL_HYPER_PARAMS, **kwargs)

    @property
    def model_name(self):
        return "TimeKAN"

    def _init_model(self):
        return TimeKANModeL(self.config)

    def _process(self, input, target, input_mark, target_mark):
        output = self.model(input)

        return {"output": output}
