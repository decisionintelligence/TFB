from typing import Type, Dict

import torch
import torch.nn as nn
from torch import optim

from ts_benchmark.models.advanced_model_base import Advanced_Model_Base

MODEL_HYPER_PARAMS = {
    "top_k": 5,
    "enc_in": 1,
    "dec_in": 1,
    "c_out": 1,
    "e_layers": 2,
    "d_layers": 1,
    "d_model": 512,
    "d_ff": 2048,
    "embed": "timeF",
    "freq": "h",
    "lradj": "type1",
    "moving_avg": 25,
    "num_kernels": 6,
    "factor": 1,
    "n_heads": 8,
    "seg_len": 6,
    "win_size": 2,
    "activation": "gelu",
    "output_attention": 0,
    "patch_len": 16,
    "stride": 8,
    "dropout": 0.1,
    "batch_size": 32,
    "lr": 0.0001,
    "num_epochs": 10,
    "num_workers": 0,
    "loss": "MSE",
    "itr": 1,
    "distil": True,
    "patience": 3,
    "p_hidden_dims": [128, 128],
    "p_hidden_layers": 2,
    "mem_dim": 32,
    "conv_kernel": [12, 16],
    "anomaly_ratio": 1.0,
    "down_sampling_windows": 2,
    "channel_independence": True,
    "down_sampling_layers": 3,
    "down_sampling_method": "avg",
    "decomp_method": "moving_avg",
    "use_norm": True,
    "parallel_strategy": "DP",
    "task_name": "short_term_forecast"
}

class TransformerAdapter(Advanced_Model_Base):
    def __init__(self, model_name, model_class, **kwargs):
        super(TransformerAdapter, self).__init__(MODEL_HYPER_PARAMS, **kwargs)
        self._model_name = model_name
        self.model_class = model_class

    @property
    def model_name(self):
        return self._model_name

    def _init_model(self):
        return self.model_class(self.config)

    def _init_criterion_and_optimizer(self):
        # Define the loss function and optimizer
        criterion = nn.MSELoss()
        # criterion = nn.L1Loss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr)
        return criterion, optimizer

    def _process(self, input, target, input_mark, target_mark):
        # decoder input
        dec_input = torch.zeros_like(target[:, -self.config.horizon:, :]).float()
        dec_input = (
            torch.cat([target[:, : self.config.label_len, :], dec_input], dim=1)
            .float()
            .to(input.device)
        )
        output = self.model(input, input_mark, dec_input, target_mark)

        return {"output": output}

def generate_model_factory(
    model_name: str, model_class: type, required_args: dict
) -> Dict:
    """
    Generate model factory information for creating Transformer Adapters model adapters.

    :param model_name: Model name.
    :param model_class: Model class.
    :param required_args: The required parameters for model initialization.
    :return: A dictionary containing model factories and required parameters.
    """

    def model_factory(**kwargs) -> TransformerAdapter:
        """
        Model factory, used to create TransformerAdapter model adapter objects.

        :param kwargs: Model initialization parameters.
        :return:  Model adapter object.
        """
        return TransformerAdapter(model_name, model_class, **kwargs)

    return {
        "model_factory": model_factory,
        "required_hyper_params": required_args,
    }


def transformer_adapter(model_info: Type[object]) -> object:
    if not isinstance(model_info, type):
        raise ValueError("the model_info does not exist")

    return generate_model_factory(
        model_name=model_info.__name__,
        model_class=model_info,
        required_args={
            "seq_len": "input_chunk_length",
            "horizon": "output_chunk_length",
            "norm": "norm",
        },
    )
