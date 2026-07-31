import math
import torch
import torch.nn as nn
from torch import optim

from ts_benchmark.baselines.phaseformer.models.PhaseFormer import Model
from ts_benchmark.baselines.deep_forecasting_model_base import DeepForecastingModelBase

# Defaults consistent with the paper's PhaseFormer scripts (run_traffic.sh).
# PhaseFormer is a phase-based forecaster with cross-phase routing attention.
MODEL_HYPER_PARAMS = {
    "seq_len": 720,
    "period_len": 24,
    "latent_dim": 32,
    "phase_encoder_hidden": 64,
    "predictor_hidden": 128,
    "phase_attn_heads": 8,
    "phase_attn_dropout": 0.0,
    "phase_attn_use_relpos": True,
    "phase_attn_window": None,
    "phase_attention_dim": None,
    "phase_num_routers": 1,
    "phase_use_pos_embed": False,
    "phase_pos_dropout": 0.0,
    "phase_encoder_use_mlp": False,
    "phase_encoder_dropout": 0.0,
    "predictor_use_mlp": False,
    "predictor_dropout": 0.0,
    "phase_layers": 2,
    "use_revin": True,
    "revin_affine": False,
    "revin_eps": 1e-5,
    "task_name": "long_term_forecast",
    "batch_size": 16,
    "lr": 0.001,
    "lradj": "constant",
    "patience": 10,
    "num_epochs": 30,
    "loss": "MSE",
    "use_gpu": True,
    "parallel_strategy": "DP",
}


class PhaseFormer(DeepForecastingModelBase):
    """
    PhaseFormer adapter class.

    Wraps the official PhaseFormer model from
    "PhaseFormer: From Patches to Phases for Efficient and Effective Time Series Forecasting"
    (arXiv:2510.04134) so it can be used inside TFB.
    """

    def __init__(self, **kwargs):
        super(PhaseFormer, self).__init__(MODEL_HYPER_PARAMS, **kwargs)

    @property
    def model_name(self):
        return "PhaseFormer"

    def _init_model(self):
        return Model(self.config)

    def _process(self, input, target, input_mark, target_mark):
        output = self.model(input, input_mark)
        return {"output": output}
