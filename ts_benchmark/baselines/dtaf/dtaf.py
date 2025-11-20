import torch
import torch.nn as nn


from ts_benchmark.baselines.dtaf.model.DTAF_model import DTAF as DTAF_Model
from ts_benchmark.baselines.deep_forecasting_model_base import DeepForecastingModelBase

MODEL_HYPER_PARAMS = {
    "sample_num": 0,
    "kan_div": 4,
    "sigma": 1,
    "expert_num": 2,
    "kl": 1,
    "k": 1,
    "alpha": 0.2,
    "aggregated_norm": 1,
    "top_k": 5,
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
    "loss": "MAE",
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
}

class DTAF(DeepForecastingModelBase):
    """
    DTAF adapter class.
    """

    def __init__(self, **kwargs):
        super(DTAF, self).__init__(MODEL_HYPER_PARAMS, **kwargs)
    @property
    def model_name(self):
        return "DTAF"
    def _init_model(self):
        return DTAF_Model(self.config)
    def klLoss(self, stables, sample_num):
        """
            输入: patches of shape [B, N, D]
            输出: KL divergence 矩阵, shape [B, N, N]
            """

        # new 采样
        if sample_num > 0:
            shuffle = torch.randint(low=0, high=stables.shape[0], size=(stables.shape[0],), device=stables.device).unsqueeze(
                -1).unsqueeze(-1)[:sample_num].repeat(1, stables.shape[1], stables.shape[2])
            stables = torch.gather(stables, dim=0, index=shuffle)
        # 转换成概率分布
        probs = stables.softmax(dim=-1)  # [B, N, D]
        log_probs = torch.log(probs + 1e-8)  # 防止 log(0)

        # 扩展维度计算 pairwise KL(p || q)
        p_i = probs.unsqueeze(2)  # [B, N, 1, D]
        log_p_i = log_probs.unsqueeze(2)  # [B, N, 1, D]
        log_q_j = log_probs.unsqueeze(1)  # [B, 1, N, D]

        kl = p_i * (log_p_i - log_q_j)  # [B, N, N, D]
        kl = kl.sum(dim=-1)  # [B, N, N]

        return kl.mean(dim=-1).mean(dim=-1).mean(dim=-1)
    def _process(self, input, target, input_mark, target_mark):
        output, stables = self.model(input)
        output_r, stables_r = self.model(input)
        target = target[:, -self.config.horizon :, :]
        output = output[:, -self.config.horizon :, :]
        addtional_loss =-nn.L1Loss()(output,target)/2+nn.L1Loss()(output_r, target)/2 + self.config.r_dropout * nn.MSELoss()(output, output_r) + self.config.kl * (self.klLoss(stables, self.config.sample_num) + self.klLoss(stables_r, self.config.sample_num))
        # (a+b)/2 = a-a/2 + b/2
        return {"output": output, "addtional_loss": addtional_loss}
        