import torch
import torch.nn as nn
from typing import Union
from ..patchs.ElasTST_Forecaster import Forecaster
from ..layers.ElasTST_backbone import ElasTST_backbone
from ..utils.ElasTST_utils import convert_to_list, weighted_average, InstanceNorm


class ElasTST(Forecaster):
    def __init__(self, config):
        """
        ElasTST model.

        Parameters
        ----------
        config : dict or object
            Configuration object or dictionary containing all parameters.
        """
        super().__init__(**config.kwargs)  # 假设 config.kwargs 包含其他扩展参数

        # 提取 config 中的参数
        if isinstance(config, dict):
            config_dict = config
        else:
            # 如果 config 是一个类实例，提取其属性
            config_dict = {
                k: v for k, v in config.__dict__.items() if not k.startswith("_")
            }

        # 从 config_dict 中提取特定参数
        self.l_patch_size = convert_to_list(config_dict.get("l_patch_size", "8_16_32"))
        self.use_norm = config_dict.get("use_norm", True)

        # 动态传递参数给 ElasTST_backbone
        self.model = ElasTST_backbone(
            l_patch_size=self.l_patch_size,
            stride=config_dict.get("stride", None),
            k_patch_size=config_dict.get("k_patch_size", 1),
            in_channels=config_dict.get("in_channels", 1),
            t_layers=config_dict.get("t_layers", 1),
            v_layers=config_dict.get("v_layers", 0),
            hidden_size=config_dict.get("f_hidden_size", 40),
            d_inner=config_dict.get("d_inner", 256),
            n_heads=config_dict.get("n_heads", 16),
            d_k=config_dict.get("d_k", 8),
            d_v=config_dict.get("d_v", 8),
            dropout=config_dict.get("dropout", 0.0),
            rotate=config_dict.get("rotate", True),
            max_seq_len=config_dict.get("max_seq_len", 1024),
            theta=config_dict.get("theta_base", 10000),
            addv=config_dict.get("addv", False),
            bin_att=config_dict.get("bin_att", False),
            learn_tem_emb=config_dict.get("learn_tem_emb", False),
            abs_tem_emb=config_dict.get("abs_tem_emb", False),
            learnable_theta=config_dict.get("learnable_rope", True),
            structured_mask=config_dict.get("structured_mask", True),
            rope_theta_init=config_dict.get("rope_theta_init", "exp"),
            min_period=config_dict.get("min_period", 1),
            max_period=config_dict.get("max_period", 1000),
            patch_share_backbone=config_dict.get("patch_share_backbone", True),
        )

        self.loss_fn = nn.MSELoss(reduction="none")
        self.instance_norm = InstanceNorm()

    def forward(self, input, input_mark, dec_input, target_mark, dataset_name=None):
        """
        适配统一输入形式的 forward 方法。
        参数:
            input: 历史目标数据 [B, L, K]
            input_mark: 历史观测值掩码 [B, L, K]
            dec_input: 解码器输入（未来目标数据的占位符）[B, pred_len, K]
            target_mark: 未来观测值掩码 [B, pred_len, K]
            dataset_name: 数据集名称（可选）
        """
        # 检查预测长度是否可被 patch size 整除
        pred_len = dec_input.shape[1]  # 从 dec_input 中获取预测长度
        new_pred_len = pred_len
        for p in self.l_patch_size:
            new_pred_len = self.check_divisibility(new_pred_len, p)

        # 如果 pred_len 被调整，需要调整 dec_input 和 target_mark
        if new_pred_len != pred_len:
            dec_input = torch.zeros(
                [dec_input.shape[0], new_pred_len, dec_input.shape[2]]
            ).to(dec_input.device)
            target_mark = torch.zeros(
                [target_mark.shape[0], new_pred_len, target_mark.shape[2]]
            ).to(target_mark.device)

        # 归一化输入数据
        if self.use_norm:
            input = self.instance_norm(input, "norm")

        # 调用模型
        x, pred_list = self.model(
            input,  # 历史目标数据
            dec_input,  # 未来目标数据的占位符
            input_mark,  # 历史观测值掩码
            target_mark,  # 未来观测值掩码
            dataset_name=dataset_name,  # 数据集名称
        )

        # 截取有效长度的输出
        dec_out = x[:, :pred_len]

        # 反归一化输出数据
        if self.use_norm:
            dec_out = self.instance_norm(dec_out, "denorm")

        return dec_out  # [B, pred_len, K]

    def loss(self, batch_data, reduce="none"):
        max_pred_len = (
            batch_data.max_prediction_length
            if batch_data.max_prediction_length is not None
            else max(self.train_prediction_length)
        )

        predict = self(
            batch_data,
            max_pred_len,
            dataset_name=None,
        )
        target = batch_data.future_target_cdf

        observed_values = batch_data.future_observed_values
        loss = self.loss_fn(target, predict)

        loss = self.get_weighted_loss(observed_values, loss, reduce=reduce)

        if reduce == "mean":
            loss = loss.mean()
        return loss

    def forecast(self, batch_data, num_samples=None):
        # max_pred_len = batch_data.max_prediction_length if batch_data.max_prediction_length is not None else max(self.prediction_length)
        max_pred_len = batch_data.future_target_cdf.shape[1]
        outputs = self(
            batch_data,
            max_pred_len,
            dataset_name=None,
        )
        return outputs.unsqueeze(1)

    def check_divisibility(self, pred_len, patch_size):
        if pred_len % patch_size == 0:
            return pred_len
        else:
            return (pred_len // patch_size + 1) * patch_size

    def get_weighted_loss(self, observed_values, loss, reduce="mean"):
        loss_weights, _ = observed_values.min(dim=-1, keepdim=True)
        loss = weighted_average(loss, weights=loss_weights, dim=1, reduce=reduce)
        return loss
