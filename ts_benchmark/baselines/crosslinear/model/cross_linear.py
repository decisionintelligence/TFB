import torch
import torch.nn as nn
import math


class Patch_Embedding(nn.Module):
    def __init__(self, seq_len, patch_num, patch_len, d_model, d_ff, variate_num):
        super(Patch_Embedding, self).__init__()
        self.pad_num = patch_num * patch_len - seq_len
        self.patch_len = patch_len
        self.linear = nn.Sequential(
            nn.LayerNorm([variate_num, patch_num, patch_len]),
            nn.Linear(patch_len, d_ff),
            nn.LayerNorm([variate_num, patch_num, d_ff]),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
            nn.LayerNorm([variate_num, patch_num, d_model]),
            nn.ReLU(),
        )

    def forward(self, x):
        x = nn.functional.pad(x, (0, self.pad_num))
        x = x.unfold(2, self.patch_len, self.patch_len)
        x = self.linear(x)
        return x


class De_Patch_Embedding(nn.Module):
    def __init__(self, pred_len, patch_num, d_model, d_ff, variate_num):
        super(De_Patch_Embedding, self).__init__()
        self.linear = nn.Sequential(
            nn.Flatten(2),
            nn.Linear(patch_num * d_model, d_ff),
            nn.LayerNorm([variate_num, d_ff]),
            nn.ReLU(),
            nn.Linear(d_ff, pred_len),
        )

    def forward(self, x):
        x = self.linear(x)
        return x


class CrossLinearModel(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.task_name = "long_term_forecast"
        self.ms = False
        self.EPS = 1e-5
        patch_len = configs.patch_len
        patch_num = math.ceil(configs.seq_len / patch_len)
        variate_num = 1 if self.ms else configs.dec_in
        # embedding
        self.alpha = nn.Parameter(torch.ones([1]) * configs.alpha)
        self.beta = nn.Parameter(torch.ones([1]) * configs.beta)
        self.correlation_embedding = nn.Conv1d(
            configs.dec_in, variate_num, 3, padding="same"
        )
        self.value_embedding = Patch_Embedding(
            configs.seq_len,
            patch_num,
            patch_len,
            configs.d_model,
            configs.d_ff,
            variate_num,
        )
        self.pos_embedding = nn.Parameter(
            torch.randn(1, variate_num, patch_num, configs.d_model)
        )
        # head
        if (
            self.task_name == "long_term_forecast"
            or self.task_name == "short_term_forecast"
        ):
            self.head = De_Patch_Embedding(
                configs.pred_len, patch_num, configs.d_model, configs.d_ff, variate_num
            )

    def forecast(self, x_enc):
        x_enc = x_enc.permute(0, 2, 1)
        # normalization
        x_obj = x_enc[:, [-1], :] if self.ms else x_enc
        mean = torch.mean(x_obj, dim=-1, keepdim=True)
        std = torch.std(x_obj, dim=-1, keepdim=True)
        x_enc = (x_enc - torch.mean(x_enc, dim=-1, keepdim=True)) / (
            torch.std(x_enc, dim=-1, keepdim=True) + self.EPS
        )
        # embedding
        x_obj = x_enc[:, [-1], :] if self.ms else x_enc
        x_obj = self.alpha * x_obj + (1 - self.alpha) * self.correlation_embedding(
            x_enc
        )
        x_obj = (
            self.beta * self.value_embedding(x_obj)
            + (1 - self.beta) * self.pos_embedding
        )
        # head
        y_out = self.head(x_obj)
        # de-normalization
        y_out = y_out * std + mean
        y_out = y_out.permute(0, 2, 1)
        return y_out

    def forward(self, x_enc, ):
        if (
            self.task_name == "long_term_forecast"
            or self.task_name == "short_term_forecast"
        ):
            dec_out = self.forecast(x_enc)
        return dec_out
