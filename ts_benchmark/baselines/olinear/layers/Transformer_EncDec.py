import random
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearEncoder(nn.Module):
    def __init__(
        self,
        d_model,
        d_ff=None,
        CovMat=None,
        dropout=0.1,
        activation="relu",
        token_num=None,
        **kwargs,
    ):
        super(LinearEncoder, self).__init__()

        d_ff = d_ff or 4 * d_model
        self.d_model = d_model
        self.d_ff = d_ff
        self.CovMat = CovMat.unsqueeze(0) if CovMat is not None else None
        self.token_num = token_num

        self.norm1 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        init_weight_mat = torch.eye(self.token_num) * 1.0 + torch.randn(
            self.token_num, self.token_num
        ) * 1.0
        self.weight_mat = nn.Parameter(init_weight_mat[None, :, :])

        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.activation = F.relu if activation == "relu" else F.gelu
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, **kwargs):
        values = self.v_proj(x)

        if self.CovMat is not None:
            A = F.softmax(self.CovMat, dim=-1) + F.softplus(self.weight_mat)
        else:
            A = F.softplus(self.weight_mat)

        A = F.normalize(A, p=1, dim=-1)
        A = self.dropout(A)

        new_x = A @ values

        x = x + self.dropout(self.out_proj(new_x))
        x = self.norm1(x)

        y = self.dropout(self.activation(self.conv1(x.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        output = self.norm2(x + y)

        return output, None


class Encoder_ori(nn.Module):
    def __init__(
        self,
        attn_layers,
        conv_layers=None,
        norm_layer=None,
        one_output=False,
        CKA_flag=False,
    ):
        super(Encoder_ori, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer
        self.one_output = one_output
        self.CKA_flag = CKA_flag

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        attns = []
        layer_len = len(self.attn_layers)
        for i, attn_layer in enumerate(self.attn_layers):
            x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
            attns.append(attn)

            if not self.training and self.CKA_flag and layer_len > 1:
                if i == layer_len - 1 and random.uniform(0, 1) < 0.0:
                    pass

        if isinstance(x, tuple) or isinstance(x, List):
            x = x[0]

        if self.norm is not None:
            x = self.norm(x)

        if self.one_output:
            return x
        else:
            return x, attns
