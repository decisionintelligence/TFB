import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils.tools import hier_half_token_weight
from ..layers.SelfAttention_Family import FullAttention, AttentionLayer
from ..layers.Embed import PatchEmbedding
from ..layers.FANLayer import FANLayer
from ..layers.newLinear import newLinear

from ..utils.tools import create_sin_pos_embed
import math
import random
from typing import List
from ..layers.RevIN import RevIN

from ..utils.CKA import CudaCKA


class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=2,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1, 2)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu", MLP_flag=True,
                 FAN_flag=False, **kwargs):
        super(EncoderLayer, self).__init__()
        self.MLP_flag = MLP_flag
        self.FAN_flag = FAN_flag
        # dff is defaulted at 4*d_model
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        if self.MLP_flag:
            self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
            self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
            self.activation = F.relu if activation == "relu" else F.gelu
        elif self.FAN_flag:
            self.FANLayers = nn.Sequential(
                FANLayer(d_model, d_ff, activation=activation),
                FANLayer(d_ff, d_model, activation=None)
            )

        if self.MLP_flag or self.FAN_flag:
            self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, attn_mask=None, tau=None, delta=None, token_weight=None, **kwargs):
        list_flag = isinstance(x, tuple) or isinstance(x, List)
        if list_flag:
            k_ori = x[1]
            if len(x) not in {2, 3}:
                raise ValueError('Input error in EncoderLayer')
            q, k, v = (x[0], x[1], x[1]) if len(x) == 2 else (x[0], x[1], x[2])
            x = q
        else:
            q, k, v = x, x, x

        new_x, attn = self.attention(
            q, k, v,
            attn_mask=attn_mask,
            tau=tau, delta=delta,
            token_weight=token_weight
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        output = y

        if self.MLP_flag:
            y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
            y = self.dropout(self.conv2(y).transpose(-1, 1))
        elif self.FAN_flag:
            y = self.FANLayers(y)

        if self.MLP_flag or self.FAN_flag:
            output = self.norm2(x + y)

        if list_flag:
            return [output, k_ori], attn
        else:
            return output, attn


class LinearEncoder(nn.Module):
    def __init__(self, d_model, d_ff=None, CovMat=None, dropout=0.1, activation="relu", token_num=None, **kwargs):
        super(LinearEncoder, self).__init__()

        d_ff = d_ff or 4 * d_model
        self.d_model = d_model
        self.d_ff = d_ff
        self.CovMat = CovMat.unsqueeze(0) if CovMat is not None else None
        self.token_num = token_num

        self.norm1 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # attention --> linear
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        init_weight_mat = torch.eye(self.token_num) * 1.0 + torch.randn(self.token_num, self.token_num) * 1.0
        self.weight_mat = nn.Parameter(init_weight_mat[None, :, :])

        # self.bias = nn.Parameter(torch.zeros(1, 1, self.d_model))

        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.activation = F.relu if activation == "relu" else F.gelu
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, **kwargs):
        # x.shape: b, l, d_model
        values = self.v_proj(x)

        if self.CovMat is not None:
            A = F.softmax(self.CovMat, dim=-1) + F.softplus(self.weight_mat)
        else:
            A = F.softplus(self.weight_mat)

        A = F.normalize(A, p=1, dim=-1)
        A = self.dropout(A)

        new_x = A @ values  # + self.bias

        x = x + self.dropout(self.out_proj(new_x))
        x = self.norm1(x)

        y = self.dropout(self.activation(self.conv1(x.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        output = self.norm2(x + y)

        return output, None


class LinearEncoder_Multihead(nn.Module):
    def __init__(self, d_model, d_ff=None, CovMat=None, dropout=0.1, activation="relu", token_num=None, n_heads=2,
                 **kwargs):
        super(LinearEncoder_Multihead, self).__init__()

        d_ff = d_ff or 4 * d_model
        self.d_model = d_model
        self.d_ff = d_ff
        self.CovMat = None  # CovMat.unsqueeze(0) if CovMat is not None else
        self.token_num = token_num
        self.n_heads = n_heads

        self.norm1 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # attention --> linear
        head_dim = d_model // n_heads
        self.v_proj = nn.Linear(d_model, head_dim * head_dim)
        self.out_proj = nn.Linear(head_dim * head_dim, d_model)

        self.weight_mat = nn.Parameter(torch.randn(self.n_heads, self.token_num, self.token_num))

        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.activation = F.relu if activation == "relu" else F.gelu
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, **kwargs):
        # x.shape: b, n, d_model
        B, N, D = x.shape
        # b,n,h,d
        values = self.v_proj(x).reshape(B, N, self.n_heads, -1)

        A = F.softplus(self.weight_mat)

        A = F.normalize(A, p=1, dim=-1)
        A = self.dropout(A)

        # cuda out of memory
        new_x = (A @ values.transpose(1, 2)).transpose(1, 2).flatten(-2)

        x = x + self.dropout(self.out_proj(new_x))
        x = self.norm1(x)

        y = self.dropout(self.activation(self.conv1(x.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        output = self.norm2(x + y)

        return output, None


class LinearEncoder_newffn(nn.Module):
    def __init__(self, d_model, d_ff=None, CovMat=None, dropout=0.1, activation="relu", token_num=None, **kwargs):
        super(LinearEncoder_newffn, self).__init__()

        d_ff = d_ff or 4 * d_model
        self.d_model = d_model
        self.d_ff = d_ff
        self.CovMat = CovMat.unsqueeze(0) if CovMat is not None else None
        self.token_num = token_num

        self.norm1 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # attention --> linear
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        init_weight_mat = torch.eye(self.token_num) * 1.0 + torch.randn(self.token_num, self.token_num) * 1.0
        self.weight_mat = nn.Parameter(init_weight_mat[None, :, :])

        # self.ffn1 = newLinear(input_dim=d_model, output_dim=d_ff, bias=True)
        # self.ffn2 = newLinear(input_dim=d_ff, output_dim=d_model, bias=True)

        self.ffn1 = nn.Linear(d_model, d_ff)
        self.ffn2 = nn.Linear(d_ff, d_model)
        self.activation = F.relu if activation == "relu" else F.gelu
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, **kwargs):
        # x.shape: b, l, d_model
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

        # ffn
        y = self.dropout(self.activation(self.ffn1(x)))
        y = self.dropout(self.ffn2(y))
        output = self.norm2(x + y)

        return output, None


class LinearEncoder_ablation(nn.Module):
    def __init__(self, d_model, d_ff=None, CovMat=None, dropout=0.1, activation="relu", token_num=None,
                 var_linear_mode='normal', temp_linear=True, temp_attn_linear=False, var_linear_enable=True, **kwargs):
        super(LinearEncoder_ablation, self).__init__()

        d_ff = d_ff or 4 * d_model
        self.d_model = d_model
        self.d_ff = d_ff
        self.CovMat = CovMat.unsqueeze(0) if CovMat is not None else None
        self.token_num = token_num

        self.var_linear_enable = var_linear_enable
        self.linear_r_mode = var_linear_mode != 'normal'
        self.temp_linear = temp_linear
        self.temp_attn_linear = temp_attn_linear

        self.dropout = nn.Dropout(dropout)

        if not self.var_linear_enable:
            assert self.temp_linear or self.temp_attn_linear

        if self.var_linear_enable:
            self.norm1 = nn.LayerNorm(d_model)

            # attention --> linear
            if self.linear_r_mode:
                self.v_proj = nn.Linear(d_model, d_model)
                self.out_proj = nn.Linear(d_model, d_model)

                init_weight_mat = torch.eye(self.token_num) * 1.0 + torch.randn(self.token_num, self.token_num) * 1.0
                self.weight_mat = nn.Parameter(init_weight_mat[None, :, :])

            else:
                self.var_lin = nn.Linear(self.token_num, self.token_num)

        if self.temp_linear:
            self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
            self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
            self.activation = F.relu if activation == "relu" else F.gelu
            self.norm2 = nn.LayerNorm(d_model)

        elif self.temp_attn_linear:
            self.ffn1 = newLinear(d_model, d_ff)
            self.ffn2 = newLinear(d_ff, d_model)
            self.activation = F.relu if activation == "relu" else F.gelu
            self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, **kwargs):
        # x.shape: b, l, d_model
        output = x  # to make PyCharm happy

        if self.var_linear_enable:

            if self.linear_r_mode:
                values = self.v_proj(x)
                if self.CovMat is not None:
                    A = F.softmax(self.CovMat, dim=-1) + F.softplus(self.weight_mat)
                else:
                    A = F.softplus(self.weight_mat)
                A = F.normalize(A, p=1, dim=-1)
                A = self.dropout(A)
                new_x = A @ values
                new_x = self.out_proj(new_x)
            else:
                new_x = self.var_lin(x.transpose(-1, -2)).transpose(-1, -2)

            x = x + self.dropout(new_x)
            output = x = self.norm1(x)

        if self.temp_linear:
            y = self.dropout(self.activation(self.conv1(x.transpose(-1, 1))))
            y = self.dropout(self.conv2(y).transpose(-1, 1))
            output = self.norm2(x + y)

        elif self.temp_attn_linear:
            y = self.dropout(self.activation(self.ffn1(x)))
            y = self.dropout(self.ffn2(y))
            output = self.norm2(x + y)

        return output, None


class LinearEncoder_pre_post_lin(nn.Module):
    def __init__(self, d_model, d_ff=None, dropout=0.1, activation="relu", token_num=None,
                 pre_lin=True, post_lin=True, **kwargs):
        super(LinearEncoder_pre_post_lin, self).__init__()

        d_ff = d_ff or 4 * d_model
        self.d_model = d_model
        self.d_ff = d_ff
        self.token_num = token_num

        self.pre_lin = pre_lin
        self.post_lin = post_lin

        self.dropout = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)

        # attention --> linear

        self.v_proj = nn.Linear(d_model, d_model) if self.pre_lin else nn.Identity()
        self.out_proj = nn.Linear(d_model, d_model) if self.post_lin else nn.Identity()

        init_weight_mat = torch.eye(self.token_num) * 1.0 + torch.randn(self.token_num, self.token_num) * 1.0
        self.weight_mat = nn.Parameter(init_weight_mat[None, :, :])

        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.activation = F.relu if activation == "relu" else F.gelu
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, **kwargs):
        # x.shape: b, l, d_model

        values = self.v_proj(x)
        A = F.softplus(self.weight_mat)
        A = F.normalize(A, p=1, dim=-1)
        A = self.dropout(A)
        new_x = A @ values
        new_x = self.out_proj(new_x)

        x = x + self.dropout(new_x)
        x = self.norm1(x)

        y = self.dropout(self.activation(self.conv1(x.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        output = self.norm2(x + y)

        return output, None


class LinearEncoder_abla_design(nn.Module):
    def __init__(self, d_model, d_ff=None, CovMat=None, dropout=0.1, activation="relu", token_num=None,
                 CovMatTrans='softmax', WeightTrans='softplus', NormSet='L1', onlyconv=False, **kwargs):
        super(LinearEncoder_abla_design, self).__init__()

        d_ff = d_ff or 4 * d_model
        self.d_model = d_model
        self.d_ff = d_ff
        self.CovMat = CovMat.unsqueeze(0) if CovMat is not None else None
        self.token_num = token_num

        self.CovMatTrans = CovMatTrans
        self.WeightTrans = WeightTrans
        self.NormSet = NormSet
        self.onlyconv = onlyconv

        self.dropout = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)

        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        assert not (self.CovMatTrans == 'none' and self.WeightTrans == 'none'), \
            'both self.CovMatTrans and self.WeightTrans are None'

        self.cov_mat_trans = 0.0
        if self.CovMatTrans != 'none' or self.onlyconv:
            if self.CovMatTrans == 'softmax':
                self.cov_mat_trans = F.softmax(self.CovMat, dim=-1)
            elif self.CovMatTrans == 'softplus':
                self.cov_mat_trans = F.softplus(self.CovMat)
            elif self.CovMatTrans == 'sigmoid':
                self.cov_mat_trans = F.sigmoid(self.CovMat)
            elif self.CovMatTrans == 'relu':
                self.cov_mat_trans = F.relu(self.CovMat)
            elif self.CovMatTrans == 'identity':
                self.cov_mat_trans = self.CovMat
            else:
                raise NotImplementedError

        # Linear design
        if self.CovMatTrans != 'none' or self.onlyconv:
            assert self.CovMat is not None, 'CovMat cannot be None in LinearEncoder.'

        self.weight_mat = None
        if self.WeightTrans != 'none' and not self.onlyconv:
            init_weight_mat = torch.eye(self.token_num) * 1.0 + torch.randn(self.token_num, self.token_num) * 1.0
            self.weight_mat = nn.Parameter(init_weight_mat[None, :, :])

        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.activation = F.relu if activation == "relu" else F.gelu
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, **kwargs):
        # x.shape: b, l, d_model

        values = self.v_proj(x)

        weight_mat_trans = 0.0
        if not self.onlyconv and self.WeightTrans != 'none' and self.weight_mat is not None:
            if self.WeightTrans == 'softmax':
                weight_mat_trans = F.softmax(self.weight_mat, dim=-1)
            elif self.WeightTrans == 'softplus':
                weight_mat_trans = F.softplus(self.weight_mat)
            elif self.WeightTrans == 'sigmoid':
                weight_mat_trans = F.sigmoid(self.weight_mat)
            elif self.WeightTrans == 'relu':
                weight_mat_trans = F.relu(self.weight_mat)
            elif self.WeightTrans == 'identity':
                weight_mat_trans = self.weight_mat
            else:
                raise NotImplementedError

        A = self.cov_mat_trans + weight_mat_trans

        if self.NormSet != 'none':
            if self.NormSet == 'L1':
                A = F.normalize(A, p=1, dim=-1)
            elif self.NormSet == 'L2':
                A = F.normalize(A, p=2, dim=-1)

        A = self.dropout(A)
        new_x = A @ values
        new_x = self.out_proj(new_x)

        x = x + self.dropout(new_x)
        x = self.norm1(x)

        y = self.dropout(self.activation(self.conv1(x.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        output = self.norm2(x + y)

        return output, None


class LinearEncoder_abla_design_nocorr(nn.Module):
    def __init__(self, d_model, d_ff=None, dropout=0.1, activation="relu", token_num=None,
                 WeightTrans='softplus', NormSet='L1', **kwargs):
        super(LinearEncoder_abla_design_nocorr, self).__init__()

        d_ff = d_ff or 4 * d_model
        self.d_model = d_model
        self.d_ff = d_ff
        self.token_num = token_num

        self.WeightTrans = WeightTrans
        self.NormSet = NormSet

        self.dropout = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)

        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # Linear design
        init_weight_mat = torch.eye(self.token_num) * 1.0 + torch.randn(self.token_num, self.token_num) * 1.0
        self.weight_mat = nn.Parameter(init_weight_mat[None, :, :])

        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.activation = F.relu if activation == "relu" else F.gelu
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, **kwargs):
        # x.shape: b, l, d_model

        values = self.v_proj(x)

        if self.WeightTrans == 'softmax':
            weight_mat_trans = F.softmax(self.weight_mat, dim=-1)
        elif self.WeightTrans == 'softplus':
            weight_mat_trans = F.softplus(self.weight_mat)
        elif self.WeightTrans == 'sigmoid':
            weight_mat_trans = F.sigmoid(self.weight_mat)
        elif self.WeightTrans == 'relu':
            weight_mat_trans = F.relu(self.weight_mat)
        elif self.WeightTrans == 'identity':
            weight_mat_trans = self.weight_mat
        else:
            raise NotImplementedError

        A = weight_mat_trans

        if self.NormSet != 'none':
            if self.NormSet == 'L1' and self.WeightTrans != 'softmax':
                A = F.normalize(A, p=1, dim=-1)
            elif self.NormSet == 'L2':
                A = F.normalize(A, p=2, dim=-1)

        A = self.dropout(A)
        new_x = A @ values
        new_x = self.out_proj(new_x)

        x = x + self.dropout(new_x)
        x = self.norm1(x)

        y = self.dropout(self.activation(self.conv1(x.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        output = self.norm2(x + y)

        return output, None


class Encoder_ori(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None, one_output=False, CKA_flag=False):
        super(Encoder_ori, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer
        self.one_output = one_output
        self.CKA_flag = CKA_flag
        if self.CKA_flag:
            print('CKA is enabled...')

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # x [B, nvars, D]
        attns = []
        X0 = None  # to make Pycharm happy
        layer_len = len(self.attn_layers)
        for i, attn_layer in enumerate(self.attn_layers):
            x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
            attns.append(attn)

            if not self.training and self.CKA_flag and layer_len > 1:
                if i == 0:
                    X0 = x

                if i == layer_len - 1 and random.uniform(0, 1) < 1e-1:
                    CudaCKA1 = CudaCKA(device=x.device)
                    cka_value = CudaCKA1.linear_CKA(X0.flatten(0, 1)[:1000], x.flatten(0, 1)[:1000])
                    print(f'CKA: \t{cka_value:.3f}')

        if isinstance(x, tuple) or isinstance(x, List):
            x = x[0]

        if self.norm is not None:
            x = self.norm(x)

        if self.one_output:
            return x
        else:
            return x, attns


class FlattenHead(nn.Module):
    def __init__(self, nf, target_window, head_dropout=0.0):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class refine_module(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.e_layers = configs.second_e_layers
        self.d_model = configs.d_model
        self.d_ff = configs.d_ff
        self.dropout = configs.dropout
        self.n_heads = configs.n_heads
        self.activation = configs.activation
        self.patch_len = configs.temp_patch_len2
        self.stride = configs.temp_stride2

        self.encoder = Encoder_ori(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(mask_flag=False, attention_dropout=self.dropout,
                                      output_attention=False, token_num=None, imp_mode=False,
                                      ij_mat_flag=False, num_heads=self.n_heads, plot_mat_flag=False),
                        d_model=self.d_model, n_heads=self.n_heads),
                    d_model=self.d_model,
                    d_ff=self.d_ff,
                    dropout=self.dropout,
                    activation=self.activation
                ) for _ in range(self.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(self.d_model)
        )

        self.revin_layer = RevIN(self.enc_in, affine=True)
        # self.revin_layer2 = RevIN(self.enc_in, affine=True)

        self.patch_embedding = PatchEmbedding(
            d_model=self.d_model, patch_len=self.patch_len, stride=self.stride, padding=self.stride,
            dropout=self.dropout)
        self.patch_embedding2 = PatchEmbedding(
            d_model=self.d_model, patch_len=self.patch_len, stride=self.stride, padding=self.stride,
            dropout=self.dropout)

        # flatten head
        self.pred_token_num = int((self.pred_len - self.patch_len) / self.stride + 2)
        self.head_nf = self.d_model * self.pred_token_num
        self.head = FlattenHead(nf=self.head_nf, target_window=self.pred_len,
                                head_dropout=configs.dropout)
        # self.dropout_layer = nn.Dropout(configs.dropout)

        print('refine_module is used')

    def forward(self, x, pred):
        # x, pred: [batch, len, vars]
        assert x.ndim == 3 and pred.ndim == 3 and x.shape[-1] == self.enc_in and pred.shape[-1] == self.enc_in

        xy_concat = torch.concat([x, pred], dim=1)
        # normalize xy_concat
        xy_concat = self.revin_layer(xy_concat, mode='norm')
        pred = xy_concat[:, -self.pred_len:, :]

        # use another revin_layer
        # pred = self.revin_layer2(pred, mode='norm')

        # patch embedding: return [b*n, token_num, d_model]
        xy_embed, _ = self.patch_embedding(xy_concat.transpose(-1, -2))
        pred = xy_embed[:, -self.pred_token_num:, :]
        # pred, _ = self.patch_embedding2(pred.transpose(-1, -2))

        # encoder  [b*n, token_num, d_model]
        pred_refine_feat, _ = self.encoder(x=(pred, xy_embed))
        pred_refine_feat = torch.reshape(
            pred_refine_feat, (-1, self.enc_in, pred_refine_feat.shape[-2], pred_refine_feat.shape[-1]))
        pred_refine_feat = pred_refine_feat.transpose(-1, -2)

        # Decoder
        pred_refine = self.head(pred_refine_feat)  # z: [bs x nvars x pred_len]
        pred_refine = pred_refine.transpose(-1, -2)

        pred_refine = self.revin_layer(pred_refine, mode='denorm')

        return pred_refine


class fix_mask_with_neighbor(nn.Module):
    def __init__(self, enc_in, kernel_size=3):
        super().__init__()
        self.enc_in = enc_in
        self.kernel_size = kernel_size
        self.alpha = nn.Parameter(torch.tensor(-3.0))
        # different channels do not mix
        self.conv1d = nn.Conv1d(self.enc_in, self.enc_in, groups=self.enc_in, kernel_size=self.kernel_size,
                                padding='same', padding_mode='zeros', bias=True)  # padding_mode: zeros circular

    def forward(self, x, mask=None):
        if mask is None:
            return x
        B, N, D = x.shape
        if D == self.enc_in:
            x = x.transpose(-1, -2)
            B, N, D = x.shape
        assert N == self.enc_in, 'N!=self.enc_in in fix_mask_with_neighbor class...'
        alpha = F.sigmoid(self.alpha)
        x2 = alpha * self.conv1d(x) + (1 - alpha) * x
        if mask.shape[1] != N:
            mask = mask.transpose(-1, -2)
        x = torch.where(mask, x2, x)
        return x


def swin_output_update2(dec_seq_i, revin_layer, mask, x_ori, dec_seq_inter, dec_seq_inter2):
    # dec_seq_i: [b, l, n]
    # update dec_seq_i, revin_layer, dec_seq_inter, dec_seq_inter2 when Swin_output is enabled
    if dec_seq_i is not None:
        if revin_layer is not None:
            dec_seq_i = revin_layer(dec_seq_i, mode='denorm')
        dec_seq_inter.append(dec_seq_i)

        dec_seq_i[~mask] = x_ori[~mask]  # calibration
        dec_seq_inter2.append(dec_seq_i)
        if revin_layer is not None:
            dec_seq_i = revin_layer(dec_seq_i, mode='norm', mask=None)  # new revin_layer
    return dec_seq_i


class Encoder(nn.Module):
    # the general class
    def __init__(self, attn_layers, temp_attn_layers=None, temp_attn_params=None, conv_layers=None, norm_layer=None,
                 hierarchyFlag=False, patch_ln=False, imp_mode=False, Swin_after_patch=0, Swin_after_iTrans=0,
                 swin_layers=None, swin_layers2=None, Patch_CI=True, neighbor_fix=False, swin_first=False):
        # , temp_token_num=None, channel_token_num=None
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer
        self.temp_attn_layers = nn.ModuleList(temp_attn_layers) if temp_attn_layers is not None else None
        self.temp_attn_params = temp_attn_params
        self.imp_mode = imp_mode
        self.hierarchyFlag = False if self.imp_mode else hierarchyFlag

        self.Swin_after_patch = Swin_after_patch
        self.Swin_after_iTrans = Swin_after_iTrans

        self.seq_len = temp_attn_params.seq_len
        self.d_model = temp_attn_params.d_model
        self.temp_token_num = temp_attn_params.temp_token_num
        self.enc_in = temp_attn_params.enc_in

        self.swin_layers = swin_layers
        self.swin_layers2 = swin_layers2

        self.Patch_CI = Patch_CI
        self.neighbor_fix = neighbor_fix
        self.swin_first = swin_first

        self.act_layer = nn.GELU()
        # self.act_layer = nn.ReLU()

        print('Encoder is initialized...')

        self.projector3 = nn.Linear(self.seq_len, self.d_model, bias=True) \
            if self.Swin_after_patch else nn.Identity()

        if self.imp_mode and self.Swin_after_iTrans and self.swin_layers2 is not None:
            self.projector3_last = nn.Linear(self.d_model, self.seq_len, bias=True)
            self.norm_post_proj3_last = nn.LayerNorm(self.d_model)
            self.projector4_last = nn.Linear(self.seq_len, self.d_model, bias=True)

        self.dropout = nn.Dropout(p=0.1)
        if temp_attn_layers is not None:
            self.len1 = len1 = len(temp_attn_layers)
            print('Temporal attention is initialized in Encoder...')
            self.patch_ln = patch_ln  # or len(attn_layers) == 0
            if self.patch_ln:
                print('LayerNorm in PatchTST is used...')

            if self.neighbor_fix:
                print('Class fix_mask_with_neighbor is used...')
                self.fix_layer = fix_mask_with_neighbor(self.enc_in)

            if self.Patch_CI:
                self.projector1 = nn.Linear(self.temp_attn_params.temp_patch_len, self.d_model)

                # patch -- swin
                self.projector2 = nn.Linear(self.d_model * self.temp_token_num,
                                            self.seq_len if self.Swin_after_patch
                                            else self.d_model)
            else:
                self.projector1 = nn.Linear(self.temp_attn_params.temp_patch_len * self.enc_in, self.d_model)
                # [b, n'*d_model] --> [b,n*len]
                self.projector2 = nn.Linear(self.d_model * self.temp_token_num,
                                            self.seq_len * self.enc_in if self.Swin_after_patch
                                            else self.d_model * self.enc_in)

            if self.imp_mode and not self.Swin_after_patch and len(self.attn_layers) == 0:
                self.proj2_d2l = nn.Linear(self.d_model, self.seq_len)  # for check

            # for shortcut
            # if self.Swin_after_patch and self.len1 >= 1 and self.swin_first:
            #     self.proj2_l2l_patch2seq = nn.Linear(self.seq_len, self.seq_len)
            #     self.alpha = nn.Parameter(torch.tensor(-3.0))

            pe_mat = torch.zeros(1, self.temp_token_num, self.d_model)
            self.positional_encoding = nn.Parameter(pe_mat)

            self.norm0 = nn.LayerNorm(self.d_model)
            self.norm1 = nn.LayerNorm(self.d_model)

            # hierarchy mode
            if self.hierarchyFlag and len1 > 1:
                self.proj = nn.ModuleList([nn.Linear(self.d_model * 2, self.d_model,
                                                     bias=True) for _ in range(len1 - 1)] +
                                          [nn.Linear(self.d_model * len1,
                                                     self.d_model,
                                                     bias=True)])
                self.norm2 = nn.ModuleList([nn.LayerNorm(self.d_model) for _ in range(len1)])
        else:
            self.projector2 = nn.Linear(self.d_model, self.seq_len, bias=True)

        if len(self.attn_layers):
            self.norm_for_i = nn.LayerNorm(self.d_model)

        self.Pi_flag = len(self.attn_layers) > 0 and self.temp_attn_layers is not None

        if self.Pi_flag:
            self.projector_shortcut = nn.Linear(self.seq_len, self.d_model, bias=True)
            self.pi_weight = nn.Parameter(torch.zeros(2))
            self.pi_weight2 = nn.Parameter(torch.zeros(2))
            self.tau = nn.Parameter(torch.ones(1) * -5)
            self.tau2 = nn.Parameter(torch.ones(1) * -5)

    def forward(self, x, attn_mask=None, tau=None, delta=None, temp_token_weight=None, ch_token_weight=None,
                revin_layer=None, mask=None, x_ori=None, dec_seq_inter=None, dec_seq_inter2=None, swin_output=0):
        # x [B, N, D] L means channel here
        B, N, D = x.shape
        input_ori = x
        attns = []
        Swin_inter_output = x.transpose(-1, -2)
        if self.conv_layers is not None:
            # not used in this project
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                # delta only work at the first layer
                delta = delta if i == 0 else None
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None)
            attns.append(attn)
        else:
            x0 = None
            # patch tst
            if self.temp_attn_layers is not None:
                if self.neighbor_fix:
                    x = self.fix_layer(x, mask)

                    if swin_output:
                        Swin_inter_output = swin_output_update2(x.transpose(-1, -2), revin_layer, mask, x_ori,
                                                                dec_seq_inter, dec_seq_inter2)

                if self.Pi_flag:
                    x0 = x

                if self.Patch_CI:
                    # [b,n,d] --> [b,n,n',p] --> [b',n',p]
                    assert D > self.temp_attn_params.temp_patch_len
                    rem = (D - self.temp_attn_params.temp_patch_len) % self.temp_attn_params.temp_stride
                    if rem != 0:
                        x = F.pad(x, pad=[0, self.temp_attn_params.temp_stride - rem])
                    x = x.unfold(dimension=-1, size=self.temp_attn_params.temp_patch_len,
                                 step=self.temp_attn_params.temp_stride).flatten(start_dim=0, end_dim=1)
                else:
                    # [b,n,d] --> [b,n,n',p] --> [b,n',n, p] --> [b,n',n*p]
                    x = x.unfold(dimension=-1, size=self.temp_attn_params.temp_patch_len,
                                 step=self.temp_attn_params.temp_stride).transpose(1, 2).flatten(start_dim=-2)

                # [b',n',p] --> [b',n',d_model]
                # self.projector1 has been modified according to self.Patch_CI
                x = self.projector1(x) + self.positional_encoding
                x = self.norm0(x)

                x_list = []
                # x0 = x
                B2, L2, D2 = x.shape

                for i, temp_attn_layer in enumerate(self.temp_attn_layers):
                    if temp_token_weight is not None:
                        assert x.shape[1] == temp_token_weight.shape[1], 'Please check temp_token_weight.'

                    x, _ = temp_attn_layer(x, token_weight=temp_token_weight)
                    if i == 0 and self.hierarchyFlag and self.len1 > 1:
                        x_list.append(x)
                    if self.hierarchyFlag and x.shape[1] < L2:
                        # x2 = x.repeat_interleave(2 ** i, dim=1)
                        x2 = x.repeat_interleave(math.ceil(L2 / x.shape[1]), dim=1)
                        x2 = x2[:, :L2, :]
                        x_list.append(x2)

                    if self.hierarchyFlag and self.len1 > 1 and x.shape[1] > 2 and i < self.len1 - 1:
                        # 240905; token num should > 2
                        # token num --> 1/2
                        B3, L3, D3 = x.shape
                        # print(f'x.shape: {x.shape}')
                        if L3 % 2 == 1:
                            x = torch.cat([x, x[:, [-1], :]], dim=1)
                        x2 = x.reshape(B3, -1, 2, D3).flatten(start_dim=-2)
                        # print(f'x2.shape: {x2.shape}')
                        # print(f'self.proj[i]: {self.proj[i]}')
                        x2 = self.proj[i](x2)
                        x = self.act_layer(x2)
                        # x = self.norm2[i](x2)

                        # temp_token_weight; for imputation
                        temp_token_weight = hier_half_token_weight(temp_token_weight)
                        # print('Token_weight num reduced to ', temp_token_weight.shape[1], '. ')
                # output
                if self.hierarchyFlag and self.len1 > 1:
                    # cat and project
                    # x2 = torch.cat(x_list, dim=-1)
                    # x = self.norm2[-1](self.proj[-1](x2))
                    x = torch.sum(torch.stack(x_list), dim=0)
                    # x = self.norm2[-1](x)
                    x = self.act_layer(x)

            # temp_attn_layers could be empty; Swin can still be used without temp_attn_layers
            if self.imp_mode and self.Swin_after_patch and self.swin_layers is not None:
                if self.temp_attn_layers is not None:
                    # self.Patch_CI: [b',n',d_model] --> [b',n'*d_model] --> [b',seq_len] --> [b,n,l]
                    # not self.Patch_CI: [b,n',d_model] --> [b,n'*d_model] --> [b,n*seq_len] --> [b,n,l]
                    x = self.projector2(x.flatten(start_dim=-2)).view(B, N, -1)

                    # shortcut
                    # if self.swin_first:
                    #     alpha = F.sigmoid(self.alpha)
                    #     x = self.proj2_l2l_patch2seq(x) + alpha * input_ori
                else:
                    # [b,n,d] --> [b,n,l]
                    x = self.projector2(x)

                # seq_len
                # Swin_inter_output has to be [b,l,n]
                Swin_inter_output = x.transpose(-1, -2)

                # revin layer
                if swin_output:
                    Swin_inter_output = swin_output_update2(Swin_inter_output, revin_layer, mask, x_ori,
                                                            dec_seq_inter, dec_seq_inter2)
                if isinstance(self.swin_layers, nn.ModuleList):
                    for layer in self.swin_layers:
                        Swin_inter_output, _ = layer((Swin_inter_output, mask))
                else:
                    Swin_inter_output, _ = self.swin_layers((Swin_inter_output, mask))

                # revin layer
                if swin_output:
                    Swin_inter_output = swin_output_update2(Swin_inter_output, revin_layer, mask, x_ori,
                                                            dec_seq_inter, dec_seq_inter2)

                x = Swin_inter_output.transpose(-1, -2)

                if len(self.attn_layers):
                    # [b,n,l] --> [b,n,dim]
                    x = self.projector3(x)
                    x = self.norm_for_i(x)
            else:
                if self.temp_attn_layers is not None:

                    # if self.Patch_CI: [b',n',d_model] --> [b',n'*d_model] --> [b',d_model] --> [b,n,dim]
                    # else: [b, n',d_model] --> [b,n'*d_model] --> [b,n*dim] --> [b,n,dim]
                    # x = x + x0
                    x = self.projector2(x.flatten(start_dim=-2)).view(B, N, -1)

                    if self.patch_ln:
                        x = self.norm1(x)
                    else:
                        x = self.act_layer(x)

                    if len(self.attn_layers) == 0 and self.imp_mode:
                        # also check this intermediate result; 240702
                        # [b, n, dim] --> [b, n, l]
                        # print('Check this....')
                        Swin_inter_output = self.proj2_d2l(x) + Swin_inter_output.transpose(-1, -2)
                        Swin_inter_output = Swin_inter_output.transpose(-1, -2)
                        # revin layer
                        if swin_output:
                            Swin_inter_output = swin_output_update2(Swin_inter_output, revin_layer, mask, x_ori,
                                                                    dec_seq_inter, dec_seq_inter2)

            if self.Pi_flag:
                # x0 = x

                pi_weight = F.softmax(self.pi_weight / F.softplus(self.tau))
                x = x0 = x * pi_weight[0] + self.projector_shortcut(x0) * pi_weight[1]
                # if not self.training and random.uniform(0, 1) < 1e-3:
                #     print(f'pi_weight: {pi_weight}')

            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta, token_weight=ch_token_weight)
                # attns.append(attn)

            # 240925
            if self.Pi_flag:
                pi_weight2 = F.softmax(self.pi_weight2 / F.softplus(self.tau2))
                x = x * pi_weight2[0] + x0 * pi_weight2[1]

                # pi_weight2 = F.softplus(self.pi_weight2)
                # x = x + x0 * pi_weight2[1]

                # x = x0

                # if not self.training and random.uniform(0, 1) < 1e-3:
                #     print(f'pi_weight2: {pi_weight2}')

            if self.imp_mode and self.Swin_after_iTrans and self.swin_layers2 is not None:

                # [b,n,dim] --> [b,n,l] --> [b,l,n]
                Swin_inter_output = self.projector3_last(self.norm_post_proj3_last(x)).transpose(-1, -2)

                # revin layer
                if swin_output:
                    Swin_inter_output = swin_output_update2(Swin_inter_output, revin_layer, mask, x_ori,
                                                            dec_seq_inter, dec_seq_inter2)

                # [b,n,l] --> [b,l,n] --> [b,n,l]
                if isinstance(self.swin_layers2, nn.ModuleList):
                    for layer in self.swin_layers2:
                        Swin_inter_output, _ = layer((Swin_inter_output, mask))
                else:
                    Swin_inter_output, _ = self.swin_layers2((Swin_inter_output, mask))

                # [b,l,n] --> [b,n,l] --> [b,n,dim]
                # x = self.projector4_last(Swin_inter_output.transpose(-1, -2))

                # keep it is
                x = Swin_inter_output.transpose(-1, -2)

        if self.norm is not None and x.shape[-1] == self.d_model:
            x = self.norm(x)

        return x, attns, Swin_inter_output.transpose(-1, -2)  # Swin_inter_output not updated for Swin_after_iTrans


class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask,
            tau=tau, delta=None
        )[0])
        x = self.norm1(x)

        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask,
            tau=tau, delta=delta
        )[0])

        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y)


class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta)

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x
