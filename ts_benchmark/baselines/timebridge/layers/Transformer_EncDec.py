import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class TSEncoder(nn.Module):
    def __init__(self, attn_layers):
        super(TSEncoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # x [B, L, D]
        attns = []
        for attn_layer in self.attn_layers:
            x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
            attns.append(attn)
        return x, attns


def PeriodNorm(x, period_len=6):
    if len(x.shape) == 3:
        x = x.unsqueeze(-2)
    b, c, n, t = x.shape
    x_patch = [x[..., period_len - 1 - i:-i + t] for i in range(0, period_len)]
    x_patch = torch.stack(x_patch, dim=-1)

    mean = x_patch.mean(4)
    mean = F.pad(mean.reshape(b * c, n, -1),
                 mode='replicate', pad=(period_len - 1, 0)).reshape(b, c, n, -1)
    out = x - mean
    return out.squeeze(-2)


class IntAttention(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, stable_len=8, attn_map=False,
                 dropout=0.1, activation="relu", stable=True, enc_in=None):
        super(IntAttention, self).__init__()
        self.stable = stable
        self.stable_len = stable_len
        self.attn_map = attn_map
        d_ff = d_ff or 4 * d_model
        self.attention = attention

        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        new_x = self.temporal_attn(x)
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.fc1(y)))
        y = self.dropout(self.fc2(y))

        return self.norm2(x + y), None

    def temporal_attn(self, x):
        b, c, n, d = x.shape
        new_x = x.reshape(-1, n, d)

        qk = new_x
        if self.stable:
            with torch.no_grad():
                qk = PeriodNorm(new_x, self.stable_len)
        new_x = self.attention(qk, qk, new_x)[0]
        new_x = new_x.reshape(b, c, n, d)
        return new_x


class PatchSampling(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu",
                 in_p=30, out_p=4, stable=False, stable_len=8):
        super(PatchSampling, self).__init__()

        d_ff = d_ff or 4 * d_model
        self.in_p = in_p
        self.out_p = out_p
        self.stable = stable
        self.stable_len = stable_len

        self.attention = attention
        self.conv1 = nn.Conv1d(
            self.in_p, self.out_p, 1, 1, 0, bias=False)
        self.conv2 = nn.Conv1d(
            self.out_p + 1, self.out_p, 1, 1, 0, bias=False)

        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        new_x = self.down_attn(x)
        y = x = self.norm1(new_x)

        y = self.dropout(self.activation(self.fc1(y)))
        y = self.dropout(self.fc2(y))

        return self.norm2(x + y), None

    def down_attn(self, x):
        b, c, n, d = x.shape
        x = x.reshape(-1, n, d)
        new_x = self.conv1(x)
        new_x = self.conv2(torch.cat(
            [new_x, x.mean(-2, keepdim=True)], dim=-2)) + new_x
        new_x = self.attention(new_x, x, x)[0] + self.dropout(new_x)
        return new_x.reshape(b, c, -1, d)


class CointAttention(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, axial=True, stable_len=8,
                 dropout=0.1, activation="relu", stable=True, enc_in=None, ):
        super(CointAttention, self).__init__()

        self.stable = stable
        self.stable_len = stable_len
        d_ff = d_ff or 4 * d_model

        self.axial_func = axial
        self.attention1 = attention
        self.attention2 = copy.deepcopy(attention)

        self.num_rc = math.ceil((enc_in + 4) ** 0.5)
        self.pad_ch = nn.ConstantPad1d(
            (0, self.num_rc ** 2 - (enc_in + 4)), 0)

        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.norm0 = nn.LayerNorm(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        if self.axial_func is True:
            new_x = self.axial_attn(x)
        else:
            new_x = self.full_attn(x)
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.fc1(y)))
        y = self.dropout(self.fc2(y))

        return self.norm2(x + y), None

    def axial_attn(self, x):
        b, c, n, d = x.shape

        new_x = rearrange(x, 'b c n d -> (b n) c d')
        new_x = (self.pad_ch(new_x.transpose(-1, -2))
                 .transpose(-1, -2).reshape(-1, self.num_rc, d))
        new_x = self.attention1(new_x, new_x, new_x)[0]
        new_x = rearrange(new_x, '(b r) c d -> (b c) r d', r=self.num_rc)
        new_x = self.attention2(new_x, new_x, new_x)[0] + new_x

        new_x = rearrange(new_x, '(b n c) r d -> b (r c) n d', b=b, n=n)
        return new_x[:, :c, ...]

    def full_attn(self, x):
        b, c, n, d = x.shape
        new_x = rearrange(x, 'b c n d -> (b n) c d')
        new_x = self.attention1(new_x, new_x, new_x)[0]
        new_x = rearrange(new_x, '(b n) c d -> b c n d', b=b, n=n)
        return new_x[:, :c, :]
