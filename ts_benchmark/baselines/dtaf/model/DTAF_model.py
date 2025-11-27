import torch
import torch.nn as nn
from ..layer.Embed import PatchEmbedding
from ..layer.Linear_extractor import Linear_extractor
from ..layer.kan import KAN, KANLinear
from ts_benchmark.utils.get_device import get_device


class Expert(nn.Module):
    def __init__(self, input_dim, div):
        super(Expert, self).__init__()
        self.network = KAN(layers_hidden=[input_dim, input_dim // div, input_dim])

    def forward(self, x):
        return self.network(x)


class MOE(nn.Module):
    def __init__(self, expert_num, input_dim, div):
        super(MOE, self).__init__()
        self.experts = nn.ModuleList()
        self.router = KANLinear(input_dim, expert_num)
        for i in range(expert_num):
            self.experts.append(Expert(input_dim=input_dim, div=div))

    def forward(self, x):
        router = self.router(x).softmax(-1)
        experts_out = torch.stack([expert(x) for expert in self.experts], dim=-2)
        return torch.einsum("bpn,bpnd->bpd", router, experts_out)


class TFS(nn.Module):
    def __init__(self, input_dim, configs, patch_num):
        super(TFS, self).__init__()
        self.configs = configs
        self.MLP = nn.Linear(input_dim, input_dim)
        self.extractor_his = Linear_extractor(configs)
        self.weight_linear = nn.Linear(input_dim, patch_num)
        self.dropout = nn.Dropout(configs.dropout)
        self.extractor_cur = Linear_extractor(configs)
        self.gate = nn.Linear(input_dim, 1)
        if self.configs.aggregated_norm == 1:
            self.norm = nn.LayerNorm(input_dim)
        self.device = get_device()

        if configs.expert_num > 0:
            self.moe = MOE(
                expert_num=configs.expert_num, input_dim=input_dim, div=configs.kan_div
            )

    def forward(self, x):
        origin = x
        if self.configs.expert_num > 0:
            x = x - self.moe(x)
        H = self.extractor_his(x)
        weight_current = self.gate(self.extractor_cur(origin)).repeat(
            1, 1, origin.shape[-1]
        )

        weight = self.weight_linear(H).softmax(dim=-1)

        adj = torch.tril(weight, diagonal=0)
        aggregated = torch.matmul(adj, x)

        H_history = self.dropout(self.MLP(aggregated))
        H_current = self.dropout(weight_current) * x

        if self.configs.aggregated_norm == 1:
            out = self.norm(H_history + H_current)
        else:
            out = H_history + H_current
        return out, x


class Attention(nn.Module):
    def __init__(self, extra_d, heads, dropout=0.1):
        super(Attention, self).__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=extra_d, num_heads=heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(extra_d)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_output, _ = self.attention(x, x, x, attn_mask=mask)
        x = x + self.dropout(attn_output)
        x = self.norm(x)
        return x


class Predict(nn.Module):
    def __init__(self, nf, target_window, dropout=0):
        super(Predict, self).__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class DTAF(nn.Module):
    def __init__(self, configs):
        super(DTAF, self).__init__()
        self.device = get_device()
        self.config = configs
        self.patch_num = int(
            (self.config.seq_len - self.config.patch_len) / self.config.stride + 2
        )

        self.TFSs = nn.ModuleList(
            [
                TFS(self.config.d_model, self.config, self.patch_num)
                for i in range(self.config.e_layers)
            ]
        )
        self.predictor = Predict(
            2 * self.config.d_model * self.patch_num,
            self.config.pred_len,
            self.config.dropout,
        )
        self.patch_embedding = PatchEmbedding(
            self.config.d_model,
            self.config.patch_len,
            self.config.stride,
            self.config.stride,
            self.config.dropout,
        )
        self.temporal_attention = Attention(
            self.config.d_model, self.config.heads, self.config.dropout
        )
        self.frequency_attention = Attention(
            self.config.d_model, self.config.heads, self.config.dropout
        )
        self.drop = nn.Dropout(self.config.dropout)
        self.norm = nn.LayerNorm(self.config.d_model)

    def _get_mean_std(self, x):
        means = x.mean(1, keepdim=True)
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x = x / stdev
        return x, means, stdev

    def forward(self, x_enc):
        B, L, D = x_enc.size()

        # Instance Norm
        x_enc, means, stdev = self._get_mean_std(x_enc)

        # Patch & Embedding
        enc_out, _ = self.patch_embedding(x_enc.transpose(1, 2))

        # TFS
        enc_out_TFS = enc_out
        for i in range(self.config.e_layers):
            agg, stables = self.TFSs[i](enc_out_TFS)
            enc_out_TFS = self.norm(self.drop(agg) + enc_out_TFS)

        # FWM
        enc_out = enc_out_TFS
        H_t = enc_out
        wave = torch.zeros(
            enc_out.shape[0], enc_out.shape[1], enc_out.shape[2] // 2 + 1
        ).to(self.device)
        freq = torch.fft.rfft(enc_out)
        wave[:, 1:, :] = torch.exp(
            torch.abs(freq[:, 1:, :]) - torch.abs(freq[:, :-1, :])
        )

        topk_values, topk_indices = torch.topk(wave, self.config.k, dim=-1)
        mask = torch.zeros_like(freq, dtype=torch.bool)  # 创建一个与 freq 形状相同的布尔掩码
        mask.scatter_(dim=-1, index=topk_indices, value=True)

        filtered_freq = torch.where(mask, freq, torch.zeros_like(freq))
        H_f = torch.fft.irfft(filtered_freq)
        H_f[:, 0, :] = enc_out[:, 0, :]

        # dual-attention
        H_f = self.frequency_attention(H_f)
        H_t = self.frequency_attention(H_t)
        enc_out = torch.cat([H_t, H_f], dim=-2)

        enc_out = torch.reshape(enc_out, (-1, D, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)
        out = self.predictor(enc_out)
        out = out.permute(0, 2, 1)
        out = out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.config.pred_len, 1))
        out = out + (means[:, 0, :].unsqueeze(1).repeat(1, self.config.pred_len, 1))

        return out, stables
