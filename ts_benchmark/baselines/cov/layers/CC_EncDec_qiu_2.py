import torch
import torch.nn.functional as F
from torch import nn

from ts_benchmark.baselines.cov.layers.Embed import DataEmbedding_inverted, CompressAndProject
from ts_benchmark.baselines.cov.layers.SelfAttention_Family import FullAttention, AttentionLayer
from ts_benchmark.baselines.cov.layers.Transformer_EncDec import Encoder, EncoderLayer


class ExogProjector(nn.Module):
    def __init__(self, input_dim, target_dim, d_model, pred_len):
        super().__init__()
        self.feature_proj = nn.Linear(input_dim, target_dim)
        self.sequence_proj = nn.Linear(d_model, pred_len)

    def forward(self, x):  # x: [B, L, input_dim]
        x = self.feature_proj(x)  # [B, L, series_dim]
        x = x.transpose(1, 2)  # [B, series_dim, L]
        x = self.sequence_proj(x)  # [B, pred_len, L]
        x = x.transpose(1, 2)  # [B, L, pred_len]
        return x


class DataEmbedding(nn.Module):
    def __init__(self, seq_len, dim, d_model, dropout=0.1):
        super(DataEmbedding, self).__init__()
        self.value_embedding = nn.Linear(seq_len, d_model)
        self.global_embedding = CompressAndProject(dim, seq_len, d_model // 4)
        self.projector = nn.Linear(d_model + d_model // 4, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        B, T, D = x.shape
        global_embedding = self.global_embedding(x).unsqueeze(1).expand(-1, D, -1)  # [Batch Variate d_model]
        x = x.permute(0, 2, 1)  # [Batch Variate Time]
        x = self.value_embedding(x)  # [Batch Variate d_model]
        x = torch.cat([x, global_embedding], dim=2)
        x = self.projector(x)  # [Batch Variate d_model]
        return self.dropout(x)


class CovCausalityEncoder(nn.Module):
    def __init__(self, enc_in, seq_len, pred_len, series_dim,
                 d_model, d_ff, n_heads, e_layers,
                 dropout, factor, activation, criterion
                 ):
        super(CovCausalityEncoder, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.series_dim = series_dim
        self.enc_in = enc_in
        self.exog_dim = enc_in - series_dim

        self.criterion = criterion
        # Embedding
        # self.history_enc_embedding = DataEmbedding_inverted(seq_len, d_model, dropout)
        # self.future_enc_embedding = DataEmbedding_inverted(pred_len, d_model, dropout)

        self.history_enc_embedding = DataEmbedding(seq_len, self.exog_dim, d_model, dropout)
        self.future_enc_embedding = DataEmbedding(pred_len, self.exog_dim, d_model, dropout)

        # Encoder
        self.future_encoder = self._build_encoder(
            d_model=d_model, d_ff=d_ff, n_heads=n_heads, dropout=dropout, activation=activation,
            output_attention=False, factor=factor, e_layers=e_layers
        )
        self.history_encoder = self._build_encoder(
            d_model=d_model, d_ff=d_ff, n_heads=n_heads, dropout=dropout, activation=activation,
            output_attention=True, factor=factor, e_layers=e_layers
        )

        self.history_exog_projector = CompressAndProject(self.exog_dim, self.seq_len, d_model)
        self.future_exog_projector = CompressAndProject(self.exog_dim, self.pred_len, d_model)
        # self.attn_alpha_raw = nn.Parameter(torch.tensor(-1.0))

        self.history_projection = ExogProjector(enc_in - series_dim, 1, d_model, seq_len)
        self.future_projection = ExogProjector(enc_in - series_dim, 1, d_model, pred_len)

    def forward(self, x, exog_future, use_exog=True):
        # Normalization from Non-stationary Transformer

        # TODO 尝试删除归一化反归一化
        exog_history = x[:, :, self.series_dim:]
        x_history = x[:, :, :self.series_dim]

        exog_history_means = exog_history.mean(1, keepdim=True).detach()
        exog_history_stdev = torch.sqrt(torch.var(exog_history, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()

        exog_history = self.sample_norm(exog_history, exog_history_means, exog_history_stdev)
        exog_future = self.sample_norm(exog_future, exog_history_means, exog_history_stdev)

        history_outs = []
        future_outs = []

        for i in range(self.series_dim):
            x_history_i = x_history[:, :, i].unsqueeze(-1)
            x_history_i_means = x_history_i.mean(1, keepdim=True).detach()
            x_history_i_stdev = torch.sqrt(torch.var(x_history_i, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()

            exog_history_i = self.sample_denorm(exog_history, x_history_i_means.repeat(1, 1, self.exog_dim),
                                                x_history_i_stdev.repeat(1, 1, self.exog_dim))
            exog_future_i = self.sample_denorm(exog_future, x_history_i_means.repeat(1, 1, self.exog_dim),
                                               x_history_i_stdev.repeat(1, 1, self.exog_dim))

            enc_exog_history_i = self.history_enc_embedding(exog_history_i)
            enc_exog_future_i = self.future_enc_embedding(exog_future_i)

            enc_history_out_i, _ = self.history_encoder(enc_exog_history_i, attn_mask=None)
            if use_exog:
                _, causality_attns_i = self.history_encoder(enc_exog_future_i)
            else:
                causality_attns_i = None

            history_exog_projection_i = self.history_exog_projector(exog_history_i)
            future_exog_projection_i = self.future_exog_projector(exog_future_i)
            attn_alpha_i = (F.sigmoid(torch.einsum('bd,bd->b', history_exog_projection_i, future_exog_projection_i))
                          .view(-1, 1, 1, 1))

            enc_future_out_i, _ = self.future_encoder(enc_exog_future_i, exog_attns=causality_attns_i,
                                                    attn_alpha=attn_alpha_i, attn_mask=None)
            enc_history_out_i = enc_history_out_i.permute(0, 2, 1)
            enc_future_out_i = enc_future_out_i.permute(0, 2, 1)

            history_out_i = self.history_projection(enc_history_out_i)
            future_out_i = self.future_projection(enc_future_out_i)

            history_outs.append(history_out_i)
            future_outs.append(future_out_i)

        history_out = torch.cat(history_outs, dim=-1)
        future_out = torch.cat(future_outs, dim=-1)

        cov_causality_loss = self.criterion(history_out, x_history) if use_exog else 0
        # TODO qiu 反归一化方式
        return future_out, cov_causality_loss

    def _build_encoder(self, d_model, d_ff, n_heads, dropout, activation, output_attention, factor, e_layers):
        return Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, factor, attention_dropout=dropout,
                                      output_attention=output_attention), d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

    def sample_norm(self, x, means, stdev):  # TODO 改为batch级别
        x = x - means
        x /= stdev
        return x

    def sample_denorm(self, x, means, stdev):
        seq_len = x.shape[1]
        x = x * (stdev[:, 0, :].unsqueeze(1).repeat(1, seq_len, 1))
        x = x + (means[:, 0, :].unsqueeze(1).repeat(1, seq_len, 1))
        return x
