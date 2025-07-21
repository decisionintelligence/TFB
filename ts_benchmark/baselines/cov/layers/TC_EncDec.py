import torch
import torch.nn.functional as F
from torch import nn

from ts_benchmark.baselines.cov.layers.Embed import PatchEmbedding, CompressAndProject
from ts_benchmark.baselines.cov.layers.SelfAttention_Family import FullAttention, AttentionLayer
from ts_benchmark.baselines.cov.layers.Transformer_EncDec import Encoder, EncoderLayer


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class TemporalCausalityEncoder(nn.Module):
    def __init__(self, enc_in, seq_len, pred_len, series_dim,
                 patch_len, stride, d_model, d_ff, n_heads, e_layers,
                 dropout, factor, activation, criterion
                 ):
        super(TemporalCausalityEncoder, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.series_dim = series_dim
        self.criterion = criterion
        padding = stride

        # self.patch_embedding = PatchEmbedding(
        #     d_model, patch_len, stride, padding, dropout
        # )

        self.exog_patch_embedding = PatchEmbedding(
            d_model, patch_len, stride, padding, dropout
        )
        self.x_patch_embedding = PatchEmbedding(
            d_model, patch_len, stride, padding, dropout
        )

        self.encoder_exg = self._build_encoder(
            d_model=d_model, d_ff=d_ff, n_heads=n_heads, dropout=dropout, activation=activation, output_attention=True,
            factor=factor, e_layers=e_layers
        )
        self.encoder_x = self._build_encoder(
            d_model=d_model, d_ff=d_ff, n_heads=n_heads, dropout=dropout, activation=activation, output_attention=False,
            factor=factor, e_layers=e_layers
        )

        self.x_projector = CompressAndProject(self.series_dim, self.seq_len, d_model)
        self.exog_projector = CompressAndProject(enc_in - series_dim, self.seq_len, d_model)

        # Prediction Head
        self.head_nf = d_model * int((seq_len - patch_len) / stride + 2)
        # self.head = FlattenHead(
        #     enc_in,
        #     self.head_nf,
        #     pred_len,
        #     head_dropout=dropout,
        # )

        self.exog_head = FlattenHead(
            enc_in,
            self.head_nf,
            pred_len,
            head_dropout=dropout,
        )
        self.x_head = FlattenHead(
            enc_in,
            self.head_nf,
            pred_len,
            head_dropout=dropout,
        )

    def forward(self, x, exog_future, use_exog=True):
        exog_history = x[:, :, self.series_dim:]
        x_history = x[:, :, :self.series_dim]

        _, _, EXOG_D = exog_history.shape
        B, L, X_D = x_history.shape

        exog_history_means = exog_history.mean(1, keepdim=True).detach()
        x_history_means = x_history.mean(1, keepdim=True).detach()

        exog_history_stdev = torch.sqrt(torch.var(exog_history, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_history_stdev = torch.sqrt(torch.var(x_history, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()

        exog_history = self.sample_norm(exog_history, exog_history_means, exog_history_stdev)
        x_history = self.sample_norm(x_history, x_history_means, x_history_stdev)

        exog_history = exog_history.permute(0, 2, 1)
        x_history = x_history.permute(0, 2, 1)

        # patch_exog, exog_vars = self.patch_embedding(exog_history)
        # patch_x, x_vars = self.patch_embedding(x_history)

        patch_exog, exog_vars = self.exog_patch_embedding(exog_history)
        patch_x, x_vars = self.x_patch_embedding(x_history)

        enc_exog_out, _ = self.encoder_exg(patch_exog)
        if use_exog:
            _, causality_attns = self.encoder_exg(patch_x)
        else:
            causality_attns = None

        exog_history = exog_history.permute(0, 2, 1)
        x_history = x_history.permute(0, 2, 1)

        exog_history_projection = self.exog_projector(exog_history)  # batch size, d_model TODO
        x_history_projection = self.x_projector(x_history)

        attn_alpha = F.sigmoid(torch.einsum('bd,bd->b', x_history_projection, exog_history_projection)).view(-1, 1, 1,
                                                                                                             1)
        # print("tc attn_alpha mean:", torch.mean(attn_alpha))
        enc_x_out, _ = self.encoder_x(patch_x, exog_attns=causality_attns, attn_alpha=attn_alpha.repeat(self.series_dim, 1, 1, 1))

        enc_exog_out = torch.reshape(
            enc_exog_out, (-1, exog_vars, enc_exog_out.shape[-2], enc_exog_out.shape[-1])
        ).permute(0, 1, 3, 2)
        enc_x_out = torch.reshape(
            enc_x_out, (-1, x_vars, enc_x_out.shape[-2], enc_x_out.shape[-1])
        ).permute(0, 1, 3, 2)

        # enc_out = torch.cat([enc_x_out, enc_exog_out], dim=1)
        # out = self.head(enc_out)
        # out = out.permute(0, 2, 1)
        #
        # exog_out = out[:, :, self.series_dim:]
        # x_out = out[:, :, :self.series_dim]

        exog_out = self.exog_head(enc_exog_out)
        x_out = self.x_head(enc_x_out)

        exog_out = exog_out.permute(0, 2, 1)
        x_out = x_out.permute(0, 2, 1)

        exog_out = self.sample_denorm(exog_out, exog_history_means, exog_history_stdev)
        x_out = self.sample_denorm(x_out, x_history_means, x_history_stdev)

        temporal_causality_loss = self.criterion(exog_out, exog_future) if use_exog else 0

        return x_out, temporal_causality_loss

    def _build_encoder(self, d_model, d_ff, n_heads, dropout, activation, output_attention, factor, e_layers):
        return Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            factor,
                            attention_dropout=dropout,
                            output_attention=output_attention,
                        ),
                        d_model,
                        n_heads,
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(e_layers)
            ],
            norm_layer=nn.LayerNorm(d_model),
        )

    def sample_norm(self, x, means, stdev):
        x = x - means
        x /= stdev
        return x

    def sample_denorm(self, x, means, stdev):
        seq_len = x.shape[1]
        x = x * (stdev[:, 0, :].unsqueeze(1).repeat(1, seq_len, 1))
        x = x + (means[:, 0, :].unsqueeze(1).repeat(1, seq_len, 1))
        return x
