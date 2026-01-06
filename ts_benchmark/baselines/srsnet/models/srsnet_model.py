import math

import torch
from torch import nn

from ts_benchmark.baselines.srsnet.layers.SRS import SRS
from ts_benchmark.baselines.srsnet.layers.RevIN import RevIN


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0, mode='linear'):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        if mode == 'linear':
            self.head = nn.Linear(nf, target_window)
        else:
            self.head = nn.Sequential(nn.Linear(nf, nf // 2), nn.SiLU(), nn.Linear(nf // 2, target_window))
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.head(x)
        x = self.dropout(x)
        return x


class SRSNetModel(nn.Module):
    def __init__(self, config):
        """
        patch_len: int, patch len for patch_embedding
        stride: int, stride for patch_embedding
        """
        super(SRSNetModel, self).__init__()
        self.seq_len = config.seq_len
        self.pred_len = config.pred_len
        self.patch_len = config.patch_len
        self.stride = config.stride

        # selective representation space
        self.patch_embedding = SRS(
            config.d_model, self.patch_len, self.stride, self.seq_len, config.dropout, config.hidden_size, config.alpha, config.pos
        )

        # Prediction Head
        self.head_nf = config.d_model * (math.ceil((config.seq_len - self.patch_len) / self.stride) + 1)
        self.head = FlattenHead(
            config.enc_in,
            self.head_nf,
            config.pred_len,
            head_dropout=config.dropout,
            mode=config.head_mode
        )

        self.revin = RevIN(num_features=config.enc_in, affine=config.affine, subtract_last=config.subtract_last)

    def forward(self, x_enc):
        x_enc = self.revin(x_enc, 'norm')
        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)
        # u: [bs * nvars x patch_num x d_model]
        enc_out, n_vars = self.patch_embedding(x_enc)

        # z: [bs x nvars x patch_num x d_model]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1])
        )
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Decoder
        dec_out = self.head(enc_out)  # z: [bs x nvars x target_window]
        dec_out = dec_out.permute(0, 2, 1)

        # De-Normalization from Non-stationary Transformer
        dec_out = self.revin(dec_out, 'denorm')
        return dec_out
