import os

import numpy as np
import torch
import torch.nn as nn

from ts_benchmark.baselines.olinear.layers.RevIN import RevIN
from ts_benchmark.baselines.olinear.layers.Transformer_EncDec import Encoder_ori, LinearEncoder


class OLinear(nn.Module):
    def __init__(self, configs):
        super(OLinear, self).__init__()
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.seq_len = configs.seq_len
        self.d_model = configs.d_model
        self.d_ff = configs.d_ff
        self.Q_chan_indep = getattr(configs, "Q_chan_indep", False)

        q_mat_dir = configs.Q_MAT_file if self.Q_chan_indep else configs.q_mat_file
        q_mat_dir = self._resolve_path(configs, q_mat_dir)
        self.register_buffer("Q_mat", torch.from_numpy(np.load(q_mat_dir)).to(torch.float32))

        q_out_mat_dir = configs.Q_OUT_MAT_file if self.Q_chan_indep else configs.q_out_mat_file
        q_out_mat_dir = self._resolve_path(configs, q_out_mat_dir)
        self.register_buffer(
            "Q_out_mat", torch.from_numpy(np.load(q_out_mat_dir)).to(torch.float32)
        )

        self.embed_size = configs.embed_size
        self.embeddings = nn.Parameter(torch.randn(1, self.embed_size))

        self.fc = nn.Sequential(
            nn.Linear(self.pred_len * self.embed_size, self.d_ff),
            nn.GELU(),
            nn.Linear(self.d_ff, self.pred_len),
        )

        self.revin_layer = RevIN(self.enc_in, affine=True)
        self.dropout = nn.Dropout(configs.dropout)

        self.encoder = Encoder_ori(
            [
                LinearEncoder(
                    d_model=configs.d_model,
                    d_ff=configs.d_ff,
                    CovMat=None,
                    dropout=configs.dropout,
                    activation=configs.activation,
                    token_num=self.enc_in,
                )
                for _ in range(configs.e_layers)
            ],
            norm_layer=nn.LayerNorm(configs.d_model),
            one_output=True,
            CKA_flag=getattr(configs, "CKA_flag", False),
        )
        self.ortho_trans = nn.Sequential(
            nn.Linear(self.seq_len * self.embed_size, self.d_model),
            self.encoder,
            nn.Linear(self.d_model, self.pred_len * self.embed_size),
        )

        self.delta1 = nn.Parameter(torch.zeros(1, self.enc_in, 1, self.seq_len))
        self.delta2 = nn.Parameter(torch.zeros(1, self.enc_in, 1, self.pred_len))

    def _resolve_path(self, configs, path: str) -> str:
        if path is None:
            raise ValueError("Q matrix file path is None")
        if os.path.isfile(path):
            return path
        root = getattr(configs, "root_path", None)
        if root is None:
            raise ValueError(f"Q matrix file not found: {path}")
        new_path = os.path.join(root, path)
        if not os.path.isfile(new_path):
            raise ValueError(f"Q matrix file not found: {path} (resolved: {new_path})")
        return new_path

    def tokenEmb(self, x, embeddings):
        if self.embed_size <= 1:
            return x.transpose(-1, -2).unsqueeze(-1)
        x = x.transpose(-1, -2)
        x = x.unsqueeze(-1)
        return x * embeddings

    def Fre_Trans(self, x):
        B, N, T, D = x.shape
        x = x.transpose(-1, -2)

        if self.Q_chan_indep:
            x_trans = torch.einsum(
                "bndt,ntv->bndv", x, self.Q_mat.transpose(-1, -2)
            )
        else:
            x_trans = (
                torch.einsum("bndt,tv->bndv", x, self.Q_mat.transpose(-1, -2))
                + self.delta1
            )

        x_trans = self.ortho_trans(x_trans.flatten(-2)).reshape(B, N, D, self.pred_len)

        if self.Q_chan_indep:
            x = torch.einsum("bndt,ntv->bndv", x_trans, self.Q_out_mat)
        else:
            x = (
                torch.einsum("bndt,tv->bndv", x_trans, self.Q_out_mat) + self.delta2
            )

        x = x.transpose(-1, -2)
        return x

    def forward(self, x, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        x = self.revin_layer(x, mode="norm")
        x_ori = x

        x = self.tokenEmb(x_ori, self.embeddings)
        x = self.Fre_Trans(x)

        out = self.fc(x.flatten(-2)).transpose(-1, -2)
        out = self.dropout(out)
        out = self.revin_layer(out, mode="denorm")
        return out
