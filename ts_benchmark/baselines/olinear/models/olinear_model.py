import os
import torch
import torch.nn as nn
import numpy as np
from ts_benchmark.baselines.olinear.layers import RevIN, Encoder_ori, LinearEncoder


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in  # channels
        self.seq_len = configs.seq_len
        self.hidden_size = self.d_model = configs.d_model  # hidden_size
        self.d_ff = configs.d_ff  # d_ff

        self.Q_chan_indep = configs.Q_chan_indep

        q_mat_dir = configs.Q_MAT_file if self.Q_chan_indep else configs.q_mat_file
        if not os.path.isfile(q_mat_dir):
            q_mat_dir = os.path.join(configs.root_path, q_mat_dir)
        assert os.path.isfile(q_mat_dir)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.Q_mat = torch.from_numpy(np.load(q_mat_dir)).to(torch.float32).to(device)

        assert (self.Q_mat.ndim == 3 if self.Q_chan_indep else self.Q_mat.ndim == 2)
        assert (self.Q_mat.shape[0] == self.enc_in if self.Q_chan_indep else self.Q_mat.shape[0] == self.seq_len)

        q_out_mat_dir = configs.Q_OUT_MAT_file if self.Q_chan_indep else configs.q_out_mat_file
        if not os.path.isfile(q_out_mat_dir):
            q_out_mat_dir = os.path.join(configs.root_path, q_out_mat_dir)
        assert os.path.isfile(q_out_mat_dir)
        self.Q_out_mat = torch.from_numpy(np.load(q_out_mat_dir)).to(torch.float32).to(device)

        assert (self.Q_out_mat.ndim == 3 if self.Q_chan_indep else self.Q_out_mat.ndim == 2)
        assert (self.Q_out_mat.shape[0] == self.enc_in if self.Q_chan_indep else
                self.Q_out_mat.shape[0] == self.pred_len)

        self.patch_len = configs.temp_patch_len
        self.stride = configs.temp_stride

        # self.channel_independence = configs.channel_independence
        self.embed_size = configs.embed_size  # embed_size
        self.embeddings = nn.Parameter(torch.randn(1, self.embed_size))

        self.fc = nn.Sequential(
            nn.Linear(self.pred_len * self.embed_size, self.d_ff),
            nn.GELU(),
            nn.Linear(self.d_ff, self.pred_len)
        )

        # for final input and output
        self.revin_layer = RevIN(self.enc_in, affine=True)
        self.dropout = nn.Dropout(configs.dropout)

        # #############  transformer related  #########
        self.encoder = Encoder_ori(
            [
                LinearEncoder(
                    d_model=configs.d_model, d_ff=configs.d_ff, CovMat=None,
                    dropout=configs.dropout, activation=configs.activation, token_num=self.enc_in,
                ) for _ in range(configs.e_layers)
            ],
            norm_layer=nn.LayerNorm(configs.d_model),
            one_output=True,
        )
        self.ortho_trans = nn.Sequential(
            nn.Linear(self.seq_len * self.embed_size, self.d_model),
            self.encoder,
            nn.Linear(self.d_model, self.pred_len * self.embed_size)
        )

        # learnable delta
        self.delta1 = nn.Parameter(torch.zeros(1, self.enc_in, 1, self.seq_len))
        self.delta2 = nn.Parameter(torch.zeros(1, self.enc_in, 1, self.pred_len))

    # dimension extension
    def tokenEmb(self, x, embeddings):
        if self.embed_size <= 1:
            return x.transpose(-1, -2).unsqueeze(-1)
        # x: [B, T, N] --> [B, N, T]
        x = x.transpose(-1, -2)
        x = x.unsqueeze(-1)
        # B*N*T*1 x 1*D = B*N*T*D
        return x * embeddings

    def Fre_Trans(self, x):
        # [B, N, T, D]
        B, N, T, D = x.shape
        assert T == self.seq_len
        # [B, N, D, T]
        x = x.transpose(-1, -2)

        # orthogonal transformation
        # [B, N, D, T]
        if self.Q_chan_indep:
            x_trans = torch.einsum('bndt,ntv->bndv', x, self.Q_mat.transpose(-1, -2))
        else:
            x_trans = torch.einsum('bndt,tv->bndv', x, self.Q_mat.transpose(-1, -2)) + self.delta1
            # added on 25/1/30
            # x_trans = F.gelu(x_trans)
            # [B, N, D, T]
        assert x_trans.shape[-1] == self.seq_len

        # ########## transformer ####
        x_trans = self.ortho_trans(x_trans.flatten(-2)).reshape(B, N, D, self.pred_len)

        # [B, N, D, tau]; orthogonal transformation
        if self.Q_chan_indep:
            x = torch.einsum('bndt,ntv->bndv', x_trans, self.Q_out_mat)
        else:
            x = torch.einsum('bndt,tv->bndv', x_trans, self.Q_out_mat) + self.delta2
            # added on 25/1/30
            # x = F.gelu(x)

        # [B, N, tau, D]
        x = x.transpose(-1, -2)
        return x

    def forward(self, x, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        # x: [Batch, Input length, Channel]
        B, T, N = x.shape

        # revin norm
        x = self.revin_layer(x, mode='norm')
        x_ori = x

        # ###########  frequency (high-level) part ##########
        # input fre fine-tuning
        # [B, T, N]
        # embedding x: [B, N, T, D]
        x = self.tokenEmb(x_ori, self.embeddings)
        # [B, N, tau, D]
        x = self.Fre_Trans(x)

        # linear
        # [B, N, tau*D] --> [B, N, dim] --> [B, N, tau] --> [B, tau, N]
        out = self.fc(x.flatten(-2)).transpose(-1, -2)

        # dropout
        out = self.dropout(out)

        # revin denorm
        out = self.revin_layer(out, mode='denorm')

        return out
