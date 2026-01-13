import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from ts_benchmark.baselines.timefilter.layers.Embed import PositionalEmbedding
from ts_benchmark.baselines.timefilter.layers.StandardNorm import Normalize
from ts_benchmark.baselines.timefilter.layers.TimeFilter_layers import TimeFilter_Backbone


class PatchEmbed(nn.Module):
    def __init__(self, dim, patch_len, stride=None, pos=True):
        super().__init__()
        self.patch_len = patch_len
        self.stride = patch_len if stride is None else stride
        self.patch_proj = nn.Linear(self.patch_len, dim)
        self.pos = pos
        if self.pos:
            pos_emb_theta = 10000
            self.pe = PositionalEmbedding(dim, pos_emb_theta)
    
    def forward(self, x):
        # x: [B, N, T]
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # x: [B, N*L, P]
        x = self.patch_proj(x) # [B, N*L, D]
        if self.pos:
            x += self.pe(x)
        return x

class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.n_vars = configs.c_out
        self.dim = configs.d_model
        self.d_ff = configs.d_ff
        self.patch_len = configs.patch_len
        self.stride = self.patch_len
        self.num_patches = int((self.seq_len - self.patch_len) / self.stride + 1) # L

        # Filter
        self.alpha = 0.1 if configs.alpha is None else configs.alpha
        self.top_p = 0.5 if configs.top_p is None else configs.top_p

        # embed
        self.patch_embed = PatchEmbed(self.dim, self.patch_len, self.stride, configs.pos)

        # TimeFilter Backbone
        self.backbone = TimeFilter_Backbone(self.dim, self.n_vars, self.d_ff,
                                  configs.n_heads, configs.e_layers, self.top_p, configs.dropout, self.seq_len * self.n_vars // self.patch_len)
        
        # head
        self.head = nn.Linear(self.dim * self.num_patches, self.pred_len)

        # Without RevIN
        self.use_RevIN = False
        self.norm = Normalize(configs.enc_in, affine=self.use_RevIN)
    
    def forward(self, x, masks, is_training=False, target=None):
        # x: [B, T, C]
        B, T, C = x.shape
        # Normalization
        x = self.norm(x, 'norm')
        # x: [B, C, T]
        x = x.permute(0, 2, 1).reshape(-1, C*T) # [B, C*T]
        x = self.patch_embed(x) # [B, N, D]  N = [C*T / P]
        x, moe_loss = self.backbone(x, masks, self.alpha, is_training)

        # [B, C, T/P, D]
        x = self.head(x.reshape(-1, self.n_vars, self.num_patches, self.dim).flatten(start_dim=-2)) # [B, C, T]
        x = x.permute(0, 2, 1)
        # De-Normalization
        x = self.norm(x, 'denorm')

        return x, moe_loss
        