import torch
import torch.nn as nn
import math
from torch import Tensor
import torch.nn.functional as F
from typing import Optional


class Leddam(nn.Module):
    def __init__(self, configs,
                 enc_in,
                 seq_len,
                 d_model,
                 dropout,
                 pe_type,
                 kernel_size,
                 n_layers=3):

        super(Leddam, self).__init__()
        self.n_layers = n_layers
        self.LD = LD(kernel_size=kernel_size)
        self.channel_attn_blocks = nn.ModuleList([
            channel_attn_block(enc_in, d_model, dropout)
            for _ in range(self.n_layers)
        ])
        self.auto_attn_blocks = nn.ModuleList([
            auto_attn_block(enc_in, d_model, dropout)
            for _ in range(self.n_layers)
        ])
        self.position_embedder = DataEmbedding(pe_type=pe_type, seq_len=seq_len,
                                               d_model=d_model, c_in=enc_in)

    def forward(self, inp):
        inp = self.position_embedder(inp.permute(0, 2, 1)).permute(0, 2, 1)
        main = self.LD(inp)
        residual = inp - main

        res_1 = residual
        res_2 = residual
        for i in range(self.n_layers):
            res_1 = self.auto_attn_blocks[i](res_1)
        for i in range(self.n_layers):
            res_2 = self.channel_attn_blocks[i](res_2)
        res = res_1 + res_2

        return res, main


class channel_attn_block(nn.Module):
    def __init__(self, enc_in, d_model, dropout):
        super(channel_attn_block, self).__init__()
        self.channel_att_norm = nn.BatchNorm1d(enc_in)
        self.fft_norm = nn.LayerNorm(d_model)
        self.channel_attn = MultiheadAttention(d_model=d_model, n_heads=1, proj_dropout=dropout)
        self.fft_layer = nn.Sequential(
            nn.Linear(d_model, int(d_model * 2)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(d_model * 2), d_model),
        )

    def forward(self, residual):
        res_2 = self.channel_att_norm(self.channel_attn(residual.permute(0, 2, 1)) + residual.permute(0, 2, 1))
        res_2 = self.fft_norm(self.fft_layer(res_2) + res_2)
        return res_2.permute(0, 2, 1)


class auto_attn_block(nn.Module):
    def __init__(self, enc_in, d_model, dropout):
        super(auto_attn_block, self).__init__()
        self.auto_attn_norm = nn.BatchNorm1d(enc_in)
        self.fft_norm = nn.LayerNorm(d_model)
        self.auto_attn = Auto_Attention(P=64, d_model=d_model, proj_dropout=dropout)
        self.fft_layer = nn.Sequential(
            nn.Linear(d_model, int(d_model * 2)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(d_model * 2), d_model),
        )

    def forward(self, residual):
        res_1 = self.auto_attn_norm((self.auto_attn(residual) + residual).permute(0, 2, 1))
        res_1 = self.fft_norm(self.fft_layer(res_1) + res_1)
        return res_1.permute(0, 2, 1)


class LD(nn.Module):
    def __init__(self, kernel_size=25):
        super(LD, self).__init__()
        # Define a shared convolution layers for all channels
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, stride=1, padding=int(kernel_size // 2),
                              padding_mode='replicate', bias=True)
        # Define the parameters for Gaussian initialization
        kernel_size_half = kernel_size // 2
        sigma = 1.0  # 1 for variance
        weights = torch.zeros(1, 1, kernel_size)
        for i in range(kernel_size):
            weights[0, 0, i] = math.exp(-((i - kernel_size_half) / (2 * sigma)) ** 2)

        # Set the weights of the convolution layer
        self.conv.weight.data = F.softmax(weights, dim=-1)
        self.conv.bias.data.fill_(0.0)

    def forward(self, inp):
        # Permute the input tensor to match the expected shape for 1D convolution (B, N, T)
        inp = inp.permute(0, 2, 1)
        # Split the input tensor into separate channels
        input_channels = torch.split(inp, 1, dim=1)

        # Apply convolution to each channel
        conv_outputs = [self.conv(input_channel) for input_channel in input_channels]

        # Concatenate the channel outputs
        out = torch.cat(conv_outputs, dim=1)
        out = out.permute(0, 2, 1)
        return out


class Auto_Attention(nn.Module):
    def __init__(self, P, d_model, proj_dropout=0.2):
        """
        Initialize the Auto-Attention module.

        Args:
            d_model (int): The input and output dimension for queries, keys, and values.
        """
        super(Auto_Attention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.out_projector = nn.Sequential(nn.Linear(d_model, d_model), nn.Dropout(proj_dropout))
        self.P = P
        self.scale = nn.Parameter(torch.tensor(d_model ** -0.5), requires_grad=False)

    def auto_attention(self, inp):
        """
        Perform auto-attention mechanism on the input.

        Args:
            inp (torch.Tensor): Input data of shape [B, N, T], where B is the batch size,
                               N is the number of features, and T is the sequence length.
        Returns:
            output (torch.Tensor): Output after auto-attention.
        """
        # Separate query and key
        query = self.W_Q(inp[:, :, 0, :].unsqueeze(-2))  # Query
        keys = self.W_K(inp)  # Keys
        values = self.W_V(inp)  # Values

        # Calculate dot product
        attn_scores = torch.matmul(query, keys.transpose(-2, -1)) * self.scale

        # Normalize attention scores
        attn_scores = F.softmax(attn_scores, dim=-1)

        # Weighted sum
        output = torch.matmul(attn_scores, values)

        return output

    def forward(self, inp):
        """
        Forward pass of the Auto-Attention module.

        Args:
            P (int): The period for autoregressive behavior.
            inp (torch.Tensor): Input data of shape [B, T, N], where B is the batch size,
                               T is the sequence length, and N is the number of features.

        Returns:
            output (torch.Tensor): Output after autoregressive self-attention.
        """
        # Permute the input for further processing
        inp = inp.permute(0, 2, 1)  # [B, T, N] -> [B, N, T]

        T = inp.size(-1)

        cat_sequences = [inp]
        index = int(T / self.P) - 1 if T % self.P == 0 else int(T / self.P)

        for i in range(index):
            end = (i + 1) * self.P

            # Concatenate sequences to support autoregressive behavior
            cat_sequence = torch.cat([inp[:, :, end:], inp[:, :, 0:end]], dim=-1)
            cat_sequences.append(cat_sequence)

        # Stack the concatenated sequences
        output = torch.stack(cat_sequences, dim=-1)

        # Permute the output for attention calculation
        output = output.permute(0, 1, 3, 2)

        # Apply autoregressive self-attention
        output = self.auto_attention(output).squeeze(-2)
        output = self.out_projector(output).permute(0, 2, 1)

        return output


class MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads=1, attn_dropout=0., proj_dropout=0.2):
        """Multi Head Attention Layer
        Input shape:
            Q:       [batch_size (bs) x max_q_len x d_model]
            K, V:    [batch_size (bs) x q_len x d_model]
            mask:    [q_len x q_len]
        """
        super().__init__()
        d_k = d_model // n_heads
        d_v = d_model // n_heads

        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v

        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)

        # Scaled Dot-Product Attention (multiple heads)
        self.sdp_attn = ScaledDotProductAttention(d_model, n_heads, attn_dropout=attn_dropout)
        # Poject output
        self.to_out = nn.Sequential(nn.Linear(n_heads * d_v, d_model), nn.Dropout(proj_dropout))

    def forward(self, Q: Tensor, K: Optional[Tensor] = None, V: Optional[Tensor] = None, prev: Optional[Tensor] = None,
                ):

        bs = Q.size(0)
        if K is None:
            K = Q
        if V is None:
            V = Q
        # Linear (+ split in multiple heads)
        # q_s    : [bs x n_heads x max_q_len x d_k]
        q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1, 2)
        # k_s    : [bs x n_heads x d_k x q_len] - transpose(1,2) + transpose(2,3)
        k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0, 2, 3, 1)
        v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1, 2)  # v_s    : [bs x n_heads x q_len x d_v]

        # Apply Scaled Dot-Product Attention (multiple heads)
        if prev is not None:
            output, prev = self.sdp_attn(q_s, k_s, v_s)
        else:
            output = self.sdp_attn(q_s, k_s, v_s)
        # output: [bs x n_heads x q_len x d_v]

        # back to the original inputs dimensions
        output = output.transpose(1, 2).contiguous().view(bs, -1,
                                                          self.n_heads * self.d_v)  # output: [bs x q_len x n_heads * d_v]
        output = self.to_out(output)
        if prev is not None:
            return output, prev
        else:
            return output


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model, n_heads, attn_dropout=0.):
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)
        head_dim = d_model // n_heads
        self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5), requires_grad=False)

    def forward(self, q: Tensor, k: Tensor, v: Tensor, prev: Optional[Tensor] = None):
        """
        Input shape:
            q               : [bs x n_heads x max_q_len x d_k]
            k               : [bs x n_heads x d_k x seq_len]
            v               : [bs x n_heads x seq_len x d_v]
            prev            : [bs x n_heads x q_len x seq_len]
        Output shape:
            output:  [bs x n_heads x q_len x d_v]
            attn   : [bs x n_heads x q_len x seq_len]
            scores : [bs x n_heads x q_len x seq_len]
        """
        # Scaled MatMul (q, k) - similarity scores for all pairs of positions in an input sequence
        attn_scores = torch.matmul(q, k) * (self.scale)  # Scale

        # Add pre-softmax attention scores from the previous layer (optional)
        if prev is not None:
            attn_scores = attn_scores + prev
        # Normalize the attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)  # attn_weights: [bs x n_heads x max_q_len x q_len]
        attn_weights = self.attn_dropout(attn_weights)

        # Compute the new values given the attention weights
        output = torch.matmul(attn_weights, v)  # output: [bs x n_heads x max_q_len x d_v]
        if prev is not None:
            return output, attn_scores
        else:
            return output


class DataEmbedding(nn.Module):
    def __init__(self, pe_type, seq_len, d_model, c_in, dropout=0.):
        super(DataEmbedding, self).__init__()

        self.value_embedding = nn.Linear(seq_len, d_model)
        self.position_embedding = positional_encoding(pe=pe_type, learn_pe=True, q_len=c_in, d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.value_embedding(x) + self.position_embedding
        return self.dropout(x)


# pos_encoding

def SinCosPosEncoding(q_len, d_model, normalize=True):
    pe = torch.zeros(q_len, d_model)
    position = torch.arange(0, q_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    if normalize:
        pe = pe - pe.mean()
        pe = pe / (pe.std() * 10)

    return pe


def Coord2dPosEncoding(q_len, d_model, exponential=False, normalize=True, eps=1e-3):
    x = .5 if exponential else 1
    i = 0
    for i in range(100):
        cpe = 2 * (torch.linspace(0, 1, q_len).reshape(-1, 1) ** x) * (
                torch.linspace(0, 1, d_model).reshape(1, -1) ** x) - 1
        if abs(cpe.mean()) <= eps:
            break
        elif cpe.mean() > eps:
            x += .001
        else:
            x -= .001
        i += 1
    if normalize:
        cpe = cpe - cpe.mean()
        cpe = cpe / (cpe.std() * 10)

    return cpe


def Coord1dPosEncoding(q_len, exponential=False, normalize=True):
    cpe = (2 * (torch.linspace(0, 1, q_len).reshape(-1, 1) ** (.5 if exponential else 1)) - 1)
    if normalize:
        cpe = cpe - cpe.mean()
        cpe = cpe / (cpe.std() * 10)

    return cpe


def positional_encoding(pe, learn_pe, q_len, d_model):
    # Positional encoding
    if pe == None or pe == 'no':
        W_pos = torch.empty((q_len, d_model))  # pe = None and learn_pe = False can be used to measure impact of pe
        nn.init.uniform_(W_pos, -0.02, 0.02)
        learn_pe = False
    elif pe == 'zero':
        W_pos = torch.empty((q_len, 1))
        nn.init.uniform_(W_pos, -0.02, 0.02)
    elif pe == 'zeros':
        W_pos = torch.empty((q_len, d_model))
        nn.init.uniform_(W_pos, -0.02, 0.02)
    elif pe == 'normal' or pe == 'gauss':
        W_pos = torch.zeros((q_len, 1))
        torch.nn.init.normal_(W_pos, mean=0.0, std=0.1)
    elif pe == 'uniform':
        W_pos = torch.zeros((q_len, 1))
        nn.init.uniform_(W_pos, a=0.0, b=0.1)
    elif pe == 'lin1d':
        W_pos = Coord1dPosEncoding(q_len, exponential=False, normalize=True)
    elif pe == 'exp1d':
        W_pos = Coord1dPosEncoding(q_len, exponential=True, normalize=True)
    elif pe == 'lin2d':
        W_pos = Coord2dPosEncoding(q_len, d_model, exponential=False, normalize=True)
    elif pe == 'exp2d':
        W_pos = Coord2dPosEncoding(q_len, d_model, exponential=True, normalize=True)
    elif pe == 'sincos':
        W_pos = SinCosPosEncoding(q_len, d_model, normalize=True)
    else:
        raise ValueError(f"{pe} is not a valid pe (positional encoder. Available types: 'gauss'=='normal', \
        'zeros', 'zero', uniform', 'lin1d', 'exp1d', 'lin2d', 'exp2d', 'sincos', None.)")
    return nn.Parameter(W_pos, requires_grad=learn_pe)
