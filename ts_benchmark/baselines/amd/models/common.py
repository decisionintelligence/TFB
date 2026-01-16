import torch
import torch.nn as nn
import torch.nn.functional as F

import math


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str, target_slice=None):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias[:]
            x = x / (self.affine_weight + self.eps * self.eps)[:]
        x = x * self.stdev[:, :, :]
        x = x + self.mean[:, :, :]
        return x


class MDM(nn.Module):
    def __init__(self, input_shape, k=3, c=2, layernorm=True):
        super(MDM, self).__init__()
        self.seq_len = input_shape[0]
        self.k = k
        if self.k > 0:
            self.k_list = [c ** i for i in range(k, 0, -1)]
            self.avg_pools = nn.ModuleList([nn.AvgPool1d(kernel_size=k, stride=k) for k in self.k_list])
            self.linears = nn.ModuleList(
                [
                    nn.Sequential(nn.Linear(self.seq_len // k, self.seq_len // k),
                                  nn.GELU(),
                                  nn.Linear(self.seq_len // k, self.seq_len * c // k),
                                  )
                    for k in self.k_list
                ]
            )
        self.layernorm = layernorm
        if self.layernorm:
            self.norm = nn.BatchNorm1d(input_shape[0] * input_shape[-1])

    def forward(self, x):
        if self.layernorm:
            x = self.norm(torch.flatten(x, 1, -1)).reshape(x.shape)
        if self.k == 0:
            return x
        # x [batch_size, feature_num, seq_len]
        sample_x = []
        for i, k in enumerate(self.k_list):
            sample_x.append(self.avg_pools[i](x))
        sample_x.append(x)
        n = len(sample_x)
        for i in range(n - 1):
            tmp = self.linears[i](sample_x[i])
            sample_x[i + 1] = torch.add(sample_x[i + 1], tmp, alpha=1.0)
        # [batch_size, feature_num, seq_len]
        return sample_x[n - 1]


class DDI(nn.Module):
    def __init__(self, input_shape, dropout=0.2, patch=12, alpha=0.0, layernorm=True):
        super(DDI, self).__init__()
        # input_shape[0] = seq_len    input_shape[1] = feature_num
        self.input_shape = input_shape
        if alpha > 0.0:
            self.ff_dim = 2 ** math.ceil(math.log2(self.input_shape[-1]))
            self.fc_block = nn.Sequential(
                nn.Linear(self.input_shape[-1], self.ff_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(self.ff_dim, self.input_shape[-1]),
                nn.GELU(),
                nn.Dropout(dropout),
            )

        self.n_history = 1
        self.alpha = alpha
        self.patch = patch

        self.layernorm = layernorm
        if self.layernorm:
            self.norm = nn.BatchNorm1d(self.input_shape[0] * self.input_shape[-1])
        self.norm1 = nn.BatchNorm1d(self.n_history * patch * self.input_shape[-1])
        if self.alpha > 0.0:
            self.norm2 = nn.BatchNorm1d(self.patch * self.input_shape[-1])

        self.agg = nn.Linear(self.n_history * self.patch, self.patch)
        self.dropout_t = nn.Dropout(dropout)

    def forward(self, x):
        # [batch_size, feature_num, seq_len]
        if self.layernorm:
            x = self.norm(torch.flatten(x, 1, -1)).reshape(x.shape)

        output = torch.zeros_like(x)
        output[:, :, :self.n_history * self.patch] = x[:, :, :self.n_history * self.patch].clone()
        for i in range(self.n_history * self.patch, self.input_shape[0], self.patch):
            # input [batch_size, feature_num, self.n_history * patch]
            input = output[:, :, i - self.n_history * self.patch: i]
            # input [batch_size, feature_num, self.n_history * patch]
            input = self.norm1(torch.flatten(input, 1, -1)).reshape(input.shape)
            # aggregation
            # [batch_size, feature_num, patch]
            input = F.gelu(self.agg(input))  # self.n_history * patch -> patch
            input = self.dropout_t(input)
            # input [batch_size, feature_num, patch]
            # input = torch.squeeze(input, dim=-1)
            tmp = input + x[:, :, i: i + self.patch]

            res = tmp

            # [batch_size, feature_num, patch]
            if self.alpha > 0.0:
                tmp = self.norm2(torch.flatten(tmp, 1, -1)).reshape(tmp.shape)
                tmp = torch.transpose(tmp, 1, 2)
                # [batch_size, patch, feature_num]
                tmp = self.fc_block(tmp)
                tmp = torch.transpose(tmp, 1, 2)
            output[:, :, i: i + self.patch] = res + self.alpha * tmp

        # [batch_size, feature_num, seq_len]
        return output

