# code from https://github.com/ts-kim/RevIN, with minor modifications

import torch
import torch.nn as nn



class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        self.mask = None
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str, mask=None):
        # x [b,l,n]
        if mode == 'norm':
            self._get_statistics(x, mask)
            x = self._normalize(x, mask)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x, mask=None):
        self.mask = mask
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)
        else:
            if mask is None:
                self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
            else:
                assert isinstance(mask, torch.Tensor)
                # print(type(mask))
                x = x.masked_fill(mask, 0)  # in case other values are filled
                self.mean = (torch.sum(x, dim=1) / torch.sum(~mask, dim=1)).unsqueeze(1).detach()
                # self.mean could be nan or inf
                self.mean = torch.nan_to_num(self.mean, nan=0.0, posinf=0.0, neginf=0.0)

        if mask is None:
            self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()
        else:
            self.stdev = (torch.sqrt(torch.sum((x - self.mean) ** 2, dim=1) / torch.sum(~mask, dim=1) + self.eps)
                          .unsqueeze(1).detach())
            self.stdev = torch.nan_to_num(self.stdev, nan=0.0, posinf=None, neginf=None)

    def _normalize(self, x, mask=None):
        self.mask = mask
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean

        x = x / self.stdev

        # x should be zero, if the values are masked
        if mask is not None:
            # forward fill
            # x, mask2 = forward_fill(x, mask)
            # x = x.masked_fill(mask2, 0)

            # mean imputation
            x = x.masked_fill(mask, 0)

        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x
