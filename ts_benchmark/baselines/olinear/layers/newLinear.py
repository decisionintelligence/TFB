import torch
import torch.nn as nn
import torch.nn.functional as F


class newLinear(nn.Module):

    def __init__(self, input_dim, output_dim, bias=False):
        super(newLinear, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bias = bias

        self.weight_mat = nn.Parameter(torch.randn(self.output_dim, self.input_dim))

        if self.bias:
            self.bias_weight = nn.Parameter(torch.zeros(1, self.output_dim))

    def forward(self, x):
        x_shape = x.shape
        assert x_shape[-1] == self.input_dim

        x_2d = x.reshape(-1, self.input_dim)

        # output_dim, input_dim
        weight_mat = F.normalize(F.softplus(self.weight_mat), p=1, dim=-1)
        # output_dim, -1
        output = weight_mat @ (x_2d.transpose(-1, -2))
        # -1, output_dim
        output = output.transpose(-1, -2)

        if self.bias:
            output = output + self.bias_weight

        new_shape = x_shape[:-1] + (self.output_dim,)

        return output.reshape(new_shape).contiguous()
