import torch
from typing import Tuple
import torch
import torch.nn as nn
import numpy as np
import sys


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        seq_len: int,
        base: float = 10000.0,
        learnable=False,
        init="exp",
        min_period=0.01,
        max_period=1000,
    ):
        super(RotaryEmbedding, self).__init__()
        if init == "linear":
            theta = get_linear_period(min_period, max_period, dim)
        elif init == "uniform":
            theta = torch.ones([dim // 2])
            periods = torch.nn.init.uniform_(theta, a=min_period, b=max_period)
            theta = 2 * np.pi / periods
        elif init == "exp":
            theta = get_exp_period(min_period, max_period, dim)
        elif init == "rope":
            theta = 1.0 / (
                base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)
            )
        else:
            print("invalid theta init")
            sys.exit(0)

        if learnable:
            self.freqs = nn.Parameter(theta)
        else:
            self.register_buffer("freqs", torch.tensor(theta))

        self.dim = dim
        self.seq_len = seq_len
        self.learnable = learnable

    def forward(self, xq: torch.Tensor, xk: torch.Tensor, xv: torch.Tensor):
        L = xq.shape[-2]
        t = torch.arange(L, device=xq.device)

        freqs = torch.outer(t, self.freqs).float()  # m * \theta
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)

        xq_ = xq.float().reshape(*xq.shape[:-1], -1, 2)
        xk_ = xk.float().reshape(*xk.shape[:-1], -1, 2)
        xv_ = xv.float().reshape(*xv.shape[:-1], -1, 2)

        xq_ = torch.view_as_complex(xq_).to(xq.device)
        xk_ = torch.view_as_complex(xk_).to(xq.device)
        xv_ = torch.view_as_complex(xv_).to(xq.device)

        # rotate and then map to real number field
        xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(2).to(xq.device)
        xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(2).to(xq.device)
        xv_out = torch.view_as_real(xv_ * freqs_cis).flatten(2).to(xq.device)
        return xq_out.type_as(xq), xk_out.type_as(xk), xv_out.type_as(xv)


def get_linear_period(min_period, max_period, dim):
    i = torch.arange(0, dim, 2)[: (dim // 2)]

    periods = min_period + ((max_period - min_period) / dim) * i
    theta = 2 * np.pi / periods
    return theta


def get_exp_period(min_period, max_period, dim):
    i = torch.arange(0, dim, 2)[: (dim // 2)]
    max_theta = 2 * np.pi / min_period
    min_theta = 2 * np.pi / max_period
    alpha = np.log(max_theta / min_theta) * (1 / (dim - 2))
    thetas = max_theta * np.exp(-alpha * i)
    return thetas


# generate rotation matrix
def precompute_freqs_cis(dim: int, seq_len: int, theta: float = 10000.0):

    # rotate \theta_i
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # generate token indexes t = [0, 1,..., seq_len-1]
    t = torch.arange(seq_len, device=freqs.device)
    # freqs.shape = [seq_len, dim // 2]
    freqs = torch.outer(t, freqs).float()  # m * \theta

    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    xv: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 2)
    xv_ = xv.float().reshape(*xv.shape[:-1], -1, 2)

    freqs_cis = freqs_cis.to(xq.device)

    xq_ = torch.view_as_complex(xq_).to(xq.device)
    xk_ = torch.view_as_complex(xk_).to(xq.device)
    xv_ = torch.view_as_complex(xv_).to(xq.device)

    # rotate and then map to real number field
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(2).to(xq.device)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(2).to(xq.device)
    xv_out = torch.view_as_real(xv_ * freqs_cis).flatten(2).to(xq.device)
    return xq_out.type_as(xq), xk_out.type_as(xk), xv_out.type_as(xv)
