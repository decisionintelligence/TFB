import torch
from torch import nn

from ..layers.ema import EMA
from ..layers.dema import DEMA

class DECOMP(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, ma_type, alpha, beta):
        super(DECOMP, self).__init__()
        if ma_type == 'ema':
            self.ma = EMA(alpha)
        elif ma_type == 'dema':
            self.ma = DEMA(alpha, beta)

    def forward(self, x):
        moving_average = self.ma(x)
        res = x - moving_average
        return res, moving_average