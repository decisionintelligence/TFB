import torch
from torch import nn

class EMA(nn.Module):
    """
    Exponential Moving Average (EMA) block to highlight the trend of time series
    """
    def __init__(self, alpha):
        super(EMA, self).__init__()
        # self.alpha = nn.Parameter(alpha)    # Learnable alpha
        self.alpha = alpha

    # Optimized implementation with O(1) time complexity
    def forward(self, x):
        # x: [Batch, Input, Channel]
        # self.alpha.data.clamp_(0, 1)        # Clamp learnable alpha to [0, 1]
        _, t, _ = x.shape
        powers = torch.flip(torch.arange(t, dtype=torch.double), dims=(0,))
        weights = torch.pow((1 - self.alpha), powers).to('cuda')
        divisor = weights.clone()
        weights[1:] = weights[1:] * self.alpha
        weights = weights.reshape(1, t, 1)
        divisor = divisor.reshape(1, t, 1)
        x = torch.cumsum(x * weights, dim=1)
        x = torch.div(x, divisor)
        return x.to(torch.float32)
    
    # # Naive implementation with O(n) time complexity
    # def forward(self, x):
    #     # self.alpha.data.clamp_(0, 1)        # Clamp learnable alpha to [0, 1]
    #     s = x[:, 0, :]
    #     res = [s.unsqueeze(1)]
    #     for t in range(1, x.shape[1]):
    #         xt = x[:, t, :]
    #         s = self.alpha * xt + (1 - self.alpha) * s
    #         res.append(s.unsqueeze(1))
    #     return torch.cat(res, dim=1)