import torch
import torch.nn as nn


class RevIN(nn.Module):
    def __init__(self, channel, output_dim):
        super(RevIN, self).__init__()
        self.stdev = None
        self.means = None
        self.output_dim = output_dim

    def forward(self, x):
        # Calculate mean and std along dim=1
        self.means = x.mean(1, keepdim=True).detach()
        self.stdev = torch.sqrt(x.var(1, keepdim=True, unbiased=False) + 1e-5)

        # Normalize using learned parameters
        x_normalized = (x - self.means) / self.stdev
        return x_normalized

    def inverse_normalize(self, x_normalized):
        x_normalized = x_normalized * \
                       (self.stdev[:, 0, :].unsqueeze(1).repeat(
                           1, self.output_dim, 1))
        x_normalized = x_normalized + \
                       (self.means[:, 0, :].unsqueeze(1).repeat(
                           1, self.output_dim, 1))
        return x_normalized
