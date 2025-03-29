import torch
import torch.nn as nn


# This is inspired by Kolmogorov-Arnold Networks but using Chebyshev polynomials instead of splines coefficients
class ChebyKANLinear(nn.Module):
    def __init__(self, input_dim, output_dim, degree):
        super(ChebyKANLinear, self).__init__()
        self.inputdim = input_dim
        self.outdim = output_dim
        self.degree = degree

        self.cheby_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        self.epsilon = 1e-7
        self.pre_mul = False
        self.post_mul = False
        nn.init.normal_(self.cheby_coeffs, mean=0.0, std=1 / (input_dim * (degree + 1)))
        self.register_buffer("arange", torch.arange(0, degree + 1, 1))

    def forward(self, x):
        # Since Chebyshev polynomial is defined in [-1, 1]
        # We need to normalize x to [-1, 1] using tanh
        # View and repeat input degree + 1 times
        b,c_in = x.shape
        if self.pre_mul:
            mul_1 = x[:,::2]
            mul_2 = x[:,1::2]
            mul_res = mul_1 * mul_2
            x = torch.concat([x[:,:x.shape[1]//2], mul_res])
        x = x.view((b, c_in, 1)).expand(
            -1, -1, self.degree + 1
        )  # shape = (batch_size, inputdim, self.degree + 1)
        # Apply acos
        x = torch.tanh(x)
        x = torch.tanh(x)
        x = torch.acos(x)
        # x = torch.acos(torch.clamp(x, -1 + self.epsilon, 1 - self.epsilon))
        # # Multiply by arange [0 .. degree]
        x = x* self.arange
        # Apply cos
        x = x.cos()
        # Compute the Chebyshev interpolation
        y = torch.einsum(
            "bid,iod->bo", x, self.cheby_coeffs
        )  # shape = (batch_size, outdim)
        y = y.view(-1, self.outdim)
        if self.post_mul:
            mul_1 = y[:,::2]
            mul_2 = y[:,1::2]
            mul_res = mul_1 * mul_2
            y = torch.concat([y[:,:y.shape[1]//2], mul_res])
        return y