# ---------------------------------------------------------------------------------
# Portions of this file are derived from PyTorch-TS
# - Source: https://github.com/zalandoresearch/pytorch-ts
# - License: MIT, Apache-2.0 license

# We thank the authors for their contributions.
# ---------------------------------------------------------------------------------

import re
import os
import torch
import numpy as np
from typing import Optional, Dict
import torch.nn as nn


class Scaler:
    def __init__(self):
        super().__init__()

    def fit(self, values):
        raise NotImplementedError

    def transform(self, values):
        raise NotImplementedError

    def fit_transform(self, values):
        raise NotImplementedError

    def inverse_transform(self, values):
        raise NotImplementedError


class StandardScaler(Scaler):
    def __init__(
        self,
        mean: float = None,
        std: float = None,
        epsilon: float = 1e-9,
        var_specific: bool = True,
    ):
        """
        The class can be used to normalize PyTorch Tensors using native functions. The module does not expect the
        tensors to be of any specific shape; as long as the features are the last dimension in the tensor, the module
        will work fine.

        Args:
            mean: The mean of the features. The property will be set after a call to fit.
            std: The standard deviation of the features. The property will be set after a call to fit.
            epsilon: Used to avoid a Division-By-Zero exception.
            var_specific: If True, the mean and standard deviation will be computed per variate.
        """
        self.mean = mean
        self.scale = std
        self.epsilon = epsilon
        self.var_specific = var_specific

    def fit(self, values):
        """
        Args:
            values: Input values should be a PyTorch tensor of shape (T, C) or (N, T, C),
                where N is the batch size, T is the timesteps and C is the number of variates.
        """
        dims = list(range(values.dim() - 1))
        if not self.var_specific:
            self.mean = torch.mean(values)
            self.scale = torch.std(values)
        else:
            self.mean = torch.mean(values, dim=dims)
            self.scale = torch.std(values, dim=dims)

    def transform(self, values):
        if self.mean is None:
            return values

        values = (values - self.mean.to(values.device)) / (
            self.scale.to(values.device) + self.epsilon
        )
        return values.to(torch.float32)

    def fit_transform(self, values):
        self.fit(values)
        return self.transform(values)

    def inverse_transform(self, values):
        if self.mean is None:
            return values

        values = values * (self.scale.to(values.device) + self.epsilon)
        values = values + self.mean.to(values.device)
        return values


class TemporalScaler(Scaler):
    def __init__(self, minimum_scale: float = 1e-10, time_first: bool = True):
        """
        The ``TemporalScaler`` computes a per-item scale according to the average
        absolute value over time of each item. The average is computed only among
        the observed values in the data tensor, as indicated by the second
        argument. Items with no observed data are assigned a scale based on the
        global average.

        Args:
            minimum_scale: default scale that is used if the time series has only zeros.
            time_first: if True, the input tensor has shape (N, T, C), otherwise (N, C, T).
        """
        super().__init__()
        self.scale = None
        self.minimum_scale = torch.tensor(minimum_scale)
        self.time_first = time_first

    def fit(self, data: torch.Tensor, observed_indicator: torch.Tensor = None):
        """
        Fit the scaler to the data.

        Args:
            data: tensor of shape (N, T, C) if ``time_first == True`` or (N, C, T)
                if ``time_first == False`` containing the data to be scaled

            observed_indicator: observed_indicator: binary tensor with the same shape as
                ``data``, that has 1 in correspondence of observed data points,
                and 0 in correspondence of missing data points.

        Note:
            Tensor containing the scale, of shape (N, 1, C) or (N, C, 1).
        """
        if self.time_first:
            dim = -2
        else:
            dim = -1

        if observed_indicator is None:
            observed_indicator = torch.ones_like(data)

        # These will have shape (N, C)
        num_observed = observed_indicator.sum(dim=dim)
        sum_observed = (data.abs() * observed_indicator).sum(dim=dim)

        # First compute a global scale per-dimension
        total_observed = num_observed.sum(dim=0)
        denominator = torch.max(total_observed, torch.ones_like(total_observed))
        default_scale = sum_observed.sum(dim=0) / denominator

        # Then compute a per-item, per-dimension scale
        denominator = torch.max(num_observed, torch.ones_like(num_observed))
        scale = sum_observed / denominator

        # Use per-batch scale when no element is observed
        # or when the sequence contains only zeros
        scale = torch.where(
            sum_observed > torch.zeros_like(sum_observed),
            scale,
            default_scale * torch.ones_like(num_observed),
        )

        self.scale = torch.max(scale, self.minimum_scale).unsqueeze(dim=dim).detach()

    def transform(self, data):
        return data / self.scale.to(data.device)

    def fit_transform(self, data, observed_indicator=None):
        self.fit(data, observed_indicator)
        return self.transform(data)

    def inverse_transform(self, data):
        return data * self.scale.to(data.device)


class IdentityScaler(Scaler):
    """
    No scaling is applied upon calling the ``IdentityScaler``.
    """

    def __init__(self, time_first: bool = True):
        super().__init__()
        self.scale = None

    def fit(self, data):
        pass

    def transform(self, data):
        return data

    def inverse_transform(self, data):
        return data


def repeat(tensor: torch.Tensor, n: int, dim: int = 0):
    return tensor.repeat_interleave(repeats=n, dim=dim)


def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def weighted_average(
    x: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    dim: int = None,
    reduce: str = "mean",
):
    """
    Computes the weighted average of a given tensor across a given dim, masking
    values associated with weight zero,
    meaning instead of `nan * 0 = nan` you will get `0 * 0 = 0`.

    Args:
        x: Input tensor, of which the average must be computed.
        weights: Weights tensor, of the same shape as `x`.
        dim: The dim along which to average `x`

    Returns:
        Tensor: The tensor with values averaged along the specified `dim`.
    """
    if weights is not None:
        weighted_tensor = torch.where(weights != 0, x * weights, torch.zeros_like(x))
        if reduce != "mean":
            return weighted_tensor
        sum_weights = torch.clamp(
            weights.sum(dim=dim) if dim else weights.sum(), min=1.0
        )
        return (
            weighted_tensor.sum(dim=dim) if dim else weighted_tensor.sum()
        ) / sum_weights
    else:
        return x.mean(dim=dim) if dim else x


class InstanceNorm(nn.Module):
    def __init__(self, eps=1e-5):
        """
        :param eps: a value added for numerical stability
        """
        super(InstanceNorm, self).__init__()
        self.eps = eps

    def forward(self, x, mode: str):
        if mode == "norm":
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == "denorm":
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(
            torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps
        ).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        return x

    def _denormalize(self, x):
        x = x * self.stdev
        x = x + self.mean
        return x


class RamdonRevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, dist="gaussian"):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RamdonRevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.dist = dist
        self._init_params()

    def forward(self, x, mode: str, mask=None):
        dim = len(x.shape)
        if dim == 4 and x.shape[1] == 1:
            x = x.squeeze(1)
            if mask is not None:
                mask = mask.squeeze(1)

        if mode == "norm":
            self._get_statistics(x, mask)
            x = self._normalize(x) * mask
        elif mode == "denorm":
            x = self._denormalize(x) * mask
        else:
            raise NotImplementedError

        if dim == 4 and len(x.shape) == 3:
            x = x.unsqueeze(1)
        return x

    def _init_params(self):
        # initialize RevIN params
        requires_grad = False
        if self.dist == "gaussian":
            affine_weight = torch.randn(self.num_features) + torch.ones(
                self.num_features
            )
            affine_bias = torch.randn(self.num_features)
        elif self.dist == "learn":
            affine_weight = torch.ones(self.num_features)
            affine_bias = torch.zeros(self.num_features)
            requires_grad = True
        elif self.dist == "uniform" or self.dist == "none":
            affine_weight = torch.rand(self.num_features)
            affine_bias = torch.rand(self.num_features)
        else:
            print(
                "Invalid distribution setting for RamdonRevIN, select from: [gaussian, learn, uniform, none]"
            )

        if requires_grad:
            self.affine_weight = nn.Parameter(affine_weight, requires_grad=True)
            self.affine_bias = nn.Parameter(affine_bias, requires_grad=True)
        else:
            self.register_buffer("affine_weight", torch.tensor(affine_weight))
            self.register_buffer("affine_bias", torch.tensor(affine_bias))

    def _get_statistics(self, x, mask):
        dim2reduce = tuple(range(1, x.ndim - 1))
        if mask is None:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
            self.stdev = torch.sqrt(
                torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps
            ).detach()
        else:
            num_observed = mask.sum(dim=dim2reduce, keepdim=True)
            self.mean = (
                (x * mask).sum(dim=dim2reduce, keepdim=True) / num_observed
            ).detach()
            squared_diffs = (x - self.mean) ** 2 * mask
            variance = squared_diffs.sum(dim=dim2reduce, keepdim=True) / num_observed
            self.stdev = torch.sqrt(variance + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev

        x = x * self.affine_weight
        x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        x = x - self.affine_bias
        x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x


def convert_to_list(s):
    """
    Convert prediction length strings into list
    e.g., '96-192-336-720' will be convert into [96,192,336,720]
    Input: str, list, int
    Returns: list
    """
    if type(s).__name__ == "int":
        return [s]
    elif type(s).__name__ == "list":
        return s
    elif type(s).__name__ == "str":
        elements = re.split(r"\D+", s)
        return list(map(int, elements))
    else:
        return None


def convert_to_float_list(s):
    """
    Convert prediction length strings into list
    e.g., '96-192-336-720' will be convert into [96,192,336,720]
    Input: str, list, int
    Returns: list
    """
    if type(s).__name__ == "int":
        return [s]
    elif type(s).__name__ == "list":
        return s
    elif type(s).__name__ == "str":
        elements = re.split("-", s)
        return list(map(float, elements))
    else:
        return None


def find_min_prediction_length(arr, x):
    if x is None:
        return x
    for value in arr:
        if value >= x:
            return value
    return None


def find_best_epoch(ckpt_folder):
    """
    Find the highest epoch in the Test Tube file structure.
    :param ckpt_folder: dir where the checpoints are being saved.
    :return: Integer of the highest epoch reached by the checkpoints.
    """
    pattern = r"val_CRPS=([0-9]*\.[0-9]+)"
    ckpt_files = os.listdir(ckpt_folder)  # list of strings
    epochs = []
    for filename in ckpt_files:
        match = re.search(pattern, filename)
        if match:
            epochs.append(match.group(1))
    # epochs = [float(filename[18:-5]) for filename in ckpt_files]  # 'epoch={int}.ckpt' filename format
    best_crps = min(epochs)

    best_epoch = epochs.index(best_crps)
    return best_epoch, ckpt_files[best_epoch]


def get_wandb_config_dict(wandb, config_args):
    config_dict = {}
    args = config_args.model.forecaster

    if "init_args" in args:
        for key, value in args["init_args"].items():
            if wandb["target_config"] is not None and key in wandb["target_config"]:
                # continue
                config_dict[key] = value

    config_dict["dataset"] = config_args.data.data_manager["init_args"]["dataset"]
    config_dict["pred_len"] = config_args.data.data_manager["init_args"][
        "prediction_length"
    ]
    config_dict["context_length"] = config_args.data.data_manager["init_args"][
        "context_length"
    ]
    config_dict["train_pred_len"] = config_args.data.data_manager["init_args"][
        "train_pred_len_list"
    ]
    config_dict["train_ctx_len_list"] = config_args.data.data_manager["init_args"][
        "train_ctx_len_list"
    ]
    config_dict["val_pred_len_list"] = config_args.data.data_manager["init_args"][
        "val_pred_len_list"
    ]
    return config_dict
