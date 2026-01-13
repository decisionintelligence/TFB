import math
import os
import torch
from ts_benchmark.baselines.timefilter.models.TimeFilter import Model
from ts_benchmark.baselines.deep_forecasting_model_base import DeepForecastingModelBase

# model hyper params
MODEL_HYPER_PARAMS = {
    "enc_in": 1,
    "dec_in": 1,
    "label_len": 48,
    "e_layers": 3,
    "d_layers": 1,
    "d_model": 512,
    "d_ff": 1024,
    "factor": 3,
    "patch_len": 720,
    "dropout": 0.3,
    "batch_size": 16,
    "lradj": "cosine",
    "patience": 3,
    "lr": 0.001,
    "alpha": 0.1,
    "pos": 1,
    "n_heads": 4,
    "top_p": 0.5,
    "num_epochs": 10,
    "loss": "MSE",
    "parallel_strategy": "DP",
    "use_gpu": True
}


class TimeFilter(DeepForecastingModelBase):
    """
    TimeFilter adapter class.

    Attributes:
        model_name (str): Name of the model for identification purposes.
        _init_model: Initializes an instance of the DUETModel.
        _adjust_lr：Adjusts the learning rate of the optimizer based on the current epoch and configuration.
        _process: Executes the model's forward pass and returns the output.
    """

    def __init__(self, **kwargs):
        super(TimeFilter, self).__init__(MODEL_HYPER_PARAMS, **kwargs)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def adjust_lr(self, optimizer, epoch, args):
        # lr = args.lr * (0.2 ** (epoch // 2))
        if args.lradj == 'type1':
            lr_adjust = {epoch: args.lr * (0.5 ** ((epoch - 1) // 1))}
        elif args.lradj == 'type2':
            lr_adjust = {
                2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
                10: 5e-7, 15: 1e-7, 20: 5e-8
            }
        elif args.lradj == "cosine":
            lr_adjust = {epoch: args.lr /2 * (1 + math.cos(epoch / args.num_epochs * math.pi))}
        elif args.lradj == 'unchanged':
            lr_adjust = {epoch: args.lr}
        if epoch in lr_adjust.keys():
            lr = lr_adjust[epoch]
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            print('Updating learning rate to {}'.format(lr))
    
    def _adjust_lr(self, optimizer, epoch, config):
        """
        Adjusts the learning rate of the optimizer based on the current epoch and configuration.

        This method is typically called to update the learning rate according to a predefined schedule.

        :param optimizer: The optimizer for which the learning rate will be adjusted.
        :param epoch: The current training epoch used to calculate the new learning rate.
        :param config: Configuration object containing parameters that control learning rate adjustment.
        """
        self.adjust_lr(optimizer, epoch, config)

    def _get_mask(self):
        dtype = torch.float32
        L = self.config.seq_len * self.config.enc_in // self.config.patch_len
        N = self.config.seq_len // self.config.patch_len
        masks = []
        for k in range(L):
            S = ((torch.arange(L) % N == k % N) & (torch.arange(L) != k)).to(dtype).to(self.device)
            T = ((torch.arange(L) >= k // N * N) & (torch.arange(L) < k // N * N + N) & (torch.arange(L) != k)).to(dtype).to(self.device)
            ST = torch.ones(L).to(dtype).to(self.device) - S - T
            ST[k] = 0.0
            masks.append(torch.stack([S, T, ST], dim=0))
        masks = torch.stack(masks, dim=0)
        return masks
    
    def _get_mask_2(self):
        dtype = torch.float32
        dtype = torch.float32
        L = self.config.seq_len * self.config.c_out // self.config.patch_len
        N = self.config.seq_len // self.config.patch_len

        mask_base = torch.eye(L, device=self.device, dtype=dtype).unsqueeze(0).unsqueeze(0)
        mask0 = torch.eye(L, device=self.device, dtype=dtype)
        mask0.view(self.config.c_out, N, self.config.c_out, N).diagonal(dim1=0, dim2=2).fill_(1)
        mask0 = mask0.unsqueeze(0).unsqueeze(0) - mask_base
        mask1 = torch.kron(torch.ones(self.config.c_out, self.config.c_out, device=self.device, dtype=dtype), 
                            torch.eye(N, device=self.device, dtype=dtype))
        mask1 = mask1.unsqueeze(0).unsqueeze(0) - mask_base
        mask2 = torch.ones((1, 1, L, L), device=self.device, dtype=dtype) - mask1 - mask0 - mask_base
        masks = torch.cat([mask0, mask1, mask2], dim=0)  # [3, 1, L, L]
        return masks
    
    @property
    def model_name(self):
        return "TimeFilter"

    def _init_model(self):
        self.masks = self._get_mask()
        return Model(self.config)

    def _process(self, input, target, input_mark, target_mark):
        output, moe_loss = self.model(input, self.masks, is_training=self.model.training)
        alpha = 0.05
        out_loss = {"output": output}
        if self.model.training:
            out_loss["additional_loss"] = alpha * moe_loss
        return out_loss
