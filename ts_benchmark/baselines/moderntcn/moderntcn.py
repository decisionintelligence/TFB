import math
import os
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
from ts_benchmark.baselines.moderntcn.models.ModernTCN import Model
from ts_benchmark.baselines.deep_forecasting_model_base import DeepForecastingModelBase

# model hyper params
MODEL_HYPER_PARAMS = {
    "stem_ratio": 6,
    "downsample_ratio": 2,
    "ffn_ratio": 2,
    "patch_size": 16,
    "stride": 8,
    "patch_stride": 8,
    "num_blocks": [1,1,1,1],
    "large_size": [31,29,27,13],
    "small_size": [5,5,5,5],
    "dims": [256,256,256,256],
    "dw_dims": [256,256,256,256],
    "small_kernel_merged": False,
    "call_structural_reparam": False,
    "use_multi_scale": True,
    "pct_start": 0.3,
    "use_gpu": True,
    "batch_size": 128,
    "lradj": "type3",
    "lr": 0.0001,
    "num_epochs": 100,
    "loss": "MSE",
    "parallel_strategy": "DP",
    "label_len": 48,
    "subtract_last": 0,
    "freq": "h",
    "individual": 0,
    "revin": 1,
    "affine": 0,
    "kernel_size": 25,
    "decomposition": 0
}


class ModernTCN(DeepForecastingModelBase):
    """
    TimeFilter adapter class.

    Attributes:
        model_name (str): Name of the model for identification purposes.
        _init_model: Initializes an instance of the DUETModel.
        _adjust_lr：Adjusts the learning rate of the optimizer based on the current epoch and configuration.
        _process: Executes the model's forward pass and returns the output.
    """

    def __init__(self, **kwargs):
        super(ModernTCN, self).__init__(MODEL_HYPER_PARAMS, **kwargs)

    def _init_criterion_and_optimizer(self):
        """
        Initializes the task loss function and optimizer.

        This method configures the task loss function and the optimizer based on the settings in `self.config`.
        Default supported loss functions include Mean Squared Error (MSE), Mean Absolute Error (MAE), and Huber Loss.
        And the Adam optimizer is used with the model's parameters and the learning rate specified in the configuration.

        :return: A tuple containing the initialized task loss function (`criterion`) and the optimizer (`optimizer`).
        """
        if self.config.loss == "MSE":
            criterion = nn.MSELoss()
        elif self.config.loss == "MAE":
            criterion = nn.L1Loss()
        else:
            criterion = nn.HuberLoss(delta=0.5)

        optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr)
        train_steps = len(self.train_data_loader)
        self.scheduler = lr_scheduler.OneCycleLR(optimizer=optimizer,
                                            steps_per_epoch=train_steps,
                                            pct_start=self.config.pct_start,
                                            epochs=self.config.num_epochs,
                                            max_lr=self.config.lr)
        return criterion, optimizer

    def adjust_learning_rate(self, optimizer, scheduler, epoch, args, printout=True):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
        if args.lradj == 'type1':
            lr_adjust = {epoch: args.lr * (0.5 ** ((epoch - 1) // 1))}
        elif args.lradj == 'type2':
            lr_adjust = {
                2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
                10: 5e-7, 15: 1e-7, 20: 5e-8
            }
        elif args.lradj == 'type3':
            lr_adjust = {epoch: args.lr if epoch < 3 else args.lr * (0.9 ** ((epoch - 3) // 1))}
        elif args.lradj == 'constant':
            lr_adjust = {epoch: args.lr}
        elif args.lradj == '3':
            lr_adjust = {epoch: args.lr if epoch < 10 else args.lr*0.1}
        elif args.lradj == '4':
            lr_adjust = {epoch: args.lr if epoch < 15 else args.lr*0.1}
        elif args.lradj == '5':
            lr_adjust = {epoch: args.lr if epoch < 25 else args.lr*0.1}
        elif args.lradj == '6':
            lr_adjust = {epoch: args.lr if epoch < 5 else args.lr*0.1}  
        elif args.lradj == 'TST':
            lr_adjust = {epoch: scheduler.get_last_lr()[0]}
        
        if epoch in lr_adjust.keys():
            lr = lr_adjust[epoch]
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            if printout: print('Updating learning rate to {}'.format(lr))
    
    def _adjust_lr(self, optimizer, epoch, config):
        """
        Adjusts the learning rate of the optimizer based on the current epoch and configuration.

        This method is typically called to update the learning rate according to a predefined schedule.

        :param optimizer: The optimizer for which the learning rate will be adjusted.
        :param epoch: The current training epoch used to calculate the new learning rate.
        :param config: Configuration object containing parameters that control learning rate adjustment.
        """
        self.adjust_learning_rate(optimizer, self.scheduler, epoch, config)
        self.scheduler.step()
    
    @property
    def model_name(self):
        return "ModernTCN"

    def _init_model(self):
        return Model(
            self.config
        )

    def _process(self, input, target, input_mark, target_mark):
        output = self.model(input, input_mark)
        out_loss = {"output": output}
        return out_loss
