import copy
import math
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import logging

from einops import rearrange
from sklearn.preprocessing import StandardScaler
from torch import optim
from torch.utils.data import DataLoader

from ts_benchmark.baselines.utils import EarlyStopping, adjust_learning_rate, DBLoss
from ts_benchmark.baselines.utils import (
    forecasting_data_provider,
    train_val_split,
    get_time_mark,
)
from ts_benchmark.models.model_base import ModelBase, BatchMaker
from ts_benchmark.utils.data_processing import split_time
from ts_benchmark.baselines.time_series_library.layers.SelfAttention_Family import FullAttention, AttentionLayer


logger = logging.getLogger(__name__)

# Default hyper parameters
DEFAULT_HYPER_PARAMS = {
    "use_amp": 0,
    "loss": "MSE",
    "batch_size": 256,
    "lradj": "type3",
    "lr": 0.0001,
    "num_workers": 0,
    "patience": 10,
    "num_epochs": 100,
    "adj_lr_in_epoch": True,
    "adj_lr_in_batch": False,
    "parallel_strategy": None,
    "fusion_method": "",
    "covariate_dim": 6,
    "cross_attention_head": 12,
    "cross_attention_dropout": 0.1,
    "cross_attention_factor": 2,
    "conv_dropout": 0.1,
    "alpha_cov": 1.0,
    "mlp_hidden_dim": 64,
}

# x_dec为未来协变量,ouput为经过series_dim处理后的目标值
class MLP(nn.Module):
    def __init__(self, configs):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(configs.input_dim, configs.mlp_hidden_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(configs.mlp_hidden_dim, configs.output_dim)

    def forward(self, x_dec, output):
        x_dec = x_dec.float()
        output = output.float()
        mlp_input = torch.cat((output, x_dec), dim=-1)
        mlp_input = self.linear1(mlp_input)
        mlp_input = self.relu(mlp_input)
        mlp_input = self.linear2(mlp_input)
        return mlp_input

class Conv(nn.Module):
    def __init__(self, configs):
        super(Conv, self).__init__()
        self.correlation_embedding = nn.Conv1d(configs.input_dim, configs.output_dim, 3, padding='same')
        self.alpha = nn.Parameter(torch.ones([1]) * configs.alpha_cov)
        self.dropout = nn.Dropout(configs.conv_dropout)
        self.norm = nn.LayerNorm(configs.horizon)

    def forward(self, x_dec, output):
        x_dec = x_dec.float()
        output = output.float()
        output1 = output
        conv_input = torch.cat((output, x_dec), dim=-1)
        conv_input = rearrange(conv_input, 'b l n -> b n l')
        conv_output = self.correlation_embedding(conv_input).permute(0, 2, 1)
        conv_output = conv_output.permute(0, 2, 1) + output.permute(0, 2, 1)
        conv_output = self.norm(conv_output)
        output = self.alpha * output1.permute(0, 2, 1) + (1 - self.alpha) * conv_output
        output = output.permute(0, 2, 1)
        return output

class CrossAttention(nn.Module):
    def __init__(self, configs):
        super(CrossAttention, self).__init__()
        self.cross_attention = AttentionLayer(
            FullAttention(False, configs.cross_attention_factor, attention_dropout=configs.cross_attention_dropout,
                          output_attention=False),
            configs.horizon, configs.cross_attention_head)
        self.alpha = nn.Parameter(torch.ones([1]) * configs.alpha_cov)
        self.dropout = nn.Dropout(configs.cross_attention_dropout)
        self.norm = nn.LayerNorm(configs.horizon)

    def forward(self, x_dec, output):
        x_dec = x_dec.float()
        output = output.float()
        output1 = output
        x_glb_attn = self.dropout(self.cross_attention(
            output.permute(0, 2, 1), x_dec.permute(0, 2, 1), x_dec.permute(0, 2, 1),
            attn_mask=None,
            tau=None, delta=None
        )[0])
        x_glb = output.permute(0, 2, 1) + x_glb_attn
        x_glb = self.norm(x_glb)
        output = self.alpha * x_glb.permute(0, 2, 1) + (1 - self.alpha) * output1
        return output



class Config:
    def __init__(self, model_config, **kwargs):

        for key, value in DEFAULT_HYPER_PARAMS.items():
            setattr(self, key, value)

        for key, value in model_config.items():
            setattr(self, key, value)

        for key, value in kwargs.items():
            setattr(self, key, value)

        if hasattr(self, "horizon"):
            logger.warning(
                "The model parameter horizon is deprecated. Please use pred_len."
            )
            setattr(self, "pred_len", self.horizon)


class DeepForecastingModelBase(ModelBase):
    """
    Base class for deep learning model in forecasting tasks, inherited from ModelBase.

    This class provides a framework and default functionalities for adapters in time series forecasting tasks,
    including model initialization, configuration of loss functions and optimizers, data processing,
    learning rate adjustment, save checkpoints and early stopping mechanisms.

    Subclasses must implement _init_model and _process methods to define specific data processing and modeling logic.

    """

    def __init__(self, model_config, **kwargs):
        super(DeepForecastingModelBase, self).__init__()
        self.config = Config(model_config, **kwargs)
        # self.scaler = StandardScaler()
        self.scaler1 = StandardScaler()
        self.scaler2 = StandardScaler()
        self.seq_len = self.config.seq_len
        self.win_size = self.config.seq_len

    def _init_model(self):
        """
        Initialize the model.

        This method is intended to be implemented by subclasses to initialize the specific model.
        The current implementation raises a NotImplementedError to indicate that this method should
        be overridden in subclasses.

        :return: The actual model object. The specific type of the return value should be defined by subclasses.
        """
        raise NotImplementedError("model must be implemented.")

    def _adjust_lr(self, optimizer, epoch, config):
        """
        Adjusts the learning rate of the optimizer based on the current epoch and configuration.

        This method is typically called to update the learning rate according to a predefined schedule.

        :param optimizer: The optimizer for which the learning rate will be adjusted.
        :param epoch: The current training epoch used to calculate the new learning rate.
        :param config: Configuration object containing parameters that control learning rate adjustment.
        """
        adjust_learning_rate(optimizer, epoch, config)

    def save_checkpoint(self, models):
        """
        Save the model checkpoint.

        This function saves the model's state dictionary (state_dict) to be used
        for restoring the model at a later time. A deep copy of the state_dict is returned.

        Parameters:
        - model (torch.nn.Module): The current instance of the model being trained.

        Returns:
        - OrderedDict: A deep copy of the model's state_dict, which can be used to restore
          the model's parameters in the future.
        """
        return {key: copy.deepcopy(model.state_dict()) for key, model in models.items()}

    def _init_criterion(self):
        """
        Initializes the task loss function.

        Supports MSELoss, L1Loss (MAE), and HuberLoss depending on `self.config.loss`.
        """
        if self.config.loss == "MSE":
            return nn.MSELoss()
        elif self.config.loss == "MAE":
            return nn.L1Loss()
        else:
            return nn.HuberLoss(delta=0.5)

    def _init_optimizer(self,CovariateFusion=None):
        """
        Initializes the optimizer using Adam.

        If `self.CovariateFusion` exists, creates parameter groups with separate learning rates.
        """
        if hasattr(self, "CovariateFusion") and self.CovariateFusion is not None:
            return optim.Adam(
                [
                    {"params": self.model.parameters(), "lr": self.config.lr},
                    {"params": self.CovariateFusion.parameters(), "lr": self.config.lr},
                ]
            )
        else:
            return optim.Adam(self.model.parameters(), lr=self.config.lr)

    def _process(self, input, target, input_mark, target_mark ,exog_future=None):
        """
        A method that needs to be implemented by subclasses to process data and model, and calculate additional loss.

        This method's purpose is to serve as a template method, defining a standard process for data processing
        and modeling, as well as calculating any additional losses. Subclasses should implement specific processing
        and calculation logic based on their own needs.

        Parameters:
        - input: The input data, the specific form and meaning depend on the implementation of the subclass.
        - target: The target data, used in conjunction with input data for processing and loss calculation.
        - input_mark: Marks or metadata for the input data, assisting in data processing or model training.
        - target_mark: Marks or metadata for the target data, similarly assisting in data processing or model training.
        - exog_future: Exogenous future data, used in conjunction with input data for processing.

        Returns:
        - dict: A dictionary containing at least one key:
            - 'output' (necessary): The model output tensor.
            - 'additional_loss' (optional): An additional loss if it exists.

        Raises:
        - NotImplementedError: If the subclass does not implement this method, a NotImplementedError will be raised
                               when calling this method.
        """
        raise NotImplementedError("Process must be implemented")

    def _post_process(self, output, target):
        """
        Performs post-processing on the output and target data.

        This function is designed to process the output and target data after the model's forward computation,
        and return them directly in this example. The specific post-processing logic may include, but is not limited to,
        data format conversion, dimensionality matching, data type conversion, etc.

        Parameters:
        - output: The output data from the model, with no specific data format or type assumed.
        - target: The target data, which is the expected result, also without a fixed data format or type.

        Returns:
        - output: The output data after post-processing, which in this case is the same as the input.
        - target: The target data after post-processing, which in this case is the same as the input.
        """
        return output, target

    def _init_early_stopping(self):
        """
        Initializes the early stopping strategy for training.

        This function is used to create an instance of EarlyStopping, which helps prevent overfitting
        during model training by halting the training process when the validation performance
        does not improve for a specified number of consecutive iterations.

        Parameters:
        None directly, but it uses self.config.patience as the patience parameter for EarlyStopping.

        Returns:
        An instance of EarlyStopping, which monitors the model's performance metrics and determines
        when to stop the training.
        """
        return EarlyStopping(patience=self.config.patience)

    @property
    def model_name(self):
        return "DeepForecastingModelBase"

    @staticmethod
    def required_hyper_params() -> dict:
        """
        Return the hyperparameters required by model.

        :return: An empty dictionary indicating that model does not require additional hyperparameters.
        """
        return {
            "seq_len": "input_chunk_length",
            "horizon": "output_chunk_length",
            "norm": "norm",
        }

    def __repr__(self) -> str:
        """
        Returns a string representation of the model name.
        """
        return self.model_name

    def multi_forecasting_hyper_param_tune(self, train_data: pd.DataFrame):
        freq = pd.infer_freq(train_data.index)
        if freq == None:
            raise ValueError("Irregular time intervals")
        elif freq[0].lower() not in ["m", "w", "b", "d", "h", "t", "s"]:
            self.config.freq = "s"
        else:
            self.config.freq = freq[0].lower()

        column_num = train_data.shape[1]
        self.config.enc_in = column_num
        self.config.dec_in = column_num
        self.config.c_out = column_num

        if self.model_name == "MICN":
            setattr(self.config, "label_len", self.config.seq_len)
        else:
            setattr(self.config, "label_len", self.config.seq_len // 2)

    def single_forecasting_hyper_param_tune(self, train_data: pd.DataFrame):
        freq = pd.infer_freq(train_data.index)
        if freq == None:
            raise ValueError("Irregular time intervals")
        elif freq[0].lower() not in ["m", "w", "b", "d", "h", "t", "s"]:
            self.config.freq = "s"
        else:
            self.config.freq = freq[0].lower()

        column_num = train_data.shape[1]
        self.config.enc_in = column_num
        self.config.dec_in = column_num
        self.config.c_out = column_num

        setattr(self.config, "label_len", self.config.horizon)

    def detect_hyper_param_tune(self, train_data: pd.DataFrame):
        freq = pd.infer_freq(train_data.index)
        if freq == None:
            raise ValueError("Irregular time intervals")
        elif freq[0].lower() not in ["m", "w", "b", "d", "h", "t", "s"]:
            self.config.freq = "s"
        else:
            self.config.freq = freq[0].lower()

        column_num = train_data.shape[1]
        self.config.enc_in = column_num
        self.config.dec_in = column_num
        self.config.c_out = column_num
        self.config.label_len = 48

    def padding_data_for_forecast(self, test):
        time_column_data = test.index
        data_colums = test.columns
        start = time_column_data[-1]
        # padding_zero = [0] * (self.config.horizon + 1)
        date = pd.date_range(
            start=start, periods=self.config.horizon + 1, freq=self.config.freq.upper()
        )
        df = pd.DataFrame(columns=data_colums)

        df.iloc[: self.config.horizon + 1, :] = 0

        df["date"] = date
        df = df.set_index("date")
        new_df = df.iloc[1:]
        test = pd.concat([test, new_df])
        return test

    def _padding_time_stamp_mark(
        self, time_stamps_list: np.ndarray, padding_len: int
    ) -> np.ndarray:
        """
        Padding time stamp mark for prediction.

        :param time_stamps_list: A batch of time stamps.
        :param padding_len: The len of time stamp need to be padded.
        :return: The padded time stamp mark.
        """
        padding_time_stamp = []
        for time_stamps in time_stamps_list:
            start = time_stamps[-1]
            expand_time_stamp = pd.date_range(
                start=start,
                periods=padding_len + 1,
                freq=self.config.freq.upper(),
            )
            padding_time_stamp.append(expand_time_stamp.to_numpy()[-padding_len:])
        padding_time_stamp = np.stack(padding_time_stamp)
        whole_time_stamp = np.concatenate(
            (time_stamps_list, padding_time_stamp), axis=1
        )
        padding_mark = get_time_mark(whole_time_stamp, 1, self.config.freq)
        return padding_mark

    def validate(
        self, valid_data_loader: DataLoader, series_dim: int, criterion: torch.nn.Module
    ) -> float:
        """
        Validates the model performance on the provided validation dataset.
        :param valid_data_loader: A PyTorch DataLoader for the validation dataset.
        :param series_dim : The number of series data‘s dimensions.
        :param criterion : The loss function to compute the loss between model predictions and ground truth.
        :returns:The mean loss computed over the validation dataset.
        """
        config = self.config
        total_loss = []
        self.model.eval()
        if self.CovariateFusion is not None:
            self.CovariateFusion.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with torch.no_grad():
            for input, target, input_mark, target_mark in valid_data_loader:
                input, target, input_mark, target_mark = (
                    input.to(device),
                    target.to(device),
                    input_mark.to(device),
                    target_mark.to(device),
                )
                exog_future = target[:, -config.horizon:, series_dim:]
                out_loss = self._process(input, target, input_mark, target_mark, exog_future)
                additional_loss = 0
                output = out_loss["output"]
                if "additional_loss" in out_loss:
                    additional_loss = out_loss["additional_loss"]
                target = target[:, -config.horizon :, :series_dim]
                output = output[:, -config.horizon :, :series_dim]
                if (config.fusion_method == 'mlp' or config.fusion_method == 'cross_attention' or config.fusion_method == 'conv') and self.CovariateFusion is not None:
                    output = self.CovariateFusion(exog_future, output)
                output, target = self._post_process(output, target)
                all_loss = criterion(output, target) + additional_loss
                loss = all_loss.detach().cpu().numpy()
                total_loss.append(loss)

        total_loss = np.mean(total_loss)
        self.model.train()
        if self.CovariateFusion is not None:
            self.CovariateFusion.train()
        return total_loss

    def forecast_fit(
        self,
        train_valid_data: pd.DataFrame,
        *,
        covariates: Optional[dict] = None,
        train_ratio_in_tv: float = 1.0,
        **kwargs,
    ) -> "ModelBase":
        """
        Train the model.
        :param train_valid_data: Time series data used for training and validation.
        :param covariates: Additional external variables.
        :param train_ratio_in_tv: Represents the splitting ratio of the training set validation set. If it is equal to 1, it means that the validation set is not partitioned.
        :return: The fitted model object.
        """
        if covariates is None:
            covariates = {}
        series_dim = train_valid_data.shape[-1]
        exog_data = covariates.get("exog", None)
        if exog_data is not None:
            train_valid_data = pd.concat([train_valid_data, exog_data], axis=1)
            exog_dim = exog_data.shape[-1]
        else:
            exog_dim = 0

        if train_valid_data.shape[1] == 1:
            train_drop_last = False
            self.single_forecasting_hyper_param_tune(train_valid_data)
        else:
            train_drop_last = True
            self.multi_forecasting_hyper_param_tune(train_valid_data)

        self.config.series_dim = series_dim
        self.config.input_dim = series_dim + exog_dim
        self.config.output_dim = series_dim

        criterion = self._init_criterion()
        self.model = self._init_model()
        if self.config.fusion_method == 'mlp':
            self.CovariateFusion = MLP(self.config)
        elif self.config.fusion_method == 'cross_attention':
            self.CovariateFusion = CrossAttention(self.config)
        elif self.config.fusion_method == 'conv':
            self.CovariateFusion = Conv(self.config)
        else:
            self.CovariateFusion = None
        device_ids = np.arange(torch.cuda.device_count()).tolist()
        if len(device_ids) > 1 and self.config.parallel_strategy == "DP":
            self.model = nn.DataParallel(self.model, device_ids=device_ids)
            if self.CovariateFusion is not None:
                self.CovariateFusion = nn.DataParallel(self.CovariateFusion, device_ids=device_ids)
        print(
            "----------------------------------------------------------",
            self.model_name,
        )
        config = self.config
        train_data, valid_data = train_val_split(
            train_valid_data, train_ratio_in_tv, config.seq_len
        )

        # 分别 fit 两个 scaler
        if exog_dim > 0:
            # Fit scaler1 for series data
            self.scaler1.fit(train_data.values[:,:series_dim])
            # Fit scaler2 for exog data
            self.scaler2.fit(train_data.values[:,series_dim:])

        # self.scaler.fit(train_data.values)

            if config.norm:
                scaled_series = self.scaler1.transform(
                    train_data.values[:, :series_dim]
                )
                scaled_exog = self.scaler2.transform(
                    train_data.values[:, series_dim:]
                )
                final_train_data = np.concatenate((scaled_series, scaled_exog), axis=1)
                train_data = pd.DataFrame(
                # self.scaler.transform(train_data.values),
                final_train_data,
                columns=train_data.columns,
                index=train_data.index,
                )
        else:
            # Only series data, use scaler1
            self.scaler1.fit(train_data.values)
            if config.norm:
                train_data = pd.DataFrame(
                    self.scaler1.transform(train_data.values),
                    columns=train_data.columns,
                    index=train_data.index,
                )


        if train_ratio_in_tv != 1:
            if config.norm:
                if exog_dim > 0:
                    scaled_series = self.scaler1.transform(
                        valid_data.values[:, :series_dim]
                    )
                    scaled_exog = self.scaler2.transform(
                        valid_data.values[:, series_dim:]
                    )
                    final_valid_data = np.concatenate((scaled_series, scaled_exog), axis=1)
                    valid_data = pd.DataFrame(
                        final_valid_data,
                        columns=valid_data.columns,
                        index=valid_data.index,
                    )
                else:
                    valid_data = pd.DataFrame(
                        self.scaler1.transform(valid_data.values),
                        columns=valid_data.columns,
                        index=valid_data.index,
                    )
            valid_dataset, valid_data_loader = forecasting_data_provider(
                valid_data,
                config,
                timeenc=1,
                batch_size=config.batch_size,
                shuffle=True,
                drop_last=False,
            )

        train_dataset, self.train_data_loader = forecasting_data_provider(
            train_data,
            config,
            timeenc=1,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=train_drop_last,
        )
        # Define optimizer
        optimizer = self._init_optimizer(CovariateFusion=self.CovariateFusion)

        if config.use_amp == 1:
            scaler = torch.cuda.amp.GradScaler()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.early_stopping = self._init_early_stopping()
        self.model.to(device)
        if self.CovariateFusion is not None:
            self.CovariateFusion.to(device)
        total_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        if self.CovariateFusion is not None:
            total_params += sum(
                p.numel() for p in self.CovariateFusion.parameters() if p.requires_grad
            )
        print(f"Total trainable parameters: {total_params}")

        for epoch in range(config.num_epochs):
            self.model.train()
            if self.CovariateFusion is not None:
                self.CovariateFusion.train()
            # for input, target, input_mark, target_mark in train_data_loader:
            for i, (input, target, input_mark, target_mark) in enumerate(
                self.train_data_loader
            ):
                optimizer.zero_grad()
                input, target, input_mark, target_mark = (
                    input.to(device),
                    target.to(device),
                    input_mark.to(device),
                    target_mark.to(device),
                )
                # decoder input
                exog_future = target[:, -config.horizon:, series_dim:].to(device)
                out_loss = self._process(input, target, input_mark, target_mark, exog_future)
                additional_loss = 0
                output = out_loss["output"]
                if "additional_loss" in out_loss:
                    additional_loss = out_loss["additional_loss"]

                target = target[:, -config.horizon :, :series_dim]
                output = output[:, -config.horizon :, :series_dim]
                output, target = self._post_process(output, target)
                if (self.config.fusion_method == "mlp" or self.config.fusion_method == "conv" or self.config.fusion_method == "cross_attention") and self.CovariateFusion is not None:
                    output = self.CovariateFusion(exog_future, output)
                loss = criterion(output, target)

                total_loss = loss + additional_loss

                if config.use_amp == 1:
                    scaler.scale(total_loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    total_loss.backward()
                    optimizer.step()

                if self.config.lradj == "TST":
                    self._adjust_lr(optimizer, epoch + 1, config)

            if train_ratio_in_tv != 1:
                valid_loss = self.validate(valid_data_loader, series_dim, criterion)
                improved = self.early_stopping(valid_loss, self.model)
                if improved:
                    if self.CovariateFusion is not None:
                        self.check_point = self.save_checkpoint({"Model": self.model, "CovariateFusion": self.CovariateFusion})
                    else:
                        self.check_point = self.save_checkpoint({"Model": self.model})
                if self.early_stopping.early_stop:
                    break

            if self.config.lradj != "TST":
                self._adjust_lr(optimizer, epoch + 1, config)

    def forecast(
        self,
        horizon: int,
        series: pd.DataFrame,
        *,
        covariates: Optional[dict] = None,
    ) -> np.ndarray:
        """
        Make predictions.
        :param horizon: The predicted length.
        :param series: Time series data used for prediction.
        :param covariates: Additional external variables
        :return: An array of predicted results.
        """
        if covariates is None:
            covariates = {}
        series_dim = series.shape[-1]
        exog_data = covariates.get("exog", None)
        if exog_data is not None:
            series = pd.concat([series, exog_data], axis=1)
            if (
                hasattr(self.config, "output_chunk_length")
                and horizon != self.config.output_chunk_length
            ):
                raise ValueError(
                    f"Error: 'exog' is enabled during training, but horizon ({horizon}) != output_chunk_length ({self.config.output_chunk_length}) during forecast."
                )

        if self.check_point is not None:
            self.model.load_state_dict(self.check_point["Model"])
            if self.CovariateFusion is not None and "CovariateFusion" in self.check_point:
                self.CovariateFusion.load_state_dict(self.check_point["CovariateFusion"])

        if self.config.norm:
            if exog_data is not None:
                # scaler series data with scaler1
                series_values = series.iloc[:,:series_dim].values
                scaled_series = self.scaler1.transform(series_values)

                # scaler exog data with scaler2
                exog_values = series.iloc[:,series_dim:].values
                scaled_exog = self.scaler2.transform(exog_values)

                # Combine scaled data
                scaled_values = np.concatenate([scaled_series, scaled_exog], axis=1)
                series = pd.DataFrame(
                    scaled_values,
                    columns=series.columns,
                    index=series.index,
                )
            else:
                series = pd.DataFrame(
                    self.scaler1.transform(series.values),
                    columns=series.columns,
                    index=series.index,
                )

        if self.model is None:
            raise ValueError("Model not trained. Call the fit() function first.")

        config = self.config
        series, test = split_time(series, len(series) - config.seq_len)
        test = self.padding_data_for_forecast(test)

        test_data_set, test_data_loader = forecasting_data_provider(
            test, config, timeenc=1, batch_size=1, shuffle=False, drop_last=False
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.eval()
        if self.CovariateFusion is not None:
            self.CovariateFusion.to(device)
            self.CovariateFusion.eval()
        with torch.no_grad():
            answer = None
            while answer is None or answer.shape[0] < horizon:
                for input, target, input_mark, target_mark in test_data_loader:
                    input, target, input_mark, target_mark = (
                        input.to(device),
                        target.to(device),
                        input_mark.to(device),
                        target_mark.to(device),
                    )
                    exog_future = target[:, -config.horizon:, series_dim:]
                    out_loss = self._process(input, target, input_mark, target_mark, exog_future)
                    output = out_loss["output"]
                    if self.CovariateFusion is not None:
                        output = output[:, -config.horizon:, :series_dim]
                        output = self.CovariateFusion(exog_future, output)
                    else:
                        output = output[:, -config.horizon :, :series_dim]

                column_num = output.shape[-1]
                temp = output.cpu().numpy().reshape(-1, column_num)[-config.horizon :]

                if answer is None:
                    answer = temp
                else:
                    answer = np.concatenate([answer, temp], axis=0)

                if answer.shape[0] >= horizon:
                    if self.config.norm:
                        answer[-horizon:] = self.scaler1.inverse_transform(
                            answer[-horizon:]
                        )
                    return answer[-horizon:, :series_dim]

                output = output.cpu().numpy()[:, -config.horizon :]
                for i in range(config.horizon):
                    test.iloc[i + config.seq_len] = output[0, i, :]

                test = test.iloc[config.horizon :]
                test = self.padding_data_for_forecast(test)

                test_data_set, test_data_loader = forecasting_data_provider(
                    test,
                    config,
                    timeenc=1,
                    batch_size=1,
                    shuffle=False,
                    drop_last=False,
                )

    def batch_forecast(
        self, horizon: int, batch_maker: BatchMaker, exog_futures, i, **kwargs
    ) -> np.ndarray:
        """
        Make predictions by batch.

        :param horizon: The length of each prediction.
        :param batch_maker: Make batch data used for prediction.
        :param exog_futures: Future exogenous data used for prediction.
        :i: The index of the batch.
        :return: An array of predicted results.
        """
        if self.check_point is not None:
            self.model.load_state_dict(self.check_point["Model"])
            if self.CovariateFusion is not None:
                self.CovariateFusion.load_state_dict(self.check_point["CovariateFusion"])
        if self.model is None:
            raise ValueError("Model not trained. Call the fit() function first.")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.eval()
        if self.CovariateFusion is not None:
            self.CovariateFusion.to(device)
            self.CovariateFusion.eval()

        input_data = batch_maker.make_batch(self.config.batch_size, self.config.seq_len)
        input_np = input_data["input"]
        series_dim = input_np.shape[-1]
        batch_size = self.config.batch_size
        if input_data["covariates"] is None:
            covariates = {}
        else:
            covariates = input_data["covariates"]
        exog_data = covariates.get("exog")
        if exog_data is not None:
            exog_dim = exog_data.shape[-1]
            input_np = np.concatenate((input_np, exog_data), axis=2)
            if (
                hasattr(self.config, "output_chunk_length")
                and horizon != self.config.output_chunk_length
            ):
                raise ValueError(
                    f"Error: 'exog' is enabled during training, but horizon ({horizon}) != output_chunk_length ({self.config.output_chunk_length}) during forecast."
                )
        else:
            exog_dim = 0
        if self.config.norm:
            if exog_dim > 0:
                # Scale series data with scaler1
                series_data = input_np[..., :series_dim]
                origin_shape1 = series_data.shape
                flattened_data = series_data.reshape((-1, series_data.shape[-1]))
                series_data = self.scaler1.transform(flattened_data).reshape(origin_shape1)
                # Scale exog data with scaler2
                exog_data = input_np[..., series_dim:]
                origin_shape2 = exog_data.shape
                flattened_data = exog_data.reshape((-1, exog_data.shape[-1]))
                exog_data = self.scaler2.transform(flattened_data).reshape(origin_shape2)

                input_np = np.concatenate((series_data, exog_data), axis=2)
            else:
                origin_shape = input_np.shape
                flattened_data = input_np.reshape((-1, input_np.shape[-1]))
                input_np = self.scaler1.transform(flattened_data).reshape(origin_shape)

        # 传入exog_futures,每个batch中对应的未来协变量
        if exog_futures is not None:
            exog_future = torch.tensor(
                exog_futures[i * batch_size: (i + 1) * batch_size, -horizon:, :]
            ).to(device)
        else:
            exog_future = None

        if self.config.norm and exog_dim > 0:
            flattened_data = exog_future.reshape((-1, exog_future.shape[-1]))
            flattened_data_np = flattened_data.cpu().numpy()
            exog_future = self.scaler2.transform(flattened_data_np).reshape(exog_future.shape)
            exog_future = torch.tensor(exog_future).to(device)
        input_index = input_data["time_stamps"]
        padding_len = (
            math.ceil(horizon / self.config.horizon) + 1
        ) * self.config.horizon
        all_mark = self._padding_time_stamp_mark(input_index, padding_len)

        answers = self._perform_rolling_predictions(horizon, input_np, exog_future, series_dim, all_mark, device)

        if self.config.norm:
            flattened_data = answers.reshape((-1, answers.shape[-1]))
            answers = self.scaler1.inverse_transform(flattened_data).reshape(
                answers.shape
            )

        return answers[..., :series_dim]

    def _perform_rolling_predictions(
        self,
        horizon: int,
        input_np: np.ndarray,
        exog_future: torch.Tensor,
        series_dim: int,
        all_mark: np.ndarray,
        device: torch.device,
    ) -> list:
        """
        Perform rolling predictions using the given input data and marks.

        :param horizon: Length of predictions to be made.
        :param input_np: Numpy array of input data.
        :param exog_future: Future exogenous data used for prediction.
        :param series_dim: Dimension of the series data.
        :param all_mark: Numpy array of all marks (time stamps mark).
        :param device: Device to run the model on.
        :return: List of predicted results for each prediction batch.
        """
        rolling_time = 0
        input_np, target_np, input_mark_np, target_mark_np = self._get_rolling_data(
            input_np, None, all_mark, rolling_time
        )
        if exog_future is not None:
            rolling_time_sum = horizon // self.config.horizon + 1
            need_horizon = rolling_time_sum * self.config.horizon - horizon
            exog_future = torch.cat((exog_future, torch.zeros((exog_future.shape[0], need_horizon, exog_future.shape[-1])).to(device)), dim=1)
            exog_future = exog_future.float()
        with torch.no_grad():
            answers = []
            while not answers or sum(a.shape[1] for a in answers) < horizon:
                input, dec_input, input_mark, target_mark = (
                    torch.tensor(input_np, dtype=torch.float32).to(device),
                    torch.tensor(target_np, dtype=torch.float32).to(device),
                    torch.tensor(input_mark_np, dtype=torch.float32).to(device),
                    torch.tensor(target_mark_np, dtype=torch.float32).to(device),
                )
                if exog_future is not None:
                    exog_future1 = exog_future[:,rolling_time*self.config.horizon:(rolling_time+1)*self.config.horizon,:]
                else:
                    exog_future1 = None
                out_loss = self._process(input, dec_input, input_mark, target_mark, exog_future1)
                output = out_loss["output"]
                if self.CovariateFusion is not None and exog_future is not None:
                    output1 = output[:, -self.config.horizon:,:series_dim]
                    output1 = self.CovariateFusion(exog_future1, output1)
                else:
                    output1 = output[:, -self.config.horizon:,:series_dim]
                column_num = output.shape[-1]
                real_batch_size = output.shape[0]
                output = torch.cat([output1, output[:, -self.config.horizon:, series_dim:]], dim=-1)
                answer = (
                    output.cpu()
                    .numpy()
                    .reshape(real_batch_size, -1, column_num)[
                        :, -self.config.horizon :, :
                    ]
                )
                answers.append(answer)
                if sum(a.shape[1] for a in answers) >= horizon:
                    break
                rolling_time += 1
                output = output.cpu().numpy()[:, -self.config.horizon :, :]
                (
                    input_np,
                    target_np,
                    input_mark_np,
                    target_mark_np,
                ) = self._get_rolling_data(input_np, output, all_mark, rolling_time)

        answers = np.concatenate(answers, axis=1)
        return answers[:, -horizon:, :]

    def _get_rolling_data(
        self,
        input_np: np.ndarray,
        output: Optional[np.ndarray],
        all_mark: np.ndarray,
        rolling_time: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare rolling data based on the current rolling time.

        :param input_np: Current input data.
        :param output: Output from the model prediction.
        :param all_mark: Numpy array of all marks (time stamps mark).
        :param rolling_time: Current rolling time step.
        :return: Updated input data, target data, input marks, and target marks for rolling prediction.
        """
        if rolling_time > 0:
            input_np = np.concatenate((input_np, output), axis=1)
            input_np = input_np[:, -self.config.seq_len :, :]
        target_np = np.zeros(
            (
                input_np.shape[0],
                self.config.label_len + self.config.horizon,
                input_np.shape[2],
            )
        )
        target_np[:, : self.config.label_len, :] = input_np[
            :, -self.config.label_len :, :
        ]
        advance_len = rolling_time * self.config.horizon
        input_mark_np = all_mark[:, advance_len : self.config.seq_len + advance_len, :]
        start = self.config.seq_len - self.config.label_len + advance_len
        end = self.config.seq_len + self.config.horizon + advance_len
        target_mark_np = all_mark[
            :,
            start:end,
            :,
        ]
        return input_np, target_np, input_mark_np, target_mark_np
