from typing import Optional, Tuple

import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from torch import optim
from torch.optim import lr_scheduler
from einops import rearrange

from ts_benchmark.baselines.pathformer.models.pathformer_model import PathformerModel
from ts_benchmark.baselines.utils import (
    forecasting_data_provider,
    train_val_split,
    get_time_mark,
)
from ts_benchmark.utils.data_processing import split_time
from .utils.tools import EarlyStopping, adjust_learning_rate
from ...models.model_base import ModelBase, BatchMaker

DEFAULT_HYPER_PARAMS = {
    "k": 2,
    "enc_in": 1,
    "dec_in": 1,
    "c_out": 1,
    "e_layers": 1,
    "d_layers": 1,
    "d_model": 4,
    "d_ff": 64,
    "embed": "timeF",
    "freq": "h",
    "lradj": "TST",
    "moving_avg": 25,
    "num_kernels": 6,
    "factor": 1,
    "n_heads": 8,
    "seg_len": 6,
    "win_size": 2,
    "activation": "gelu",
    "output_attention": 0,
    "patch_len": 16,
    "stride": 8,
    "dropout": 0.1,
    "batch_size": 512,
    "learning_rate": 0.0001,
    "train_epochs": 30,
    "num_workers": 0,
    "loss": "MAE",
    "itr": 1,
    "distil": True,
    "patience": 5,
    "p_hidden_dims": [128, 128],
    "p_hidden_layers": 2,
    "mem_dim": 32,
    "conv_kernel": [12, 16],
    "individual": False,
    "num_nodes": 21,
    "layer_nums": 3,
    "num_experts_list": [4, 4, 4],
    "patch_size_list": [[56, 28, 12, 24], [42, 28, 16, 21], [56, 16, 28, 42]],
    "revin": 1,
    "drop": 0.1,
    "pct_start": 0.4,
    "residual_connection": 0,
    "gpu": 0,
    "seq_len": 336,
    "batch_norm": 0,
    "use_mlp": False,  # New parameter
}


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size1, output_size):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size1)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size1, output_size)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


class PathformerConfig:
    def __init__(self, **kwargs):
        for key, value in DEFAULT_HYPER_PARAMS.items():
            setattr(self, key, value)

        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def pred_len(self):
        return self.horizon


class Pathformer(ModelBase):
    def __init__(self, **kwargs):
        super(Pathformer, self).__init__()
        self.config = PathformerConfig(**kwargs)
        self.scaler1 = StandardScaler()  # For series data
        self.scaler2 = StandardScaler()  # For exog data
        self.seq_len = self.config.seq_len
        self.win_size = self.config.seq_len
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    @property
    def model_name(self):
        return "Pathformer"

    def multi_forecasting_hyper_param_tune(self, train_data: np.ndarray):
        self.config.freq = "h"  # Simplified frequency setting
        column_num = train_data.shape[1]
        self.config.enc_in = column_num
        self.config.dec_in = column_num
        self.config.c_out = column_num

        if self.model_name == "MICN":
            setattr(self.config, "label_len", self.config.seq_len)
        else:
            setattr(self.config, "label_len", self.config.seq_len // 2)

    def single_forecasting_hyper_param_tune(self, train_data: np.ndarray):
        # When using numpy arrays, frequency inference is not directly available
        # Use default frequency or require it to be passed in config
        self.config.freq = getattr(self.config, "freq", "h")

        column_num = train_data.shape[1]
        self.config.enc_in = column_num
        self.config.dec_in = column_num
        self.config.c_out = column_num

        setattr(self.config, "label_len", self.config.horizon)

    def detect_hyper_param_tune(self, train_data: np.ndarray):
        # When using numpy arrays, frequency inference is not directly available
        # Use default frequency or require it to be passed in config
        self.config.freq = getattr(self.config, "freq", "h")

        column_num = train_data.shape[1]
        self.config.enc_in = column_num
        self.config.dec_in = column_num
        self.config.c_out = column_num
        self.config.label_len = 48

    def validate(
            self, valid_data_loader: DataLoader, series_dim: int, criterion: torch.nn.Module
    ) -> float:
        """
        Validates the model performance on the provided validation dataset.
        :param valid_data_loader: A PyTorch DataLoader for the validation dataset.
        :param series_dim : The number of series data's dimensions.
        :param criterion : The loss function to compute the loss between model predictions and ground truth.
        :returns:The mean loss computed over the validation dataset.
        """
        config = self.config
        total_loss = []
        self.model.eval()
        if self.MLP is not None:
            self.MLP.eval()

        for input, target in valid_data_loader:
            input, target = (
                input.to(self.device),
                target.to(self.device),
            )
            # decoder input
            dec_input = torch.zeros_like(target[:, -config.horizon:, :]).float()
            dec_input = (
                torch.cat([target[:, : config.label_len, :], dec_input], dim=1)
                .float()
                .to(self.device)
            )

            exog_future = target[:, -config.horizon:, series_dim:].to(self.device)
            output, balance_loss = self.model(input)

            if self.config.use_mlp and self.MLP is not None:
                transformer_output = output[:, -config.horizon:, :series_dim]
                output = self.MLP(torch.cat((transformer_output, exog_future), dim=-1))
            else:
                output = output[:, -config.horizon:, :series_dim]

            target = target[:, -config.horizon:, :series_dim]
            loss = criterion(output, target).detach().cpu().numpy()
            total_loss.append(loss)

        total_loss = np.mean(total_loss)
        self.model.train()
        if self.MLP is not None:
            self.MLP.train()
        return total_loss

    def forecast_fit(
            self,
            train_valid_data: np.ndarray,
            *,
            covariates: Optional[dict] = None,
            train_ratio_in_tv: float = 1.0,
            **kwargs,
    ) -> "ModelBase":
        """
        Train the model.
        :param train_valid_data: Time series data used for training and validation as numpy array.
        :param covariates: Additional external variables.
        :param train_ratio_in_tv: Represents the splitting ratio of the training set validation set. If it is equal to 1, it means that the validation set is not partitioned.
        :return: The fitted model object.
        """
        if covariates is None:
            covariates = {}
        series_dim = train_valid_data.shape[-2]
        exog_data = covariates.get("exog", None)
        if exog_data is not None:
            train_valid_data = np.concatenate((train_valid_data, exog_data), axis=1)
            exog_dim = exog_data.shape[-2]
        else:
            exog_dim = 0

        if train_valid_data.shape[1] == 1:
            train_drop_last = False
            self.single_forecasting_hyper_param_tune(train_valid_data)
        else:
            train_drop_last = True
            self.multi_forecasting_hyper_param_tune(train_valid_data)

        setattr(self.config, "task_name", "short_term_forecast")
        self.model = PathformerModel(self.config)

        # Initialize MLP if enabled
        if self.config.use_mlp:
            input_size = series_dim + exog_dim
            output_size = series_dim
            self.MLP = MLP(input_size=input_size, hidden_size1=2048, output_size=output_size)
            self.MLP.to(self.device)
        else:
            self.MLP = None

        print(
            "----------------------------------------------------------",
            self.model_name,
        )
        config = self.config
        train_data, valid_data = train_val_split(
            train_valid_data, train_ratio_in_tv, config.seq_len
        )
        train_data_l = train_data.shape[0]
        valid_data_l = valid_data.shape[0]

        # Fit scalers based on whether we have exog data
        if exog_dim > 0:
            # Fit scaler1 for series data
            self.scaler1.fit(rearrange(train_data[:, :series_dim, :], 'l c n->(l n) c'))
            # Fit scaler2 for exog data
            self.scaler2.fit(rearrange(train_data[:, series_dim:, :], 'l c n->(l n) c'))

            if config.norm:
                # Scale series data
                scaled_series = self.scaler1.transform(rearrange(train_data[:, :series_dim, :], 'l c n->(l n) c'))
                train_series = rearrange(scaled_series, '(l n) c -> l c n', l=train_data_l)

                # Scale exog data
                scaled_exog = self.scaler2.transform(rearrange(train_data[:, series_dim:, :], 'l c n->(l n) c'))
                train_exog = rearrange(scaled_exog, '(l n) c -> l c n', l=train_data_l)

                # Concatenate scaled data
                train_data = np.concatenate([train_series, train_exog], axis=1)
        else:
            # Only series data, use scaler1
            self.scaler1.fit(rearrange(train_data, 'l c n->(l n) c'))
            if config.norm:
                scaled_data = self.scaler1.transform(rearrange(train_data, 'l c n->(l n) c'))
                train_data = rearrange(scaled_data, '(l n) c -> l c n', l=train_data_l)

        if train_ratio_in_tv != 1:
            if config.norm:
                if exog_dim > 0:
                    # Scale validation series data
                    scaled_series = self.scaler1.transform(rearrange(valid_data[:, :series_dim, :], 'l c n->(l n) c'))
                    valid_series = rearrange(scaled_series, '(l n) c -> l c n', l=valid_data_l)

                    # Scale validation exog data
                    scaled_exog = self.scaler2.transform(rearrange(valid_data[:, series_dim:, :], 'l c n->(l n) c'))
                    valid_exog = rearrange(scaled_exog, '(l n) c -> l c n', l=valid_data_l)

                    # Concatenate scaled data
                    valid_data = np.concatenate([valid_series, valid_exog], axis=1)
                else:
                    scaled_data = self.scaler1.transform(rearrange(valid_data, 'l c n->(l n) c'))
                    valid_data = rearrange(scaled_data, '(l n) c -> l c n', l=valid_data_l)
            valid_dataset, valid_data_loader = forecasting_data_provider(
                valid_data,
                config,
                timeenc=1,
                batch_size=config.batch_size,
                shuffle=True,
                drop_last=False,
            )

        train_dataset, train_data_loader = forecasting_data_provider(
            train_data,
            config,
            timeenc=1,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=train_drop_last,
        )

        # Define the loss function and optimizer
        if config.loss == "MSE":
            criterion = nn.MSELoss()
        elif config.loss == "MAE":
            criterion = nn.L1Loss()
        else:
            criterion = nn.HuberLoss(delta=0.5)

        # Configure optimizer based on whether MLP is used
        if self.MLP is not None:
            optimizer = optim.Adam([
                {'params': self.model.parameters(), 'lr': config.learning_rate},
                {'params': self.MLP.parameters(), 'lr': config.learning_rate * 0.1}
            ])
        else:
            optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)

        self.early_stopping = EarlyStopping(patience=config.patience)
        self.model.to(self.device)

        total_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        if self.MLP is not None:
            total_params += sum(p.numel() for p in self.MLP.parameters() if p.requires_grad)

        print(f"Total trainable parameters: {total_params}")
        train_steps = len(train_data_loader)
        scheduler = lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            steps_per_epoch=train_steps,
            pct_start=config.pct_start,
            epochs=config.train_epochs,
            max_lr=config.learning_rate,
        )

        for epoch in range(config.train_epochs):
            self.model.train()
            if self.MLP is not None:
                self.MLP.train()

            for i, (input, target) in enumerate(train_data_loader):
                optimizer.zero_grad()
                input, target = (
                    input.to(self.device),
                    target.to(self.device),
                )
                # decoder input
                dec_input = torch.zeros_like(target[:, -config.horizon:, :]).float()
                dec_input = (
                    torch.cat([target[:, : config.label_len, :], dec_input], dim=1)
                    .float()
                    .to(self.device)
                )

                exog_future = target[:, -config.horizon:, series_dim:].to(self.device)
                output, balance_loss = self.model(input)

                if self.config.use_mlp and self.MLP is not None:
                    transformer_output = output[:, -config.horizon:, :series_dim]
                    output = self.MLP(torch.cat((transformer_output, exog_future), dim=-1))
                else:
                    output = output[:, -config.horizon:, :series_dim]

                target = target[:, -config.horizon:, :series_dim]
                loss = criterion(output, target)

                loss.backward()
                optimizer.step()

                if config.lradj == "TST":
                    adjust_learning_rate(
                        optimizer, scheduler, epoch + 1, config, printout=False
                    )
                    scheduler.step()

            if train_ratio_in_tv != 1:
                valid_loss = self.validate(valid_data_loader, series_dim, criterion)
                if self.MLP is not None:
                    self.early_stopping(valid_loss, {'transformer': self.model, 'mlp': self.MLP})
                else:
                    self.early_stopping(valid_loss, {'transformer': self.model})
                if self.early_stopping.early_stop:
                    break
            if config.lradj != "TST":
                adjust_learning_rate(optimizer, scheduler, epoch + 1, config)

    def forecast(
            self,
            horizon: int,
            series: np.ndarray,
            *,
            covariates: Optional[dict] = None,
    ) -> np.ndarray:
        """
        Make predictions.
        :param horizon: The predicted length.
        :param series: Time series data used for prediction as numpy array.
        :param covariates: Additional external variables
        :return: An array of predicted results.
        """
        if covariates is None:
            covariates = {}
        series_dim = series.shape[-1]
        exog_data = covariates.get("exog", None)
        if exog_data is not None:
            series = np.concatenate([series, exog_data], axis=1)
            if (
                    hasattr(self.config, "output_chunk_length")
                    and horizon != self.config.output_chunk_length
            ):
                raise ValueError(
                    f"Error: 'exog' is enabled during training, but horizon ({horizon}) != output_chunk_length ({self.config.output_chunk_length}) during forecast."
                )

        if self.early_stopping.check_point is not None:
            if isinstance(self.early_stopping.check_point, dict):
                self.model.load_state_dict(self.early_stopping.check_point['transformer'])
                if self.MLP is not None and 'mlp' in self.early_stopping.check_point:
                    self.MLP.load_state_dict(self.early_stopping.check_point['mlp'])
            else:
                # Backward compatibility
                self.model.load_state_dict(self.early_stopping.check_point)

        if self.config.norm:
            if exog_data is not None:
                # Scale series data with scaler1
                series_values = series[:, :series_dim]
                scaled_series = self.scaler1.transform(series_values)

                # Scale exog data with scaler2
                exog_values = series[:, series_dim:]
                scaled_exog = self.scaler2.transform(exog_values)

                # Combine scaled data
                series = np.concatenate([scaled_series, scaled_exog], axis=1)
            else:
                series = self.scaler1.transform(series)

        if self.model is None:
            raise ValueError("Model not trained. Call the fit() function first.")

        config = self.config
        # For numpy arrays, we need to handle the time splitting differently
        # Assuming series has shape (time_steps, features)
        test = series[-config.seq_len:, :]

        # Pad test data for forecast
        test = self._padding_data_for_forecast_numpy(test)

        test_data_set, test_data_loader = forecasting_data_provider(
            test, config, timeenc=1, batch_size=1, shuffle=False, drop_last=False
        )

        self.model.to(self.device)
        self.model.eval()
        if self.MLP is not None:
            self.MLP.to(self.device)
            self.MLP.eval()

        with torch.no_grad():
            answer = None
            while answer is None or answer.shape[0] < horizon:
                for input, target in test_data_loader:
                    input, target = (
                        input.to(self.device),
                        target.to(self.device),
                    )
                    dec_input = torch.zeros_like(
                        target[:, -config.horizon:, :]
                    ).float()
                    dec_input = (
                        torch.cat([target[:, : config.label_len, :], dec_input], dim=1)
                        .float()
                        .to(self.device)
                    )

                    exog_future = target[:, -config.horizon:, series_dim:]
                    output, balance_loss = self.model(input)

                    if self.config.use_mlp and self.MLP is not None:
                        transformer_output = output[:, -config.horizon:, :series_dim]
                        output = self.MLP(torch.cat((transformer_output, exog_future), dim=-1))
                    else:
                        output = output[:, -config.horizon:, :series_dim]

                column_num = output.shape[-1]
                temp = output.cpu().numpy().reshape(-1, column_num)[-config.horizon:]

                if answer is None:
                    answer = temp
                else:
                    answer = np.concatenate([answer, temp], axis=0)

                if answer.shape[0] >= horizon:
                    if self.config.norm:
                        # Only inverse transform series data with scaler1
                        answer[-horizon:] = self.scaler1.inverse_transform(
                            answer[-horizon:]
                        )
                    return answer[-horizon:, :series_dim]

                output = output.cpu().numpy()[:, -config.horizon:]
                for i in range(config.horizon):
                    test[i + config.seq_len] = output[0, i, :]

                test = test[config.horizon:]
                test = self._padding_data_for_forecast_numpy(test)

                test_data_set, test_data_loader = forecasting_data_provider(
                    test,
                    config,
                    timeenc=1,
                    batch_size=1,
                    shuffle=False,
                    drop_last=False,
                )

    def batch_forecast(
            self, horizon: int, batch_maker: BatchMaker, **kwargs
    ) -> np.ndarray:
        """
        Make predictions by batch.

        :param horizon: The length of each prediction.
        :param batch_maker: Make batch data used for prediction.
        :return: An array of predicted results.
        """
        if self.early_stopping.check_point is not None:
            if isinstance(self.early_stopping.check_point, dict):
                self.model.load_state_dict(self.early_stopping.check_point['transformer'])
                if self.MLP is not None and 'mlp' in self.early_stopping.check_point:
                    self.MLP.load_state_dict(self.early_stopping.check_point['mlp'])
            else:
                # Backward compatibility
                self.model.load_state_dict(self.early_stopping.check_point)

        if self.model is None:
            raise ValueError("Model not trained. Call the fit() function first.")

        self.model.to(self.device)
        self.model.eval()
        if self.MLP is not None:
            self.MLP.to(self.device)
            self.MLP.eval()

        input_data = batch_maker.make_batch(self.config.batch_size, self.config.seq_len)
        input_np = input_data["input"]
        real_batch_size = self.config.batch_size * input_np.shape[3]
        series_dim = input_np.shape[-2]

        if input_data["covariates"] is None:
            covariates = {}
        else:
            covariates = input_data["covariates"]
        exog_data = covariates.get("exog")
        exog_dim = 0

        if exog_data is not None:
            exog_dim = exog_data.shape[-2]
            input_np = np.concatenate((input_np, exog_data), axis=2)
            if (
                    hasattr(self.config, "output_chunk_length")
                    and horizon != self.config.output_chunk_length
            ):
                raise ValueError(
                    f"Error: 'exog' is enabled during training, but horizon ({horizon}) != output_chunk_length ({self.config.output_chunk_length}) during forecast."
                )

        # Reshape input to match expected format
        input_np = rearrange(input_np, 'b l c n -> (b n) l c')
        input_np_b = input_np.shape[0]

        if self.config.norm:
            if exog_dim > 0:
                # Scale series data with scaler1
                series_data = input_np[:, :, :series_dim]
                scaled_series = self.scaler1.transform(rearrange(series_data, 'b l c->(b l) c'))
                scaled_series = rearrange(scaled_series, '(b l) c -> b l c', b=input_np_b)

                # Scale exog data with scaler2
                exog_data = input_np[:, :, series_dim:]
                scaled_exog = self.scaler2.transform(rearrange(exog_data, 'b l c->(b l) c'))
                scaled_exog = rearrange(scaled_exog, '(b l) c -> b l c', b=input_np_b)

                # Combine scaled data
                input_np = np.concatenate([scaled_series, scaled_exog], axis=2)
            else:
                scaled_data = self.scaler1.transform(rearrange(input_np, 'b l c->(b l) c'))
                input_np = rearrange(scaled_data, '(b l) c -> b l c', b=input_np_b)

        # Get exog future data if available
        exog_future = None
        if 'exog_futures' in kwargs and exog_dim > 0:
            exog_futures = kwargs['exog_futures']
            i = kwargs.get('i', 0)
            exog_future = torch.tensor(
                exog_futures[i * real_batch_size: (i + 1) * real_batch_size, -horizon:, :]
            ).to(self.device)

            if self.config.norm:
                exog_future_np = exog_future.cpu().numpy()
                exog_future_b = exog_future_np.shape[0]
                scaled_exog_future = self.scaler2.transform(rearrange(exog_future_np, 'b l c->(b l) c'))
                scaled_exog_future = rearrange(scaled_exog_future, '(b l) c -> b l c', b=exog_future_b)
                exog_future = torch.tensor(scaled_exog_future).to(self.device)

        # Simplified rolling predictions
        answers = self._perform_rolling_predictions(horizon, input_np, exog_future, series_dim)

        if self.config.norm:
            # Only inverse transform series data with scaler1
            answers_b = answers.shape[0]
            scaled_data = self.scaler1.inverse_transform(rearrange(answers, 'b l c->(b l) c'))
            answers = rearrange(scaled_data, '(b l) c -> b l c', b=answers_b)

        return answers[..., :series_dim]

    def _perform_rolling_predictions(
            self,
            horizon: int,
            input_np: np.ndarray,
            exog_future: Optional[torch.Tensor],
            series_dim: int,
    ) -> np.ndarray:
        """
        Perform rolling predictions using the given input data.

        :param horizon: Length of predictions to be made.
        :param input_np: Numpy array of input data.
        :param exog_future: Optional future exogenous variables.
        :param series_dim: Number of series dimensions.
        :return: Array of predicted results.
        """
        rolling_time = 0
        with torch.no_grad():
            answers = []
            while not answers or sum(a.shape[1] for a in answers) < horizon:
                input = torch.tensor(input_np, dtype=torch.float32).to(self.device)
                output, balance_loss = self.model(input)

                if self.config.use_mlp and self.MLP is not None and exog_future is not None:
                    transformer_output = output[:, -self.config.horizon:, :series_dim]
                    current_exog = exog_future[:,
                                   rolling_time * self.config.horizon:(rolling_time + 1) * self.config.horizon, :]
                    output = self.MLP(torch.cat((transformer_output, current_exog), dim=-1))
                else:
                    output = output[:, -self.config.horizon:, :series_dim]

                column_num = output.shape[-1]
                real_batch_size = output.shape[0]
                answer = (
                    output.cpu()
                    .numpy()
                    .reshape(real_batch_size, -1, column_num)[:, -self.config.horizon:, :]
                )
                answers.append(answer)

                if sum(a.shape[1] for a in answers) >= horizon:
                    break

                rolling_time += 1
                output_np = output.cpu().numpy()[:, -self.config.horizon:, :]

                # Pad output to match input dimensions if needed
                if output_np.shape[-1] < input_np.shape[-1]:
                    padding_size = input_np.shape[-1] - output_np.shape[-1]
                    padding = np.zeros((output_np.shape[0], output_np.shape[1], padding_size))
                    output_np = np.concatenate((output_np, padding), axis=-1)

                input_np = np.concatenate((input_np, output_np), axis=1)
                input_np = input_np[:, -self.config.seq_len:, :]

        answers = np.concatenate(answers, axis=1)
        return answers[:, -horizon:, :]

    def _padding_data_for_forecast_numpy(self, test: np.ndarray) -> np.ndarray:
        """
        Pad numpy array data for forecast.

        :param test: Numpy array to pad
        :return: Padded numpy array
        """
        # Create padding of zeros
        padding = np.zeros((self.config.horizon, test.shape[1]))
        # Concatenate the original data with padding
        return np.concatenate([test, padding], axis=0)