from typing import Optional, Tuple

import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch import optim
from torch.utils.data import DataLoader
from einops import rearrange

from ts_benchmark.baselines.patchmlp.models.patchmlp_model import PatchMLPModel
from ts_benchmark.baselines.patchmlp.utils.tools import (
    EarlyStopping,
    adjust_learning_rate,
)
from ts_benchmark.baselines.utils import (
    forecasting_data_provider,
    train_val_split,
    get_time_mark,
)
from ts_benchmark.utils.data_processing import split_time
from ...models.model_base import ModelBase, BatchMaker

DEFAULT_HYPER_PARAMS = {
    "data": "custom",
    "use_amp": 0,
    "label_len": 48,
    "enc_in": 21,
    "dec_in": 21,
    "c_out": 21,
    "e_layers": 1,
    "d_layers": 1,
    "d_model": 1024,
    "d_ff": 2048,
    "freq": "h",
    "factor": 1,
    "distil": 1,
    "embed": "timeF",
    "n_heads": 8,
    "seq_len": 96,
    "activation": "gelu",
    "output_attention": 0,
    "dropout": 0.1,
    "moving_avg": 13,
    "batch_size": 32,
    "lradj": "type1",
    "lr": 0.0001,
    "num_epochs": 100,
    "num_workers": 0,
    "loss": "MSE",
    "patience": 5,
    "use_mlp": False,
    "task_name": "short_term_forecast",
    "parallel_strategy": "DP",
    "norm": True,
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


class PatchMLPConfig:
    def __init__(self, **kwargs):
        for key, value in DEFAULT_HYPER_PARAMS.items():
            setattr(self, key, value)

        for key, value in kwargs.items():
            setattr(self, key, value)

        if self.parallel_strategy not in [None, "DP"]:
            raise ValueError(
                "Invalid value for parallel_strategy. Supported values are 'DP' and None."
            )

    @property
    def pred_len(self):
        return self.horizon


class PatchMLP(ModelBase):
    def __init__(self, **kwargs):
        super(PatchMLP, self).__init__()
        self.config = PatchMLPConfig(**kwargs)
        self.scaler1 = StandardScaler()
        self.scaler2 = StandardScaler()
        self.seq_len = self.config.seq_len
        self.win_size = self.config.seq_len
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def model_name(self):
        return "PatchMLP"

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

    def multi_forecasting_hyper_param_tune(self, train_data: np.ndarray):
        self.config.freq = "h"
        column_num = train_data.shape[1]
        self.config.enc_in = column_num
        self.config.dec_in = column_num
        self.config.c_out = column_num

        if self.model_name == "MICN":
            setattr(self.config, "label_len", self.config.seq_len)
        else:
            setattr(self.config, "label_len", self.config.seq_len // 2)

    def single_forecasting_hyper_param_tune(self, train_data: np.ndarray):
        # When using numpy array, we cannot infer frequency automatically
        # Default to 'h' or it should be passed in config
        self.config.freq = getattr(self.config, 'freq', 'h')

        column_num = train_data.shape[1]
        self.config.enc_in = column_num
        self.config.dec_in = column_num
        self.config.c_out = column_num

        setattr(self.config, "label_len", self.config.horizon)

    def detect_hyper_param_tune(self, train_data: np.ndarray):
        # When using numpy array, we cannot infer frequency automatically
        # Default to 'h' or it should be passed in config
        self.config.freq = getattr(self.config, 'freq', 'h')

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
        with torch.no_grad():
            for input, target in valid_data_loader:
                input, target = input.to(self.device), target.to(self.device)
                exog_future = target[:, -config.horizon:, series_dim:].to(self.device)

                # For PatchMLP, we pass input and get output
                output, _ = self.model(input)

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
        :param train_valid_data: Time series data used for training and validation (numpy array).
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
        self.model = PatchMLPModel(self.config)

        # Initialize MLP if enabled
        if self.config.use_mlp:
            input_size = series_dim + exog_dim
            output_size = series_dim
            self.MLP = MLP(input_size=input_size, hidden_size1=2048, output_size=output_size)
            self.MLP.to(self.device)
        else:
            self.MLP = None

        # Handle multi-GPU if available
        device_ids = np.arange(torch.cuda.device_count()).tolist()
        if len(device_ids) > 1 and self.config.parallel_strategy == "DP":
            self.model = nn.DataParallel(self.model, device_ids=device_ids)
            if self.MLP is not None:
                self.MLP = nn.DataParallel(self.MLP, device_ids=device_ids)

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

        # Fit scalers separately for series and exog data
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
        criterion = nn.MSELoss()

        # Mixed optimizer for MLP support
        if self.MLP is not None:
            optimizer = optim.Adam([
                {'params': self.model.parameters(), 'lr': config.lr},
                {'params': self.MLP.parameters(), 'lr': config.lr * 0.1}
            ])
        else:
            optimizer = optim.Adam(self.model.parameters(), lr=config.lr)

        self.early_stopping = EarlyStopping(patience=config.patience)
        self.model.to(self.device)
        if self.MLP is not None:
            self.MLP.to(self.device)

        total_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        if self.MLP is not None:
            total_params += sum(p.numel() for p in self.MLP.parameters() if p.requires_grad)

        print(f"Total trainable parameters: {total_params}")

        for epoch in range(config.num_epochs):
            self.model.train()
            if self.MLP is not None:
                self.MLP.train()

            # Training loop - Note: only unpacking 2 values (input, target)
            for i, (input, target) in enumerate(train_data_loader):
                optimizer.zero_grad()
                input, target = input.to(self.device), target.to(self.device)

                # Prepare exog_future
                exog_future = target[:, -config.horizon:, series_dim:].to(self.device)

                # Forward pass - PatchMLP only takes input
                output, _ = self.model(input)

                if self.config.use_mlp and self.MLP is not None:
                    transformer_output = output[:, -config.horizon:, :series_dim]
                    output = self.MLP(torch.cat((transformer_output, exog_future), dim=-1))
                else:
                    output = output[:, -config.horizon:, :series_dim]

                target = target[:, -config.horizon:, :series_dim]
                loss = criterion(output, target)

                loss.backward()
                optimizer.step()

            if train_ratio_in_tv != 1:
                valid_loss = self.validate(valid_data_loader, series_dim, criterion)
                if self.MLP is not None:
                    self.early_stopping(valid_loss, {'patchmlp': self.model, 'mlp': self.MLP})
                else:
                    self.early_stopping(valid_loss, {'patchmlp': self.model})
                if self.early_stopping.early_stop:
                    break

            adjust_learning_rate(optimizer, epoch + 1, config)

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
        :param series: Time series data used for prediction (numpy array).
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
            self.model.load_state_dict(self.early_stopping.check_point['patchmlp'])
            if self.MLP is not None and 'mlp' in self.early_stopping.check_point:
                self.MLP.load_state_dict(self.early_stopping.check_point['mlp'])

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
        # Split series for test
        test = series[-config.seq_len:]

        # Perform rolling predictions
        self.model.to(self.device)
        self.model.eval()
        if self.MLP is not None:
            self.MLP.to(self.device)
            self.MLP.eval()

        with torch.no_grad():
            answers = []
            current_input = test.copy()

            while not answers or sum(len(a) for a in answers) < horizon:
                input_tensor = torch.tensor(current_input, dtype=torch.float32).unsqueeze(0).to(self.device)

                output, _ = self.model(input_tensor)

                if self.config.use_mlp and self.MLP is not None and exog_data is not None:
                    transformer_output = output[:, -config.horizon:, :series_dim]
                    # Note: This assumes exog_future is available, which might need to be passed
                    # For now, using zeros as placeholder
                    exog_future = torch.zeros(1, config.horizon, exog_data.shape[-1]).to(self.device)
                    output = self.MLP(torch.cat((transformer_output, exog_future), dim=-1))
                else:
                    output = output[:, -config.horizon:, :series_dim]

                column_num = output.shape[-1]
                temp = output.cpu().numpy().reshape(-1, column_num)[-config.horizon:]
                answers.append(temp)

                if sum(len(a) for a in answers) >= horizon:
                    break

                # Update input for next iteration
                output_np = output.cpu().numpy()[0, -config.horizon:]
                current_input = np.concatenate([current_input[config.horizon:], output_np], axis=0)

        answers = np.concatenate(answers, axis=0)
        if self.config.norm:
            # Only inverse transform series data with scaler1
            answers = self.scaler1.inverse_transform(answers)

        return answers[:horizon, :series_dim]

    def batch_forecast(
            self, horizon: int, batch_maker: BatchMaker, exog_futures, i, **kwargs
    ) -> np.ndarray:
        """
        Make predictions by batch.

        :param horizon: The length of each prediction.
        :param batch_maker: Make batch data used for prediction.
        :param exog_futures: Future exogenous variables.
        :param i: Batch index.
        :return: An array of predicted results.
        """
        if self.early_stopping.check_point is not None:
            self.model.load_state_dict(self.early_stopping.check_point['patchmlp'])
            if self.MLP is not None and 'mlp' in self.early_stopping.check_point:
                self.MLP.load_state_dict(self.early_stopping.check_point['mlp'])

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
        else:
            exog_dim = 0

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

        exog_future = torch.tensor(exog_futures[i * real_batch_size: (i + 1) * real_batch_size, -horizon:, :]).to(
            self.device)

        if self.config.norm and exog_dim > 0:
            exog_future_np = exog_future.cpu().numpy()
            exog_future_b = exog_future_np.shape[0]
            scaled_exog_future = self.scaler2.transform(rearrange(exog_future_np, 'b l c->(b l) c'))
            scaled_exog_future = rearrange(scaled_exog_future, '(b l) c -> b l c', b=exog_future_b)
            exog_future = torch.tensor(scaled_exog_future).to(self.device)

        answers = torch.tensor(self._perform_rolling_predictions(horizon, input_np, exog_future, series_dim))
        answers = answers[:, -horizon:, :series_dim].to(self.device)

        if self.config.norm:
            # Only inverse transform series data with scaler1
            answers_b = answers.shape[0]
            scaled_data = self.scaler1.inverse_transform(rearrange(answers.cpu().detach().numpy(), 'b l c->(b l) c'))
            answers = rearrange(scaled_data, '(b l) c -> b l c', b=answers_b)

        return answers

    def _perform_rolling_predictions(
            self,
            horizon: int,
            input_np: np.ndarray,
            exog_future: torch.Tensor,
            series_dim: int,
    ) -> list:
        """
        Perform simplified rolling predictions using the given input data.

        :param horizon: Length of predictions to be made.
        :param input_np: Numpy array of input data.
        :param exog_future: Future exogenous variables.
        :param series_dim: Number of series dimensions.
        :return: Array of predicted results.
        """
        rolling_time = 0
        answers = []
        with torch.no_grad():
            while not answers or sum(a.shape[1] for a in answers) < horizon:
                input = torch.tensor(input_np, dtype=torch.float32).to(self.device)

                output, _ = self.model(input)

                if self.config.use_mlp and self.MLP is not None:
                    output = torch.tensor(output[:, -horizon:, :series_dim]).to(self.device)
                    output = self.MLP(torch.cat((output.to(torch.float32), exog_future.to(torch.float32)), dim=-1))
                else:
                    output = output[:, -horizon:, :series_dim]

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
                output = output.cpu().numpy()[:, -self.config.horizon:, :]
                input_np, _ = self._get_rolling_data(input_np, output, rolling_time)
        answers = np.concatenate(answers, axis=1)
        return answers[:, -horizon:, :]

    def _get_rolling_data(
            self,
            input_np: np.ndarray,
            output: Optional[np.ndarray],
            rolling_time: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare rolling data based on the current rolling time.

        :param input_np: Current input data.
        :param output: Output from the model prediction.
        :param rolling_time: Current rolling time step.
        :return: Updated input data and target data for rolling prediction.
        """
        if rolling_time > 0 and output is not None:
            input_np = np.concatenate((input_np, output), axis=1)
            input_np = input_np[:, -self.config.seq_len:, :]
        target_np = np.zeros(
            (
                input_np.shape[0],
                self.config.label_len + self.config.horizon,
                input_np.shape[2],
            )
        )
        target_np[:, : self.config.label_len, :] = input_np[
                                                   :, -self.config.label_len:, :
                                                   ]
        return input_np, target_np