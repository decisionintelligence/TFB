import torch
from torch import nn
from typing import List
from ..utils.ElasTST_utils import weighted_average, TemporalScaler
from typing import Union


class Forecaster(nn.Module):
    def __init__(
        self,
        target_dim: int,
        context_length: Union[list, int],
        prediction_length: Union[list, int],
        freq: str,
        use_lags: bool = False,
        use_feat_idx_emb: bool = False,
        use_time_feat: bool = False,
        lags_list: List[int] = None,
        feat_idx_emb_dim: int = 1,
        time_feat_dim: int = 1,
        use_scaling: bool = False,
        autoregressive: bool = False,
        no_training: bool = False,
        dataset: str = None,
        **kwargs,
    ):
        super().__init__()

        self.context_length = context_length
        self.prediction_length = prediction_length

        if isinstance(self.context_length, list):
            self.max_context_length = max(self.context_length)
        else:
            self.max_context_length = self.context_length

        if isinstance(self.prediction_length, list):
            self.max_prediction_length = max(self.prediction_length)
        else:
            self.max_prediction_length = self.prediction_length

        self.target_dim = target_dim
        self.freq = freq
        self.use_lags = use_lags
        self.use_feat_idx_emb = use_feat_idx_emb
        self.use_time_feat = use_time_feat
        self.feat_idx_emb_dim = feat_idx_emb_dim
        self.time_feat_dim = time_feat_dim
        self.autoregressive = autoregressive
        self.no_training = no_training
        self.use_scaling = use_scaling
        self.dataset = dataset
        # Lag parameters
        self.lags_list = lags_list
        if self.use_scaling:
            self.scaler = TemporalScaler()
        else:
            self.scaler = None

        self.lags_dim = len(self.lags_list) * target_dim
        self.feat_idx_emb = nn.Embedding(
            num_embeddings=self.target_dim, embedding_dim=self.feat_idx_emb_dim
        )
        self.input_size = self.get_input_size()

    @property
    def name(self):
        return self.__class__.__name__

    def get_input_size(self):
        input_size = self.target_dim if not self.use_lags else self.lags_dim
        if self.use_feat_idx_emb:
            input_size += self.use_feat_idx_emb * self.target_dim
        if self.use_time_feat:
            input_size += self.time_feat_dim
        return input_size

    def get_lags(self, sequence, lags_list, lags_length=1):
        """
        Get several lags from the sequence of shape (B, L, C) to (B, L', C*N),
        where L' = lag_length and N = len(lag_list).
        """
        assert max(lags_list) + lags_length <= sequence.shape[1]

        lagged_values = []
        for lag_index in lags_list:
            begin_index = -lag_index - lags_length
            end_index = -lag_index if lag_index > 0 else None
            lagged_value = sequence[:, begin_index:end_index, ...]
            if self.use_scaling:
                lagged_value = lagged_value / self.scaler.scale
            lagged_values.append(lagged_value)
        return torch.cat(lagged_values, dim=-1)

    def get_input_sequence(self, past_target_cdf, future_target_cdf, mode):
        if mode == "all":
            sequence = torch.cat((past_target_cdf, future_target_cdf), dim=1)
            seq_length = self.max_context_length + self.max_prediction_length
        elif mode == "encode":
            sequence = past_target_cdf
            seq_length = self.max_context_length
        elif mode == "decode":
            sequence = past_target_cdf
            seq_length = 1
        else:
            raise ValueError(f"Unsupported input mode: {mode}")

        if self.use_lags:
            input_seq = self.get_lags(sequence, self.lags_list, seq_length)
        else:
            input_seq = sequence[:, -seq_length:, ...]
            if self.use_scaling:
                input_seq = input_seq / self.scaler.scale
        return input_seq

    def get_input_feat_idx_emb(self, target_dimension_indicator, input_length):
        input_feat_idx_emb = self.feat_idx_emb(target_dimension_indicator)  # [B K D]

        input_feat_idx_emb = (
            input_feat_idx_emb.unsqueeze(1)
            .expand(-1, input_length, -1, -1)
            .reshape(-1, input_length, self.target_dim * self.feat_idx_emb_dim)
        )
        return input_feat_idx_emb  # [B L K*D]

    def get_input_time_feat(self, past_time_feat, future_time_feat, mode):
        if mode == "all":
            time_feat = torch.cat(
                (past_time_feat[:, -self.max_context_length :, ...], future_time_feat),
                dim=1,
            )
        elif mode == "encode":
            time_feat = past_time_feat[:, -self.max_context_length :, ...]
        elif mode == "decode":
            time_feat = future_time_feat
        return time_feat

    def get_inputs(self, batch_data, mode):
        inputs_list = []

        input_seq = self.get_input_sequence(
            batch_data.past_target_cdf, batch_data.future_target_cdf, mode=mode
        )
        input_length = input_seq.shape[1]  # [B L n_lags*K]
        inputs_list.append(input_seq)

        if self.use_feat_idx_emb:
            input_feat_idx_emb = self.get_input_feat_idx_emb(
                batch_data.target_dimension_indicator, input_length
            )  # [B L K*D]
            inputs_list.append(input_feat_idx_emb)

        if self.use_time_feat:
            input_time_feat = self.get_input_time_feat(
                batch_data.past_time_feat, batch_data.future_time_feat, mode=mode
            )  # [B L Dt]
            inputs_list.append(input_time_feat)
        return torch.cat(inputs_list, dim=-1).to(dtype=torch.float32)

    def get_scale(self, batch_data):
        self.scaler.fit(
            batch_data.past_target_cdf[:, -self.max_context_length :, ...],
            batch_data.past_observed_values[:, -self.max_context_length :, ...],
        )

    def get_weighted_loss(self, batch_data, loss):
        observed_values = batch_data.future_observed_values
        loss_weights, _ = observed_values.min(dim=-1, keepdim=True)
        loss = weighted_average(loss, weights=loss_weights, dim=1)
        return loss

    def loss(self, batch_data):
        raise NotImplementedError

    def forecast(self, batch_data=None, num_samples=None):
        raise NotImplementedError
