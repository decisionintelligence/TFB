import torch
import torch.nn as nn
import torch.nn.functional as F
from ts_benchmark.baselines.timekan.layers.Autoformer_EncDec import series_decomp
from ts_benchmark.baselines.timekan.layers.Embed import DataEmbedding_wo_pos
from ts_benchmark.baselines.timekan.layers.StandardNorm import Normalize
from ts_benchmark.baselines.timekan.layers.ChebyKANLayer import ChebyKANLinear
import math


class ChebyKANLayer(nn.Module):
    def __init__(self, in_features, out_features, order):
        super().__init__()
        self.fc1 = ChebyKANLinear(
            in_features,
            out_features,
            order)

    def forward(self, x):
        B, N, C = x.shape
        x = self.fc1(x.reshape(B * N, C))
        x = x.reshape(B, N, -1).contiguous()
        return x


class FrequencyDecomp(nn.Module):

    def __init__(self, configs):
        super(FrequencyDecomp, self).__init__()
        self.configs = configs

    def forward(self, level_list):

        level_list_reverse = level_list.copy()
        level_list_reverse.reverse()
        out_low = level_list_reverse[0]
        out_high = level_list_reverse[1]
        out_level_list = [out_low]
        for i in range(len(level_list_reverse) - 1):
            out_high_res = self.frequency_interpolation(out_low.transpose(1, 2),
                                                        self.configs.seq_len // (self.configs.down_sampling_window ** (
                                                                    self.configs.down_sampling_layers - i)),
                                                        self.configs.seq_len // (self.configs.down_sampling_window ** (
                                                                    self.configs.down_sampling_layers - i - 1))
                                                        ).transpose(1, 2)
            out_high_left = out_high - out_high_res
            out_low = out_high
            if i + 2 <= len(level_list_reverse) - 1:
                out_high = level_list_reverse[i + 2]
            out_level_list.append(out_high_left)
        out_level_list.reverse()
        return out_level_list

    def frequency_interpolation(self, x, seq_len, target_len):
        len_ratio = seq_len / target_len
        x_fft = torch.fft.rfft(x, dim=2)
        out_fft = torch.zeros([x_fft.size(0), x_fft.size(1), target_len // 2 + 1], dtype=x_fft.dtype).to(x_fft.device)
        out_fft[:, :, :seq_len // 2 + 1] = x_fft
        out = torch.fft.irfft(out_fft, dim=2)
        out = out * len_ratio
        return out


class FrequencyMixing(nn.Module):

    def __init__(self, configs):
        super(FrequencyMixing, self).__init__()
        self.configs = configs
        self.front_block = M_KAN(configs.d_model,
                                 self.configs.seq_len // (
                                             self.configs.down_sampling_window ** (self.configs.down_sampling_layers)),
                                 order=configs.begin_order)

        self.front_blocks = torch.nn.ModuleList(
            [
                M_KAN(configs.d_model,
                      self.configs.seq_len // (
                                  self.configs.down_sampling_window ** (self.configs.down_sampling_layers - i - 1)),
                      order=i + configs.begin_order + 1)
                for i in range(configs.down_sampling_layers)
            ])

    def forward(self, level_list):
        level_list_reverse = level_list.copy()
        level_list_reverse.reverse()
        out_low = level_list_reverse[0]
        out_high = level_list_reverse[1]
        out_low = self.front_block(out_low)
        out_level_list = [out_low]
        for i in range(len(level_list_reverse) - 1):
            out_high = self.front_blocks[i](out_high)
            out_high_res = self.frequency_interpolation(out_low.transpose(1, 2),
                                                        self.configs.seq_len // (self.configs.down_sampling_window ** (
                                                                    self.configs.down_sampling_layers - i)),
                                                        self.configs.seq_len // (self.configs.down_sampling_window ** (
                                                                    self.configs.down_sampling_layers - i - 1))
                                                        ).transpose(1, 2)
            out_high = out_high + out_high_res
            out_low = out_high
            if i + 2 <= len(level_list_reverse) - 1:
                out_high = level_list_reverse[i + 2]
            out_level_list.append(out_low)
        out_level_list.reverse()
        return out_level_list

    def frequency_interpolation(self, x, seq_len, target_len):
        len_ratio = seq_len / target_len
        x_fft = torch.fft.rfft(x, dim=2)
        out_fft = torch.zeros([x_fft.size(0), x_fft.size(1), target_len // 2 + 1], dtype=x_fft.dtype).to(x_fft.device)
        out_fft[:, :, :seq_len // 2 + 1] = x_fft
        out = torch.fft.irfft(out_fft, dim=2)
        out = out * len_ratio
        return out


class M_KAN(nn.Module):
    def __init__(self, d_model, seq_len, order):
        super().__init__()
        self.channel_mixer = nn.Sequential(
            ChebyKANLayer(d_model, d_model, order)
        )
        self.conv = BasicConv(d_model, d_model, kernel_size=3, degree=order, groups=d_model)

    def forward(self, x):
        x1 = self.channel_mixer(x)
        x2 = self.conv(x)
        out = x1 + x2
        return out


class BasicConv(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, degree, stride=1, padding=0, dilation=1, groups=1, act=False, bn=False,
                 bias=False, dropout=0.):
        super(BasicConv, self).__init__()
        self.out_channels = c_out
        self.conv = nn.Conv1d(c_in, c_out, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm1d(c_out) if bn else None
        self.act = nn.GELU() if act else None
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        if self.bn is not None:
            x = self.bn(x)
        x = self.conv(x.transpose(-1, -2)).transpose(-1, -2)
        if self.act is not None:
            x = self.act(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x


class TimeKANModeL(nn.Module):

    def __init__(self, configs):
        super(TimeKANModeL, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.down_sampling_window = configs.down_sampling_window
        self.channel_independence = configs.channel_independence
        self.res_blocks = nn.ModuleList([FrequencyDecomp(configs)
                                         for _ in range(configs.e_layers)])
        self.add_blocks = nn.ModuleList([FrequencyMixing(configs)
                                         for _ in range(configs.e_layers)])

        self.preprocess = series_decomp(configs.moving_avg)
        self.enc_in = configs.enc_in
        self.use_future_temporal_feature = configs.use_future_temporal_feature

        self.enc_embedding = DataEmbedding_wo_pos(1, configs.d_model, configs.embed, configs.freq,
                                                  configs.dropout)
        self.layer = configs.e_layers
        self.normalize_layers = torch.nn.ModuleList(
            [
                Normalize(self.configs.enc_in, affine=True, non_norm=True if configs.use_norm == 0 else False)
                for i in range(configs.down_sampling_layers + 1)
            ]
        )
        self.projection_layer = nn.Linear(
            configs.d_model, 1, bias=True)
        self.predict_layer = nn.Linear(
            configs.seq_len,
            configs.pred_len,
        )

    def forecast(self, x_enc):
        x_enc = self.__multi_level_process_inputs(x_enc)
        x_list = []
        for i, x in zip(range(len(x_enc)), x_enc, ):
            B, T, N = x.size()
            x = self.normalize_layers[i](x, 'norm')
            x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
            x_list.append(x)

        enc_out_list = []
        for i, x in zip(range(len(x_list)), x_list):
            enc_out = self.enc_embedding(x, None)  # [B,T,C]
            enc_out_list.append(enc_out)

        for i in range(self.layer):
            enc_out_list = self.res_blocks[i](enc_out_list)
            enc_out_list = self.add_blocks[i](enc_out_list)

        dec_out = enc_out_list[0]
        dec_out = self.predict_layer(dec_out.permute(0, 2, 1)).permute(
            0, 2, 1)
        dec_out = self.projection_layer(dec_out).reshape(B, self.configs.c_out, self.pred_len).permute(0, 2,
                                                                                                       1).contiguous()
        dec_out = self.normalize_layers[0](dec_out, 'denorm')
        return dec_out

    def __multi_level_process_inputs(self, x_enc):
        down_pool = torch.nn.AvgPool1d(self.configs.down_sampling_window)
        # B,T,C -> B,C,T
        x_enc = x_enc.permute(0, 2, 1)
        x_enc_ori = x_enc
        x_enc_sampling_list = []
        x_enc_sampling_list.append(x_enc.permute(0, 2, 1))
        for i in range(self.configs.down_sampling_layers):
            x_enc_sampling = down_pool(x_enc_ori)
            x_enc_sampling_list.append(x_enc_sampling.permute(0, 2, 1))
            x_enc_ori = x_enc_sampling
        x_enc = x_enc_sampling_list
        return x_enc

    def forward(self, x_enc):
        if self.task_name == 'long_term_forecast':
            dec_out = self.forecast(x_enc)
            return dec_out
        else:
            raise ValueError('Other tasks implemented yet')