import torch
import torch.nn as nn
import torch.nn.functional as F
import torch


def cal_orthogonal_loss(matrix):
    gram_matrix = torch.matmul(matrix.transpose(-2, -1), matrix)
    one_diag = torch.diagonal(gram_matrix, dim1=-2, dim2=-1)
    two_diag = torch.diag_embed(one_diag)
    off_diagonal = gram_matrix - two_diag
    loss = torch.norm(off_diagonal, dim=(-2, -1))
    return loss.mean()


class TimeBaseModel(nn.Module):
    def __init__(self, configs):
        super(TimeBaseModel, self).__init__()
        self.use_period_norm = configs.use_period_norm
        self.use_orthogonal = configs.use_orthogonal

        self.seq_len = configs.seq_len
        self.pred_len = configs.horizon
        self.enc_in = configs.enc_in
        self.period_len = configs.period_len
        self.pad_seq_len = 0
        self.basis_num = configs.basis_num

        self.seg_num_x = self.seq_len // self.period_len
        self.seg_num_y = self.pred_len // self.period_len
        if self.seq_len > self.seg_num_x * self.period_len:
            self.pad_seq_len = (self.seg_num_x + 1) * self.period_len - self.seq_len
            self.seg_num_x += 1
        if self.pred_len > self.seg_num_y * self.period_len:
            self.seg_num_y += 1

        self.individual = configs.individual
        if self.individual:
            self.ts2basis = nn.ModuleList()
            self.basis2ts = nn.ModuleList()
            for i in range(self.enc_in):
                self.ts2basis.append(nn.Linear(self.seg_num_x, configs.basis_num))
                self.basis2ts.append(nn.Linear(configs.basis_num, self.seg_num_y))
        else:
            self.ts2basis = nn.Linear(self.seg_num_x, configs.basis_num)
            self.basis2ts = nn.Linear(configs.basis_num, self.seg_num_y)

    def _normalize_input(self, x, b, c):
        """
        input: x (bc, p, n)
        output: x_normalized, norm_stats (dict)
        """
        if self.use_period_norm:
            period_mean = torch.mean(x, dim=-1, keepdim=True)
            x = x - period_mean
            return x, {"period_mean": period_mean}
        else:
            x = x.reshape(b, c, -1)
            mean = torch.mean(x, dim=-1, keepdim=True)
            x = x - mean
            x = x.reshape(-1, self.period_len, self.seg_num_x)
            return x, {"mean": mean}

    def _denormalize_output(self, x, norm_stats, b, c):
        """
        input: x (bc, p, n)
        output: x_denorm
        """
        if self.use_period_norm:
            x = x + norm_stats["period_mean"]
        else:
            x = x.reshape(b, c, -1)
            x = x + norm_stats["mean"]
            x = x.reshape(-1, self.period_len, self.seg_num_y)
        return x

    def forward(self, x):
        """
        x: b t c
        out: b t c
        """
        b, t, c = x.shape
        batch_size = b
        x = x.permute(0, 2, 1)  # b c t

        # Padding
        if self.pad_seq_len > 0:
            pad_start = (self.seg_num_x - 1) * self.period_len
            x = torch.cat(
                [x, x[:, :, pad_start - self.pad_seq_len : pad_start]], dim=-1
            )

        # Reshape
        x = x.reshape(batch_size, self.enc_in, self.seg_num_x, self.period_len)
        x = x.permute(0, 1, 3, 2).reshape(
            -1, self.period_len, self.seg_num_x
        )  # (bc, p, n)

        # Normalize
        x, norm_stats = self._normalize_input(x, b, c)

        # Basis transformation
        if self.individual:
            x = x.reshape(b, c, self.period_len, self.seg_num_x)
            x_pred = torch.zeros(
                [b, c, self.period_len, self.seg_num_y], dtype=x.dtype, device=x.device
            )
            x_basis = torch.zeros(
                [b, c, self.period_len, self.basis_num], dtype=x.dtype, device=x.device
            )
            for i in range(self.enc_in):
                x_basis[:, i, :, :] = self.ts2basis[i](x[:, i, :, :])
                x_pred[:, i, :, :] = self.basis2ts[i](x_basis[:, i, :, :])
            x_basis = x_basis.reshape(-1, self.period_len, self.basis_num)
            x = x_pred.reshape(-1, self.period_len, self.seg_num_y)
        else:
            x_basis = self.ts2basis(x)
            x = self.basis2ts(x_basis)

        # Denormalize
        x = self._denormalize_output(x, norm_stats, b, c)

        # Reshape back
        x = x.reshape(batch_size, self.enc_in, self.period_len, self.seg_num_y).permute(
            0, 1, 3, 2
        )
        x = x.reshape(batch_size, self.enc_in, -1).permute(0, 2, 1)  # b t c

        if self.use_orthogonal:
            orthogonal_loss = cal_orthogonal_loss(x_basis)
            return x[:, : self.pred_len, :], orthogonal_loss
        else:
            return x[:, : self.pred_len, :]
