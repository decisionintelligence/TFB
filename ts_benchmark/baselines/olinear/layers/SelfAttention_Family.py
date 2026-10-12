import os
import random
import time
from math import sqrt

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from einops import rearrange, reduce
from reformer_pytorch import LSHSelfAttention

from ..utils.masking import TriangularCausalMask, ProbMask
from ..utils.tools import plot_mat, moore_penrose_iter_pinv


# from entmax import sparsemax, entmax15


# Code implementation from https://github.com/thuml/Flowformer
class FlowAttention(nn.Module):
    def __init__(self, attention_dropout=0.1):
        super(FlowAttention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        print('FlowAttention is used...')

    def kernel_method(self, x):
        return torch.sigmoid(x)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None, *args, **kwargs):
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        # kernel
        queries = self.kernel_method(queries)
        keys = self.kernel_method(keys)
        # incoming and outgoing
        normalizer_row = 1.0 / (torch.einsum("nhld,nhd->nhl", queries + 1e-6, keys.sum(dim=2) + 1e-6))
        normalizer_col = 1.0 / (torch.einsum("nhsd,nhd->nhs", keys + 1e-6, queries.sum(dim=2) + 1e-6))
        # reweighting
        normalizer_row_refine = (
            torch.einsum("nhld,nhd->nhl", queries + 1e-6, (keys * normalizer_col[:, :, :, None]).sum(dim=2) + 1e-6))
        normalizer_col_refine = (
            torch.einsum("nhsd,nhd->nhs", keys + 1e-6, (queries * normalizer_row[:, :, :, None]).sum(dim=2) + 1e-6))
        # competition and allocation
        normalizer_row_refine = torch.sigmoid(
            normalizer_row_refine * (float(queries.shape[2]) / float(keys.shape[2])))
        normalizer_col_refine = torch.softmax(normalizer_col_refine, dim=-1) * keys.shape[2]  # B h L vis
        # multiply
        kv = keys.transpose(-2, -1) @ (values * normalizer_col_refine[:, :, :, None])
        x = (((queries @ kv) * normalizer_row[:, :, :, None]) * normalizer_row_refine[:, :, :, None]).transpose(1,
                                                                                                                2).contiguous()
        return x, None


# Code implementation from https://github.com/shreyansh26/FlashAttention-PyTorch
class FlashAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FlashAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        print('FlashAttention is used...')

    def flash_attention_forward(self, Q, K, V, mask=None):
        # BLOCK_SIZE = 32
        BLOCK_SIZE = 128
        NEG_INF = -1e10  # -infinity
        EPSILON = 1e-10
        # mask = torch.randint(0, 2, (128, 8)).to(device='cuda')
        O = torch.zeros_like(Q, requires_grad=True)
        l = torch.zeros(Q.shape[:-1])[..., None]
        m = torch.ones(Q.shape[:-1])[..., None] * NEG_INF

        O = O.to(device='cuda')
        l = l.to(device='cuda')
        m = m.to(device='cuda')

        Q_BLOCK_SIZE = min(BLOCK_SIZE, Q.shape[-1])
        KV_BLOCK_SIZE = BLOCK_SIZE

        Q_BLOCKS = torch.split(Q, Q_BLOCK_SIZE, dim=2)
        K_BLOCKS = torch.split(K, KV_BLOCK_SIZE, dim=2)
        V_BLOCKS = torch.split(V, KV_BLOCK_SIZE, dim=2)
        if mask is not None:
            mask_BLOCKS = list(torch.split(mask, KV_BLOCK_SIZE, dim=1))

        Tr = len(Q_BLOCKS)
        Tc = len(K_BLOCKS)

        O_BLOCKS = list(torch.split(O, Q_BLOCK_SIZE, dim=2))
        l_BLOCKS = list(torch.split(l, Q_BLOCK_SIZE, dim=2))
        m_BLOCKS = list(torch.split(m, Q_BLOCK_SIZE, dim=2))

        for j in range(Tc):
            Kj = K_BLOCKS[j]
            Vj = V_BLOCKS[j]
            if mask is not None:
                maskj = mask_BLOCKS[j]

            for i in range(Tr):
                Qi = Q_BLOCKS[i]
                Oi = O_BLOCKS[i]
                li = l_BLOCKS[i]
                mi = m_BLOCKS[i]

                scale = 1 / np.sqrt(Q.shape[-1])
                Qi_scaled = Qi * scale

                S_ij = torch.einsum('... i d, ... j d -> ... i j', Qi_scaled, Kj)
                if mask is not None:
                    # Masking
                    maskj_temp = rearrange(maskj, 'b j -> b 1 1 j')
                    S_ij = torch.where(maskj_temp > 0, S_ij, NEG_INF)

                m_block_ij, _ = torch.max(S_ij, dim=-1, keepdims=True)
                P_ij = torch.exp(S_ij - m_block_ij)
                if mask is not None:
                    # Masking
                    P_ij = torch.where(maskj_temp > 0, P_ij, 0.)

                l_block_ij = torch.sum(P_ij, dim=-1, keepdims=True) + EPSILON

                P_ij_Vj = torch.einsum('... i j, ... j d -> ... i d', P_ij, Vj)

                mi_new = torch.maximum(m_block_ij, mi)
                li_new = torch.exp(mi - mi_new) * li + torch.exp(m_block_ij - mi_new) * l_block_ij

                O_BLOCKS[i] = (li / li_new) * torch.exp(mi - mi_new) * Oi + (
                        torch.exp(m_block_ij - mi_new) / li_new) * P_ij_Vj
                l_BLOCKS[i] = li_new
                m_BLOCKS[i] = mi_new

        O = torch.cat(O_BLOCKS, dim=2)
        l = torch.cat(l_BLOCKS, dim=2)
        m = torch.cat(m_BLOCKS, dim=2)
        return O, l, m

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None, token_weight=None, *args, **kwargs):
        res = \
            self.flash_attention_forward(queries.permute(0, 2, 1, 3), keys.permute(0, 2, 1, 3),
                                         values.permute(0, 2, 1, 3),
                                         attn_mask)[0]
        return res.permute(0, 2, 1, 3).contiguous(), None


class FullAttention(nn.Module):
    def __init__(self, mask_flag=False, factor=5, scale=None, attention_dropout=0.1, output_attention=False,
                 token_num=None, imp_mode=False, ij_mat_flag=False, ij_attn_adjust_init=10.0, ij_mat_para=0,
                 num_heads=None, weight_plus=False, plot_mat_flag=False, save_folder='./'):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.imp_mode = imp_mode
        self.token_num = token_num
        self.ij_mat_flag = ij_mat_flag  # False  #
        self.ij_mat_para = ij_mat_para
        self.weight_plus = weight_plus
        self.plot_mat = plot_mat_flag
        self.save_folder = os.path.join(save_folder, 'attn_mat')
        if self.plot_mat:
            os.makedirs(self.save_folder, exist_ok=True)
        self.num_heads = 1

        print(f'self.weight_plus in FullAttention: {self.weight_plus}')

        print(f'ij_mat_flag in FullAttention:{self.ij_mat_flag}')
        print(f'ij_mat_para in FullAttention:{self.ij_mat_para}')

        if self.imp_mode and self.token_num is not None:
            self.token_contribution = nn.Parameter(torch.zeros(1, self.num_heads or 1, 1, self.token_num))
            # [1,1,N,N]
            self.tau = nn.Parameter(torch.ones(self.token_num, 1))
            if self.ij_mat_para:
                print('self.ij_mat_para in FullAttention is enabled...')

                # self.weight_mat = nn.Parameter(torch.randn(1, self.token_num, self.token_num) * 1.0)
                self.weight_mat = nn.Parameter((torch.eye(self.token_num) * 1.0 +
                                                torch.randn(self.token_num, self.token_num) * 1.0)
                                               [None, :, :].repeat(self.num_heads or 1, 1, 1))

                # ablation study: ones(token_num, token_num) sucks
                # self.weight_mat = nn.Parameter((torch.ones(token_num, token_num) * 1.0)[None, :, :].
                #                                repeat(self.num_heads or 1, 1, 1))

            elif self.ij_mat_flag:
                print('distance-based short-sight-attention in FullAttention is enabled...')

                self.attn_tau = nn.Parameter(torch.ones(token_num, 1) * ij_attn_adjust_init)
                # ij_mat = torch.ones(token_num, token_num)
                self.exp_para = nn.Parameter(torch.tensor(-10.0))
                range_tensor = torch.arange(token_num)
                ij_mat = (range_tensor.view(token_num, 1) -
                          range_tensor.view(1, token_num)).abs()

                self.register_buffer('ij_mat', ij_mat)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None, token_weight=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)
        # this_device = queries.device

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = scale * scores

        weight_mat = None
        token_contribution = None

        # attention matrix adjustment; 240507
        if self.imp_mode and self.token_num is not None:
            # token_contribution = self.token_contribution.to(queries.device)

            if token_weight is None:
                # 2d
                if self.ij_mat_para:
                    weight_mat = F.softmax(self.weight_mat / F.softplus(self.tau), dim=-1)[None, :, :, :]
                    # softplus 241014
                    # weight_mat = F.softplus(self.weight_mat / F.softplus(self.tau))[None, :, :, :]
                elif self.ij_mat_flag:
                    ij_mat = self.ij_mat.pow(F.softplus(self.exp_para)) if self.ij_mat_flag else self.ij_mat
                    weight_mat = (-ij_mat.unsqueeze(0).unsqueeze(0)
                                  / F.softplus(self.attn_tau).unsqueeze(0).unsqueeze(0))
                    # weight_mat = weight_mat.exp()
                    weight_mat = F.softmax(weight_mat / F.softplus(self.tau), dim=-1)
            else:
                # 4d
                if token_weight.shape[-1] != self.token_num:
                    print(f'token_weight ({token_weight.shape[-1]}) does not match token_num ({self.token_num})!')
                    raise ValueError
                # token_weight: [b,l] --> [b,1,1,l], in case that there is 0 in token_weight
                token_weight = torch.maximum(token_weight.unsqueeze(1).unsqueeze(1), torch.tensor(1e-5))
                if self.ij_mat_para:
                    weight_mat = self.weight_mat.unsqueeze(0)  # [1,h,l,l]
                    # token_weight is considered
                    weight_mat = F.softmax(weight_mat * token_weight /
                                           F.softplus(self.tau.unsqueeze(0).unsqueeze(0)), dim=-1)

                else:
                    if self.ij_mat_flag:
                        ij_mat = self.ij_mat.pow(F.softplus(self.exp_para))
                        weight_mat = (-ij_mat.unsqueeze(0).unsqueeze(0)
                                      / F.softplus(self.attn_tau).unsqueeze(0).unsqueeze(0) / token_weight)
                        weight_mat = F.softmax(weight_mat, dim=-1)

                    else:
                        weight_mat = F.softmax(token_weight / F.softplus(self.tau), dim=-1)

        if token_contribution is not None:
            A = A + token_contribution

        if self.weight_plus and weight_mat is not None:
            A = A + weight_mat

        # attention matrix [b,h,l,s]
        A = torch.softmax(A, dim=-1)

        if not self.weight_plus and weight_mat is not None:
            A = A * weight_mat
            A = F.normalize(A, p=1, dim=-1)

        # plot
        if not self.training and self.plot_mat and random.random() < 0.08 and A.shape[-1] > 10:
            batch_idx = random.randint(0, A.shape[0] - 1)
            head_idx = random.randint(0, A.shape[1] - 1)
            att_mat_2d = A[batch_idx, head_idx, :, :]
            time_or_channel = 'temporal' if self.ij_mat_flag else 'channel'
            plot_mat(att_mat_2d, str_cat='attn_mat', str0=f"batch_{batch_idx}_head_{head_idx}_{time_or_channel}",
                     save_folder=self.save_folder)

            if weight_mat is not None:
                batch_idx2 = min(batch_idx, weight_mat.shape[0] - 1)
                head_idx2 = min(head_idx, weight_mat.shape[1] - 1)
                weight_mat_2d = weight_mat[batch_idx2, head_idx2, :, :]

                #
                range_tensor = torch.arange(self.token_num)
                ij_mat = (range_tensor.view(self.token_num, 1) -
                          range_tensor.view(1, self.token_num)).abs()
                manual_weight = (ij_mat + 1).pow(1)
                weight_mat_2d = weight_mat_2d * manual_weight.to(weight_mat_2d.device)

                weight_mat_2d = F.normalize(weight_mat_2d, p=1, dim=-1)

                plot_mat(weight_mat_2d, str_cat='weight_mat_2d',
                         str0=f"batch_{batch_idx2}_head_{head_idx2}_{time_or_channel}",
                         save_folder=self.save_folder)

            # only the ij
            if self.ij_mat_flag:
                ij_mat = self.ij_mat.pow(F.softplus(self.exp_para))
                weight_mat = (-ij_mat.unsqueeze(0).unsqueeze(0)
                              / F.softplus(self.attn_tau).unsqueeze(0).unsqueeze(0))
                weight_mat = F.softmax(weight_mat, dim=-1)
                weight_mat_2d_ij = weight_mat[0, 0, :, :]

                #
                range_tensor = torch.arange(self.token_num)
                ij_mat = (range_tensor.view(self.token_num, 1) -
                          range_tensor.view(1, self.token_num)).abs()
                manual_weight = (ij_mat + 1).pow(0.5)
                weight_mat_2d_ij = weight_mat_2d_ij * manual_weight.to(weight_mat_2d_ij.device)

                weight_mat_2d_ij = F.normalize(weight_mat_2d_ij, p=1, dim=-1)
                plot_mat(weight_mat_2d_ij, str_cat='weight_mat_2d_ij', str0=f"batch_{0}_head_{0}_{time_or_channel}",
                         save_folder=self.save_folder)

            # i will handle this
            if self.ij_mat_flag:
                # channel does not need this
                # mask
                if hasattr(self, 'ij_mat'):
                    ij_mat = self.ij_mat
                else:
                    range_tensor = torch.arange(self.token_num)
                    ij_mat = (range_tensor.view(self.token_num, 1) -
                              range_tensor.view(1, self.token_num)).abs()
                manual_weight = (ij_mat + 1).pow(-0.5)
                att_mat_2d = att_mat_2d * manual_weight.to(att_mat_2d.device)

                att_mat_2d = F.normalize(att_mat_2d, p=1, dim=-1)
                plot_mat(att_mat_2d, str_cat='attn_mat', str0=f"manual_batch_{batch_idx}_head_{head_idx}_"
                                                              f"{time_or_channel}",
                         save_folder=self.save_folder)

            print('Attention matrix in FullAttention has been saved...')

        # dropout, reserved
        A = self.dropout(A)

        # print(f'A.shape: {A.shape}')
        # print(f'values.shape: {values.shape}')
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class FullAttention_SF(nn.Module):
    def __init__(self, mask_flag=False, factor=5, scale=None, attention_dropout=0.1, output_attention=False,
                 token_num=None, SF_mode=False, contri_flag=False, ij_mat_flag=False, ij_attn_adjust_init=10.0,
                 ij_mat_para=0, weight_plus=False, plot_mat_flag=False, save_folder='./'):
        super(FullAttention_SF, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.SF_mode = SF_mode
        self.contri_flag = contri_flag
        self.token_num = token_num
        self.ij_mat_flag = ij_mat_flag  # False  #
        self.ij_mat_para = ij_mat_para
        self.weight_plus = weight_plus
        self.plot_mat_flag = plot_mat_flag
        self.save_folder = os.path.join(save_folder)
        if self.plot_mat_flag and not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder, exist_ok=True)
        self.num_heads = 1

        print(f'self.weight_plus in FullAttention: {self.weight_plus}')

        print(f'ij_mat_flag in FullAttention:{self.ij_mat_flag}')
        print(f'ij_mat_para in FullAttention:{self.ij_mat_para}')

        if self.contri_flag:
            # [1,1,1,N]
            self.token_contri = nn.Parameter(torch.ones(1, self.num_heads or 1, 1, self.token_num))
        else:
            self.token_contri = None

        # self.tau_general = nn.Parameter(torch.ones(1, 1, self.token_num, 1))

        if self.SF_mode and self.weight_plus:
            self.sum_weight = nn.Parameter(torch.tensor(1.0))

        if self.SF_mode and self.token_num is not None:
            # [1,1,N,1]
            self.tau = nn.Parameter(torch.ones(1, 1, self.token_num, 1))
            # scalar
            # self.tau = nn.Parameter(torch.ones(1))
            # [1,1,1,N]
            # self.tau = nn.Parameter(torch.ones(1, 1, 1, self.token_num))
            # [1,1,N,N]
            # self.tau = nn.Parameter(torch.ones(1, 1, self.token_num, self.token_num))
            if self.ij_mat_para:
                print('self.ij_mat_para in FullAttention is enabled...')

                # self.weight_mat = nn.Parameter(torch.randn(1, self.token_num, self.token_num) * 1.0)
                self.weight_mat = nn.Parameter((torch.eye(self.token_num) * 1.0 +
                                                torch.randn(self.token_num, self.token_num) * 1.0)
                                               [None, None, :, :].repeat(1, self.num_heads or 1, 1, 1))

                # ablation study: ones(token_num, token_num) sucks
                # self.weight_mat = nn.Parameter((torch.ones(token_num, token_num) * 1.0)[None, :, :].
                #                                repeat(self.num_heads or 1, 1, 1))

            elif self.ij_mat_flag:
                print('distance-based short-sight-attention in FullAttention is enabled...')

                self.attn_tau = nn.Parameter(torch.ones(token_num, 1) * ij_attn_adjust_init)
                # ij_mat = torch.ones(token_num, token_num)
                self.exp_para = nn.Parameter(torch.tensor(-10.0))
                range_tensor = torch.arange(token_num)
                ij_mat = (range_tensor.view(token_num, 1) -
                          range_tensor.view(1, token_num)).abs()

                self.register_buffer('ij_mat', ij_mat)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None, token_weight=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)
        # this_device = queries.device

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = scale * scores

        weight_mat = None

        # attention matrix adjustment; 240507
        if self.SF_mode and self.token_num is not None:
            # token_contribution = self.token_contribution.to(queries.device)

            # 2d
            if self.ij_mat_para:
                if self.contri_flag and self.token_contri is not None:
                    weight_mat = F.softmax(self.weight_mat / F.softplus(self.tau) / F.softplus(self.token_contri),
                                           dim=-1)
                else:
                    weight_mat = F.softmax(self.weight_mat / F.softplus(self.tau), dim=-1)
                    # softplus 241014
                    # weight_mat = F.softplus(self.weight_mat / F.softplus(self.tau))[None, :, :, :]
                    # only use weight_mat: not good
                    # weight_mat = F.sigmoid(self.weight_mat)
                    # weight_mat = F.softplus(self.weight_mat)
            elif self.ij_mat_flag:
                # [N, N]
                ij_mat = self.ij_mat.pow(F.softplus(self.exp_para))
                # [1,1,N,N]
                weight_mat = (-ij_mat.unsqueeze(0).unsqueeze(0)
                              / F.softplus(self.attn_tau).unsqueeze(0).unsqueeze(0))
                # weight_mat = weight_mat.exp()
                weight_mat = F.softmax(weight_mat / F.softplus(self.tau), dim=-1)

        if self.SF_mode and weight_mat is not None:
            if self.weight_plus:
                A = A + self.sum_weight * weight_mat
            else:
                # ablation: this is a better choice
                A = A * weight_mat

        elif self.contri_flag:
            # not SF mode but use token contribution
            token_contri = F.softmax(self.token_contri, dim=-1)
            A = A * token_contri

        # attention matrix [b,h,l,s]
        A = torch.softmax(A, dim=-1)
        # not helpful
        # A = torch.softmax(A / F.softplus(self.tau_general), dim=-1)
        # double softmax
        # A = A.softmax(dim=-1).softmax(dim=-1)

        # if not self.weight_plus and weight_mat is not None:
        #     A = A * weight_mat
        #     A = F.normalize(A, p=1, dim=-1)

        # plot
        if not self.training and self.plot_mat_flag and random.random() < 0.01:  # and A.shape[-1] > 10
            batch_idx = random.randint(0, A.shape[0] - 1)
            head_idx = random.randint(0, A.shape[1] - 1)
            att_mat_2d = A[batch_idx, head_idx, :, :]
            time_or_channel = 'channel'
            plot_mat(att_mat_2d, str_cat='self_attn_SF', str0=f"batch_{batch_idx}_head_{head_idx}_{time_or_channel}",
                     save_folder=self.save_folder)

            if weight_mat is not None:
                batch_idx2 = min(batch_idx, weight_mat.shape[0] - 1)
                head_idx2 = min(head_idx, weight_mat.shape[1] - 1)
                weight_mat_2d = weight_mat[batch_idx2, head_idx2, :, :]

                plot_mat(weight_mat_2d, str_cat='weight_mat_2d',
                         str0=f"batch_{batch_idx2}_head_{head_idx2}_{time_or_channel}",
                         save_folder=self.save_folder)

            print(f'Attention matrix in FullAttention has been saved to {self.save_folder}...')

        # dropout, reserved
        A = self.dropout(A)

        # print(f'A.shape: {A.shape}')
        # print(f'values.shape: {values.shape}')
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class FullAttention_ablation(nn.Module):
    def __init__(self, mask_flag=False, factor=5, scale=None, attention_dropout=0.1, output_attention=False,
                 token_num=None, SF_mode=1, softmax_flag=1, weight_plus=0, outside_softmax=0,
                 plot_mat_flag=False, save_folder='./', plot_grad_flag=False, **kwargs):
        super(FullAttention_ablation, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.enhance_mode = SF_mode
        self.softmax_flag = softmax_flag
        self.token_num = token_num
        self.outside_softmax = outside_softmax  # False  #
        self.weight_plus = weight_plus
        self.plot_mat_flag = plot_mat_flag
        self.plot_grad_flag = plot_grad_flag
        self.save_folder = os.path.join(save_folder)
        if self.plot_mat_flag and not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder, exist_ok=True)
        self.num_heads = 1

        print(f'self.weight_plus in FullAttention_ablation: {self.weight_plus}')
        print(f'self.softmax_flag in FullAttention_ablation: {self.softmax_flag}')
        print(f'self.outside_softmax in FullAttention_ablation: {self.outside_softmax}')

        if not self.enhance_mode:
            print('Vanilla attention is used...')
        else:
            print('Enhanced attention is used...')

        if self.enhance_mode and self.token_num is not None:
            # [1,1,N,1]
            if self.softmax_flag:
                self.tau = nn.Parameter(torch.ones(1, 1, self.token_num, 1))

            init_weight_mat = (torch.eye(self.token_num) * 1.0 +
                               torch.randn(self.token_num, self.token_num) * 1.0)
            self.weight_mat = nn.Parameter(init_weight_mat[None, None, :, :].repeat(1, self.num_heads or 1, 1, 1))

            self.tau2 = nn.Parameter(torch.ones(1, 1, self.token_num, 1))

    def forward(self, queries, keys, values, attn_mask, **kwargs):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = scale * scores

        weight_mat = None
        ori_attn_mat = None
        if self.enhance_mode:
            if not self.training and self.plot_mat_flag:
                ori_attn_mat = torch.softmax(A, dim=-1)

        # attention matrix adjustment; 240507
        if self.enhance_mode and self.token_num is not None:
            # 2d
            if self.softmax_flag:
                weight_mat = F.softmax(self.weight_mat / F.softplus(self.tau), dim=-1)
            else:
                # use scale or not
                weight_mat = F.softplus(self.weight_mat)  # / sqrt(self.token_num)

        if self.enhance_mode and weight_mat is not None:
            if self.outside_softmax:
                if self.weight_plus:
                    A = A.softmax(dim=-1) + weight_mat

                    # ablations: two softmax
                    # A = A.softmax(dim=-1).softmax(dim=-1) + weight_mat

                    # ablations: * tau
                    # A = (A * F.softplus(self.tau2)).softmax(dim=-1) + weight_mat
                else:
                    A = A.softmax(dim=-1) * weight_mat
                A = F.normalize(A, p=1, dim=-1)
            else:
                if self.weight_plus:
                    A = A + weight_mat
                else:
                    # ablation: this is a better choice
                    A = A * weight_mat

                # attention matrix [b,h,l,s]
                A = torch.softmax(A, dim=-1)

        else:
            A = torch.softmax(A, dim=-1)

        # plot mat
        if not self.training and self.plot_mat_flag and random.random() < 1:  # and A.shape[-1] > 10
            batch_idx = random.randint(0, A.shape[0] - 1)
            head_idx = random.randint(0, A.shape[1] - 1)

            # final
            att_mat_2d = A[batch_idx, head_idx, :, :]
            time_or_channel = 'channel'
            plot_mat(att_mat_2d, str_cat='final_attn_mat', str0=f"batch_{batch_idx}_head_{head_idx}_{time_or_channel}",
                     save_folder=self.save_folder)

            if ori_attn_mat is not None and self.enhance_mode:
                ori_att_mat_2d = ori_attn_mat[batch_idx, head_idx, :, :]
                time_or_channel = 'channel'
                plot_mat(ori_att_mat_2d, str_cat='ori_attn_mat', str0=f"batch_{batch_idx}_head_{head_idx}_"
                                                                      f"{time_or_channel}",
                         save_folder=self.save_folder)

            if weight_mat is not None and self.enhance_mode:
                batch_idx2 = min(batch_idx, weight_mat.shape[0] - 1)
                head_idx2 = min(head_idx, weight_mat.shape[1] - 1)
                weight_mat_2d = weight_mat[batch_idx2, head_idx2, :, :]

                plot_mat(weight_mat_2d, str_cat='adding_mat_2d',
                         str0=f"batch_{batch_idx2}_head_{head_idx2}_{time_or_channel}",
                         save_folder=self.save_folder)

            print(f'Attention matrix in FullAttention has been saved to {self.save_folder}...')

        # dropout, reserved
        A = self.dropout(A)

        # print(f'A.shape: {A.shape}')
        # print(f'values.shape: {values.shape}')
        V = torch.einsum("bhls,bshd->blhd", A, values)

        # plot gradient
        if self.plot_grad_flag and random.random() < 0.01 and self.weight_mat.grad is not None:
            batch_idx = random.randint(0, self.weight_mat.shape[0] - 1)
            head_idx = random.randint(0, self.weight_mat.shape[1] - 1)

            # final
            weight_mat_grad = self.weight_mat.grad[batch_idx, head_idx, :, :]
            time_or_channel = 'channel'
            plot_mat(weight_mat_grad, str_cat='weight_grad_mat',
                     str0=f"batch_{batch_idx}_head_{head_idx}_{time_or_channel}",
                     save_folder=self.save_folder)

            print(f'Attention gradient matrix in FullAttention has been saved to {self.save_folder}...')

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class FullAttention_L1(nn.Module):
    def __init__(self, mask_flag=False, factor=5, scale=None, attention_dropout=0.1, output_attention=False,
                 token_num=None, dim=None, plot_mat_flag=False, save_folder='./'):
        super(FullAttention_L1, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.token_num = token_num
        self.dropout = nn.Dropout(attention_dropout)
        self.plot_mat_flag = plot_mat_flag
        self.save_folder = save_folder
        if self.plot_mat_flag and not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder, exist_ok=True)

        # self.tau1 = nn.Parameter(torch.ones(1, self.token_num, 1, 1))
        # self.tau2 = nn.Parameter(torch.ones(1, self.token_num, 1, 1))
        # self.power = nn.Parameter(torch.tensor(2.0))

        init_weight_mat = torch.eye(self.token_num) * 1.0 + torch.randn(self.token_num, self.token_num) * 1.0
        self.weight_mat = nn.Parameter(init_weight_mat[None, None, :, :])
        self.weight_mat2 = nn.Parameter(torch.randn(1, 1, self.token_num, self.token_num))
        # self.dim = dim
        # if self.dim is not None:
        #     self.norm1 = nn.LayerNorm(self.dim)
        #     self.norm2 = nn.LayerNorm(self.dim)

        self.lin_v = nn.Linear(self.token_num, self.token_num)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None, token_weight=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape

        scale = self.scale or 1. / sqrt(E)

        # queries, keys = ((queries / F.softplus(self.tau1)).softmax(dim=-1),
        #                  (keys / F.softplus(self.tau2)).softmax(dim=-1))

        # queries, keys = self.norm1(queries), self.norm2(keys)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = scale * scores
        # A = scores

        # softmax --> L1
        # A = F.normalize(F.softplus(A) ** 3, p=1, dim=-1)

        # /tau (seems helpless)
        # A = (A / F.softplus(self.tau)).softmax(dim=-1)

        # softmax & softmax
        # A = (A.softmax(dim=-1)).softmax(dim=-1)  # .softmax(dim=-1)
        # A = A.exp().softmax(dim=-1)  # could produce nan, probably due to overlarge values

        # softmax --> x^3: failed

        # enhanced attention
        A = A.softmax(dim=-1) + F.softplus(self.weight_mat)
        # A = entmax15(A, dim=-1) + F.softplus(self.weight_mat)
        # A = sparsemax(A, dim=-1) + F.softplus(self.weight_mat)
        A = F.normalize(A, p=1, dim=-1)

        # x --> (x+3)^3 --> softmax: failed
        # A = ((A + self.bias) ** 3).softmax(dim=-1)

        # plot mat
        if not self.training and self.plot_mat_flag and random.random() < 0.01:  # and A.shape[-1] > 10
            batch_idx = random.randint(0, A.shape[0] - 1)
            head_idx = random.randint(0, A.shape[1] - 1)

            # final
            att_mat_2d = A[batch_idx, head_idx, :, :]
            time_or_channel = 'channel'
            plot_mat(att_mat_2d, str_cat='final_attn_mat', str0=f"batch_{batch_idx}_head_{head_idx}_{time_or_channel}",
                     save_folder=self.save_folder)

            print(f'Attention matrix in FullAttention_L1 has been saved to {self.save_folder}...')

        # dropout, reserved
        A = self.dropout(A)

        V = torch.einsum("bhls,bshd->blhd", A + F.softmax(self.weight_mat2, dim=-1), values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class EnhancedAttention(nn.Module):
    def __init__(self, mask_flag=False, factor=5, scale=None, attention_dropout=0.1, output_attention=False,
                 token_num=None, CovMat=None, plot_mat_flag=False, save_folder='./',
                 **kwargs):
        super(EnhancedAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.token_num = token_num
        self.CovMat = CovMat.unsqueeze(0).unsqueeze(0) if CovMat is not None else None
        self.dropout = nn.Dropout(attention_dropout)
        self.plot_mat_flag = plot_mat_flag
        self.save_folder = save_folder
        if self.plot_mat_flag and not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder, exist_ok=True)

        init_weight_mat = torch.eye(self.token_num) * 1.0 + torch.randn(self.token_num, self.token_num) * 1.0
        self.weight_mat = nn.Parameter(init_weight_mat[None, None, :, :])

        print('Enhanced Attention is used...')

    def forward(self, queries, keys, values, attn_mask=None, **kwargs):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape

        # original enhanced attention
        if self.CovMat is not None:
            A = F.softmax(self.CovMat, dim=-1) + F.softplus(self.weight_mat)
        else:
            scale = self.scale or 1. / sqrt(E)

            scores = torch.einsum("blhe,bshe->bhls", queries, keys)

            if self.mask_flag:
                if attn_mask is None:
                    attn_mask = TriangularCausalMask(B, L, device=queries.device)

                scores.masked_fill_(attn_mask.mask, -np.inf)

            A = scale * scores
            A = A.softmax(dim=-1) + F.softplus(self.weight_mat)

        A = F.normalize(A, p=1, dim=-1)

        # plot mat
        if not self.training and self.plot_mat_flag and random.random() < 0.01:  # and A.shape[-1] > 10
            batch_idx = random.randint(0, A.shape[0] - 1)
            head_idx = random.randint(0, A.shape[1] - 1)

            # final
            att_mat_2d = A[batch_idx, head_idx, :, :]
            time_or_channel = 'channel'
            plot_mat(att_mat_2d, str_cat='final_attn_mat', str0=f"batch_{batch_idx}_head_{head_idx}_{time_or_channel}",
                     save_folder=self.save_folder)

            print(f'Attention matrix in Enhanced Attention has been saved to {self.save_folder}...')

        # dropout, reserved
        A = self.dropout(A)

        V = torch.einsum("bhls,bshd->blhd", A, values)
        # V = self.v_proj(V)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class VanillaAttention(nn.Module):
    def __init__(self, mask_flag=False, factor=5, scale=None, attention_dropout=0.1, output_attention=False,
                 plot_mat_flag=False, save_folder='./', **kwargs):
        super(VanillaAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.plot_mat_flag = plot_mat_flag
        self.save_folder = save_folder
        if self.plot_mat_flag and not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder, exist_ok=True)

        print('Vanilla attention is used...')

    def forward(self, queries, keys, values, attn_mask=None, **kwargs):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape

        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = scale * scores
        A = A.softmax(dim=-1)

        # plot mat
        if not self.training and self.plot_mat_flag and random.random() < 0.01:  # and A.shape[-1] > 10
            batch_idx = random.randint(0, A.shape[0] - 1)
            head_idx = random.randint(0, A.shape[1] - 1)

            # final
            att_mat_2d = A[batch_idx, head_idx, :, :]
            time_or_channel = 'channel'
            plot_mat(att_mat_2d, str_cat='final_attn_mat', str0=f"batch_{batch_idx}_head_{head_idx}_{time_or_channel}",
                     save_folder=self.save_folder)

            print(f'Attention matrix in vanilla Attention has been saved to {self.save_folder}...')

        # dropout, reserved
        A = self.dropout(A)

        V = torch.einsum("bhls,bshd->blhd", A, values)
        # V = self.v_proj(V)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class EnhancedAttention2(nn.Module):
    def __init__(self, mask_flag=False, factor=5, scale=None, attention_dropout=0.1, output_attention=False,
                 token_num=None, head_dim=None,
                 plot_mat_flag=False, save_folder='./', **kwargs):
        super(EnhancedAttention2, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.token_num = token_num
        self.head_dim = head_dim
        self.dropout = nn.Dropout(attention_dropout)
        self.plot_mat_flag = plot_mat_flag
        self.save_folder = save_folder
        if self.plot_mat_flag and not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder, exist_ok=True)

        init_weight_mat = torch.eye(self.token_num) * 1.0 + torch.randn(self.token_num, self.token_num) * 1.0
        self.weight_mat = nn.Parameter(init_weight_mat[None, None, :, :])

        self.q_k_layer = nn.Sequential(
            nn.Linear(self.head_dim, self.head_dim * 4),
            nn.GELU(),
            nn.Linear(self.head_dim * 4, self.head_dim),
        )

        # self.Q_learn = nn.Parameter(torch.randn(1, self.token_num, 1, self.head_dim))
        # self.K_learn = nn.Parameter(torch.randn(1, self.token_num, 1, self.head_dim))

        # self.tau = nn.Parameter(torch.tensor(3.0))

    def forward(self, queries, keys, values, attn_mask=None, token_weight=None, **kwargs):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape

        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys * F.sigmoid(self.q_k_layer(queries)))

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = scale * scores

        # enhanced attention 2
        # alpha = F.sigmoid(self.tau)
        # A = alpha * A.softmax(dim=-1) + (1 - alpha) * F.normalize(self.weight_mat, p=1, dim=-1)

        # QK^T
        # delta_mat = torch.einsum("blhe,bshe->bhls", self.Q_learn, self.K_learn)
        # A = A.softmax(dim=-1) + F.softplus(delta_mat)
        # A = F.normalize(A, p=1, dim=-1)

        # original enhanced attention
        A = A.softmax(dim=-1) + F.softplus(self.weight_mat)
        A = F.normalize(A, p=1, dim=-1)

        # plot mat
        if not self.training and self.plot_mat_flag and random.random() < 0.01:  # and A.shape[-1] > 10
            batch_idx = random.randint(0, A.shape[0] - 1)
            head_idx = random.randint(0, A.shape[1] - 1)

            # final
            att_mat_2d = A[batch_idx, head_idx, :, :]
            time_or_channel = 'channel'
            plot_mat(att_mat_2d, str_cat='final_attn_mat', str0=f"batch_{batch_idx}_head_{head_idx}_{time_or_channel}",
                     save_folder=self.save_folder)

            print(f'Attention matrix in Enhanced Attention has been saved to {self.save_folder}...')

        # dropout, reserved
        A = self.dropout(A)

        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class BlockWiseEnhancedAttention(nn.Module):
    def __init__(self, mask_flag=False, factor=5, scale=None, attention_dropout=0.1, output_attention=False,
                 token_num=None, plot_mat_flag=False, save_folder='./', block_size=150, shuffle=True, **kwargs):
        super(BlockWiseEnhancedAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.token_num = token_num
        self.dropout = nn.Dropout(attention_dropout)
        self.plot_mat_flag = plot_mat_flag
        self.save_folder = save_folder
        self.block_size = int(block_size)
        self.shuffle = shuffle
        if self.plot_mat_flag and not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder, exist_ok=True)

        self.block_num = int(np.ceil(self.token_num / self.block_size))

        if not self.shuffle:
            init_weight_mat = torch.randn(1, 1, self.block_num, self.block_size, self.block_size)
            self.weight_mat = nn.Parameter(init_weight_mat)

    def forward(self, queries, keys, values, attn_mask, **kwargs):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape

        # shuffle
        if self.shuffle:
            if self.training:
                g = torch.Generator()
                seed = int(time.time_ns()) % (2 ** 32)
                g.manual_seed(seed)
                perm = torch.randperm(L, generator=g)
            else:
                perm = torch.randperm(L)
            queries = queries[:, perm, :, :]
            keys = keys[:, perm, :, :]
            values = values[:, perm, :, :]

        if self.token_num != self.block_num * self.block_size:
            delta = self.block_num * self.block_size - self.token_num
            queries = F.pad(queries, pad=(0, 0, 0, 0, 0, delta), mode='constant', value=0)
            keys = F.pad(keys, pad=(0, 0, 0, 0, 0, delta), mode='constant', value=0)
            values = F.pad(values, pad=(0, 0, 0, 0, 0, delta), mode='constant', value=0)

        queries = rearrange(queries, 'b (n s) h e -> b n s h e', n=self.block_num)
        keys = rearrange(keys, 'b (n s) h e -> b n s h e', n=self.block_num)

        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("b n s h e, b n l h e -> b h n s l", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = scale * scores

        if self.shuffle:
            # vanilla attention
            A = A.softmax(dim=-1)
        else:
            # enhanced attention
            A = A.softmax(dim=-1) + F.softplus(self.weight_mat)
            A = F.normalize(A, p=1, dim=-1)

        # plot mat
        if not self.training and self.plot_mat_flag and random.random() < 0.01:  # and A.shape[-1] > 10
            batch_idx = random.randint(0, A.shape[0] - 1)
            head_idx = random.randint(0, A.shape[1] - 1)

            # final
            att_mat_2d = A[batch_idx, head_idx, :, :]
            time_or_channel = 'channel'
            plot_mat(att_mat_2d, str_cat='final_attn_mat', str0=f"batch_{batch_idx}_head_{head_idx}_{time_or_channel}",
                     save_folder=self.save_folder)

            print(f'Attention matrix in Enhanced Attention has been saved to {self.save_folder}...')

        # dropout, reserved
        A = self.dropout(A)

        values = rearrange(values, 'b (n s) h e -> b n s h e', n=self.block_num)
        # print(f'values.shape: {values.shape}')
        # print(f'A.shape: {A.shape}')

        V = torch.einsum("b h n s l, b n l h e -> b n s h e", A, values)

        V = rearrange(V, 'b n s h e -> b (n s) h e')[:, :self.token_num, :, :]

        if self.shuffle:
            # back to original order
            inv_perm = torch.argsort(perm)
            V = V[:, inv_perm, :, :]

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class RowWiseEnhancedAttention(nn.Module):
    # aim for no performance loss
    def __init__(self, mask_flag=False, factor=5, scale=None, attention_dropout=0.1, output_attention=False,
                 token_num=None, enhanced=True, plot_mat_flag=False, save_folder='./', block_size=150, **kwargs):
        super(RowWiseEnhancedAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.token_num = token_num
        self.enhanced = enhanced
        self.dropout = nn.Dropout(attention_dropout)
        self.plot_mat_flag = plot_mat_flag
        self.save_folder = save_folder
        self.block_size = int(block_size)
        if self.plot_mat_flag and not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder, exist_ok=True)

        self.block_num = int(np.ceil(self.token_num / self.block_size))

        if self.enhanced:
            print('Enhanced attention in RowWiseEnhancedAttention is used...')
            init_weight_mat = torch.randn(1, 1, self.token_num, self.token_num)
            self.weight_mat = nn.Parameter(init_weight_mat)
        else:
            print('Vanilla attention in RowWiseEnhancedAttention is used...')

    def forward(self, queries, keys, values, attn_mask, **kwargs):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape

        V_list = []
        scale = self.scale or 1. / sqrt(E)

        for i in range(self.block_num):
            start_idx = i * self.block_size
            end_idx = min(L, (i + 1) * self.block_size)
            query = queries[:, start_idx:end_idx, :, :]

            A = torch.einsum("blhe,bshe->bhls", query, keys) * scale

            if self.enhanced:
                weight_mat = self.weight_mat[:, :, start_idx:end_idx, :]
                A = A.softmax(dim=-1) + F.softplus(weight_mat)
                A = F.normalize(A, p=1, dim=-1)
            else:
                A = A.softmax(dim=-1)

            A = self.dropout(A)

            V = torch.einsum("bhls,bshd->blhd", A, values)

            V_list.append(V)

        V = torch.concat(V_list, dim=1)

        return V.contiguous(), None


class NystromAttention(nn.Module):
    def __init__(self, mask_flag=False, factor=5, scale=None, attention_dropout=0.1, output_attention=False,
                 token_num=None, plot_mat_flag=False, save_folder='./', num_landmarks=64, pinv_iterations=6, **kwargs):
        super(NystromAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.token_num = token_num
        self.dropout = nn.Dropout(attention_dropout)
        self.plot_mat_flag = plot_mat_flag
        self.save_folder = save_folder
        self.num_landmarks = int(num_landmarks)
        self.pinv_iterations = int(pinv_iterations)
        if self.plot_mat_flag and not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder, exist_ok=True)

        self.merge_size = int(np.ceil(self.token_num / self.num_landmarks))

    def forward(self, queries, keys, values, attn_mask, **kwargs):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape

        if self.token_num != self.num_landmarks * self.merge_size:
            delta = self.num_landmarks * self.merge_size - self.token_num
            queries2 = F.pad(queries, pad=(0, 0, 0, 0, 0, delta), mode='constant', value=0)
            keys2 = F.pad(keys, pad=(0, 0, 0, 0, 0, delta), mode='constant', value=0)
        else:
            queries2, keys2 = queries, keys

        # reduce the number of tokens
        landmark_einops_eq = '... (n l) h e -> ... n h e'
        q_landmarks = reduce(queries2, landmark_einops_eq, 'sum', l=self.merge_size) / self.merge_size
        k_landmarks = reduce(keys2, landmark_einops_eq, 'sum', l=self.merge_size) / self.merge_size

        einops_eq = 'b l h e, b s h e -> b h l s'
        sim1 = einsum(einops_eq, queries, k_landmarks)
        sim2 = einsum(einops_eq, q_landmarks, k_landmarks)
        sim3 = einsum(einops_eq, q_landmarks, keys)

        # eq (15) in the paper and aggregate values

        attn1, attn2, attn3 = map(lambda t: t.softmax(dim=-1), (sim1, sim2, sim3))
        attn2_inv = moore_penrose_iter_pinv(attn2, self.pinv_iterations)

        # b, h, l, e
        V = (attn1 @ attn2_inv) @ (attn3 @ values.transpose(1, 2))

        # b, l, h, e
        V = V.transpose(1, 2)

        return V.contiguous(), None


class FullAttention_Gauss(nn.Module):
    def __init__(self, mask_flag=False, factor=5, scale=None, attention_dropout=0.1, output_attention=False,
                 token_num=None):
        super(FullAttention_Gauss, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.token_num = token_num
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

        if token_num is not None:
            self.tau = nn.Parameter(torch.ones(1, 1, self.token_num, 1))
        else:
            self.tau = nn.Parameter(torch.ones(1))

    def forward(self, queries, keys, values, attn_mask, **kwargs):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape

        # gauss RBF Kernel Weighting: could incur out of memory
        # b,h,l,1,e - b,h,1,l,e --> b,h,l,l,e
        diff = queries.transpose(1, 2).unsqueeze(-2) - keys.transpose(1, 2).unsqueeze(2)
        # b,h,l,l
        dist_sq = torch.sum(diff ** 2, dim=-1)
        A = torch.exp(-dist_sq / F.softplus(self.tau))

        # dropout, reserved
        A = self.dropout(A)

        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class FullAttention_ablation_2501(nn.Module):
    def __init__(self, mask_flag=False, factor=5, scale=None, attention_dropout=0.1, output_attention=False,
                 token_num=None, SF_mode=1, softmax_flag=1, weight_plus=0, outside_softmax=0, enhance_alpha=1,
                 plot_mat_flag=False, save_folder='./', plot_grad_flag=False):  # './utils/corr_mat/traffic.npy'
        super(FullAttention_ablation_2501, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.SF_mode = SF_mode
        self.softmax_flag = softmax_flag
        self.token_num = token_num
        self.outside_softmax = outside_softmax  # False  #
        self.enhance_alpha = enhance_alpha  # False  #
        self.weight_plus = weight_plus
        self.plot_mat_flag = plot_mat_flag
        self.plot_grad_flag = plot_grad_flag
        self.save_folder = os.path.join(save_folder)
        if self.plot_mat_flag and not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder, exist_ok=True)
        self.num_heads = 1

        print(f'self.weight_plus in FullAttention_ablation: {self.weight_plus}')
        print(f'self.softmax_flag in FullAttention_ablation: {self.softmax_flag}')
        print(f'self.outside_softmax in FullAttention_ablation: {self.outside_softmax}')

        if self.weight_plus and self.enhance_alpha:
            self.alpha = nn.Parameter(torch.tensor(0.0))

        if not self.SF_mode:
            print('Vanilla attention is used...')
        else:
            print('Enhanced attention is used...')

        if self.SF_mode and self.token_num is not None:
            # [1,1,N,1]
            if self.softmax_flag:
                self.tau = nn.Parameter(torch.ones(1, 1, self.token_num, 1))

            init_weight_mat = (torch.eye(self.token_num) * 1.0 +
                               torch.randn(self.token_num, self.token_num) * 1.0)
            # ablation
            # init_weight_mat = (torch.eye(self.token_num) * 0.0 +
            #                    torch.randn(self.token_num, self.token_num) * 1.0)
            self.weight_mat = nn.Parameter(init_weight_mat[None, None, :, :].repeat(1, self.num_heads or 1, 1, 1))

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None, token_weight=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)
        # this_device = queries.device

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = scale * scores

        weight_mat = None
        ori_attn_mat = None
        if self.SF_mode:
            if not self.training and self.plot_mat_flag:
                ori_attn_mat = torch.softmax(A, dim=-1)

        # attention matrix adjustment; 240507
        if self.SF_mode and self.token_num is not None:
            # 2d
            if self.softmax_flag:
                weight_mat = F.softmax(self.weight_mat / F.softplus(self.tau), dim=-1)
            else:
                # use scale or not
                weight_mat = F.softplus(self.weight_mat)  # / sqrt(self.token_num)

        if self.SF_mode and weight_mat is not None:
            if self.outside_softmax:
                if self.weight_plus:
                    A = A.softmax(dim=-1) + weight_mat
                else:
                    A = A.softmax(dim=-1) * weight_mat
                A = F.normalize(A, p=1, dim=-1)
            else:
                if self.weight_plus:
                    if self.enhance_alpha:
                        alpha = F.softplus(self.alpha)
                        A = alpha * A + (1 - alpha) * weight_mat
                    else:
                        A = A + weight_mat
                else:
                    # ablation: this is a better choice
                    A = A * weight_mat

                # attention matrix [b,h,l,s]
                A = torch.softmax(A, dim=-1)

        else:
            A = torch.softmax(A, dim=-1)

        # plot mat
        if not self.training and self.plot_mat_flag and random.random() < 1:  # and A.shape[-1] > 10
            batch_idx = random.randint(0, A.shape[0] - 1)
            head_idx = random.randint(0, A.shape[1] - 1)

            # final
            att_mat_2d = A[batch_idx, head_idx, :, :]
            time_or_channel = 'channel'
            plot_mat(att_mat_2d, str_cat='final_attn_mat', str0=f"batch_{batch_idx}_head_{head_idx}_{time_or_channel}",
                     save_folder=self.save_folder)

            if ori_attn_mat is not None and self.SF_mode:
                ori_att_mat_2d = ori_attn_mat[batch_idx, head_idx, :, :]
                time_or_channel = 'channel'
                plot_mat(ori_att_mat_2d, str_cat='ori_attn_mat', str0=f"batch_{batch_idx}_head_{head_idx}_"
                                                                      f"{time_or_channel}",
                         save_folder=self.save_folder)

            if weight_mat is not None and self.SF_mode:
                batch_idx2 = min(batch_idx, weight_mat.shape[0] - 1)
                head_idx2 = min(head_idx, weight_mat.shape[1] - 1)
                weight_mat_2d = weight_mat[batch_idx2, head_idx2, :, :]

                plot_mat(weight_mat_2d, str_cat='adding_mat_2d',
                         str0=f"batch_{batch_idx2}_head_{head_idx2}_{time_or_channel}",
                         save_folder=self.save_folder)

            print(f'Attention matrix in FullAttention has been saved to {self.save_folder}...')

        # dropout, reserved
        A = self.dropout(A)

        # print(f'A.shape: {A.shape}')
        # print(f'values.shape: {values.shape}')
        V = torch.einsum("bhls,bshd->blhd", A, values)

        # plot gradient
        if self.plot_grad_flag and random.random() < 0.01 and self.weight_mat.grad is not None:
            batch_idx = random.randint(0, self.weight_mat.shape[0] - 1)
            head_idx = random.randint(0, self.weight_mat.shape[1] - 1)

            # final
            weight_mat_grad = self.weight_mat.grad[batch_idx, head_idx, :, :]
            time_or_channel = 'channel'
            plot_mat(weight_mat_grad, str_cat='weight_grad_mat',
                     str0=f"batch_{batch_idx}_head_{head_idx}_{time_or_channel}",
                     save_folder=self.save_folder)

            print(f'Attention gradient matrix in FullAttention has been saved to {self.save_folder}...')

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class FullAttention_ori(nn.Module):
    def __init__(self, mask_flag=False, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention_ori, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class LaserAttention(nn.Module):
    def __init__(self, mask_flag=False, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(LaserAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None, *args, **kwargs):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        # laser attention
        value_max = values.max(dim=1, keepdims=True)[0]
        values = (values - value_max).exp()

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values).contiguous()

        # laser attention
        V = V.log() + value_max

        if self.output_attention:
            return V, A
        else:
            return V, None


class LinearAttention(nn.Module):
    def __init__(self, mask_flag=False, factor=5, scale=None, attention_dropout=0.1, output_attention=False,
                 mapping_fun='softmax_learn', token_num=None, imp_mode=False, d_model=None,
                 plot_mat_flag=False, save_folder='./'):
        super(LinearAttention, self).__init__()
        # mask_flag, factor, scale, attention_dropout are not used
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.mapping_fun = mapping_fun
        self.plot_mat_flag = plot_mat_flag
        self.save_folder = save_folder

        self.softmax = nn.Softmax(dim=-1)

        self.imp_mode = imp_mode
        self.token_num = token_num
        self.d_model = d_model

        print(f'Linear Attention is used. mapping_fun: {self.mapping_fun}')

        if self.imp_mode and self.token_num:
            # [1,1,1,N]
            self.attn_adjust = nn.Parameter(torch.zeros(self.token_num)).unsqueeze(0).unsqueeze(0).unsqueeze(0)
            # [1,1,N,N]
            ij_mat = torch.ones(1, 1, token_num, token_num)
            # ij_mat = (torch.arange(token_num).view(token_num, 1) -
            #           torch.arange(token_num).view(1, token_num)).pow(2).unsqueeze(0).unsqueeze(0)
            self.register_buffer('ij_mat', ij_mat)

        if self.mapping_fun == 'softmax_learn':
            self.delta1 = nn.Parameter(torch.tensor(0.0))
            # self.delta1 = torch.tensor(0.0)
            # self.delta2 = nn.Parameter(torch.tensor(-1.0))
        elif self.mapping_fun == 'softmax_learn_v2':
            # assert self.token_num is not None, 'token_num should not be NAN'
            self.tau_q = nn.Parameter(torch.zeros(1))
            # self.tau_k = nn.Parameter(torch.zeros(1))
        if self.mapping_fun == 'agent':
            self.n_agent = 4
            assert self.d_model is not None, 'self.d_model is None'
            self.agent_linear = nn.Linear(self.d_model, self.d_model)
            self.agent_conv1d = nn.Conv1d(self.token_num, self.n_agent, kernel_size=1)

        assert mapping_fun in ['softmax_learn', 'softmax_q_k', 'x_3', 'relu', 'elu_plus_1', 'agent', 'softmax_learn_v2']

    def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None, token_weight=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape

        if self.mapping_fun == 'agent':
            assert L == self.token_num, 'input does not match with LinearAttention'
        # [b l h d] --> [b h l d]
        q, k, v = queries.transpose(1, 2), keys.transpose(1, 2), values.transpose(1, 2)
        if self.mapping_fun == 'softmax_learn':
            q[q < 0] = -20
            k[k < 0] = -20
            q = self.softmax(q / nn.Softplus()(self.delta1))
            k = self.softmax(k / nn.Softplus()(self.delta1))
        elif self.mapping_fun == 'softmax_learn_v2':
            q = torch.where(q < 0, -20, q)
            k = torch.where(k < 0, -20, k)
            q = self.softmax(q / F.softplus(self.tau_q))
            k = self.softmax(k / F.softplus(self.tau_q))

        elif self.mapping_fun == 'softmax_q_k':
            q = self.softmax(q)
            # softmax2
            softmax2 = nn.Softmax(dim=-2)
            k = softmax2(k)
        elif self.mapping_fun == 'x_3':
            # x**3 and relu
            q = nn.ReLU()(q)
            k = nn.ReLU()(k)
            # x**3
            q_norm = q.norm(dim=-1, keepdim=True)
            k_norm = k.norm(dim=-1, keepdim=True)
            q = q ** 3
            k = k ** 3
            q = q / (q.norm(dim=-1, keepdim=True) + 1e-6) * q_norm.clone()  # clone() to make it independent
            k = k / (k.norm(dim=-1, keepdim=True) + 1e-6) * k_norm.clone()
        elif self.mapping_fun == 'relu':
            q = nn.ReLU()(q)
            k = nn.ReLU()(k)
        elif self.mapping_fun == 'elu_plus_1':
            # elu+1
            q = F.elu(q) + 1
            k = F.elu(k) + 1
        elif self.mapping_fun == 'agent':
            # agent [b,h,n,d] --> [b,h,N,d]
            agent1 = (self.agent_conv1d(self.agent_linear(v).flatten(start_dim=0, end_dim=1))
                      .view(B, H, -1, q.shape[-1]))

            # q * agent1.T  -->  softmax
            q = self.softmax(torch.matmul(q, agent1.transpose(-1, -2)))  # [b,h,N,n]
            # agent1.T * k.T  -->  softmax
            k = self.softmax(torch.matmul(agent1, k.transpose(-1, -2))).transpose(-1, -2)  # [b,h,N,n]

        special_mode = self.imp_mode and self.token_num is not None and token_weight is not None

        output = None  # to make pycharm happy
        if not special_mode:
            kv = torch.einsum("bhdl, bhle -> bhde", k.transpose(-1, -2), v)
            z = 1 / (torch.einsum("bhld, bhd -> bhl", q, k.transpose(-1, -2).sum(dim=-1)) + 1e-6)
            # output should be blhd. Bug here! 0508
            output = torch.einsum("bhle, bhed, bhl -> blhd", q, kv, z)

            if not self.training and self.plot_mat_flag and random.random() < 0.01:
                A = torch.einsum("bhle,bhse->bhls", q, k)
                A = F.normalize(A, p=1, dim=-1)
                batch_idx = random.randint(0, A.shape[0] - 1)
                head_idx = random.randint(0, A.shape[1] - 1)
                att_mat_2d = A[batch_idx, head_idx, :, :]
                plot_mat(att_mat_2d, str_cat='linear_attn_mat', str0=f"{self.mapping_fun}_"
                                                                     f"batch_{batch_idx}_head_{head_idx}",
                         save_folder=self.save_folder)
                print(f'Attention matrix in LinearAttention has been saved to {self.save_folder}...')

        attn_weights = None
        if self.output_attention or special_mode:
            # even though we do not explicitly visit attention matrix usually in linear attention, but here we are:
            attn_scores = torch.einsum("bhld, bhLd -> bhlL", q, k)
            attn_weights = F.normalize(attn_scores, p=1, dim=-1)

            if special_mode:
                # token_weight: [b,l] --> [b,1,1,l]
                # attn_adjust has to be positive
                weight_mat = -self.ij_mat / F.softplus(self.attn_adjust) * token_weight.unsqueeze(1).unsqueeze(1)
                weight_mat = weight_mat.exp()
                attn_weights = attn_weights * weight_mat
                attn_weights = F.normalize(attn_weights, p=1, dim=-1)

                output = torch.einsum("bhls,bhsd->blhd", attn_weights, v)

        return output.contiguous(), attn_weights


# Code implementation from https://github.com/zhouhaoyi/Informer2020
class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        print('ProbAttention is used...')

    def _prob_QK(self, Q, K, sample_k, n_top):  # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        # real U = U_part(factor*ln(L_k))*L_q
        index_sample = torch.randint(L_K, (L_Q, sample_k))
        K_sample = K_expand[:, :, torch.arange(
            L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(
            Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   M_top, :]  # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H,
                                                L_Q, V_sum.shape[-1]).clone()
        else:  # use mask
            # requires that L_Q == L_V, i.e. for self-attention only
            assert (L_Q == L_V)
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None],
        torch.arange(H)[None, :, None],
        index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) /
                     L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[
                                                  None, :, None], index, :] = attn
            return context_in, attns
        else:
            return context_in, None

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None, *args, **kwargs):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        U_part = self.factor * \
                 np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k)
        u = self.factor * \
            np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        scores_top, index = self._prob_QK(
            queries, keys, sample_k=U_part, n_top=u)

        # add scale factor
        scale = self.scale or 1. / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(
            context, values, scores_top, index, L_Q, attn_mask)

        return context.contiguous(), attn


class dynamic_projection(nn.Module):
    def __init__(self, dim1, dim2):
        super().__init__()
        self.dim1 = dim1
        self.mlp = nn.Linear(dim1, dim2)

    def forward(self, src):
        # src: b, n, d
        assert src.shape[-1] == self.dim1
        src_dp = self.mlp(src)
        src_dp = F.softmax(src_dp, dim=-1)
        src_dp = torch.einsum('bef,bec -> bcf', src, src_dp)
        return src_dp


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None, dp_rank=None, imp_mode=False):
        super(AttentionLayer, self).__init__()

        self.imp_mode = imp_mode
        self.n_heads = n_heads
        self.dp_rank = dp_rank

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)

        if dp_rank:
            self.dp_key = dynamic_projection(d_keys * n_heads, dp_rank)
            self.dp_value = dynamic_projection(d_values * n_heads, dp_rank)

        self.out_projection = nn.Linear(d_values * n_heads, d_model)

    def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None, token_weight=None, **kwargs):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys)
        values = self.value_projection(values)

        if self.dp_rank:
            S = self.dp_rank
            keys = self.dp_key(keys)
            values = self.dp_value(values)

        keys = keys.view(B, S, H, -1)
        values = values.view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask=attn_mask,
            tau=tau,
            delta=delta,
            token_weight=token_weight.to(queries.device) if token_weight is not None else None
        )
        # [b,l,h,s]
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


class ReformerLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None, causal=False, bucket_size=4, n_hashes=4):
        super().__init__()
        self.bucket_size = bucket_size
        self.attn = LSHSelfAttention(
            dim=d_model,
            heads=n_heads,
            bucket_size=bucket_size,
            n_hashes=n_hashes,
            causal=causal
        )
        print('ReformerLayer is used...')

    def fit_length(self, queries):
        # inside reformer: assert N % (bucket_size * 2) == 0
        B, N, C = queries.shape
        if N % (self.bucket_size * 2) == 0:
            return queries
        else:
            # fill the time series
            fill_len = (self.bucket_size * 2) - (N % (self.bucket_size * 2))
            return torch.cat([queries, torch.zeros([B, fill_len, C]).to(queries.device)], dim=1)

    def forward(self, queries, keys, values, attn_mask, tau, delta, *args, **kwargs):
        # in Reformer: defalut queries=keys
        if queries.ndim > 3:
            queries = queries.flatten(2)
        B, N, C = queries.shape

        queries = self.attn(self.fit_length(queries))[:, :N, :]
        return queries, None
