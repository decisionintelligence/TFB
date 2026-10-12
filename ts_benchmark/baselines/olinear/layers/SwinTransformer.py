import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.tools import create_sin_pos_embed, create_swin_relative_index, get_relative_coords_table
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


def window_partition(x, window_size):
    """
    Args:
    x: (B, H, W, C)
    window_size (int): window size

    Returns:
    windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    window_size = var2tuple(window_size)
    window_size = fix_window(window_size, H, W)
    assert H % window_size[0] == 0 and W % window_size[1] == 0
    # print('x.shape: ', x.shape)
    x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
    windows: (num_windows*B, window_size, window_size, C)
    window_size (int): Window size
    H (int): Height of image
    W (int): Width of image

    Returns:
    x: (B, H, W, C)
    """
    window_size = var2tuple(window_size)
    window_size = fix_window(window_size, H, W)
    B = int(windows.shape[0] / (H * W / window_size[0] / window_size[1]))
    x = windows.view(B, H // window_size[0], W // window_size[1], window_size[0], window_size[1], -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


def fix_window(x, h, w, max_prod=math.inf):
    x = (x[0] if 0 < x[0] <= h else h, x[1] if 0 < x[1] <= w else w)
    if x[0] * x[1] > max_prod:
        if x[0] == h:
            x = (h, max(max_prod // h, 1))
        elif x[1] == w:
            x = (max(max_prod // w, 1), w)
    return x


def fix_shift(shift_size, window_size):
    shift_size = [min(a, b) for a, b in zip(shift_size, [(window_size[0] + 1) // 2, (window_size[1] + 1) // 2])]
    return shift_size


def win_padding(x, window_size):
    if x is None:
        return x, 0, 0

    B, H, W, C = x.shape
    window_size = var2tuple(window_size)
    window_size = fix_window(window_size, H, W)
    padding = [0, 0, 0, 0, 0, 0]

    rem1 = H % window_size[0]
    if rem1 != 0:
        padding[5] = window_size[0] - rem1

    rem2 = W % window_size[1]
    if rem2 != 0:
        padding[3] = window_size[1] - rem2

    x = F.pad(x, padding)
    return x, x.shape[1], x.shape[2]


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, shift_size=0, mask_flag=False, rel_pos_flag=True, DPB=True,
                 seq_len=96, mask_weight_flag=True):
        super().__init__()
        self.row_att_mat_index = None
        self.num_win = None
        self.dim = dim
        self.window_size = var2tuple(window_size)
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.shift_size = var2tuple(shift_size)
        self.mask_flag = mask_flag
        self.rel_pos_flag_ori = rel_pos_flag
        self.rel_pos_flag = rel_pos_flag and any(i > 0 for i in self.window_size)
        self.seq_len = seq_len
        self.mask_weight_flag = mask_weight_flag
        self.DPB = DPB

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

        if self.DPB:
            self.pow_mode = True

            self.cpb_mlp = nn.Sequential(nn.Linear(2, 512, bias=True),
                                         nn.ReLU(),
                                         nn.Linear(512, self.num_heads, bias=False))  # self.n_heads
            self.h_times = nn.Parameter(torch.tensor(2.0))
            # no log
            self.pow_para = nn.Parameter(torch.tensor(0.0))
            self.ws_scalar = nn.Parameter(torch.tensor(1.0)) if self.pow_mode else nn.Parameter(torch.tensor(5.0))
            self.ws_scalar2 = nn.Parameter(torch.tensor(2.0))  # 5-->3
            self.period_scale = nn.Parameter(torch.tensor(-3.0))
            self.relative_coords_table = None
            self.relative_position_bias_table = None
        elif self.rel_pos_flag:
            w_s = [a if a > 0 else self.seq_len for a in self.window_size]
            # define a parameter table of relative position bias # (2*Wh-1) * (2*Ww-1), nH
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * w_s[0] - 1) * (2 * w_s[1] - 1), num_heads))
            trunc_normal_(self.relative_position_bias_table, std=.02)

            if all(i > 0 for i in self.window_size):
                relative_position_index = create_swin_relative_index(self.window_size)
                self.relative_position_index = relative_position_index
            else:
                self.relative_position_index = None

        elif self.rel_pos_flag_ori:
            self.pos_table = nn.Parameter(torch.zeros(1, self.seq_len, self.dim))

        # for different rows
        self.tau = nn.Parameter(torch.tensor(-1.0))

    def create_mask(self, window_size, shift_size, res):
        if any(shift_size) and all(i > 0 for i in window_size):
            # there was a bug here!!!!
            img_mask = torch.zeros((res[0], res[1]))
            if 0 < shift_size[0] < window_size[0]:
                h_slices = (slice(0, -window_size[0]),
                            slice(-window_size[0], -shift_size[0]),
                            slice(-shift_size[0], None))
            else:
                h_slices = (slice(0, None),)
            if 0 < shift_size[1] < window_size[1]:
                w_slices = (slice(0, -window_size[1]),
                            slice(-window_size[1], -shift_size[1]),
                            slice(-shift_size[1], None))
            else:
                w_slices = (slice(0, None),)
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[h, w] = cnt
                    cnt += 1
        else:
            img_mask = torch.zeros((res[0], res[1]))

        # [num_windows, ws0, ws1]
        # window_partition: x: (1, H, W, 1), window_size --> (num_windows, window_size, window_size, 1)
        img_masks = window_partition(img_mask.unsqueeze(0).unsqueeze(-1), window_size).squeeze(-1)
        self.num_win = num_win = img_masks.shape[0]
        # mask the attention score: [num_windows, ws0*ws1,ws0*ws1]
        mask = img_masks.view(num_win, 1, -1) - img_masks.view(num_win, -1, 1)
        mask = mask.masked_fill(mask != 0, float('-inf')).masked_fill(mask == 0, float(0.0))
        # mask cannot be self.mask
        return mask

    def forward(self, x, window_size=None, shift_size=None, res=None, imp_mask=None):
        # imp_mask: [B*nW, w0*w1]

        window_change = 0 if window_size == self.window_size else 1
        if window_size is not None and window_change:
            self.window_size = window_size
        shift_change = 0 if shift_size == self.shift_size else 1
        if shift_size is not None and shift_change:
            self.shift_size = shift_size

        if self.mask_flag and any(self.shift_size) and all(i > 0 for i in self.window_size):
            swin_mask = self.create_mask(self.window_size, self.shift_size, res=res)
        else:
            swin_mask = None

        B, N, C = x.shape
        if self.rel_pos_flag_ori and not self.rel_pos_flag:
            x = x + self.pos_table[:, :N, :]

        # 3, B, num_heads, N, d
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # b,h,n,d
        q, k, v = qkv[0], qkv[1], qkv[2]

        # b,h,n,n
        attn = (q @ k.transpose(-2, -1)) * self.scale

        if self.DPB or self.rel_pos_flag:
            if window_change or self.relative_position_index is None:
                self.relative_position_index = create_swin_relative_index(self.window_size)

            relative_position_index = self.relative_position_index

            if self.DPB:
                # 1, 2*Wh-1, 2*Ww-1, 2  #  + F.sigmoid(self.period_scale) *  period / seq_len
                self.relative_coords_table = get_relative_coords_table(self.window_size,
                                                                       h_times=F.softplus(self.h_times),
                                                                       ws_scalar=F.softplus(self.ws_scalar),
                                                                       ws_scalar2=F.softplus(self.ws_scalar2),
                                                                       pow_para=F.sigmoid(self.pow_para),
                                                                       pow_mode=self.pow_mode).to(x.device)

                # 1, 2*Wh-1, 2*Ww-1, n_heads --> ()*(), n_heads
                self.relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.n_heads)

            if relative_position_index is not None:
                # relative position  # Wh*Ww,Wh*Ww,nH
                relative_position_bias = self.relative_position_bias_table[relative_position_index.view(-1)].view(
                    self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
                relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
                attn = attn + relative_position_bias.unsqueeze(0)

        if swin_mask is not None and self.num_win is not None:
            size1 = attn.shape
            # self.mask: [num_windows, ws0*ws1,ws0*ws1]
            attn = (attn.view(-1, self.num_win, self.num_heads, N, N) + swin_mask.unsqueeze(0).unsqueeze(2).to(
                attn.device))
            attn = attn.view(size1)  # Apply the mask

        if imp_mask is not None:
            # [b,n] --> [b,1,1,n]
            attn = attn * imp_mask.unsqueeze(1).unsqueeze(1)

        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


def var2tuple(x, num=2):
    num = int(num)
    if isinstance(x, tuple):
        if len(x) == num:
            return x
        elif len(x) > num:
            return x[:num]
        else:
            return x + (x[-1],) * (num - len(x))
    return (x,) * num


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution=None, num_heads=8, window_size=(5, 5), shift_size=(0, 0), mask_flag=False,
                 seq_len=96, mask_weight_flag=True, series_shift=False, pad_first=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.window_size = var2tuple(window_size)

        self.shift_size = var2tuple(shift_size)
        self.num_heads = num_heads
        self.mask_weight_flag = mask_weight_flag
        self.series_shift = series_shift
        self.pad_first = pad_first

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, self.window_size, num_heads, shift_size=self.shift_size, mask_flag=mask_flag,
                                    seq_len=seq_len, mask_weight_flag=mask_weight_flag)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
        )
        # self.pe = nn.Parameter(torch.zeros(1, 100, dim))

        self.tau = nn.Parameter(torch.tensor(0.0))

    def forward(self, x_tuple):
        # args: x:[b, h, w, c], mask: [b, h, w]

        # parse input
        if isinstance(x_tuple, tuple):
            x, mask = x_tuple
            mask = mask if self.mask_weight_flag else None
        else:
            x, mask = x_tuple, None

        mask_ori = mask

        B, H0, W0, C = x.shape
        shortcut = x

        if mask is not None:
            assert (B, H0, W0) == (mask.shape[0], mask.shape[1], mask.shape[2]), \
                "Check x and mask in SwinTransformerBlock..."

        if self.input_resolution is not None:
            H, W = self.input_resolution
            assert H == H0 and W == W0, "Input feature has wrong size in SwinTransformerBlock..."

        # adjust if window_size is too big
        window_size2 = fix_window(self.window_size, H0, W0)
        self.shift_size = fix_shift(self.shift_size, window_size2)
        if self.window_size[0] != window_size2[0] or self.window_size[1] != window_size2[1]:
            self.window_size = window_size2

        if self.pad_first:
            # pad --> shift
            # do some padding if needed
            x, H, W = win_padding(x, self.window_size)
            if mask is not None:
                mask = win_padding(mask.unsqueeze(-1), self.window_size)[0].squeeze(-1)
            # Shift window partition
            if any(self.shift_size):

                if self.series_shift:
                    # new way
                    # [B, H, W, C] --> [B, H*W, C] --> [B, shift(H*W), C] --> [B, H, W, C]
                    shifted_x = (torch.roll(x.flatten(start_dim=1, end_dim=2), shifts=-self.shift_size[1], dims=1)
                                 .view(B, H, W, -1))
                    if mask is not None:
                        mask = (torch.roll(mask.flatten(start_dim=1, end_dim=2), shifts=-self.shift_size[1], dims=1)
                                .view(B, H, W, -1))
                else:
                    # old method: use the traditional swin
                    shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
                    if mask is not None:
                        mask = torch.roll(mask, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))

            else:
                shifted_x = x
        else:
            # shift --> pad
            # Shift window partition
            if any(self.shift_size):

                if self.series_shift:
                    # new way
                    # [B, H, W, C] --> [B, H*W, C] --> [B, shift(H*W), C] --> [B, H, W, C]
                    shifted_x = (torch.roll(x.flatten(start_dim=1, end_dim=2), shifts=-self.shift_size[1], dims=1)
                                 .view(B, H0, W0, -1))
                    if mask is not None:
                        mask = (torch.roll(mask.flatten(start_dim=1, end_dim=2), shifts=-self.shift_size[1], dims=1)
                                .view(B, H0, W0, -1))
                else:
                    # old method: use the traditional swin
                    shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
                    if mask is not None:
                        mask = torch.roll(mask, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))

            else:
                shifted_x = x

            # do some padding if needed
            shifted_x, H, W = win_padding(shifted_x, self.window_size)
            if mask is not None:
                mask = win_padding(mask.unsqueeze(-1), self.window_size)[0].squeeze(-1)

        shifted_x = self.norm1(shifted_x)

        # Window partition
        # (B, H, W, C) --> (num_windows * B, window_size, window_size, C) -->
        # (num_windows * B, window_size * window_size, C)
        windows = window_partition(shifted_x, self.window_size).flatten(start_dim=1, end_dim=2)
        if mask is not None:
            # [B*nW, w0*w1]
            mask = window_partition(mask.unsqueeze(-1), self.window_size).flatten(start_dim=1, end_dim=2).squeeze(-1)
            # normalize
            mask = F.normalize(mask.pow(F.softplus(self.tau)), p=1, dim=-1)

        # Window attention: input b,n,c
        attn_windows = self.attn(windows, window_size=self.window_size, shift_size=self.shift_size,
                                 res=(H, W), imp_mask=mask)

        # Merge windows
        # (num_windows * B, window_size * window_size, C) --> (B, H, W, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        if self.pad_first:
            # Reverse shift
            if any(self.shift_size):

                if self.series_shift:
                    # new way
                    x = (torch.roll(shifted_x.flatten(start_dim=1, end_dim=2), shifts=self.shift_size[1], dims=1)
                         .view(B, H0, W0, -1))
                else:
                    # old way:
                    x = torch.roll(shifted_x, shifts=(self.shift_size[0], self.shift_size[1]), dims=(1, 2))

            else:
                x = shifted_x

            if H != H0 or W != W0:
                x = x[:, :H0, :W0, :]
        else:
            if H != H0 or W != W0:
                shifted_x = shifted_x[:, :H0, :W0, :]

            # Reverse
            if any(self.shift_size):

                if self.series_shift:
                    # new way
                    x = (torch.roll(shifted_x.flatten(start_dim=1, end_dim=2), shifts=self.shift_size[1], dims=1)
                         .view(B, H0, W0, -1))
                else:
                    # old way:
                    x = torch.roll(shifted_x, shifts=(self.shift_size[0], self.shift_size[1]), dims=(1, 2))

            else:
                x = shifted_x

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        return x, mask_ori


class SwinTransformerBlockTwice(nn.Module):
    def __init__(self, dim, input_resolution=None, c_in=7, c_out=7, num_heads=8,
                 window_size1=(5, 5), window_size2=(5, 5), shift_size=(3, 3), block_num=3,
                 attn_mask_flag=False, conv_patch_flag=True, seq_len=96, mask_weight_flag=True,
                 all_attn_flag=False):
        super().__init__()
        self.c_in = c_in
        self.num_heads = num_heads
        self.block_num = block_num
        self.attn_mask_flag = attn_mask_flag
        self.conv_patch_flag = conv_patch_flag
        self.all_attn_flag = all_attn_flag

        self.window_size1 = var2tuple(window_size1)
        self.window_size2 = var2tuple(window_size2)
        self.shift_size = var2tuple(shift_size)

        print(f'self.window_size1: {self.window_size1}')
        print(f'self.window_size2: {self.window_size2}')
        print(f'self.shift_size: {self.shift_size}')

        # conv as patch; input should be [n,c,h,w]
        if self.conv_patch_flag:
            self.conv_patch = nn.Conv2d(c_in, c_in, kernel_size=3, stride=1, padding='same', bias=True)

        self.proj0 = nn.Linear(c_in, dim) if c_in != dim else nn.Identity()

        self.proj1 = nn.Linear(dim, c_out) if c_out != dim else nn.Identity()

        # column and row attention
        self.block1 = nn.ModuleList([SwinTransformerBlock(dim, input_resolution, num_heads,
                                                          window_size=self.window_size1,
                                                          shift_size=(0, 0), seq_len=seq_len,
                                                          mask_weight_flag=mask_weight_flag, mask_flag=False)
                                     for _ in range(self.block_num)])

        if self.window_size2 is not None and not all(i == -1 for i in self.window_size1):

            self.block2 = nn.ModuleList([SwinTransformerBlock(dim, input_resolution, num_heads,
                                                              window_size=self.window_size2,
                                                              shift_size=self.shift_size,
                                                              mask_flag=self.attn_mask_flag, seq_len=seq_len,
                                                              mask_weight_flag=mask_weight_flag)
                                         for _ in range(self.block_num)])
            if self.all_attn_flag:
                self.block3 = nn.ModuleList([SwinTransformerBlock(dim, input_resolution, num_heads,
                                                                  window_size=(-1, -1),
                                                                  shift_size=0,
                                                                  mask_flag=self.attn_mask_flag, seq_len=seq_len,
                                                                  mask_weight_flag=mask_weight_flag)
                                             for _ in range(self.block_num)])

                self.all_swin_layers = nn.Sequential(*[layer for pair in zip(self.block1, self.block2, self.block3)
                                                       for layer in pair])  # , self.block3
            else:
                self.all_swin_layers = nn.Sequential(*[layer for pair in zip(self.block1, self.block2)
                                                       for layer in pair])
        else:
            print('Because of the setting of window_size2 or window_size1, only block1 is employed '
                  'in SwinTransformerBlockTwice...')
            self.all_swin_layers = nn.Sequential(*self.block1)

    def forward(self, x, mask=None):
        # args: x:[b,h,w,c], mask:[b,h,w]
        # [b,h,w,c] --> [b,c,h,w] --> [b,h,w,c]
        if self.conv_patch_flag:
            x = self.conv_patch(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        # [b,h,w,c]
        x = self.proj0(x)
        x, _ = self.all_swin_layers((x, mask))

        x = self.proj1(x)
        return x
