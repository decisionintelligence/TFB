__all__ = ["PatchTST_backbone"]

# Cell
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import numpy as np
from einops import rearrange, repeat
from ..utils.ElasTST_position_emb import Time_Encoder, sin_cos_encoding
from .ElasTST_Layers import EncoderLayer


# Cell
class ElasTST_backbone(nn.Module):
    def __init__(
        self,
        l_patch_size: list,
        stride: int = None,
        k_patch_size: int = 1,
        in_channels: int = 1,
        n_layers: int = 0,
        t_layers: int = 1,
        v_layers: int = 1,
        hidden_size: int = 256,
        n_heads: int = 16,
        d_k: Optional[int] = None,
        d_v: Optional[int] = None,
        d_inner: int = 256,
        dropout: float = 0.0,
        rotate: bool = False,
        max_seq_len=1000,
        theta=10000,
        learnable_theta=False,
        addv: bool = False,
        bin_att: bool = False,
        abs_tem_emb: bool = False,
        learn_tem_emb: bool = False,
        structured_mask: bool = True,
        rope_theta_init: str = "exp",
        min_period: float = 1,
        max_period: float = 1000,
        patch_share_backbone: bool = True,
    ):

        super().__init__()

        if rotate:
            print(
                f"Using Rotary Embedding... [theta init]: {rope_theta_init}, [period range]: [{min_period},{max_period}], [learnable]: {learnable_theta}"
            )
        print(
            "[Binary Att.]: ",
            bin_att,
            " [Learned time emb]: ",
            learn_tem_emb,
            " [Abs time emb]: ",
            abs_tem_emb,
        )
        print("[Multi Patch Share Backbone]: ", patch_share_backbone)
        print("[Structured Mask]: ", not structured_mask)
        # Patching
        self.l_patch_size = l_patch_size
        self.k_patch_size = k_patch_size
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_share_backbone = patch_share_backbone
        self.abs_tem_emb = abs_tem_emb

        self.hidden_size = hidden_size
        if stride is not None:
            self.stride = stride
        else:
            self.stride = self.l_patch_size

        x_embedder = []
        final_layer = []
        backbone = []
        for p in self.l_patch_size:
            print(f"=== Patch {p} Branch ===")
            x_embedder.append(
                TimePatchEmbed(
                    p,
                    self.k_patch_size,
                    self.in_channels,
                    self.hidden_size,
                    bias=True,
                    stride=p,
                )
            )
            final_layer.append(
                MLP_FinalLayer(
                    self.hidden_size, p, self.k_patch_size, self.out_channels
                )
            )

            if not patch_share_backbone:
                backbone.append(
                    DoublyAtt(
                        d_model=self.hidden_size,
                        n_layers=n_layers,
                        t_layers=t_layers,
                        v_layers=v_layers,
                        d_inner=d_inner,
                        n_heads=n_heads,
                        d_k=d_k,
                        d_v=d_v,
                        dropout=dropout,
                        rotate=rotate,
                        max_seq_len=max_seq_len,
                        theta=theta,
                        addv=addv,
                        bin_att=bin_att,
                        learnable_theta=learnable_theta,
                        structured_mask=structured_mask,
                        rope_theta_init=rope_theta_init,
                        min_period=min_period,
                        max_period=max_period,
                    )
                )

        self.x_embedder = nn.ModuleList(x_embedder)
        self.final_layer = nn.ModuleList(final_layer)

        if not patch_share_backbone:
            self.backbone = nn.ModuleList(backbone)
        else:
            self.backbone = DoublyAtt(
                d_model=self.hidden_size,
                n_layers=n_layers,
                t_layers=t_layers,
                v_layers=v_layers,
                d_inner=d_inner,
                n_heads=n_heads,
                d_k=d_k,
                d_v=d_v,
                dropout=dropout,
                rotate=rotate,
                max_seq_len=max_seq_len,
                theta=theta,
                addv=addv,
                bin_att=bin_att,
                learnable_theta=learnable_theta,
                structured_mask=structured_mask,
                rope_theta_init=rope_theta_init,
                min_period=min_period,
                max_period=max_period,
            )

        self.learn_tem_emb = learn_tem_emb
        if self.learn_tem_emb:
            self.learn_time_embedding = Time_Encoder(self.hidden_size)

    def get_patch_num(self, dim_size, len_size, l_patch_size):
        num_k_patches = int((dim_size - self.k_patch_size) / self.k_patch_size + 1)
        num_l_patches = int((len_size - l_patch_size) / l_patch_size + 1)
        return num_k_patches, num_l_patches

    def forward(
        self,
        past_target,
        future_placeholder,
        past_observed_values,
        future_observed_values,
        dataset_name=None,
    ):  # z: [bs x nvars x seq_len]

        pred_shape = future_placeholder.shape
        future_observed_indicator = torch.zeros(future_observed_values.shape).to(
            future_observed_values.device
        )

        x = torch.cat((past_target, future_placeholder), dim=1)  # B L+T K

        past_value_indicator = torch.cat(
            (past_observed_values, future_observed_indicator), dim=1
        )  # B L+T K
        observed_value_indicator = torch.cat(
            (past_observed_values, future_observed_values), dim=1
        )  # B L+T K

        pred_list = []

        for idx in range(len(self.l_patch_size)):

            x_p = x.clone()

            num_k_patches, num_l_patches = self.get_patch_num(
                x_p.shape[-1], x_p.shape[-2], self.l_patch_size[idx]
            )

            # do patching
            x_p, past_value_indicator_p, observed_value_indicator_p = self.x_embedder[
                idx
            ](
                x_p, past_value_indicator, observed_value_indicator
            )  # b k l d

            if self.learn_tem_emb:
                grid_len = np.arange(num_l_patches, dtype=np.float32)
                grid_len = (
                    torch.tensor(grid_len, requires_grad=False)
                    .float()
                    .unsqueeze(0)
                    .to(x.device)
                )
                pos_embed = repeat(grid_len, "1 l -> b l", b=pred_shape[0])
                pos_embed = self.learn_time_embedding(pos_embed)  # b l 1 d
                pos_embed = rearrange(pos_embed, "b l 1 d -> b 1 l d")
                x_p = x_p + pos_embed

            # use a absolute position embedding
            if self.abs_tem_emb:
                B, K, L, embed_dim = x_p.shape
                pos_embed = sin_cos_encoding(B, K, L, embed_dim).float()  # b k l d
                x_p = x_p + pos_embed.to(x_p.device)

            # model
            if self.patch_share_backbone:
                x_p = self.backbone(
                    x_p, past_value_indicator_p, observed_value_indicator_p
                )  # b k l d
            else:
                x_p = self.backbone[idx](
                    x_p, past_value_indicator_p, observed_value_indicator_p
                )  # b k l d

            x_p = self.final_layer[idx](x_p)  # b k l p

            x_p = rearrange(x_p, "b k t p -> b (t p) k")

            x_p = x_p[:, -pred_shape[1] :, :]

            pred_list.append(x_p.unsqueeze(-1))

        pred_list = torch.cat(pred_list, dim=-1)
        multi_patch_mean_res = torch.mean(pred_list, dim=-1)

        return multi_patch_mean_res, pred_list


class DoublyAtt(nn.Module):
    def __init__(
        self,
        d_model,
        n_layers,
        d_inner,
        n_heads,
        d_k,
        d_v,
        dropout,
        rotate=False,
        max_seq_len=1024,
        theta=10000,
        t_layers=2,
        v_layers=1,
        bin_att=False,
        addv=False,
        learnable_theta=False,
        structured_mask=True,
        rope_theta_init="exp",
        min_period=0.1,
        max_period=10,
    ):
        super().__init__()
        # assert n_layers <= (t_layers + v_layers) <= 2*n_layers , "Sum of t_layers and n_layers must be between 1 and 2"

        # Configuration based on temporal and variate ratios
        self.layer_stack = nn.ModuleList()
        num_t = t_layers
        num_v = v_layers
        num_both = min(t_layers, v_layers)

        num_t = num_t - num_both
        num_v = num_v - num_both

        t_count = 0
        v_count = 0
        for _ in range(num_t + num_v):
            if t_count < num_t:
                self.layer_stack.append(
                    EncoderLayer(
                        d_model,
                        d_inner,
                        n_heads,
                        d_k,
                        d_v,
                        dropout=dropout,
                        tem_att=True,
                        type_att=False,
                        structured_mask=structured_mask,
                        rotate=rotate,
                        max_seq_len=max_seq_len,
                        theta=theta,
                        addv=addv,
                        learnable_theta=learnable_theta,
                        bin_att=bin_att,
                        rope_theta_init=rope_theta_init,
                        min_period=min_period,
                        max_period=max_period,
                    )
                )
                t_count = t_count + 1
                print(f"[Encoder Layer {t_count+v_count}] Use tem att")
            if v_count < num_v:
                self.layer_stack.append(
                    EncoderLayer(
                        d_model,
                        d_inner,
                        n_heads,
                        d_k,
                        d_v,
                        dropout=dropout,
                        tem_att=False,
                        type_att=True,
                        structured_mask=structured_mask,
                        rotate=rotate,
                        max_seq_len=max_seq_len,
                        theta=theta,
                        addv=addv,
                        learnable_theta=learnable_theta,
                        bin_att=bin_att,
                        rope_theta_init=rope_theta_init,
                        min_period=min_period,
                        max_period=max_period,
                    )
                )
                v_count = v_count + 1
                print(f"[Encoder Layer {t_count+v_count}] Use var att")

        for idx in range(num_both):
            self.layer_stack.append(
                EncoderLayer(
                    d_model,
                    d_inner,
                    n_heads,
                    d_k,
                    d_v,
                    dropout=dropout,
                    tem_att=True,
                    type_att=True,
                    structured_mask=structured_mask,
                    rotate=rotate,
                    max_seq_len=max_seq_len,
                    theta=theta,
                    addv=addv,
                    learnable_theta=learnable_theta,
                    bin_att=bin_att,
                    rope_theta_init=rope_theta_init,
                    min_period=min_period,
                    max_period=max_period,
                )
            )

            print(f"[Encoder Layer {idx+t_count+v_count}] Use tem and var att")

    def forward(self, x, past_value_indicator, observed_indicator) -> Tensor:

        for enc_layer in self.layer_stack:
            x = enc_layer(
                x,
                past_value_indicator=past_value_indicator,
                observed_indicator=observed_indicator,
            )

        return x


class MLP_FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, hidden_size, l_patch_size, k_patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(
            hidden_size, l_patch_size * k_patch_size * out_channels, bias=True
        )

    def forward(self, x):
        x = self.norm_final(x)
        x = self.linear(x)
        return x


class TimePatchEmbed(nn.Module):
    """Time Patch Embedding"""

    def __init__(
        self,
        l_patch_size: int = 16,
        k_patch_size=1,
        in_chans: int = 1,
        embed_dim: int = 768,
        norm_layer: Optional[Callable] = None,
        flatten: bool = False,
        bias: bool = True,
        # padding_patch = None,
        stride=None,
        # strict_img_size: bool = True,
    ):
        super().__init__()
        self.l_patch_size = l_patch_size
        self.k_patch_size = k_patch_size
        if stride is None:
            stride = l_patch_size

        self.flatten = flatten

        padding = 0
        kernel_size = (l_patch_size, k_patch_size)
        stride_size = (stride, k_patch_size)

        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=kernel_size,
            stride=stride_size,
            bias=bias,
            padding=padding,
        )
        self.mask_proj = nn.Conv2d(
            1,
            1,
            kernel_size=kernel_size,
            stride=stride_size,
            bias=False,
            padding=padding,
        )

        self.mask_proj.weight.data.fill_(1.0)

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x, future_mask, obv_mask):
        """
        future_mask: only past values are set to 1
        obv_mask: past values and values to be predicted are set to 1
        """

        # B, C, K, L = x.shape
        if len(x.shape) == 3:
            x = rearrange(x, "b l k -> b 1 l k")

        future_mask = rearrange(future_mask, "b l k -> b 1 l k")
        obv_mask = rearrange(obv_mask, "b l k -> b 1 l k")

        x = self.proj(x)  # B C L K -> B C L' K

        with torch.no_grad():
            future_mask = self.mask_proj(future_mask)
            obv_mask = self.mask_proj(obv_mask)

        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
            future_mask = future_mask.flatten(2).transpose(1, 2)  # NCHW -> NLC
            obv_mask = obv_mask.flatten(2).transpose(1, 2)  # NCHW -> NLC

        x = self.norm(x)

        x = rearrange(x, "b d l k -> b k l d")
        future_mask = rearrange(future_mask, "b 1 l k -> b k l")
        obv_mask = rearrange(obv_mask, "b 1 l k -> b k l")
        return x, future_mask, obv_mask
