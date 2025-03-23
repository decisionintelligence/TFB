import torch.nn as nn
import sys
import torch
from .ElasTST_SubLayers import (
    PositionwiseFeedForward,
    MultiHeadAttention_tem_bias,
    MultiHeadAttention_type_bias,
)
from einops import rearrange, repeat


PAD = 0


def get_attn_key_pad_mask_K(
    past_value_indicator, observed_indicator, transpose=False, structured_mask=False
):
    """For masking out the padding part of key sequence.
    input: mask: transpose=False: [b k l]
    """

    if structured_mask:
        mask = past_value_indicator
    else:
        mask = observed_indicator

    if transpose:
        mask = rearrange(mask, "b l k -> b k l")
    padding_mask = repeat(mask, "b k l1 -> b k l2 l1", l2=mask.shape[-1]).eq(PAD)

    return padding_mask


class EncoderLayer(nn.Module):
    """Compose with two layers"""

    def __init__(
        self,
        d_model,
        d_inner,
        n_head,
        d_k,
        d_v,
        dropout=0.1,
        tem_att=True,
        type_att=False,
        structured_mask=True,
        rotate=False,
        max_seq_len=100,
        theta=10000,
        addv=False,
        learnable_theta=False,
        bin_att=False,
        rope_theta_init="exp",
        min_period=0.1,
        max_period=10,
    ):
        super(EncoderLayer, self).__init__()

        self.structured_mask = structured_mask
        self.tem_att = tem_att
        self.type_att = type_att

        if tem_att:
            self.slf_tem_attn = MultiHeadAttention_tem_bias(
                n_head,
                d_model,
                d_k,
                d_v,
                dropout=dropout,
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

        if type_att:
            self.slf_type_attn = MultiHeadAttention_type_bias(
                n_head,
                d_model,
                d_k,
                d_v,
                dropout=dropout,
                rotate=False,
                max_seq_len=max_seq_len,
                bin_att=bin_att,
            )

        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, input, past_value_indicator=None, observed_indicator=None):
        # time attention
        # [B, K, L, D]
        if self.tem_att:
            tem_mask = get_attn_key_pad_mask_K(
                past_value_indicator=past_value_indicator,
                observed_indicator=observed_indicator,
                transpose=False,
                structured_mask=self.structured_mask,
            )
            tem_output = self.layer_norm(input)

            tem_output, enc_tem_attn = self.slf_tem_attn(
                tem_output, tem_output, tem_output, mask=tem_mask
            )

            tem_output = tem_output + input
        else:
            tem_output = input

        tem_output = rearrange(tem_output, "b k l d -> b l k d")

        # type attention
        # [B, L, K, D]
        if self.type_att:
            type_mask = get_attn_key_pad_mask_K(
                past_value_indicator=past_value_indicator,
                observed_indicator=observed_indicator,
                transpose=True,
                structured_mask=self.structured_mask,
            )

            type_output = self.layer_norm(tem_output)

            type_output, enc_type_attn = self.slf_type_attn(
                type_output, type_output, type_output, mask=type_mask
            )

            enc_output = type_output + tem_output
        else:
            enc_output = tem_output

        # FFNN
        output = self.layer_norm(enc_output)

        output = self.pos_ffn(output)

        output = output + enc_output

        output = rearrange(output, "b l k d -> b k l d")

        # optional
        output = self.layer_norm(output)

        return output  # , enc_tem_attn, enc_type_attn
