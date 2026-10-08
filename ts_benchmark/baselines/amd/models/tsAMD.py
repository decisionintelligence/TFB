import torch
import torch.nn as nn

from ts_benchmark.baselines.amd.models.common import RevIN
from ts_benchmark.baselines.amd.models.common import DDI
from ts_benchmark.baselines.amd.models.common import MDM
from ts_benchmark.baselines.amd.models.tsmoe import AMS


class Model(nn.Module):
    """Implementation of AMD."""

    def __init__(self, configs):
        super(Model, self).__init__()

        input_shape = (configs.seq_len, configs.enc_in)
        pred_len = configs.pred_len
        n_block = configs.n_block
        dropout = configs.dropout
        patch = configs.patch
        k = configs.mix_layer_num
        c = configs.mix_layer_scale
        alpha = configs.alpha
        norm=configs.norm,
        layernorm=configs.layernorm
        self.norm = norm

        if self.norm:
            self.rev_norm = RevIN(input_shape[-1])

        self.pastmixing = MDM(input_shape, k=k, c=c, layernorm=layernorm)

        self.fc_blocks = nn.ModuleList([DDI(input_shape, dropout=dropout, patch=patch, alpha=alpha, layernorm=layernorm)
                                        for _ in range(n_block)])

        self.moe = AMS(input_shape, pred_len, ff_dim=2048, dropout=dropout, num_experts=8, top_k=2)

    def forward(self, x):
        # [batch_size, seq_len, feature_num]
        # layer norm    
        if self.norm:
            x = self.rev_norm(x, 'norm')
        # [batch_size, seq_len, feature_num]

        # [batch_size, seq_len, feature_num]
        x = x.permute(0,2,1)
        # [batch_size, feature_num, seq_len]

        time_embedding = self.pastmixing(x)

        for fc_block in self.fc_blocks:
            x = fc_block(x)

        # MOE
        x, moe_loss = self.moe(x, time_embedding)  # seq_len -> pred_len

        # [batch_size, feature_num, pred_len]
        x = x.permute(0,2,1)
        # [batch_size, pred_len, feature_num]

        if self.norm:
            x = self.rev_norm(x, 'denorm')
        # [batch_size, pred_len, feature_num]
        # x = x.permute(0,2,1)
        return x, moe_loss