import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from ts_benchmark.baselines.phaseformer.layers.SelfAttention_Family import AttentionLayer, FullAttention


class RevIN(nn.Module):
    """
    Reversible Instance Normalization over time (per-sample, per-variable).

    Normalizes inputs along the temporal axis for each sample and variable.
    Input is shaped (B, L, C). The stored statistics allow exact de-normalization
    at the output stage so predictions can be mapped back to the original scale.
    """

    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = False):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(1, 1, num_features))
            self.bias = nn.Parameter(torch.zeros(1, 1, num_features))

    def normalize(self, x):  # x: (B, L, C)
        mu = x.mean(dim=1, keepdim=True)  # (B,1,C)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        sigma = (var + self.eps).sqrt()
        xn = (x - mu) / sigma
        if self.affine:
            xn = xn * self.weight + self.bias
        return xn, (mu, sigma)

    def denormalize(self, y, stats):  # y: (B, L', C)
        mu, sigma = stats
        return y * sigma + mu


class CrossPhaseRoutingLayer(nn.Module):

    def __init__(
        self,
        latent_dim: int,
        num_routers: int = 8,
        num_heads: int = 4,
        dropout: float = 0.0,
        use_relpos: bool = True,
        period_len: int = 24,
        window_size: Optional[int] = None,
        attention_dim: Optional[int] = None,
        use_pos_embed: bool = False,
        pos_dropout: float = 0.0,
    ):
        super().__init__()
        # The attention_dim parameter is kept for interface compatibility; it does not
        # change the projection dimensions in this implementation.
        self.attention_dim = attention_dim or latent_dim
        assert (
            self.attention_dim % num_heads == 0
        ), "attention_dim must be divisible by num_heads"

        self.latent_dim = latent_dim
        self.num_routers = num_routers
        self.num_heads = num_heads
        self.head_dim = self.attention_dim // num_heads
        self.dropout = dropout
        self.use_pos_embed = use_pos_embed
        self.period_len = period_len

        # Learnable routers shared across batch and channels
        self.router = nn.Parameter(torch.randn(num_routers, latent_dim))
        nn.init.trunc_normal_(self.router, std=0.02)

        # Optional phase positional embeddings (length equals period_len)
        if self.use_pos_embed:
            self.pos_embedding = nn.Parameter(torch.zeros(period_len, latent_dim))
            nn.init.trunc_normal_(self.pos_embedding, std=0.02)
            self.pos_dropout = nn.Dropout(pos_dropout)

        # Two-stage attention: routers aggregate then distribute
        self.router_sender = AttentionLayer(
            FullAttention(
                False, factor=5, attention_dropout=dropout, output_attention=False
            ),
            latent_dim,
            num_heads,
        )
        self.router_receiver = AttentionLayer(
            FullAttention(
                False, factor=5, attention_dropout=dropout, output_attention=False
            ),
            latent_dim,
            num_heads,
        )

        # Post-attention residual + LayerNorm + MLP
        self.norm1 = nn.LayerNorm(latent_dim)
        self.norm2 = nn.LayerNorm(latent_dim)
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, 4 * latent_dim),
            nn.GELU(),
            nn.Linear(4 * latent_dim, latent_dim),
        )

        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, Z):  # Z: (B, C, L, D)
        B, C, L, D = Z.shape
        x = Z.view(B * C, L, D)

        # Optional positional embedding
        if self.use_pos_embed:
            if L == self.period_len:
                pe = self.pos_embedding.unsqueeze(0).expand(B * C, -1, -1)
            elif L < self.period_len:
                pe = self.pos_embedding[:L, :].unsqueeze(0).expand(B * C, -1, -1)
            else:
                repeat_factor = (L + self.period_len - 1) // self.period_len
                expanded_pe = self.pos_embedding.repeat(repeat_factor, 1)
                pe = expanded_pe[:L, :].unsqueeze(0).expand(B * C, -1, -1)
            x = x + pe
            x = self.pos_dropout(x)

        # Stage 1: routers aggregate token information
        batch_router = self.router.unsqueeze(0).expand(B * C, -1, -1)  # (BC, R, D)
        router_buffer, _ = self.router_sender(batch_router, x, x, attn_mask=None)

        # Stage 2: routers distribute information back to tokens
        router_receive, _ = self.router_receiver(
            x, router_buffer, router_buffer, attn_mask=None
        )

        # Residual + LayerNorm
        out = x + self.dropout_layer(router_receive)
        out = self.norm1(out)

        # MLP block + Residual + LayerNorm
        mlp_out = self.mlp(out)
        out = out + self.dropout_layer(mlp_out)
        out = self.norm2(out)

        # Restore shape back to (B, C, L, D)
        out = out.view(B, C, L, D)
        return out



class PhaseEmbedding(nn.Module):
    """Projects phase-series tokens (P_in) into the latent space (D) with optional MLP.

    This layer applies a linear projection (or small MLP) across the phase dimension
    and then normalizes with LayerNorm to stabilize training.
    """
    def __init__(
        self,
        p_in: int,
        latent_dim: int,
        hidden: int = 32,
        use_mlp: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.use_mlp = use_mlp
        self.norm = nn.LayerNorm(latent_dim)
        if use_mlp:
            self.projection = nn.Sequential(
                nn.Linear(p_in, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, latent_dim),
            )
        else:
            self.projection = nn.Linear(p_in, latent_dim)

    def forward(self, phase_series):  # (B, C, L, P_in)
        return self.norm(self.projection(phase_series))




class PhasePredictor(nn.Module):
    """Maps latent features to the desired number of output phase steps (P_out).

    By default this is a single linear layer with optional dropout. An optional
    small MLP can be enabled via configuration, but the default matches the
    original implementation (use_mlp=False).
    """
    
    def __init__(self, p_out: int, latent_dim: int, hidden: int, use_mlp: bool = False, dropout: float = 0.0):
        super().__init__()
        self.p_out = p_out
        self.use_mlp = use_mlp
        
        if use_mlp:
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, hidden),
                nn.ReLU(),
                nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
                nn.Linear(hidden, p_out)
            )
        else:
            self.decoder = nn.Linear(latent_dim, p_out)
            self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        
    def forward(self, z):
        """
        Input: z (B, C, L, latent_dim)
        Output: (B, C, L, p_out)
        """
        if self.use_mlp:
            return self.decoder(z)  # (B, C, L, p_out)
        else:
            z = self.dropout(z)
            return self.decoder(z)  # (B, C, L, p_out)


class CrossPhaseRoutingUnit(nn.Module):
    """Routing unit with optional linear in/out projections around cross-phase routing.

    Composition to preserve the original information flow:
    - First unit: apply_in_proj=False, apply_out_proj=True (produce P_in for next layer)
    - Middle units: apply_in_proj=True, apply_out_proj=True
    - Last unit: apply_in_proj=True, apply_out_proj=False (final P_out via top-level predictor)
    """

    def __init__(
        self,
        *,
        apply_in_proj: bool,
        apply_out_proj: bool,
        num_periods_input: int,
        latent_dim: int,
        phase_encoder_hidden: int,
        predictor_hidden: int,
        phase_attn_heads: int,
        phase_attn_dropout: float,
        phase_attn_use_relpos: bool,
        period_len: int,
        phase_attn_window=None,
        phase_attention_dim=None,
        phase_num_routers: int = 8,
        phase_use_pos_embed: bool = False,
        phase_pos_dropout: float = 0.0,
        phase_encoder_use_mlp: bool = False,
        phase_encoder_dropout: float = 0.0,
        predictor_use_mlp: bool = False,
        predictor_dropout: float = 0.0,
    ):
        super().__init__()
        self.apply_in_proj = apply_in_proj
        self.apply_out_proj = apply_out_proj

        if self.apply_in_proj:
            # Linear in-projection from P_in to latent_dim; include LayerNorm to match PhaseEmbedding(use_mlp=False)
            self.in_proj = nn.Sequential(
                nn.Linear(num_periods_input, latent_dim),
                nn.LayerNorm(latent_dim),
            )
        else:
            self.in_proj = None

        self.interact = CrossPhaseRoutingLayer(
            latent_dim=latent_dim,
            num_routers=phase_num_routers,
            num_heads=phase_attn_heads,
            dropout=phase_attn_dropout,
            use_relpos=phase_attn_use_relpos,
            period_len=period_len,
            window_size=phase_attn_window,
            attention_dim=phase_attention_dim,
            use_pos_embed=phase_use_pos_embed,
            pos_dropout=phase_pos_dropout,
        )

        if self.apply_out_proj:
            # Linear out-projection back to P_in for chaining to the next layer
            self.out_proj = nn.Linear(latent_dim, num_periods_input)
        else:
            self.out_proj = None

    def forward(self, phase_series, z_prev=None):
        # Inputs:
        #   phase_series: (B, C, L, P_in)
        #   z_prev: (B, C, L, D) or None (must be provided if apply_in_proj is False)
        if self.apply_in_proj:
            z_curr = self.in_proj(phase_series)
            if z_prev is not None:
                z = z_prev + z_curr
            else:
                z = z_curr
        else:
            assert z_prev is not None, "z_prev must be provided when apply_in_proj is False"
            z = z_prev

        z = self.interact(z)

        if self.out_proj is not None:
            y_phase_steps = self.out_proj(z)  # (B, C, L, P_in)
        else:
            y_phase_steps = None

        return z, y_phase_steps


class PhaseFormerBlock(nn.Module):
    """Legacy block kept for reference and minimal disruption of imports.

    It represents a single layer of the original design: Encoder -> Interaction -> Decoder.
    The new implementation uses top-level embedding and predictor with routing units
    in between. This class is unused in the new data path but preserved to avoid
    breaking external references.
    """

    def __init__(
        self,
        num_periods_input: int,
        num_periods_output: int,
        latent_dim: int,
        phase_encoder_hidden: int,
        predictor_hidden: int,
        phase_attn_heads: int,
        phase_attn_dropout: float,
        phase_attn_use_relpos: bool,
        period_len: int,
        phase_attn_window=None,
        phase_attention_dim=None,
        phase_num_routers: int = 8,
        phase_use_pos_embed: bool = False,
        phase_pos_dropout: float = 0.0,
        phase_encoder_use_mlp: bool = False,
        phase_encoder_dropout: float = 0.0,
        predictor_use_mlp: bool = False,
        predictor_dropout: float = 0.0,
    ):
        super().__init__()

        self.encoder = PhaseEmbedding(
            p_in=num_periods_input,
            latent_dim=latent_dim,
            hidden=phase_encoder_hidden,
            use_mlp=phase_encoder_use_mlp,
            dropout=phase_encoder_dropout,
        )

        self.interact = CrossPhaseRoutingLayer(
            latent_dim=latent_dim,
            num_routers=phase_num_routers,
            num_heads=phase_attn_heads,
            dropout=phase_attn_dropout,
            use_relpos=phase_attn_use_relpos,
            period_len=period_len,
            window_size=phase_attn_window,
            attention_dim=phase_attention_dim,
            use_pos_embed=phase_use_pos_embed,
            pos_dropout=phase_pos_dropout,
        )

        self.decoder = PhasePredictor(
            p_out=num_periods_output,
            latent_dim=latent_dim,
            hidden=predictor_hidden,
            use_mlp=predictor_use_mlp,
            dropout=predictor_dropout,
        )

    def forward(self, phase_series, z_prev=None):
        # phase_series: (B, C, L, P_in)
        # z_prev: (B, C, L, D) or None
        z_curr = self.encoder(phase_series)  # (B, C, L, D)
        if z_prev is not None:
            # residual aggregation across layers
            z = z_prev + z_curr
        else:
            z = z_curr

        z = self.interact(z)
        y_phase_steps = self.decoder(z)  # (B, C, L, P_out)
        return z, y_phase_steps


class Model(nn.Module):
    """
    PhaseFormer: phase-based modeling without cross-channel fusion.

    Pipeline:
    1) RevIN over time per variable
    2) Embedding -> [CrossPhaseRouting] x N -> Predictor produce future phase steps
    3) Reassemble to forecasting sequence and de-normalize
    """

    def __init__(self, configs):
        super().__init__()

        # basic configs
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.period_len = getattr(configs, "period_len", 24)

        # phase dimensions
        self.latent_dim = getattr(configs, "latent_dim", 8)
        self.phase_encoder_hidden = getattr(configs, "phase_encoder_hidden", 32)
        self.predictor_hidden = getattr(configs, "predictor_hidden", 64)

        # phase attention parameters
        self.phase_attn_heads = getattr(configs, "phase_attn_heads", 4)
        self.phase_attn_dropout = getattr(configs, "phase_attn_dropout", 0.0)
        self.phase_attn_use_relpos = getattr(configs, "phase_attn_use_relpos", True)
        self.phase_attn_window = getattr(configs, "phase_attn_window", None)
        self.phase_attention_dim = getattr(configs, "phase_attention_dim", None)
        self.phase_num_routers = getattr(configs, "phase_num_routers", 8)
        self.phase_use_pos_embed = getattr(configs, "phase_use_pos_embed", False)
        self.phase_pos_dropout = getattr(configs, "phase_pos_dropout", 0.0)

        # period calculations
        self.num_periods_input = (self.seq_len + self.period_len - 1) // self.period_len
        self.num_periods_output = (self.pred_len + self.period_len - 1) // self.period_len
        self.total_len_in = self.num_periods_input * self.period_len
        self.pad_seq_len = self.total_len_in - self.seq_len

        # RevIN normalization
        self.use_revin = getattr(configs, "use_revin", True)
        self.revin_affine = getattr(configs, "revin_affine", False)
        self.revin_eps = getattr(configs, "revin_eps", 1e-5)
        if self.use_revin:
            self.revin = RevIN(num_features=self.enc_in, eps=self.revin_eps, affine=self.revin_affine)

        # task name
        self.task_name = getattr(configs, "task_name", "long_term_forecast")

        # expose: embedding -> [CrossPhaseRouting] x N -> predictor (P_out)
        self.phase_layers = getattr(configs, "phase_layers", 1)

        # Top-level embedding: projects (B, C, L, P_in) -> (B, C, L, D)
        self.embedding = PhaseEmbedding(
            p_in=self.num_periods_input,
            latent_dim=self.latent_dim,
            hidden=self.phase_encoder_hidden,
            use_mlp=getattr(configs, "phase_encoder_use_mlp", False),
            dropout=getattr(configs, "phase_encoder_dropout", 0.0),
        )

        # Routing layers: Cross-phase routing with optional linear in/out projections
        routing_units = []
        if self.phase_layers == 1:
            routing_units.append(
                CrossPhaseRoutingUnit(
                    apply_in_proj=False,
                    apply_out_proj=False,
                    num_periods_input=self.num_periods_input,
                    latent_dim=self.latent_dim,
                    phase_encoder_hidden=self.phase_encoder_hidden,
                    predictor_hidden=self.predictor_hidden,
                    phase_attn_heads=self.phase_attn_heads,
                    phase_attn_dropout=self.phase_attn_dropout,
                    phase_attn_use_relpos=self.phase_attn_use_relpos,
                    period_len=self.period_len,
                    phase_attn_window=self.phase_attn_window,
                    phase_attention_dim=self.phase_attention_dim,
                    phase_num_routers=self.phase_num_routers,
                    phase_use_pos_embed=self.phase_use_pos_embed,
                    phase_pos_dropout=self.phase_pos_dropout,
                    phase_encoder_use_mlp=getattr(configs, "phase_encoder_use_mlp", False),
                    phase_encoder_dropout=getattr(configs, "phase_encoder_dropout", 0.0),
                    predictor_use_mlp=getattr(configs, "predictor_use_mlp", False),
                    predictor_dropout=getattr(configs, "predictor_dropout", 0.0),
                )
            )
        else:
            for li in range(self.phase_layers):
                is_first = li == 0
                is_last = li == self.phase_layers - 1
                routing_units.append(
                    CrossPhaseRoutingUnit(
                        apply_in_proj=not is_first,
                        apply_out_proj=not is_last,
                        num_periods_input=self.num_periods_input,
                        latent_dim=self.latent_dim,
                        phase_encoder_hidden=self.phase_encoder_hidden,
                        predictor_hidden=self.predictor_hidden,
                        phase_attn_heads=self.phase_attn_heads,
                        phase_attn_dropout=self.phase_attn_dropout,
                        phase_attn_use_relpos=self.phase_attn_use_relpos,
                        period_len=self.period_len,
                        phase_attn_window=self.phase_attn_window,
                        phase_attention_dim=self.phase_attention_dim,
                        phase_num_routers=self.phase_num_routers,
                        phase_use_pos_embed=self.phase_use_pos_embed,
                        phase_pos_dropout=self.phase_pos_dropout,
                        phase_encoder_use_mlp=getattr(configs, "phase_encoder_use_mlp", False),
                        phase_encoder_dropout=getattr(configs, "phase_encoder_dropout", 0.0),
                        predictor_use_mlp=getattr(configs, "predictor_use_mlp", False),
                        predictor_dropout=getattr(configs, "predictor_dropout", 0.0),
                    )
                )
        self.routing_layers = nn.ModuleList(routing_units)

        # Top-level predictor to P_out: maps (B, C, L, D) -> (B, C, L, P_out)
        self.predictor = PhasePredictor(
            p_out=self.num_periods_output,
            latent_dim=self.latent_dim,
            hidden=self.predictor_hidden,
            use_mlp=getattr(configs, "predictor_use_mlp", False),
            dropout=getattr(configs, "predictor_dropout", 0.0),
        )

    # phase rearrangement helpers
    @staticmethod
    def _to_phase_series(x_periods):
        """(B, C, P_in, L) -> (B, C, L, P_in)"""
        return x_periods.permute(0, 1, 3, 2).contiguous()

    @staticmethod
    def _from_phase_steps_to_periods(y_phase_steps):
        """(B, C, L, P_out) -> (B, C, P_out, L)"""
        return y_phase_steps.permute(0, 1, 3, 2).contiguous()

    # forward pass
    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, *args, **kwargs):
        """
        Input:  x_enc (B, seq_len, C)
        Output: y_hat (B, pred_len, C)
        Also returns intermediate Z (B,C,L,D) and future phase values (B,C,L,P_out) for analysis.
        """
        # 1) RevIN normalization
        if self.use_revin:
            # RevIN expects (B, C, L)
            x_in, stats = self.revin.normalize(x_enc)
        else:
            x_in = x_enc.float()
        # 2) Use original input (no cross-channel fusion)
        x_fused = x_in  # (B, L, C)

        # 3) Ring padding to full periods
        x = x_fused.permute(0, 2, 1)  # (B, C, L_total)
        B, C, L = x.shape
        if self.pad_seq_len > 0:
            x = F.pad(x, (0, self.pad_seq_len), mode="circular")  # (B, C, total_len_in)

        # 4) Split to periods (B, C, P_in, L)
        x_periods = x.view(B, C, self.num_periods_input, self.period_len)

        # 5) Parallel by phase view (B, C, L, P_in)
        phase_series = self._to_phase_series(x_periods)

        # 6-8) Embedding -> routing layers -> top predictor
        # Initial latent from embedding
        Z = self.embedding(phase_series)  # (B, C, L, D)
        phase_series_cur = phase_series

        for layer_index, unit in enumerate(self.routing_layers):
            Z, y_phase_steps_p_in = unit(phase_series_cur, Z)
            if layer_index < len(self.routing_layers) - 1:
                # intermediate layers must produce P_in for the next layer
                phase_series_cur = y_phase_steps_p_in

        # final predictor to produce P_out
        y_phase_steps = self.predictor(Z)  # (B, C, L, P_out)

        # 9) Reassemble to sequence (B, pred_len, C)
        y_periods = self._from_phase_steps_to_periods(y_phase_steps)  # (B, C, P_out, L)
        y_full = y_periods.reshape(B, C, -1)[..., : self.pred_len]  # (B, C, pred_len)
        y_hat = y_full.permute(0, 2, 1)  # (B, pred_len, C)

        # 10) De-normalization
        if self.use_revin:
            y_hat = self.revin.denormalize(y_hat, stats)

        return y_hat



