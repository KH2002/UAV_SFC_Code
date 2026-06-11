# -*- coding: utf-8 -*-
"""MAPPO 网络公共模块。"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        dropout: float = 0.0,
        use_layer_norm: bool = False,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        layer_dims = [input_dim]
        if num_layers == 1:
            layer_dims.append(output_dim)
        else:
            layer_dims.extend([hidden_dim] * (num_layers - 1))
            layer_dims.append(output_dim)

        self.layers = nn.ModuleList(
            [nn.Linear(layer_dims[i], layer_dims[i + 1]) for i in range(num_layers)]
        )
        self.pre_norms = nn.ModuleList(
            [nn.LayerNorm(layer_dims[i]) for i in range(num_layers)]
        ) if use_layer_norm else None
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, linear in enumerate(self.layers):
            if self.pre_norms is not None:
                x = self.pre_norms[i](x)
            x = linear(x)
            if i < len(self.layers) - 1:
                x = torch.relu(x)
                if self.dropout is not None:
                    x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """标准 pre-norm Transformer block (MHA + FFN)。"""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        ffn_multiplier: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * ffn_multiplier),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * ffn_multiplier, hidden_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        norm_x = self.norm1(x)
        attn_out, _ = self.mha(
            query=norm_x,
            key=norm_x,
            value=norm_x,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + self.dropout(attn_out)
        norm_x = self.norm2(x)
        ffn_out = self.ffn(norm_x)
        x = x + self.dropout(ffn_out)
        return x


def masked_mean(
    x: torch.Tensor,
    valid_mask: Optional[torch.Tensor],
    dim: int,
    eps: float = 1e-8,
) -> torch.Tensor:
    """对 dim 维做 masked mean。

    Args:
        x: [..., D]
        valid_mask: 与 x 在 dim 维同长度的 0/1 或 bool mask。True/1 表示有效。
    """
    if valid_mask is None:
        return x.mean(dim=dim)

    mask = valid_mask.float()
    while mask.ndim < x.ndim:
        mask = mask.unsqueeze(-1)

    numerator = (x * mask).sum(dim=dim)
    denominator = mask.sum(dim=dim).clamp_min(eps)
    return numerator / denominator
