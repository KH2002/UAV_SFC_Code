# -*- coding: utf-8 -*-
"""MAPPO Critic：中心化价值网络。"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .common import MLP, TransformerBlock, masked_mean


class MAPPOCritic(nn.Module):
    def __init__(
        self,
        uav_state_dim: int,
        task_dim: int,
        context_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 4,
        num_uav_blocks: int = 1,
        num_task_blocks: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.uav_encoder = MLP(uav_state_dim, hidden_dim, hidden_dim, num_layers=2, dropout=dropout)
        self.task_encoder = MLP(task_dim, hidden_dim, hidden_dim, num_layers=2, dropout=dropout)
        self.context_encoder = MLP(context_dim, hidden_dim, hidden_dim, num_layers=2, dropout=dropout)

        self.uav_blocks = nn.ModuleList(
            [TransformerBlock(hidden_dim, num_heads=num_heads, dropout=dropout) for _ in range(num_uav_blocks)]
        )
        self.task_blocks = nn.ModuleList(
            [TransformerBlock(hidden_dim, num_heads=num_heads, dropout=dropout) for _ in range(num_task_blocks)]
        )

        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        uav_states: torch.Tensor,
        tasks: torch.Tensor,
        context: torch.Tensor,
        agent_avail_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """返回 V(s)。

        Args:
            uav_states: [B, N, uav_state_dim]
            tasks: [B, P, task_dim]
            context: [B, context_dim]
            agent_avail_mask: [B, N]，1/True 表示有效 UAV
        """
        uav_tokens = self.uav_encoder(uav_states)  # [B,N,H]
        task_tokens = self.task_encoder(tasks)      # [B,P,H]
        context_token = self.context_encoder(context)  # [B,H]

        uav_padding_mask = None
        if agent_avail_mask is not None:
            uav_padding_mask = ~agent_avail_mask.bool()

        for block in self.uav_blocks:
            uav_tokens = block(uav_tokens, key_padding_mask=uav_padding_mask)

        task_padding_mask = (tasks.abs().sum(dim=-1) <= 0)
        for block in self.task_blocks:
            task_tokens = block(task_tokens, key_padding_mask=task_padding_mask)

        uav_pool = masked_mean(uav_tokens, valid_mask=~uav_padding_mask if uav_padding_mask is not None else None, dim=1)
        task_pool = masked_mean(task_tokens, valid_mask=~task_padding_mask, dim=1)

        fusion = torch.cat([uav_pool, task_pool, context_token], dim=-1)
        value = self.value_head(fusion)
        return value
