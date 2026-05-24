# -*- coding: utf-8 -*-
"""MAPPO Actor：参数共享策略网络。"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.distributions import Categorical

from .common import MLP, TransformerBlock


class MAPPOActor(nn.Module):
    def __init__(
        self,
        self_dim: int,
        task_dim: int,
        context_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 4,
        num_agent_blocks: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        self.self_encoder = MLP(
            self_dim, hidden_dim, hidden_dim, num_layers=2, dropout=dropout, use_layer_norm=True
        )
        self.task_encoder = MLP(
            task_dim, hidden_dim, hidden_dim, num_layers=2, dropout=dropout, use_layer_norm=True
        )
        self.context_encoder = MLP(
            context_dim, hidden_dim, hidden_dim, num_layers=2, dropout=dropout, use_layer_norm=True
        )

        self.agent_blocks = nn.ModuleList(
            [TransformerBlock(hidden_dim, num_heads=num_heads, dropout=dropout) for _ in range(num_agent_blocks)]
        )

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_query_norm = nn.LayerNorm(hidden_dim)
        self.cross_kv_norm = nn.LayerNorm(hidden_dim)
        self.cross_dropout = nn.Dropout(dropout)
        self.cross_norm = nn.LayerNorm(hidden_dim)

        self.policy_head = MLP(
            hidden_dim * 3,
            hidden_dim,
            action_dim,
            num_layers=2,
            dropout=dropout,
            use_layer_norm=True,
        )

    def _encode_agents(
        self,
        agent_self: torch.Tensor,
        agent_avail_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # agent_self: [B, N, self_dim]
        x = self.self_encoder(agent_self)

        key_padding_mask = None
        if agent_avail_mask is not None:
            # MHA 的 key_padding_mask: True 表示需要忽略
            key_padding_mask = ~agent_avail_mask.bool()

        for block in self.agent_blocks:
            x = block(x, key_padding_mask=key_padding_mask)
        return x

    def _select_current_agent_token(
        self,
        agent_tokens: torch.Tensor,
        current_agent_id: torch.Tensor,
    ) -> torch.Tensor:
        # agent_tokens: [B, N, H], current_agent_id: [B]
        bsz, _, hidden = agent_tokens.shape
        gather_idx = current_agent_id.view(bsz, 1, 1).expand(-1, 1, hidden)
        selected = agent_tokens.gather(dim=1, index=gather_idx).squeeze(1)
        return selected

    def forward(
        self,
        agent_self: torch.Tensor,
        task_matrix: torch.Tensor,
        context: torch.Tensor,
        current_agent_id: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        agent_avail_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """前向输出 logits。"""
        # shapes:
        # agent_self [B,N,self_dim], task_matrix [B,P,task_dim], context [B,ctx_dim]
        # current_agent_id [B], action_mask [B,A]
        agent_tokens = self._encode_agents(agent_self, agent_avail_mask)  # [B,N,H]
        selected_agent = self._select_current_agent_token(agent_tokens, current_agent_id)  # [B,H]

        task_tokens = self.task_encoder(task_matrix)  # [B,P,H]
        context_token = self.context_encoder(context)  # [B,H]

        query = (selected_agent + context_token).unsqueeze(1)  # [B,1,H]
        norm_query = self.cross_query_norm(query)
        norm_task_tokens = self.cross_kv_norm(task_tokens)

        task_padding_mask = (task_matrix.abs().sum(dim=-1) <= 0)  # [B,P]
        cross_out, _ = self.cross_attn(
            query=norm_query,
            key=norm_task_tokens,
            value=norm_task_tokens,
            key_padding_mask=task_padding_mask,
            need_weights=False,
        )
        cross_out = self.cross_norm(query + self.cross_dropout(cross_out)).squeeze(1)  # [B,H]

        fusion = torch.cat([selected_agent, cross_out, context_token], dim=-1)  # [B,3H]
        logits = self.policy_head(fusion)  # [B,A]

        if action_mask is not None:
            logits = logits.masked_fill(~action_mask.bool(), -1e9)

        return {
            "logits": logits,
            "selected_agent": selected_agent,
            "cross_token": cross_out,
            "context_token": context_token,
        }

    def get_dist(
        self,
        agent_self: torch.Tensor,
        task_matrix: torch.Tensor,
        context: torch.Tensor,
        current_agent_id: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        agent_avail_mask: Optional[torch.Tensor] = None,
    ) -> Categorical:
        out = self.forward(
            agent_self=agent_self,
            task_matrix=task_matrix,
            context=context,
            current_agent_id=current_agent_id,
            action_mask=action_mask,
            agent_avail_mask=agent_avail_mask,
        )
        return Categorical(logits=out["logits"])

    @torch.no_grad()
    def act(
        self,
        agent_self: torch.Tensor,
        task_matrix: torch.Tensor,
        context: torch.Tensor,
        current_agent_id: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        agent_avail_mask: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Dict[str, torch.Tensor]:
        dist = self.get_dist(
            agent_self=agent_self,
            task_matrix=task_matrix,
            context=context,
            current_agent_id=current_agent_id,
            action_mask=action_mask,
            agent_avail_mask=agent_avail_mask,
        )

        if deterministic:
            action = dist.probs.argmax(dim=-1)
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return {
            "action": action,
            "log_prob": log_prob,
            "entropy": entropy,
            "probs": dist.probs,
        }
