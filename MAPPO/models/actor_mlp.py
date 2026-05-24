# -*- coding: utf-8 -*-
"""MAPPO Actor (MLP baseline)：参数共享策略网络。"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.distributions import Categorical

from .common import MLP, masked_mean


class MAPPOMLPActor(nn.Module):
    def __init__(
        self,
        self_dim: int,
        task_dim: int,
        context_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        mlp_hidden_dim: Optional[int] = None,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.mlp_hidden_dim = mlp_hidden_dim if mlp_hidden_dim is not None else hidden_dim
        self.num_layers = num_layers

        self.self_encoder = MLP(
            self_dim,
            self.mlp_hidden_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            use_layer_norm=True,
        )
        self.task_encoder = MLP(
            task_dim,
            self.mlp_hidden_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            use_layer_norm=True,
        )
        self.context_encoder = MLP(
            context_dim,
            self.mlp_hidden_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            use_layer_norm=True,
        )

        self.policy_head = MLP(
            hidden_dim * 4,
            hidden_dim,
            action_dim,
            num_layers=2,
            dropout=dropout,
            use_layer_norm=True,
        )

    def _select_current_agent_token(
        self,
        agent_tokens: torch.Tensor,
        current_agent_id: torch.Tensor,
    ) -> torch.Tensor:
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
        agent_tokens = self.self_encoder(agent_self)  # [B,N,H]
        selected_agent = self._select_current_agent_token(agent_tokens, current_agent_id)  # [B,H]
        agent_pool = masked_mean(agent_tokens, valid_mask=agent_avail_mask, dim=1)  # [B,H]

        task_tokens = self.task_encoder(task_matrix)  # [B,P,H]
        task_padding_mask = (task_matrix.abs().sum(dim=-1) <= 0)  # [B,P]
        task_pool = masked_mean(task_tokens, valid_mask=~task_padding_mask, dim=1)  # [B,H]

        context_token = self.context_encoder(context)  # [B,H]

        fusion = torch.cat([selected_agent, agent_pool, task_pool, context_token], dim=-1)  # [B,4H]
        logits = self.policy_head(fusion)  # [B,A]

        if action_mask is not None:
            logits = logits.masked_fill(~action_mask.bool(), -1e9)

        return {
            "logits": logits,
            "selected_agent": selected_agent,
            "agent_pool": agent_pool,
            "task_pool": task_pool,
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
