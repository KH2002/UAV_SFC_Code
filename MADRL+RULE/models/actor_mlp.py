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
        self.vnf_encoder = MLP(
            7,
            self.mlp_hidden_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            use_layer_norm=True,
        )
        self.vnf_score_head = MLP(
            hidden_dim * 3,
            hidden_dim,
            1,
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

    def _build_vnf_features(self, task_matrix: torch.Tensor) -> torch.Tensor:
        """将每个 SFC request 的两个 VNF 拆成 VNF token 特征。"""
        vnf0 = torch.stack(
            [
                task_matrix[..., 0],
                task_matrix[..., 1],
                task_matrix[..., 2],
                task_matrix[..., 6],
                task_matrix[..., 7],
                task_matrix[..., 5],
                task_matrix[..., 10],
            ],
            dim=-1,
        )
        vnf1 = torch.stack(
            [
                task_matrix[..., 3],
                task_matrix[..., 4],
                task_matrix[..., 5],
                task_matrix[..., 8],
                task_matrix[..., 9],
                task_matrix[..., 2],
                task_matrix[..., 10],
            ],
            dim=-1,
        )
        return torch.stack([vnf0, vnf1], dim=2)  # [B,P,2,7]

    def score_vnfs_for_all_agents(
        self,
        agent_self: torch.Tensor,
        task_matrix: torch.Tensor,
        context: torch.Tensor,
        agent_avail_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """复用 MLP Actor 的 VNF scoring head，计算所有 UAV 对所有 VNF 的得分。"""
        agent_tokens = self.self_encoder(agent_self)  # [B,N,H]
        context_token = self.context_encoder(context)  # [B,H]
        vnf_features = self._build_vnf_features(task_matrix)  # [B,P,2,7]
        bsz, num_tasks, num_vnfs, _ = vnf_features.shape

        flat_vnf_features = vnf_features.view(bsz, num_tasks * num_vnfs, -1)
        vnf_tokens = self.vnf_encoder(flat_vnf_features)  # [B,2P,H]

        num_agents = agent_tokens.shape[1]
        agent_expand = agent_tokens.unsqueeze(2).expand(-1, -1, num_tasks * num_vnfs, -1)
        vnf_expand = vnf_tokens.unsqueeze(1).expand(-1, num_agents, -1, -1)
        context_expand = context_token.unsqueeze(1).unsqueeze(2).expand_as(agent_expand)
        fusion = torch.cat([agent_expand, vnf_expand, context_expand], dim=-1)
        return self.vnf_score_head(fusion).squeeze(-1)  # [B,N,2P]

    def forward(
        self,
        agent_self: torch.Tensor,
        task_matrix: torch.Tensor,
        context: torch.Tensor,
        current_agent_id: torch.Tensor,
        vnf_location_ids: Optional[torch.Tensor] = None,
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

        if vnf_location_ids is not None:
            vnf_features = self._build_vnf_features(task_matrix)  # [B,P,2,7]
            bsz, num_tasks, num_vnfs, _ = vnf_features.shape
            flat_vnf_features = vnf_features.view(bsz, num_tasks * num_vnfs, -1)
            flat_location_ids = vnf_location_ids.view(bsz, num_tasks * num_vnfs)

            vnf_tokens = self.vnf_encoder(flat_vnf_features)  # [B,2P,H]
            selected_expand = selected_agent.unsqueeze(1).expand_as(vnf_tokens)
            context_expand = context_token.unsqueeze(1).expand_as(vnf_tokens)
            vnf_fusion = torch.cat([selected_expand, vnf_tokens, context_expand], dim=-1)
            vnf_scores = self.vnf_score_head(vnf_fusion).squeeze(-1)  # [B,2P]

            vnf_state = flat_vnf_features[..., 2]
            valid_vnf = (
                (flat_location_ids >= 0)
                & (flat_location_ids < self.action_dim)
                & (vnf_state < 2.0)
                & (flat_vnf_features.abs().sum(dim=-1) > 0)
            )
            safe_location_ids = flat_location_ids.clamp(min=0, max=self.action_dim - 1)

            logits = torch.zeros(
                bsz,
                self.action_dim,
                dtype=vnf_scores.dtype,
                device=vnf_scores.device,
            )
            logits.scatter_add_(dim=1, index=safe_location_ids, src=vnf_scores.masked_fill(~valid_vnf, 0.0))

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
        vnf_location_ids: Optional[torch.Tensor] = None,
        action_mask: Optional[torch.Tensor] = None,
        agent_avail_mask: Optional[torch.Tensor] = None,
    ) -> Categorical:
        out = self.forward(
            agent_self=agent_self,
            task_matrix=task_matrix,
            context=context,
            current_agent_id=current_agent_id,
            vnf_location_ids=vnf_location_ids,
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
        vnf_location_ids: Optional[torch.Tensor] = None,
        action_mask: Optional[torch.Tensor] = None,
        agent_avail_mask: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Dict[str, torch.Tensor]:
        dist = self.get_dist(
            agent_self=agent_self,
            task_matrix=task_matrix,
            context=context,
            current_agent_id=current_agent_id,
            vnf_location_ids=vnf_location_ids,
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
