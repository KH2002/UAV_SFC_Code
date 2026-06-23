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
        self.context_encoder = MLP(
            context_dim, hidden_dim, hidden_dim, num_layers=2, dropout=dropout, use_layer_norm=True
        )

        self.agent_blocks = nn.ModuleList(
            [TransformerBlock(hidden_dim, num_heads=num_heads, dropout=dropout) for _ in range(num_agent_blocks)]
        )
        self.vnf_blocks = nn.ModuleList(
            [TransformerBlock(hidden_dim, num_heads=num_heads, dropout=dropout) for _ in range(num_agent_blocks)]
        )

        self.vnf_encoder = MLP(
            7,
            hidden_dim,
            hidden_dim,
            num_layers=2,
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

    def _encode_agents(
        self,
        agent_self: torch.Tensor,
        agent_avail_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        x = self.self_encoder(agent_self)

        key_padding_mask = None
        if agent_avail_mask is not None:
            key_padding_mask = ~agent_avail_mask.bool()

        for block in self.agent_blocks:
            x = block(x, key_padding_mask=key_padding_mask)
        return x

    def _select_current_agent_token(
        self,
        agent_tokens: torch.Tensor,
        current_agent_id: torch.Tensor,
    ) -> torch.Tensor:
        bsz, _, hidden = agent_tokens.shape
        gather_idx = current_agent_id.view(bsz, 1, 1).expand(-1, 1, hidden)
        return agent_tokens.gather(dim=1, index=gather_idx).squeeze(1)

    def _encode_vnfs(self, task_matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        vnf_features = self._build_vnf_features(task_matrix)
        bsz, num_tasks, num_vnfs, _ = vnf_features.shape
        flat_vnf_features = vnf_features.reshape(bsz, num_tasks * num_vnfs, -1)
        vnf_tokens = self.vnf_encoder(flat_vnf_features)
        vnf_padding_mask = flat_vnf_features.abs().sum(dim=-1) <= 0
        for block in self.vnf_blocks:
            vnf_tokens = block(vnf_tokens, key_padding_mask=vnf_padding_mask)
        return vnf_tokens, flat_vnf_features, vnf_padding_mask

    def _build_vnf_features(self, task_matrix: torch.Tensor) -> torch.Tensor:
        """将每个 SFC request 的两个 VNF 拆成 VNF token 特征。"""
        vnf0 = torch.stack(
            [
                task_matrix[..., 0],   # workload
                task_matrix[..., 1],   # cpu frequency
                task_matrix[..., 2],   # own state
                task_matrix[..., 6],   # location x
                task_matrix[..., 7],   # location y
                task_matrix[..., 5],   # partner state
                task_matrix[..., 10],  # request communication demand
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
        """复用 Actor 的 VNF scoring head，计算所有 UAV 对所有 VNF 的得分。

        Returns:
            vnf_scores: [B, N, 2P]，顺序为 req0-vnf0, req0-vnf1, req1-vnf0, ...
        """
        agent_tokens = self._encode_agents(agent_self, agent_avail_mask)  # [B,N,H]
        context_token = self.context_encoder(context)  # [B,H]
        vnf_tokens, _, _ = self._encode_vnfs(task_matrix)  # [B,2P,H]

        num_agents = agent_tokens.shape[1]
        num_vnf_tokens = vnf_tokens.shape[1]
        agent_expand = agent_tokens.unsqueeze(2).expand(-1, -1, num_vnf_tokens, -1)
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
        if vnf_location_ids is None:
            raise ValueError("vnf_location_ids is required: 动作 logits 由 VNF 得分散射得到。")

        agent_tokens = self._encode_agents(agent_self, agent_avail_mask)  # [B,N,H]
        selected_agent = self._select_current_agent_token(agent_tokens, current_agent_id)  # [B,H]
        context_token = self.context_encoder(context)  # [B,H]

        vnf_tokens, flat_vnf_features, _ = self._encode_vnfs(task_matrix)  # [B,2P,H]
        bsz = vnf_tokens.shape[0]
        flat_location_ids = vnf_location_ids.reshape(bsz, -1)

        # 动作空间大小在运行时确定：优先用 action_mask 的宽度（=num_locations），
        # 没有 mask 时回退到 location_id 的最大值。网络权重与该值无关，可推理期动态变化。
        if action_mask is not None:
            action_dim = int(action_mask.shape[-1])
        else:
            action_dim = int(flat_location_ids.max().item()) + 1

        selected_expand = selected_agent.unsqueeze(1).expand_as(vnf_tokens)
        context_expand = context_token.unsqueeze(1).expand_as(vnf_tokens)
        vnf_fusion = torch.cat([selected_expand, vnf_tokens, context_expand], dim=-1)
        vnf_scores = self.vnf_score_head(vnf_fusion).squeeze(-1)  # [B,2P]

        vnf_state = flat_vnf_features[..., 2]
        valid_vnf = (
            (flat_location_ids >= 0)
            & (flat_location_ids < action_dim)
            & (vnf_state < 2.0)
            & (flat_vnf_features.abs().sum(dim=-1) > 0)
        )
        safe_location_ids = flat_location_ids.clamp(min=0, max=action_dim - 1)

        logits = torch.zeros(
            bsz,
            action_dim,
            dtype=vnf_scores.dtype,
            device=vnf_scores.device,
        )
        logits.scatter_add_(dim=1, index=safe_location_ids, src=vnf_scores.masked_fill(~valid_vnf, 0.0))

        if action_mask is not None:
            logits = logits.masked_fill(~action_mask.bool(), -1e9)

        return {
            "logits": logits,
            "selected_agent": selected_agent,
            "context_token": context_token,
            "vnf_scores": vnf_scores,
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
            vnf_location_ids=vnf_location_ids,
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
        vnf_location_ids: Optional[torch.Tensor] = None,
        action_mask: Optional[torch.Tensor] = None,
        agent_avail_mask: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Dict[str, torch.Tensor]:
        dist = self.get_dist(
            agent_self=agent_self,
            task_matrix=task_matrix,
            vnf_location_ids=vnf_location_ids,
            context=context,
            current_agent_id=current_agent_id,
            action_mask=action_mask,
            agent_avail_mask=agent_avail_mask,
        )

        action = dist.probs.argmax(dim=-1) if deterministic else dist.sample()
        return {
            "action": action,
            "log_prob": dist.log_prob(action),
            "entropy": dist.entropy(),
            "probs": dist.probs,
        }
