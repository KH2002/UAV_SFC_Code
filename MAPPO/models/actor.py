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
        if task_dim < 11:
            raise ValueError(f"MAPPOActor expects task_dim >= 11, got {task_dim}")
        # 每个 VNF token 的输入特征：
        # own(5) + vnf_role(2) + request_idx_norm(1) + paired_vnf(5) + comm(1) = 14
        self.vnf_token_dim = 14

        self.self_encoder = MLP(
            self_dim, hidden_dim, hidden_dim, num_layers=2, dropout=dropout, use_layer_norm=True
        )
        self.task_encoder = MLP(
            self.vnf_token_dim, hidden_dim, hidden_dim, num_layers=2, dropout=dropout, use_layer_norm=True
        )
        self.context_encoder = MLP(
            context_dim, hidden_dim, hidden_dim, num_layers=2, dropout=dropout, use_layer_norm=True
        )

        self.agent_blocks = nn.ModuleList(
            [TransformerBlock(hidden_dim, num_heads=num_heads, dropout=dropout) for _ in range(num_agent_blocks)]
        )

        # Pointer/scoring head：query 对每个 VNF token 打分
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_norm = nn.LayerNorm(hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.score_scale = hidden_dim ** -0.5
        self.end_head = MLP(
            hidden_dim * 2,
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

    def _build_vnf_tokens(self, task_matrix: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """由每行 SFC 特征构造 2 个独立 VNF token，并补充 request/paired 关系特征。

        Args:
            task_matrix: [B, P, 11]

        Returns:
            vnf_tokens: [B, 2P, 14]
            vnf_valid_mask: [B, 2P] (bool)
        """
        # task_matrix layout:
        # [v1_w, v1_f, v1_s, v2_w, v2_f, v2_s, loc1_x, loc1_y, loc2_x, loc2_y, comm]
        bsz, max_pending, _ = task_matrix.shape
        req_valid = (task_matrix.abs().sum(dim=-1) > 0)  # [B,P]

        v1_w = task_matrix[:, :, 0:1]
        v1_f = task_matrix[:, :, 1:2]
        v1_s = task_matrix[:, :, 2:3]
        v2_w = task_matrix[:, :, 3:4]
        v2_f = task_matrix[:, :, 4:5]
        v2_s = task_matrix[:, :, 5:6]
        loc1_x = task_matrix[:, :, 6:7]
        loc1_y = task_matrix[:, :, 7:8]
        loc2_x = task_matrix[:, :, 8:9]
        loc2_y = task_matrix[:, :, 9:10]
        comm = task_matrix[:, :, 10:11]

        req_idx = torch.arange(max_pending, device=task_matrix.device, dtype=task_matrix.dtype).view(1, max_pending, 1)
        req_idx_norm = req_idx / max(max_pending - 1, 1)
        req_idx_norm = req_idx_norm.expand(bsz, -1, -1)

        is_v1 = torch.ones_like(v1_w)
        is_v2 = torch.zeros_like(v1_w)

        # token(v1): own(v1) + role + request + paired(v2) + comm
        token_v1 = torch.cat(
            [v1_w, v1_f, v1_s, loc1_x, loc1_y, is_v1, is_v2, req_idx_norm, v2_w, v2_f, v2_s, loc2_x, loc2_y, comm],
            dim=-1,
        )
        # token(v2): own(v2) + role + request + paired(v1) + comm
        token_v2 = torch.cat(
            [v2_w, v2_f, v2_s, loc2_x, loc2_y, is_v2, is_v1, req_idx_norm, v1_w, v1_f, v1_s, loc1_x, loc1_y, comm],
            dim=-1,
        )

        # [B,P,2,14] -> [B,2P,14], 顺序为 (req0-v1, req0-v2, req1-v1, req1-v2, ...)
        vnf_tokens = torch.stack([token_v1, token_v2], dim=2).reshape(bsz, max_pending * 2, self.vnf_token_dim)
        vnf_valid_mask = req_valid.unsqueeze(-1).expand(-1, -1, 2).reshape(bsz, max_pending * 2)
        return vnf_tokens, vnf_valid_mask

    def forward(
        self,
        agent_self: torch.Tensor,
        task_matrix: torch.Tensor,
        context: torch.Tensor,
        current_agent_id: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        agent_avail_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """前向输出 logits（pointer/scoring head）。"""
        # shapes:
        # agent_self [B,N,self_dim], task_matrix [B,P,task_dim], context [B,ctx_dim]
        # current_agent_id [B], action_mask [B,A]
        agent_tokens = self._encode_agents(agent_self, agent_avail_mask)  # [B,N,H]
        selected_agent = self._select_current_agent_token(agent_tokens, current_agent_id)  # [B,H]

        context_token = self.context_encoder(context)  # [B,H]
        query = self.query_norm(selected_agent + context_token)  # [B,H]
        query = self.query_proj(query)  # [B,H]

        vnf_raw_tokens, vnf_valid_mask = self._build_vnf_tokens(task_matrix)  # [B,2P,14], [B,2P]
        vnf_tokens = self.task_encoder(vnf_raw_tokens)  # [B,2P,H]
        vnf_tokens = self.key_proj(self.key_norm(vnf_tokens))  # [B,2P,H]
        vnf_scores = torch.einsum("bh,bth->bt", query, vnf_tokens) * self.score_scale  # [B,2P]
        vnf_scores = vnf_scores.masked_fill(~vnf_valid_mask.bool(), -1e9)

        # END 动作单独打分
        end_logit = self.end_head(torch.cat([selected_agent, context_token], dim=-1)).squeeze(-1)  # [B]

        # 组装最终 logits，前 2P 对应 VNF 动作，最后一维对应 END
        bsz, vnf_action_count = vnf_scores.shape
        logits = torch.full(
            (bsz, self.action_dim),
            fill_value=-1e9,
            dtype=vnf_scores.dtype,
            device=vnf_scores.device,
        )
        capped = min(vnf_action_count, self.action_dim - 1)
        logits[:, :capped] = vnf_scores[:, :capped]
        logits[:, self.action_dim - 1] = end_logit

        if action_mask is not None:
            logits = logits.masked_fill(~action_mask.bool(), -1e9)

        return {
            "logits": logits,
            "selected_agent": selected_agent,
            "vnf_scores": vnf_scores,
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
