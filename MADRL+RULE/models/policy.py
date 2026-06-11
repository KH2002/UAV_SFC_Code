# -*- coding: utf-8 -*-
"""MAPPO 策略封装：Actor + Critic。"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from .actor import MAPPOActor
from .actor_mlp import MAPPOMLPActor
from .critic import MAPPOCritic


class MAPPOPolicy(nn.Module):
    def __init__(
        self,
        self_dim: int,
        task_dim: int,
        context_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
        num_actor_agent_blocks: int = 1,
        num_critic_uav_blocks: int = 1,
        num_critic_task_blocks: int = 1,
        actor_type: str = "attn",
        actor_mlp_hidden_dim: Optional[int] = None,
        actor_mlp_num_layers: int = 2,
    ):
        super().__init__()
        actor_type = str(actor_type).lower()
        if actor_type in {"attn", "attention", "transformer"}:
            self.actor = MAPPOActor(
                self_dim=self_dim,
                task_dim=task_dim,
                context_dim=context_dim,
                action_dim=action_dim,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                num_agent_blocks=num_actor_agent_blocks,
                dropout=dropout,
            )
        elif actor_type == "mlp":
            self.actor = MAPPOMLPActor(
                self_dim=self_dim,
                task_dim=task_dim,
                context_dim=context_dim,
                action_dim=action_dim,
                hidden_dim=hidden_dim,
                mlp_hidden_dim=actor_mlp_hidden_dim,
                num_layers=actor_mlp_num_layers,
                dropout=dropout,
            )
        else:
            raise ValueError(f"Unsupported actor_type: {actor_type}. Expected one of: attn, mlp.")

        self.critic = MAPPOCritic(
            uav_state_dim=self_dim,
            task_dim=task_dim,
            context_dim=context_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_uav_blocks=num_critic_uav_blocks,
            num_task_blocks=num_critic_task_blocks,
            dropout=dropout,
        )

    def act(
        self,
        obs_t: Dict[str, torch.Tensor],
        deterministic: bool = False,
    ) -> Dict[str, torch.Tensor]:
        return self.actor.act(
            agent_self=obs_t["agent_self"],
            task_matrix=obs_t["task_matrix"],
            vnf_location_ids=obs_t.get("vnf_location_ids"),
            context=obs_t["context"],
            current_agent_id=obs_t["current_agent_id"],
            action_mask=obs_t.get("action_mask"),
            agent_avail_mask=obs_t.get("agent_avail_mask"),
            deterministic=deterministic,
        )

    def get_value(
        self,
        obs_t: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        return self.critic(
            uav_states=obs_t["agent_self"],
            tasks=obs_t["task_matrix"],
            context=obs_t["context"],
            agent_avail_mask=obs_t.get("agent_avail_mask"),
        )

    @torch.no_grad()
    def score_vnfs_for_deployment(
        self,
        obs_t: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """返回部署排序使用的 policy VNF 分数，shape=[B,N,2P]。"""
        return self.actor.score_vnfs_for_all_agents(
            agent_self=obs_t["agent_self"],
            task_matrix=obs_t["task_matrix"],
            context=obs_t["context"],
            agent_avail_mask=obs_t.get("agent_avail_mask"),
        )

    def evaluate_actions(
        self,
        obs_t: Dict[str, torch.Tensor],
        actions: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """用于 PPO 更新：给定动作计算 log_prob / entropy / value。"""
        dist = self.actor.get_dist(
            agent_self=obs_t["agent_self"],
            task_matrix=obs_t["task_matrix"],
            vnf_location_ids=obs_t.get("vnf_location_ids"),
            context=obs_t["context"],
            current_agent_id=obs_t["current_agent_id"],
            action_mask=obs_t.get("action_mask"),
            agent_avail_mask=obs_t.get("agent_avail_mask"),
        )
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        value = self.get_value(obs_t).squeeze(-1)
        return {
            "log_prob": log_prob,
            "entropy": entropy,
            "value": value,
        }
