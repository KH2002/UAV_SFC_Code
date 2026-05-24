# -*- coding: utf-8 -*-
"""将环境返回的字典观测转换为 PyTorch 张量。"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch


def _to_tensor(x, dtype=None, device=None):
    if isinstance(x, torch.Tensor):
        t = x
    else:
        t = torch.as_tensor(x)
    if dtype is not None:
        t = t.to(dtype=dtype)
    if device is not None:
        t = t.to(device)
    return t


def obs_to_tensors(
    obs: Dict[str, object],
    device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    """单条观测转张量并自动加 batch 维。"""
    agent_self = _to_tensor(obs["agent_self"], dtype=torch.float32, device=device).unsqueeze(0)  # [1,N,self_dim]
    task_matrix = _to_tensor(obs["task_matrix"], dtype=torch.float32, device=device).unsqueeze(0)  # [1,P,task_dim]
    context = _to_tensor(obs["context"], dtype=torch.float32, device=device).unsqueeze(0)  # [1,ctx_dim]

    current_agent_id = _to_tensor([obs["current_agent_id"]], dtype=torch.long, device=device)  # [1]

    action_mask = _to_tensor(obs["action_mask"], dtype=torch.bool, device=device).unsqueeze(0)  # [1,A]
    agent_avail_mask = _to_tensor(obs.get("agent_avail_mask", np.ones(agent_self.shape[1])), dtype=torch.bool, device=device).unsqueeze(0)  # [1,N]

    return {
        "agent_self": agent_self,
        "task_matrix": task_matrix,
        "context": context,
        "current_agent_id": current_agent_id,
        "action_mask": action_mask,
        "agent_avail_mask": agent_avail_mask,
    }


def collate_obs_batch(
    obs_batch: List[Dict[str, object]],
    device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    """观测列表堆叠成 batch。"""
    agent_self = torch.stack([
        _to_tensor(o["agent_self"], dtype=torch.float32) for o in obs_batch
    ], dim=0)
    task_matrix = torch.stack([
        _to_tensor(o["task_matrix"], dtype=torch.float32) for o in obs_batch
    ], dim=0)
    context = torch.stack([
        _to_tensor(o["context"], dtype=torch.float32) for o in obs_batch
    ], dim=0)

    current_agent_id = torch.as_tensor([int(o["current_agent_id"]) for o in obs_batch], dtype=torch.long)
    action_mask = torch.stack([
        _to_tensor(o["action_mask"], dtype=torch.bool) for o in obs_batch
    ], dim=0)
    agent_avail_mask = torch.stack([
        _to_tensor(o.get("agent_avail_mask", np.ones(agent_self.shape[1])), dtype=torch.bool) for o in obs_batch
    ], dim=0)

    if device is not None:
        agent_self = agent_self.to(device)
        task_matrix = task_matrix.to(device)
        context = context.to(device)
        current_agent_id = current_agent_id.to(device)
        action_mask = action_mask.to(device)
        agent_avail_mask = agent_avail_mask.to(device)

    return {
        "agent_self": agent_self,
        "task_matrix": task_matrix,
        "context": context,
        "current_agent_id": current_agent_id,
        "action_mask": action_mask,
        "agent_avail_mask": agent_avail_mask,
    }
