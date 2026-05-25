# -*- coding: utf-8 -*-
"""
MAPPO 环境（多智能体版本）

目标：
- 配置默认值来自 config.py
- 可被 DRL/training/config.yaml 覆盖
- 动作空间保持离散：0..(2*max_pending-1) 为认领 VNF，最后一位为 END_TOKEN
"""

from __future__ import annotations

import os
import sys
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import yaml

# 让脚本在仓库任意位置执行时都能导入根目录模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import config
import utils
from entities import Request, UAV
from DRL.training.dataset import EpisodeData, generate_locations, generate_requests, generate_uavs


@dataclass
class MAPPOEnvConfig:
    """环境与场景配置（由 config.py + YAML 合并得到）"""

    # scene
    area_size: float
    num_locations: int
    num_uavs: int
    num_requests: int
    num_time_slots: int
    vnfs_per_request: int

    # env
    max_pending: int
    max_rounds_per_slot: int
    invalid_action_penalty: float
    claim_reward: float
    complete_sfc_reward: float

    # training
    max_steps_per_episode: int

    @property
    def action_dim(self) -> int:
        # 2 * max_pending 个 VNF + 1 个 END
        return 2 * self.max_pending + 1


def load_env_config(yaml_path: Optional[str] = None) -> MAPPOEnvConfig:
    """加载并合并环境配置。

    优先级：YAML > config.py
    """
    yaml_path = yaml_path or os.path.join("DRL", "training", "config.yaml")

    cfg = {
        "scene": {
            "area_size": float(config.AREA_SIZE),
            "num_locations": int(config.NUM_LOCATIONS),
            "num_uavs": int(config.NUM_UAVS),
            "num_requests": int(config.NUM_REQUESTS),
            "num_time_slots": int(config.NUM_TIME_SLOTS),
            "vnfs_per_request": int(config.VNFS_PER_REQUEST),
        },
        "env": {
            "max_pending": 20,
            "max_steps_per_slot": 50,
            "invalid_action_penalty": 0.1,
            "claim_reward": 0.05,
            "complete_sfc_reward": 2.0,
        },
        "training": {
            "max_steps_per_episode": 800,
        },
    }

    if yaml_path and os.path.exists(yaml_path):
        with open(yaml_path, "r", encoding="utf-8") as f:
            user_cfg = yaml.safe_load(f) or {}
        for block in ("scene", "env", "training"):
            if isinstance(user_cfg.get(block), dict):
                cfg[block].update(user_cfg[block])

    return MAPPOEnvConfig(
        area_size=float(cfg["scene"]["area_size"]),
        num_locations=int(cfg["scene"]["num_locations"]),
        num_uavs=int(cfg["scene"]["num_uavs"]),
        num_requests=int(cfg["scene"]["num_requests"]),
        num_time_slots=int(cfg["scene"]["num_time_slots"]),
        vnfs_per_request=int(cfg["scene"]["vnfs_per_request"]),
        max_pending=int(cfg["env"]["max_pending"]),
        max_rounds_per_slot=int(cfg["env"]["max_steps_per_slot"]),
        invalid_action_penalty=float(cfg["env"]["invalid_action_penalty"]),
        claim_reward=float(cfg["env"]["claim_reward"]),
        complete_sfc_reward=float(cfg["env"]["complete_sfc_reward"]),
        max_steps_per_episode=int(cfg["training"]["max_steps_per_episode"]),
    )


class MAPPOSFCEnv:
    """MAPPO 训练环境（多智能体轮转单步决策）。

    step 输入：
        - (agent_id, action)
        - 每次只执行一个 agent 的动作；一个 round 包含所有 agent 各执行一次

    动作定义：
        - [0, 2*max_pending-1]: claim 某个 (request_idx, vnf_idx)
        - 2*max_pending: END_TOKEN
    """

    def __init__(
        self,
        env_config: Optional[MAPPOEnvConfig] = None,
        config_yaml_path: Optional[str] = None,
        episode_data: Optional[EpisodeData] = None,
        seed: Optional[int] = None,
    ):
        self.cfg = env_config or load_env_config(config_yaml_path)
        self.rng = np.random.default_rng(seed)

        self._episode_data = episode_data
        self.locations: Dict[int, Tuple[float, float]] = {}
        self.uavs: List[UAV] = []
        self.requests: List[Request] = []

        self._original_uavs: Optional[List[UAV]] = None
        self._original_requests: Optional[List[Request]] = None

        self.agent_ids = list(range(self.cfg.num_uavs))
        self.end_token = self.cfg.action_dim - 1
        self.max_distance = float(np.sqrt(self.cfg.area_size**2 + self.cfg.area_size**2))

        # 动态状态
        self.current_time_slot = 1
        self.current_round = 0
        self.episode_step = 0
        self.completed_request_ids: List[int] = []

        # slot 临时状态
        self.slot_claims: Dict[int, Dict[int, int]] = {}  # req_id -> {vnf_idx: agent_id}
        self.agent_slot_location: Dict[int, Optional[int]] = {}  # agent_id -> location_id
        self.turn_in_round = 0  # 当前 round 已执行到第几个 agent
        self.round_end_votes = 0  # 当前 round 中选择 END_TOKEN 的 agent 数
        self.shuffle_requests_on_reset = False

        self._init_scene()

    # --------------------------- public API ---------------------------

    def set_episode_data(self, episode_data: EpisodeData) -> None:
        self._episode_data = episode_data
        self._init_scene()

    def set_request_shuffle_on_reset(self, enabled: bool) -> None:
        """是否在每个 episode reset 时随机打乱 request 顺序。"""
        self.shuffle_requests_on_reset = bool(enabled)

    def reset(self) -> Dict[str, object]:
        if self._original_uavs is not None and self._original_requests is not None:
            self.uavs = self._copy_uavs(self._original_uavs)
            self.requests = self._copy_requests(self._original_requests)
        else:
            self._init_scene_runtime_random()

        # 仅打乱动作索引与 request 的映射顺序，不改变 request 本身内容
        if self.shuffle_requests_on_reset and len(self.requests) > 1:
            order = self.rng.permutation(len(self.requests)).tolist()
            self.requests = [self.requests[i] for i in order]

        self.current_time_slot = 1
        self.current_round = 0
        self.episode_step = 0
        self.completed_request_ids = []

        self._reset_slot_state()

        return self.get_obs()

    def step(self, agent_id: int, action: int) -> Tuple[Dict[str, object], float, bool, Dict[str, object]]:
        expected_agent_id = self.agent_ids[self.turn_in_round]
        if int(agent_id) != expected_agent_id:
            raise ValueError(
                f"当前轮到 agent {expected_agent_id} 决策，但收到 agent {agent_id}。"
            )

        self.episode_step += 1
        invalid_count = 0
        valid_claim_count = 0
        action = int(action)

        # 仅执行当前 agent 动作
        if action == self.end_token:
            self.round_end_votes += 1
        else:
            valid, _ = self._try_claim(agent_id, action)
            if not valid:
                invalid_count += 1
            else:
                valid_claim_count = 1

        # 推进 round 内 turn
        self.turn_in_round += 1
        round_completed = False
        slot_ended = False
        all_end = False
        slot_forced_end = False
        no_feasible_left = False

        slot_completed = 0
        if self.turn_in_round >= self.cfg.num_uavs:
            # 一个完整 round 结束
            round_completed = True
            self.current_round += 1

            all_end = self.round_end_votes >= self.cfg.num_uavs
            slot_forced_end = self.current_round >= self.cfg.max_rounds_per_slot
            no_feasible_left = not self._has_any_feasible_claim()
            slot_ended = all_end or slot_forced_end or no_feasible_left

            # 重置 round 级统计，进入下一 round
            self.turn_in_round = 0
            self.round_end_votes = 0

            if slot_ended:
                slot_completed = self._settle_slot()
                self._advance_time_slot()

        done = (
            self.current_time_slot > self.cfg.num_time_slots
            or self._pending_count() == 0
            or self.episode_step >= self.cfg.max_steps_per_episode
        )

        # 奖励：合法认领小奖励 + 完成SFC大奖励 - 非法惩罚
        reward = self.cfg.claim_reward * valid_claim_count
        reward += self.cfg.complete_sfc_reward * float(slot_completed)
        reward -= self.cfg.invalid_action_penalty * invalid_count

        if done and len(self.requests) > 0:
            success_rate = len(self.completed_request_ids) / float(len(self.requests))
            reward += success_rate * 20.0
        else:
            success_rate = len(self.completed_request_ids) / float(max(len(self.requests), 1))

        obs = self.get_obs()
        info = {
            "slot_ended": slot_ended,
            "round_completed": round_completed,
            "all_end_in_round": all_end,
            "slot_forced_end": slot_forced_end,
            "no_feasible_left": no_feasible_left,
            "slot_completed": slot_completed,
            "invalid_count": invalid_count,
            "valid_claim_count": valid_claim_count,
            "current_slot": self.current_time_slot,
            "current_round": self.current_round,
            "turn_in_round": self.turn_in_round,
            "next_agent_id": None if done else self.agent_ids[self.turn_in_round],
            "executed_agent_id": agent_id,
            "executed_action": action,
            "pending_count": self._pending_count(),
            "completed_count": len(self.completed_request_ids),
            "success_rate": success_rate,
            "action_mask": obs["action_mask"],
            "action_masks_all": obs["action_masks_all"],
        }

        return obs, reward, done, info

    def get_obs(self) -> Dict[str, object]:
        """返回共享观测结构，便于直接编码为神经网络输入。"""
        current_agent_id = self.agent_ids[self.turn_in_round]
        agent_self = self._build_all_uav_features()  # [N, self_dim]
        task_matrix = self._build_task_matrix()      # [P, task_dim]
        context = self._build_context_features()     # [ctx_dim]
        action_mask = self.get_action_mask(current_agent_id)  # [A]
        action_masks_all = self.get_action_masks()             # {agent_id: [A]}

        return {
            "agent_self": agent_self,
            "agent_avail_mask": np.ones(self.cfg.num_uavs, dtype=np.int32),
            "task_matrix": task_matrix,
            "context": context,
            "current_agent_id": current_agent_id,
            "turn_in_round": self.turn_in_round,
            "action_mask": action_mask,
            "action_masks_all": action_masks_all,
            # 兼容中心化 critic 调用
            "global_state": {
                "uav_states": agent_self,
                "tasks": task_matrix,
                "context": context,
            },
        }

    def get_local_obs(self, agent_id: int) -> Dict[str, np.ndarray]:
        return {
            "self": self._build_self_features(agent_id),
            "tasks": self._build_task_matrix(),
            "context": self._build_context_features(),
        }

    def get_global_state(self) -> Dict[str, np.ndarray]:
        return {
            "uav_states": self._build_all_uav_features(),
            "tasks": self._build_task_matrix(),
            "context": self._build_context_features(),
        }

    def get_action_mask(self, agent_id: int) -> np.ndarray:
        """单 agent 动作掩码，shape = [2*max_pending + 1]。"""
        n_vnf_actions = 2 * self.cfg.max_pending
        mask = np.zeros(n_vnf_actions + 1, dtype=np.int32)

        for action in range(n_vnf_actions):
            req_idx, vnf_idx = self.decode_action(action)
            if self._is_valid_claim(agent_id, req_idx, vnf_idx):
                mask[action] = 1

        mask[self.end_token] = 1
        return mask

    def get_action_masks(self) -> Dict[int, np.ndarray]:
        return {agent_id: self.get_action_mask(agent_id) for agent_id in self.agent_ids}

    def decode_action(self, action: int) -> Tuple[int, int]:
        """将离散动作映射到 (request_idx, vnf_idx)。"""
        req_idx = action // 2
        vnf_idx = action % 2
        return int(req_idx), int(vnf_idx)

    # --------------------------- core logic ---------------------------

    def _try_claim(self, agent_id: int, action: int) -> Tuple[bool, Dict[str, object]]:
        req_idx, vnf_idx = self.decode_action(action)
        if not self._is_valid_claim(agent_id, req_idx, vnf_idx):
            return False, {"error": "invalid_claim"}

        req = self.requests[req_idx]
        vnf = req.vnfs[vnf_idx]
        uav = self.uavs[agent_id]

        # 扣减资源并更新 UAV 位置锁
        self._consume_resources_for_claim(uav, vnf)

        self.slot_claims.setdefault(req.id, {})[vnf_idx] = agent_id
        if self.agent_slot_location[agent_id] is None:
            self.agent_slot_location[agent_id] = int(vnf.location_id)

        return True, {"request_id": req.id, "vnf_idx": vnf_idx}

    def _settle_slot(self) -> int:
        """结算当前 slot：仅当同一请求双 VNF 均认领时判定完成。"""
        completed_this_slot: List[Request] = []

        for req in self.requests:
            if req.is_serviced:
                continue
            claim = self.slot_claims.get(req.id, {})
            if 0 not in claim or 1 not in claim:
                continue

            # 按当前业务约束：一个 SFC 两个 VNF 必须由不同 UAV
            if claim[0] == claim[1]:
                continue

            req.is_serviced = True
            completed_this_slot.append(req)

        if completed_this_slot:
            completed_ids = {r.id for r in completed_this_slot}
            existing = set(self.completed_request_ids)
            self.completed_request_ids.extend(sorted([rid for rid in completed_ids if rid not in existing]))

        return len(completed_this_slot)

    def _advance_time_slot(self) -> None:
        self.current_time_slot += 1

        for uav in self.uavs:
            # 按时隙统一扣除悬停基础能耗（同一时隙内执行多个VNF仅扣一次）
            if uav.is_busy:
                uav.energy = max(0.0, uav.energy - self._estimate_hover_energy())

            # 空闲且不在基站：返回基站并扣返航能耗
            if not uav.is_busy and uav.location_id != 0:
                return_distance = self._distance_to_base(uav)
                return_energy = utils.calculate_travel_energy(return_distance)
                uav.energy = max(0.0, uav.energy - return_energy)
                uav.location = config.BASE_STATION_LOCATION
                uav.location_id = 0

            # 在基站可充满电
            if uav.location_id == 0:
                uav.energy = config.UAV_BATTERY_CAPACITY

            # 每个新时隙恢复可部署计算资源
            uav.cpu_capacity = config.UAV_COMPUTATION_CAPACITY
            uav.is_busy = False

        self._reset_slot_state()

    def _reset_slot_state(self) -> None:
        self.current_round = 0
        self.turn_in_round = 0
        self.round_end_votes = 0
        self.slot_claims = {}
        self.agent_slot_location = {agent_id: None for agent_id in self.agent_ids}

    # --------------------------- feasibility ---------------------------

    def _is_valid_claim(self, agent_id: int, req_idx: int, vnf_idx: int) -> bool:
        if req_idx < 0 or vnf_idx not in (0, 1):
            return False
        if req_idx >= self._actionable_request_count():
            return False

        req = self.requests[req_idx]
        if req.is_serviced:
            return False

        # 同一 VNF 不可重复认领
        if req.id in self.slot_claims and vnf_idx in self.slot_claims[req.id]:
            return False

        # 如同一请求另一 VNF 已被该 agent 认领，则非法（双 UAV 约束）
        other_idx = 1 - vnf_idx
        if req.id in self.slot_claims and self.slot_claims[req.id].get(other_idx) == agent_id:
            return False

        vnf = req.vnfs[vnf_idx]
        uav = self.uavs[agent_id]

        # 同一 slot 位置锁：一个 agent 不能跨位置认领
        locked_loc = self.agent_slot_location.get(agent_id)
        if locked_loc is not None and locked_loc != vnf.location_id:
            return False

        # busy UAV 不能移动到不同位置
        if uav.is_busy and uav.location_id != vnf.location_id:
            return False

        # CPU 约束
        if uav.cpu_capacity < vnf.workload:
            return False

        # 能量约束：飞行 + 服务 + 返航预留
        target = self.locations.get(vnf.location_id, uav.location)
        if uav.is_busy:
            travel_distance = 0.0
        else:
            travel_distance = float(utils.calculate_distance(uav.location, target))

        # 首次认领时预留一个 slot 的悬停基础能耗；同一时隙后续认领不重复预留
        hover_reserve = self._estimate_hover_energy() if not uav.is_busy else 0.0
        need_energy = (
            self._estimate_travel_energy(travel_distance)
            + self._estimate_service_energy(vnf)
            + hover_reserve
            + self._estimate_return_energy(uav)
        )
        return uav.energy >= need_energy

    def _has_any_feasible_claim(self) -> bool:
        for agent_id in self.agent_ids:
            mask = self.get_action_mask(agent_id)
            # 除 END 外是否有合法动作
            if int(mask[:-1].sum()) > 0:
                return True
        return False

    def _consume_resources_for_claim(self, uav: UAV, vnf) -> None:
        target = self.locations.get(vnf.location_id, uav.location)

        # 飞行
        if not uav.is_busy:
            dist = float(utils.calculate_distance(uav.location, target))
            uav.energy -= self._estimate_travel_energy(dist)
            uav.location = target
            uav.location_id = int(vnf.location_id)

        # 服务消耗
        uav.energy -= self._estimate_service_energy(vnf)
        uav.cpu_capacity -= vnf.workload
        uav.is_busy = True

    # --------------------------- feature builders ---------------------------

    def _build_self_features(self, agent_id: int) -> np.ndarray:
        uav = self.uavs[agent_id]
        return np.array(
            [
                agent_id / max(self.cfg.num_uavs - 1, 1),
                uav.location[0] / self.cfg.area_size,
                uav.location[1] / self.cfg.area_size,
                uav.energy / config.UAV_BATTERY_CAPACITY,
                uav.cpu_capacity / config.UAV_COMPUTATION_CAPACITY,
                float(uav.is_busy),
                self._distance_to_base(uav) / self.max_distance,
            ],
            dtype=np.float32,
        )

    def _build_all_uav_features(self) -> np.ndarray:
        return np.stack([self._build_self_features(agent_id) for agent_id in self.agent_ids], axis=0)

    def _build_task_matrix(self) -> np.ndarray:
        """每个请求 11 维：
        [v1_workload, v1_freq, v1_state, v2_workload, v2_freq, v2_state,
         loc1_x, loc1_y, loc2_x, loc2_y, comm]
        """
        task_dim = 11
        tasks = np.zeros((self.cfg.max_pending, task_dim), dtype=np.float32)

        for i in range(self.cfg.max_pending):
            if i >= len(self.requests):
                continue

            req = self.requests[i]
            v1, v2 = req.vnfs[0], req.vnfs[1]

            loc1 = self.locations.get(v1.location_id, config.BASE_STATION_LOCATION)
            loc2 = self.locations.get(v2.location_id, config.BASE_STATION_LOCATION)

            c = req.communication_demands
            if c:
                comm = float(sum(c.values()))
            else:
                comm = 0.0

            # 状态编码：0 未认领，1 本时隙已认领，2 已完成
            claim = self.slot_claims.get(req.id, {})
            if req.is_serviced:
                v1_state = 2.0
                v2_state = 2.0
            else:
                v1_state = 1.0 if 0 in claim else 0.0
                v2_state = 1.0 if 1 in claim else 0.0

            tasks[i] = [
                v1.workload / max(config.VNF_WORKLOAD_RANGE[1], 1),
                float(v1.cpu_freq) / max(config.VNF_CPU_FREQUENCY_RANGE[1], 1e-6),
                v1_state,
                v2.workload / max(config.VNF_WORKLOAD_RANGE[1], 1),
                float(v2.cpu_freq) / max(config.VNF_CPU_FREQUENCY_RANGE[1], 1e-6),
                v2_state,
                loc1[0] / self.cfg.area_size,
                loc1[1] / self.cfg.area_size,
                loc2[0] / self.cfg.area_size,
                loc2[1] / self.cfg.area_size,
                comm / max(config.VNF_COMMUNICATION_DEMAND_RANGE[1], 1),
            ]

        return tasks

    def _build_context_features(self) -> np.ndarray:
        return np.array(
            [
                self.current_time_slot / max(self.cfg.num_time_slots, 1),
                (self.cfg.num_time_slots - self.current_time_slot) / max(self.cfg.num_time_slots, 1),
                self.current_round / max(self.cfg.max_rounds_per_slot, 1),
                self._pending_count() / max(len(self.requests), 1),
                len(self.completed_request_ids) / max(len(self.requests), 1),
            ],
            dtype=np.float32,
        )

    # --------------------------- scene / util ---------------------------

    def _init_scene(self) -> None:
        if self._episode_data is not None:
            self.uavs = self._copy_uavs(self._episode_data.uavs)
            self.requests = self._copy_requests(self._episode_data.requests)
            self.locations = deepcopy(self._episode_data.locations)
            self._original_uavs = self._copy_uavs(self._episode_data.uavs)
            self._original_requests = self._copy_requests(self._episode_data.requests)
        else:
            self._init_scene_runtime_random()
            self._original_uavs = self._copy_uavs(self.uavs)
            self._original_requests = self._copy_requests(self.requests)

    def _init_scene_runtime_random(self) -> None:
        self.locations = generate_locations(self.cfg.num_locations, self.cfg.area_size)
        self.uavs = generate_uavs(self.cfg.num_uavs, self.locations)
        self.requests = generate_requests(self.cfg.num_requests, self.locations)

    def _copy_uavs(self, uavs: List[UAV]) -> List[UAV]:
        copied = []
        for uav in uavs:
            new_uav = UAV(uav.id, uav.location)
            new_uav.location_id = uav.location_id
            new_uav.energy = uav.energy
            new_uav.cpu_capacity = uav.cpu_capacity
            new_uav.is_busy = uav.is_busy
            copied.append(new_uav)
        return copied

    def _copy_requests(self, requests: List[Request]) -> List[Request]:
        copied = []
        for req in requests:
            req_copy = deepcopy(req)
            copied.append(req_copy)
        return copied

    def _normalize_actions(self, actions: Union[List[int], Dict[int, int]]) -> Dict[int, int]:
        if isinstance(actions, list):
            if len(actions) != self.cfg.num_uavs:
                raise ValueError(f"actions(list) 长度应为 {self.cfg.num_uavs}，实际 {len(actions)}")
            return {agent_id: int(actions[agent_id]) for agent_id in self.agent_ids}

        if isinstance(actions, dict):
            missing = [agent_id for agent_id in self.agent_ids if agent_id not in actions]
            if missing:
                raise ValueError(f"actions(dict) 缺少 agent_id: {missing}")
            return {agent_id: int(actions[agent_id]) for agent_id in self.agent_ids}

        raise TypeError("actions 必须是 list[int] 或 dict[int,int]")

    def _pending_count(self) -> int:
        return sum(1 for req in self.requests if not req.is_serviced)

    def _actionable_request_count(self) -> int:
        return min(self.cfg.max_pending, len(self.requests))

    def _distance_to_base(self, uav: UAV) -> float:
        dx = uav.location[0] - config.BASE_STATION_LOCATION[0]
        dy = uav.location[1] - config.BASE_STATION_LOCATION[1]
        return float(np.sqrt(dx**2 + dy**2))

    def _estimate_travel_energy(self, distance: float) -> float:
        return float(utils.calculate_travel_energy(distance))

    def _estimate_service_energy(self, vnf) -> float:
        compute_energy = config.CHIP_ARCHITECTURE_CONSTANT * (vnf.cpu_freq ** 2) * vnf.workload
        # 悬停能耗已在时隙推进时统一扣除；这里仅保留每个VNF的计算+通信能耗
        return float(compute_energy + config.COMM_ENERGY_PER_TIMESLOT)

    def _estimate_hover_energy(self) -> float:
        return float((config.UAV_BLADE_PROFILE_POWER + config.UAV_INDUCED_POWER) * config.TIME_SLOT_DURATION)

    def _estimate_return_energy(self, uav: UAV) -> float:
        return float(utils.calculate_travel_energy(self._distance_to_base(uav)))


if __name__ == "__main__":
    env = MAPPOSFCEnv(config_yaml_path=os.path.join("DRL", "training", "config.yaml"), seed=42)
    obs = env.reset()
    print("Reset done")
    print(f"num_agents={env.cfg.num_uavs}, action_dim={env.cfg.action_dim}, pending={env._pending_count()}")

    done = False
    for _ in range(8):
        if done:
            break
        # 单 agent 轮转演示
        current_agent_id = obs["current_agent_id"]
        valid = np.where(obs["action_mask"] == 1)[0]
        action = int(valid[0]) if len(valid) > 0 else env.end_token

        obs, reward, done, info = env.step(current_agent_id, action)
        print(
            f"agent={info['executed_agent_id']} action={info['executed_action']} "
            f"slot={info['current_slot']} round={info['current_round']} turn={info['turn_in_round']} "
            f"reward={reward:.3f} completed={info['completed_count']} pending={info['pending_count']}"
        )
