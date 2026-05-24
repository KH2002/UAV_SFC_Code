# -*- coding: utf-8 -*-
"""
UAV-SFC 部署问题的 Gym 环境实现
参考: docs/DRL_Solution_Design.md
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import gym
from gym import spaces
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, TYPE_CHECKING
from copy import deepcopy

import config
from entities import UAV, VNF, Request
import utils

if TYPE_CHECKING:
    from DRL.training.dataset import EpisodeData


class UAVSFCEnv(gym.Env):
    """
    UAV-SFC 部署环境
    
    状态空间:
        - uav_states: [NUM_UAVS, 6] - UAV特征向量
        - pending_requests: [MAX_PENDING, 5] - 待处理请求特征
        - global_features: [4] - 全局信息（当前时隙、剩余时隙等）
    
    动作空间:
        - request_selection: int - 选择的请求索引 [0, MAX_PENDING-1]
        - uav_for_vnf1: int - 为VNF1分配的UAV [0, NUM_UAVS-1]
        - uav_for_vnf2: int - 为VNF2分配的UAV [0, NUM_UAVS-1]
    
    支持从固定种子数据集加载场景数据，确保不同RL算法可以使用相同的数据进行对比。
    """
    
    metadata = {'render.modes': ['human']}
    
    def __init__(self, 
                 uavs: Optional[List[UAV]] = None,
                 requests: Optional[List[Request]] = None,
                 locations: Optional[Dict] = None,
                 max_pending: int = 20,
                 max_steps_per_episode: int = 200,
                 max_steps_per_slot: int = 50,
                 num_time_slots: Optional[int] = None,
                 num_locations: Optional[int] = None,
                 area_size: Optional[float] = None,
                 invalid_action_penalty: float = 0.1,
                 episode_data: Optional['EpisodeData'] = None):
        """
        初始化环境
        
        Args:
            uavs: UAV列表，若为None则根据config生成
            requests: 请求列表，若为None则根据config生成
            locations: 位置字典 {location_id: (x, y)}
            max_pending: 最大待处理请求数（观察窗口）
            max_steps_per_slot: 每个时隙允许的最大决策步数
            num_time_slots: 时间槽总数（None时使用config.NUM_TIME_SLOTS）
            num_locations: 位置总数（None时使用config.NUM_LOCATIONS）
            area_size: 区域大小（None时使用config.AREA_SIZE）
            invalid_action_penalty: 无效动作惩罚系数
            episode_data: 从数据集加载的回合数据（用于固定种子训练）
        """
        super().__init__()
        
        self.max_pending = max_pending
        self.max_steps_per_episode = max_steps_per_episode
        self.num_time_slots = int(num_time_slots) if num_time_slots is not None else int(config.NUM_TIME_SLOTS)
        self.num_locations = int(num_locations) if num_locations is not None else int(config.NUM_LOCATIONS)
        self.area_size = float(area_size) if area_size is not None else float(config.AREA_SIZE)
        self.invalid_action_penalty = float(invalid_action_penalty)
        
        # 存储原始数据（用于reset时恢复）
        self._episode_data = episode_data
        
        # 初始化UAV、请求和位置
        if episode_data is not None:
            # 从数据集加载
            self.uavs = self._copy_uavs(episode_data.uavs)
            self.requests = self._copy_requests(episode_data.requests)
            self.locations = episode_data.locations
            self._original_uavs = self._copy_uavs(episode_data.uavs)
            self._original_requests = self._copy_requests(episode_data.requests)
        else:
            # 随机生成
            self.uavs = uavs if uavs is not None else self._generate_uavs()
            self.requests = requests if requests is not None else self._generate_requests()
            self.locations = locations if locations is not None else self._generate_locations()
            self._original_uavs = None
            self._original_requests = None
        
        self.num_uavs = len(self.uavs) if self.uavs else config.NUM_UAVS
        
        # 计算最大距离（用于归一化）
        self.max_distance = (self.area_size**2 + self.area_size**2)**0.5
        
        # 动作空间: 请求选择 + 两个VNF的UAV分配
        self.action_space = spaces.Dict({
            'request_idx': spaces.Discrete(self.max_pending),
            'uav_for_vnf1': spaces.Discrete(self.num_uavs),
            'uav_for_vnf2': spaces.Discrete(self.num_uavs),
        })
        
        # 观察空间
        self.observation_space = spaces.Dict({
            'uav_states': spaces.Box(
                low=0, high=1, 
                shape=(self.num_uavs, 6),   # 移除了 location_id
                dtype=np.float32
            ),
            'pending_requests': spaces.Box(
                low=0, high=1, 
                shape=(self.max_pending, 5),  # 移除了 wait_time
                dtype=np.float32
            ),
            'global_features': spaces.Box(
                low=0, high=1, 
                shape=(4,), 
                dtype=np.float32
            ),
        })
        
        # 内部状态
        self.current_time_slot = 1
        self.pending_queue = []
        self.completed_requests = []
        self.rejected_requests = []
        self.episode_step = 0
        self.max_steps_per_slot = max_steps_per_slot
        self.current_slot_step = 0
        
    def _generate_uavs(self) -> List[UAV]:
        """生成UAV列表（初始位于基站）"""
        uavs = []
        for i in range(self.num_uavs):
            uav = UAV(i, config.BASE_STATION_LOCATION)
            uav.location_id = 0  # 基站位置ID为0
            uavs.append(uav)
        return uavs
    
    def _generate_requests(self) -> List[Request]:
        """生成请求列表（简化实现，实际应从外部传入或使用生成器）"""
        # 这里返回空列表，实际使用时应该传入预生成的请求
        return []
    
    def _generate_locations(self) -> Dict[int, Tuple[float, float]]:
        """生成位置字典"""
        locations = {0: config.BASE_STATION_LOCATION}
        # 其他位置应由外部传入
        return locations
    
    def _copy_uavs(self, uavs: List[UAV]) -> List[UAV]:
        """深拷贝UAV列表"""
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
        """深拷贝请求列表"""
        copied = []
        for req in requests:
            # 深拷贝VNF列表
            vnfs_copy = []
            for vnf in req.vnfs:
                new_vnf = VNF(vnf.id, vnf.location_id, vnf.cpu_freq, vnf.workload)
                vnfs_copy.append(new_vnf)
            
            # 深拷贝通信需求
            comm_demands_copy = deepcopy(req.communication_demands)
            
            # 创建新的请求
            new_req = Request(req.id, vnfs_copy, comm_demands_copy, req.reward)
            new_req.is_serviced = req.is_serviced
            copied.append(new_req)
        return copied
    
    def set_episode_data(self, episode_data: 'EpisodeData'):
        """
        设置回合数据（用于从数据集加载）
        
        Args:
            episode_data: 回合数据
        """
        self._episode_data = episode_data
        self.uavs = self._copy_uavs(episode_data.uavs)
        self.requests = self._copy_requests(episode_data.requests)
        self.locations = episode_data.locations
        self._original_uavs = self._copy_uavs(episode_data.uavs)
        self._original_requests = self._copy_requests(episode_data.requests)
    
    def reset(self) -> Dict[str, np.ndarray]:
        """
        重置环境状态
        
        Returns:
            初始观察状态
        """
        # 如果有原始数据（从数据集加载），则从原始数据恢复
        if self._original_uavs is not None and self._original_requests is not None:
            self.uavs = self._copy_uavs(self._original_uavs)
            self.requests = self._copy_requests(self._original_requests)
        else:
            # 否则重置当前UAV状态
            for uav in self.uavs:
                uav.energy = config.UAV_BATTERY_CAPACITY
                uav.cpu_capacity = config.UAV_COMPUTATION_CAPACITY
                uav.location = config.BASE_STATION_LOCATION
                uav.location_id = 0
                uav.is_busy = False
            
            # 重置请求状态
            for req in self.requests:
                req.is_serviced = False
        
        # 重置环境状态
        self.current_time_slot = 1
        self.current_slot_step = 0
        self.episode_step = 0
        self.pending_queue = [req for req in self.requests if not req.is_serviced]
        self.completed_requests = []
        self.rejected_requests = []
        
        return self._get_observation()
    
    def step(self, action: Dict) -> Tuple[Dict, float, bool, Dict]:
        """
        执行一步动作
        
        Args:
            action: {
                'request_idx': int,  # 选择的请求索引
                'uav_for_vnf1': int, # VNF1分配的UAV索引
                'uav_for_vnf2': int  # VNF2分配的UAV索引
            }
        
        Returns:
            observation: 下一个状态
            reward: 奖励值
            done: 是否结束
            info: 额外信息
        """
        self.episode_step += 1
        self.current_slot_step += 1
        
        # 解析动作
        req_idx = action['request_idx']
        uav1_idx = action['uav_for_vnf1']
        uav2_idx = action['uav_for_vnf2']
        
        # 执行分配
        success, info = self._execute_assignment(req_idx, uav1_idx, uav2_idx)
        
        # 检查是否进入下一时间槽
        if self._should_advance_time_slot():
            self._advance_time_slot()
        
        # 检查是否结束
        done = (
            self.current_time_slot > self.num_time_slots or 
            len(self.pending_queue) == 0 or
            self.episode_step >= self.max_steps_per_episode
        )
        
        # 计算奖励（需要在检查done之后，因为奖励计算依赖done）
        reward = self._compute_reward(success, info, done)
        
        obs = self._get_observation()
        info.update({
            'completed_count': len(self.completed_requests),
            'current_time_slot': self.current_time_slot,
            'pending_count': len(self.pending_queue),
        })
        
        return obs, reward, done, info
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """
        构建观察状态
        
        Returns:
            observation: 包含uav_states, pending_requests, global_features的字典
        """
        # UAV状态 [NUM_UAVS, 6] - 移除了 location_id
        uav_states = np.zeros((self.num_uavs, 6), dtype=np.float32)
        for i, uav in enumerate(self.uavs):
            uav_states[i] = [
                uav.energy / config.UAV_BATTERY_CAPACITY,
                uav.cpu_capacity / config.UAV_COMPUTATION_CAPACITY,
                uav.location[0] / self.area_size,
                uav.location[1] / self.area_size,
                float(uav.is_busy),
                self._distance_to_base(uav) / self.max_distance,
            ]
        
        # 待处理请求状态 [MAX_PENDING, 5] - 移除了 wait_time
        pending_states = np.zeros((self.max_pending, 5), dtype=np.float32)
        for i, req in enumerate(self.pending_queue[:self.max_pending]):
            # 请求特征
            num_vnfs = len(req.vnfs)
            total_workload = sum(v.workload for v in req.vnfs)
            total_comm = sum(req.communication_demands.values()) if req.communication_demands else 0
            
            loc1_id = req.vnfs[0].location_id if num_vnfs > 0 else 0
            loc2_id = req.vnfs[1].location_id if num_vnfs > 1 else loc1_id
            
            pending_states[i] = [
                num_vnfs / config.VNFS_PER_REQUEST,
                total_workload / (config.VNF_WORKLOAD_RANGE[1] * config.VNFS_PER_REQUEST),
                total_comm / config.VNF_COMMUNICATION_DEMAND_RANGE[1],
                loc1_id / self.num_locations if self.num_locations > 0 else 0,
                loc2_id / self.num_locations if self.num_locations > 0 else 0,
            ]
        
        # 全局特征 [4]
        global_features = np.array([
            self.current_time_slot / self.num_time_slots,
            (self.num_time_slots - self.current_time_slot) / self.num_time_slots,
            len(self.completed_requests) / max(len(self.requests), 1),
            len(self.pending_queue) / max(len(self.requests), 1),
        ], dtype=np.float32)
        
        return {
            'uav_states': uav_states,
            'pending_requests': pending_states,
            'global_features': global_features,
        }
    
    def _execute_assignment(self, req_idx: int, uav1_idx: int, uav2_idx: int) -> Tuple[bool, Dict]:
        """
        执行UAV分配
        
        Args:
            req_idx: 请求索引
            uav1_idx: VNF1的UAV索引
            uav2_idx: VNF2的UAV索引
        
        Returns:
            (success, info): 是否成功，以及信息字典
        """
        # 检查请求索引有效性
        if req_idx >= len(self.pending_queue):
            return False, {'error': 'invalid_request_idx', 'action_invalid': True}
        
        request = self.pending_queue[req_idx]
        uav1 = self.uavs[uav1_idx]
        uav2 = self.uavs[uav2_idx]
        
        # 检查约束
        if not self._check_constraints(request, uav1, uav2):
            return False, {'error': 'constraint_violation', 'action_invalid': True}
        
        # 执行分配
        self._allocate(request, uav1, uav2)
        
        # 从待处理队列移除
        self.pending_queue.pop(req_idx)
        self.completed_requests.append(request.id)
        request.is_serviced = True
        
        # 计算能耗
        energy_consumed = self._calculate_energy(request, uav1, uav2)
        cpu_used = request.vnfs[0].workload + request.vnfs[1].workload
        
        return True, {
            'sfc_completed': True,
            'energy_consumed': energy_consumed,
            'cpu_used': cpu_used,
            'request_id': request.id,
        }
    
    def _check_constraints(self, request: Request, uav1: UAV, uav2: UAV) -> bool:
        """
        检查约束条件
        
        约束:
        1. UAV计算能力必须足够
        2. UAV电量必须足够（飞行到目标 + 服务 + 返回基站）
        
        注意：
        - UAV可以飞到目标位置，不需要已经在那里
        - UAV在服务完成后需要返回基站充电
        - 需要预留返回基站的能量
        - busy的UAV不能移动，但若VNF就在当前位置可继续部署
        """
        vnf1, vnf2 = request.vnfs[0], request.vnfs[1]
        
        # 检查两个VNF的目标位置是否不同
        if vnf1.location_id == vnf2.location_id:
            return False
        
        # 检查：同一SFC的两个VNF在不同位置，必须由不同UAV服务
        # 因为一个UAV在一个时隙内只能在一个位置停留
        if uav1.id == uav2.id:
            return False
        
        # busy的UAV不能移动，只能在当前位置继续服务
        target1_id = vnf1.location_id
        target2_id = vnf2.location_id
        if uav1.is_busy and uav1.location_id != target1_id:
            return False
        if uav2.is_busy and uav2.location_id != target2_id:
            return False
        
        # 检查计算能力
        if uav1.cpu_capacity < vnf1.workload:
            return False
        if uav2.cpu_capacity < vnf2.workload:
            return False
        
        # 检查电量：飞行到目标 + 服务 + 返回基站
        for uav, vnf in [(uav1, vnf1), (uav2, vnf2)]:
            target_location = self.locations.get(vnf.location_id, uav.location)
            if uav.is_busy:
                travel_distance = 0.0
            else:
                travel_distance = utils.calculate_distance(uav.location, target_location)
            
            travel_energy = self._estimate_travel_energy(travel_distance)
            service_energy = self._estimate_service_energy(vnf)
            return_energy = self._estimate_return_energy(uav)
            
            if uav.energy < (travel_energy + service_energy + return_energy):
                return False
        
        return True
    
    def _allocate(self, request: Request, uav1: UAV, uav2: UAV):
        """
        执行资源分配（扣除资源）
        
        UAV若空闲会飞到目标位置；若busy则保持原位置不动。
        服务后UAV保持busy直到时隙结束。
        """
        vnf1, vnf2 = request.vnfs[0], request.vnfs[1]
        
        # 扣除计算能力
        uav1.cpu_capacity -= vnf1.workload
        if uav2.id != uav1.id:
            uav2.cpu_capacity -= vnf2.workload
        else:
            uav1.cpu_capacity -= vnf2.workload
        
        # 让UAV飞到目标位置，扣除电量，更新位置
        for uav, vnf in [(uav1, vnf1), (uav2, vnf2)]:
            target_location = self.locations.get(vnf.location_id, uav.location)
            if not uav.is_busy:
                travel_distance = utils.calculate_distance(uav.location, target_location)

                # 扣除飞行能耗并更新位置
                travel_energy = self._estimate_travel_energy(travel_distance)
                uav.energy -= travel_energy
                uav.location = target_location
                uav.location_id = vnf.location_id
            
            # 扣除服务能耗
            service_energy = self._estimate_service_energy(vnf)
            uav.energy -= service_energy
        
        # 标记忙碌（只要有VNF服务就不能移动）
        uav1.is_busy = True
        if uav2.id != uav1.id:
            uav2.is_busy = True
        
        # 注意：返回基站的能耗在时隙结束时统一计算
    
    def _compute_reward(self, success: bool, info: Dict, done: bool) -> float:
        """
        计算奖励函数 - 稠密奖励设计
        
        策略：
        - 每次成功完成SFC时立即给予奖励
        - 无效动作给予惩罚
        - Episode结束时给予基于成功率的额外奖励
        """
        reward = 0.0
        
        # 1. 无效动作惩罚（帮助智能体快速学习约束）
        if info.get('action_invalid'):
            reward -= self.invalid_action_penalty
        
        # 2. 稠密奖励：每次成功完成SFC时立即给予奖励
        if success and info.get('sfc_completed'):
            # 基础完成奖励
            reward += 1.0
            
            # 效率奖励：完成的请求数占总请求的比例
            progress_bonus = len(self.completed_requests) / max(len(self.requests), 1)
            reward += progress_bonus * 0.5
        
        # 3. Episode结束时的额外奖励（可选，用于鼓励高完成率）
        if done:
            total_requests = len(self.requests)
            completed_requests = len(self.completed_requests)
            
            if total_requests > 0:
                success_rate = completed_requests / total_requests
                # 高折扣因子的完成率奖励
                reward += success_rate * 5.0
                
                # 完美完成的额外奖励
                if success_rate >= 0.95:
                    reward += 10.0  # 高折扣的额外奖励
            
            # 记录信息
            info['final_success_rate'] = completed_requests / max(total_requests, 1)
            info['total_completed'] = completed_requests
        
        return reward
    
    def _should_advance_time_slot(self) -> bool:
        """
        判断是否应该进入下一时隙
        
        条件：当前时隙已经没有可处理的SFC（所有剩余请求都因资源约束无法满足）
        """
        # 如果没有剩余请求，不需要推进
        if not self.pending_queue:
            return False

        # 达到当前时隙步数上限，强制推进到下一时隙
        if self.current_slot_step >= self.max_steps_per_slot:
            return True
        
        # 检查是否还有任何请求可以被服务
        for request in self.pending_queue:
            if self._can_service_request(request):
                return False
        
        # 没有可服务的请求了，推进时隙
        return True
    
    def _can_service_request(self, request: Request) -> bool:
        """
        检查给定请求是否可以被当前的UAV资源满足
        """
        vnf1, vnf2 = request.vnfs[0], request.vnfs[1]

        # 同一SFC两个VNF若在同一位置，按当前约束不可行
        if vnf1.location_id == vnf2.location_id:
            return False

        vnf1_mask, vnf2_mask = self._get_request_aware_uav_masks(request)
        return self._has_feasible_uav_pair(vnf1_mask, vnf2_mask)

    def _has_feasible_uav_pair(self, vnf1_mask: np.ndarray, vnf2_mask: np.ndarray) -> bool:
        """根据两个VNF各自的UAV可选集，判断是否存在满足 uav1 != uav2 的可行组合。"""
        count_vnf1 = int(vnf1_mask.sum())
        count_vnf2 = int(vnf2_mask.sum())

        if count_vnf1 == 0 or count_vnf2 == 0:
            return False

        # 若两侧都只有一个候选，需要确保不是同一台UAV
        if count_vnf1 == 1 and count_vnf2 == 1:
            idx_vnf1 = int(np.argmax(vnf1_mask))
            idx_vnf2 = int(np.argmax(vnf2_mask))
            return idx_vnf1 != idx_vnf2

        return True

    def _can_uav_service_vnf(self, uav: UAV, vnf: VNF) -> bool:
        """检查单台UAV是否可服务指定VNF（不包含跨VNF耦合约束）。"""
        target_location = self.locations.get(vnf.location_id, uav.location)

        # busy的UAV不能移动，只能服务当前位置VNF
        if uav.is_busy and uav.location_id != vnf.location_id:
            return False

        # 计算能力约束
        if uav.cpu_capacity < vnf.workload:
            return False

        # 电量约束：飞行到目标 + 服务 + 返回基站
        if uav.is_busy:
            travel_distance = 0.0
        else:
            travel_distance = utils.calculate_distance(uav.location, target_location)

        travel_energy = self._estimate_travel_energy(travel_distance)
        service_energy = self._estimate_service_energy(vnf)
        return_energy = self._estimate_return_energy(uav)

        return uav.energy >= (travel_energy + service_energy + return_energy)

    def _get_request_aware_uav_masks(self, request: Request) -> Tuple[np.ndarray, np.ndarray]:
        """
        为单个请求构建双掩码：
        - vnf1_mask: 可服务VNF1的UAV
        - vnf2_mask: 可服务VNF2的UAV

        返回形状均为 [num_uavs] 的0/1向量。
        """
        vnf1, vnf2 = request.vnfs[0], request.vnfs[1]
        vnf1_mask = np.zeros(self.num_uavs, dtype=np.int32)
        vnf2_mask = np.zeros(self.num_uavs, dtype=np.int32)

        for uav_idx, uav in enumerate(self.uavs):
            if self._can_uav_service_vnf(uav, vnf1):
                vnf1_mask[uav_idx] = 1
            if self._can_uav_service_vnf(uav, vnf2):
                vnf2_mask[uav_idx] = 1

        return vnf1_mask, vnf2_mask
    
    def _get_eligible_uavs(self, vnf: VNF) -> List[UAV]:
        """
        获取可以部署指定VNF的UAV列表
        
        注意：UAV可以飞到目标位置，不需要已经在那里。
        需要检查是否有足够的能量飞到目标位置、完成服务并返回基站。
        busy的UAV不能移动，但若VNF就在当前位置可继续服务。
        """
        eligible = []
        for uav in self.uavs:
            if self._can_uav_service_vnf(uav, vnf):
                eligible.append(uav)

        return eligible
    
    def _advance_time_slot(self):
        """
        推进到下一时隙，更新UAV状态
        
        严格的能耗模型：
        - 空闲的UAV必须飞回基站才能充电（扣除返回能耗）
        - 忙碌的UAV保持当前位置
        - 在基站的UAV充满电
        - 重置计算能力
        """
        self.current_time_slot += 1
        self.current_slot_step = 0
        
        # 重置UAV状态
        for uav in self.uavs:
            # 1. 空闲且不在基站的UAV飞回基站充电
            if not uav.is_busy and uav.location_id != 0:
                return_distance = self._distance_to_base(uav)
                return_energy = utils.calculate_travel_energy(return_distance)
                uav.energy = max(0, uav.energy - return_energy)  # 扣除返回能耗
                uav.location = config.BASE_STATION_LOCATION
                uav.location_id = 0
            
            # 2. 在基站的UAV充满电
            if uav.location_id == 0:
                uav.energy = config.UAV_BATTERY_CAPACITY
            
            # 3. 重置计算能力（新时隙开始时重新部署）
            uav.cpu_capacity = config.UAV_COMPUTATION_CAPACITY
            
            # 4. 重置忙碌状态
            uav.is_busy = False
    
    def _distance_to_base(self, uav: UAV) -> float:
        """计算UAV到基站的距离"""
        dx = uav.location[0] - config.BASE_STATION_LOCATION[0]
        dy = uav.location[1] - config.BASE_STATION_LOCATION[1]
        return (dx**2 + dy**2)**0.5
    
    def _estimate_travel_energy(self, distance: float) -> float:
        """估算飞行能耗 - 使用论文公式18"""
        return utils.calculate_travel_energy(distance)
    
    def _estimate_service_energy(self, vnf: VNF, communication_partners: dict = None) -> float:
        """估算服务能耗 - 使用论文公式19（悬停+计算+通信）"""
        if communication_partners is None:
            # 如果没有提供通信伙伴，只计算计算能耗+悬停能耗
            hover_energy = (config.UAV_BLADE_PROFILE_POWER + 
                          config.UAV_INDUCED_POWER) * config.TIME_SLOT_DURATION
            computation_energy = (config.CHIP_ARCHITECTURE_CONSTANT * 
                                 (vnf.cpu_freq)**2 * vnf.workload)
            return hover_energy + computation_energy + config.COMM_ENERGY_PER_TIMESLOT
        else:
            return utils.calculate_servicing_energy(vnf, communication_partners, self.locations)
    
    def _estimate_return_energy(self, uav: UAV) -> float:
        """估算返回基站的能耗"""
        distance = self._distance_to_base(uav)
        return utils.calculate_travel_energy(distance)
    
    def _calculate_energy(self, request: Request, uav1: UAV, uav2: UAV) -> float:
        """计算完成请求的总能耗（飞行+服务+返回）"""
        total_energy = 0
        for uav, vnf in [(uav1, request.vnfs[0]), (uav2, request.vnfs[1])]:
            # 飞行到目标位置的能耗
            target_location = self.locations.get(vnf.location_id, uav.location)
            travel_distance = utils.calculate_distance(uav.location, target_location)
            travel_energy = self._estimate_travel_energy(travel_distance)
            
            # 服务能耗
            service_energy = self._estimate_service_energy(vnf)
            
            # 返回基站能耗
            return_energy = self._estimate_return_energy(uav)
            
            total_energy += (travel_energy + service_energy + return_energy)
        return total_energy
    
    def get_action_mask(self) -> Dict[str, np.ndarray]:
        """
        获取动作掩码（用于屏蔽无效动作）
        
        Returns:
            masks: {
                'request': [MAX_PENDING] - 1表示可选，0表示不可选
                'uav': [NUM_UAVS] - 1表示可选，0表示不可选
                'uav_by_request_vnf1': [MAX_PENDING, NUM_UAVS] - VNF1条件化UAV掩码
                'uav_by_request_vnf2': [MAX_PENDING, NUM_UAVS] - VNF2条件化UAV掩码
                'uav_by_request': [MAX_PENDING, NUM_UAVS] - 兼容旧逻辑的并集掩码
            }
        """
        # 请求选择掩码
        request_mask = np.zeros(self.max_pending, dtype=np.int32)
        uav_by_request_mask_vnf1 = np.zeros((self.max_pending, self.num_uavs), dtype=np.int32)
        uav_by_request_mask_vnf2 = np.zeros((self.max_pending, self.num_uavs), dtype=np.int32)
        uav_by_request_mask = np.zeros((self.max_pending, self.num_uavs), dtype=np.int32)

        for i, req in enumerate(self.pending_queue[:self.max_pending]):
            req_uav_mask_vnf1, req_uav_mask_vnf2 = self._get_request_aware_uav_masks(req)
            uav_by_request_mask_vnf1[i] = req_uav_mask_vnf1
            uav_by_request_mask_vnf2[i] = req_uav_mask_vnf2
            uav_by_request_mask[i] = np.maximum(req_uav_mask_vnf1, req_uav_mask_vnf2)

            if self._has_feasible_uav_pair(req_uav_mask_vnf1, req_uav_mask_vnf2):
                request_mask[i] = 1

        # 全局UAV掩码：取所有可服务请求对应掩码的并集
        uav_mask = ((uav_by_request_mask_vnf1 + uav_by_request_mask_vnf2).sum(axis=0) > 0).astype(np.int32)
        
        return {
            'request': request_mask,
            'uav': uav_mask,
            'uav_by_request_vnf1': uav_by_request_mask_vnf1,
            'uav_by_request_vnf2': uav_by_request_mask_vnf2,
            'uav_by_request': uav_by_request_mask,
        }
    
    def render(self, mode='human'):
        """渲染环境状态"""
        print(f"\n=== Time Slot {self.current_time_slot} ===")
        print(f"Completed: {len(self.completed_requests)}/{len(self.requests)}")
        print(f"Pending: {len(self.pending_queue)}")
        print(f"UAV Energy (avg): {np.mean([uav.energy for uav in self.uavs]):.2f}")
    
    def close(self):
        """关闭环境"""
        pass
