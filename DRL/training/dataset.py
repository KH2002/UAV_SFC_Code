# -*- coding: utf-8 -*-
"""
固定种子训练数据集

用于生成和存储固定种子的训练场景，确保不同RL算法可以使用相同的数据进行对比。
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
import pickle
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

import config
from entities import UAV, VNF, Request


# ==================== 场景生成函数 ====================

def generate_locations(num_locations: int, area_size: float) -> Dict[int, Tuple[float, float]]:
    """
    生成监控点位置（包含基站）
    
    Args:
        num_locations: 监控点数量（不包括基站）
        area_size: 区域大小
    
    Returns:
        位置字典 {location_id: (x, y)}
    """
    locations = {0: config.BASE_STATION_LOCATION}  # 基站位置ID为0
    
    # 随机生成监控点位置
    for i in range(1, num_locations + 1):
        x = np.random.uniform(0, area_size)
        y = np.random.uniform(0, area_size)
        locations[i] = (x, y)
    
    return locations


def generate_uavs(num_uavs: int, locations: Dict[int, Tuple[float, float]]) -> List[UAV]:
    """
    生成UAV列表（初始位于基站）
    
    Args:
        num_uavs: UAV数量
        locations: 位置字典
    
    Returns:
        UAV列表
    """
    uavs = []
    for i in range(num_uavs):
        uav = UAV(i, config.BASE_STATION_LOCATION)
        uav.location_id = 0  # 基站位置ID为0
        uavs.append(uav)
    return uavs


def generate_requests(num_requests: int, locations: Dict[int, Tuple[float, float]]) -> List[Request]:
    """
    生成请求列表
    
    Args:
        num_requests: 请求数量
        locations: 位置字典
    
    Returns:
        请求列表
    """
    requests = []
    location_ids = list(locations.keys())
    location_ids.remove(0)  # 移除基站（0是基站，不作为监控点）
    
    for i in range(num_requests):
        # 随机选择两个不同的位置
        loc_ids = np.random.choice(location_ids, size=config.VNFS_PER_REQUEST, replace=False)
        
        # 断言：确保两个位置不同
        assert len(set(loc_ids)) == len(loc_ids), f"生成的位置有重复: {loc_ids}"
        
        # 创建VNF
        vnfs = []
        for j, loc_id in enumerate(loc_ids):
            cpu_freq = np.random.uniform(*config.VNF_CPU_FREQUENCY_RANGE)
            workload = np.random.randint(*config.VNF_WORKLOAD_RANGE)
            vnf = VNF(
                vnf_id=i * config.VNFS_PER_REQUEST + j,
                location_id=loc_id,
                cpu_freq=cpu_freq,
                workload=workload
            )
            vnfs.append(vnf)
        
        # 创建通信需求
        communication_demands = {}
        if len(vnfs) >= 2:
            demand = np.random.randint(*config.VNF_COMMUNICATION_DEMAND_RANGE)
            communication_demands[(vnfs[0].id, vnfs[1].id)] = demand
        
        # 创建请求
        request = Request(
            request_id=i,
            vnfs=vnfs,
            communication_demands=communication_demands,
            reward=config.DEFAULT_REQUEST_REWARD
        )
        requests.append(request)
    
    return requests


@dataclass
class EpisodeData:
    """
    单个训练回合的数据
    
    包含该回合的所有场景信息（UAV、请求、位置）
    """
    episode_id: int
    seed: int
    locations: Dict[int, Tuple[float, float]]
    uavs: List[UAV]
    requests: List[Request]
    
    def copy(self) -> 'EpisodeData':
        """创建数据的深拷贝（用于环境重置）"""
        # 深拷贝locations
        locations_copy = {k: v for k, v in self.locations.items()}
        
        # 深拷贝UAVs
        uavs_copy = []
        for uav in self.uavs:
            new_uav = UAV(uav.id, uav.location)
            new_uav.location_id = uav.location_id
            new_uav.energy = uav.energy
            new_uav.cpu_capacity = uav.cpu_capacity
            new_uav.is_busy = uav.is_busy
            uavs_copy.append(new_uav)
        
        # 深拷贝Requests
        requests_copy = []
        for req in self.requests:
            # 拷贝VNF列表
            vnfs_copy = []
            for vnf in req.vnfs:
                new_vnf = VNF(vnf.id, vnf.location_id, vnf.cpu_freq, vnf.workload)
                vnfs_copy.append(new_vnf)
            
            # 拷贝通信需求
            comm_demands_copy = {k: v for k, v in req.communication_demands.items()}
            
            # 创建新的请求
            new_req = Request(req.id, vnfs_copy, comm_demands_copy, req.reward)
            new_req.is_serviced = req.is_serviced
            requests_copy.append(new_req)
        
        return EpisodeData(
            episode_id=self.episode_id,
            seed=self.seed,
            locations=locations_copy,
            uavs=uavs_copy,
            requests=requests_copy
        )


def generate_episode_data(
    episode_id: int, 
    seed: int,
    num_locations: int = None,
    area_size: float = None,
    num_uavs: int = None,
    num_requests: int = None
) -> EpisodeData:
    """
    生成单个回合的训练数据
    
    Args:
        episode_id: 回合ID
        seed: 随机种子
        num_locations: 监控点数量（默认从config读取）
        area_size: 区域大小（默认从config读取）
        num_uavs: UAV数量（默认从config读取）
        num_requests: 请求数量（默认从config读取）
    
    Returns:
        EpisodeData 实例
    """
    # 使用默认值
    if num_locations is None:
        num_locations = config.NUM_LOCATIONS
    if area_size is None:
        area_size = config.AREA_SIZE
    if num_uavs is None:
        num_uavs = config.NUM_UAVS
    if num_requests is None:
        num_requests = config.NUM_REQUESTS
    
    # 设置随机种子
    np.random.seed(seed)
    
    # 生成场景数据
    locations = generate_locations(num_locations, area_size)
    uavs = generate_uavs(num_uavs, locations)
    requests = generate_requests(num_requests, locations)
    
    return EpisodeData(
        episode_id=episode_id,
        seed=seed,
        locations=locations,
        uavs=uavs,
        requests=requests
    )


class TrainingDataset:
    """
    训练数据集
    
    生成和管理固定种子的训练数据，支持保存和加载。
    """
    
    def __init__(
        self, 
        base_seed: int = 42, 
        num_episodes: int = 1000,
        num_locations: int = None,
        area_size: float = None,
        num_uavs: int = None,
        num_requests: int = None
    ):
        """
        初始化训练数据集
        
        Args:
            base_seed: 基础随机种子，每个回合的种子为 base_seed + episode_id
            num_episodes: 数据集包含的回合数
            num_locations: 监控点数量（默认从config读取）
            area_size: 区域大小（默认从config读取）
            num_uavs: UAV数量（默认从config读取）
            num_requests: 请求数量（默认从config读取）
        """
        self.base_seed = base_seed
        self.num_episodes = num_episodes
        self.num_locations = num_locations if num_locations is not None else config.NUM_LOCATIONS
        self.area_size = area_size if area_size is not None else config.AREA_SIZE
        self.num_uavs = num_uavs if num_uavs is not None else config.NUM_UAVS
        self.num_requests = num_requests if num_requests is not None else config.NUM_REQUESTS
        self.episodes: List[EpisodeData] = []
        self.current_index = 0
        
    def generate(self, verbose: bool = True):
        """
        生成所有回合的训练数据
        
        Args:
            verbose: 是否打印进度
        """
        if verbose:
            print(f"生成训练数据集 (base_seed={self.base_seed}, num_episodes={self.num_episodes})...")
            print(f"  每个episode包含 {self.num_uavs} 个UAV 和 {self.num_requests} 个请求")
            print(f"  这可能需要一些时间，请耐心等待...")
        
        self.episodes = []
        for i in range(self.num_episodes):
            seed = self.base_seed + i
            episode_data = generate_episode_data(
                i, seed,
                num_locations=self.num_locations,
                area_size=self.area_size,
                num_uavs=self.num_uavs,
                num_requests=self.num_requests
            )
            self.episodes.append(episode_data)
            
            if verbose and (i + 1) % 10 == 0:
                print(f"  已生成 {i + 1}/{self.num_episodes} 个回合 ({(i+1)/self.num_episodes*100:.1f}%)")
        
        if verbose:
            print(f"数据集生成完成！共 {self.num_episodes} 个回合")
    
    def __len__(self) -> int:
        return len(self.episodes)
    
    def __getitem__(self, idx: int) -> EpisodeData:
        """获取指定回合的数据（创建拷贝以避免修改原始数据）"""
        if idx < 0 or idx >= len(self.episodes):
            raise IndexError(f"Index {idx} out of range [0, {len(self.episodes)})")
        return self.episodes[idx].copy()
    
    def get_batch(self, start_idx: int, batch_size: int) -> List[EpisodeData]:
        """获取一批回合数据"""
        end_idx = min(start_idx + batch_size, len(self.episodes))
        return [self.episodes[i].copy() for i in range(start_idx, end_idx)]
    
    def reset_iterator(self):
        """重置迭代器"""
        self.current_index = 0
    
    def next(self) -> Optional[EpisodeData]:
        """获取下一个回合数据（用于迭代）"""
        if self.current_index >= len(self.episodes):
            return None
        data = self.episodes[self.current_index].copy()
        self.current_index += 1
        return data
    
    def save(self, filepath: str):
        """
        保存数据集到文件
        
        Args:
            filepath: 保存路径
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump({
                'base_seed': self.base_seed,
                'num_episodes': self.num_episodes,
                'num_locations': self.num_locations,
                'area_size': self.area_size,
                'num_uavs': self.num_uavs,
                'num_requests': self.num_requests,
                'episodes': self.episodes
            }, f)
        print(f"数据集已保存到: {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'TrainingDataset':
        """
        从文件加载数据集
        
        Args:
            filepath: 文件路径
        
        Returns:
            TrainingDataset 实例
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        dataset = cls(
            base_seed=data['base_seed'],
            num_episodes=data['num_episodes'],
            num_locations=data.get('num_locations'),
            area_size=data.get('area_size'),
            num_uavs=data.get('num_uavs'),
            num_requests=data.get('num_requests')
        )
        dataset.episodes = data['episodes']

        # 兼容旧缓存文件（没有保存场景参数）
        if dataset.episodes:
            first_ep = dataset.episodes[0]
            if 'num_locations' not in data:
                dataset.num_locations = max(len(first_ep.locations) - 1, 0)
            if 'num_uavs' not in data:
                dataset.num_uavs = len(first_ep.uavs)
            if 'num_requests' not in data:
                dataset.num_requests = len(first_ep.requests)
        
        print(f"数据集已从 {filepath} 加载")
        print(f"  base_seed: {dataset.base_seed}")
        print(f"  num_episodes: {dataset.num_episodes}")
        print(f"  num_locations: {dataset.num_locations}")
        print(f"  area_size: {dataset.area_size}")
        print(f"  num_uavs: {dataset.num_uavs}")
        print(f"  num_requests: {dataset.num_requests}")
        
        return dataset


def create_or_load_dataset(
    base_seed: int = 42,
    num_episodes: int = 1000,
    data_dir: str = './data',
    force_regenerate: bool = False,
    num_locations: int = None,
    area_size: float = None,
    num_uavs: int = None,
    num_requests: int = None
) -> TrainingDataset:
    """
    创建或加载数据集
    
    如果存在已保存的数据集且参数匹配，则直接加载；
    否则生成新的数据集并保存。
    
    Args:
        base_seed: 基础随机种子
        num_episodes: 回合数
        data_dir: 数据保存目录
        force_regenerate: 是否强制重新生成
        num_locations: 监控点数量（默认从config读取）
        area_size: 区域大小（默认从config读取）
        num_uavs: UAV数量（默认从config读取）
        num_requests: 请求数量（默认从config读取）
    
    Returns:
        TrainingDataset 实例
    """
    # 构建包含场景参数的文件名
    locs_str = num_locations if num_locations is not None else config.NUM_LOCATIONS
    area_val = float(area_size if area_size is not None else config.AREA_SIZE)
    area_str = str(int(area_val)) if area_val.is_integer() else format(area_val, 'g').replace('.', 'p')
    uavs_str = num_uavs if num_uavs is not None else config.NUM_UAVS
    reqs_str = num_requests if num_requests is not None else config.NUM_REQUESTS
    dataset_filename = (
        f"dataset_seed{base_seed}_ep{num_episodes}"
        f"_loc{locs_str}_area{area_str}_uav{uavs_str}_req{reqs_str}.pkl"
    )
    dataset_path = os.path.join(data_dir, dataset_filename)
    
    # 检查是否存在已保存的数据集
    if not force_regenerate and os.path.exists(dataset_path):
        try:
            dataset = TrainingDataset.load(dataset_path)
            # 验证参数是否匹配
            if (
                dataset.base_seed == base_seed and
                dataset.num_episodes == num_episodes and
                int(dataset.num_locations) == int(locs_str) and
                abs(float(dataset.area_size) - area_val) < 1e-9 and
                int(dataset.num_uavs) == int(uavs_str) and
                int(dataset.num_requests) == int(reqs_str)
            ):
                return dataset
            else:
                print(f"已存在的数据集参数不匹配，重新生成...")
        except Exception as e:
            print(f"加载数据集失败: {e}，重新生成...")
    
    # 生成新的数据集
    dataset = TrainingDataset(
        base_seed=base_seed, 
        num_episodes=num_episodes,
        num_locations=num_locations,
        area_size=area_size,
        num_uavs=num_uavs,
        num_requests=num_requests
    )
    dataset.generate()
    dataset.save(dataset_path)
    
    return dataset
