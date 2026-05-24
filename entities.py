# -*- coding: utf-8 -*-
"""
定义模拟中的核心实体：UAV, VNF, Request
"""
import config

class UAV:
    """定义无人机 (UAV) 类"""
    def __init__(self, uav_id, location):
        self.id = uav_id
        self.location = location # 当前位置 (x, y)
        self.location_id = 0 # 0 代表基站
        self.energy = config.UAV_BATTERY_CAPACITY # 当前电量
        self.cpu_capacity = config.UAV_COMPUTATION_CAPACITY # 剩余计算能力
        self.is_busy = False

    def __repr__(self):
        return f"UAV(id={self.id}, loc_id={self.location_id}, energy={self.energy:.2f})"

class VNF:
    """定义虚拟网络功能 (VNF) 类"""
    def __init__(self, vnf_id, location_id, cpu_freq, workload):
        self.id = vnf_id
        self.location_id = location_id # 需要部署的目标位置 ID
        self.cpu_freq = cpu_freq # CPU 频率需求 (GHz)
        self.workload = workload # 计算工作量 (MCCs)

    def __repr__(self):
        return f"VNF(id={self.id}, loc_id={self.location_id})"

class Request:
    """定义服务请求类"""
    def __init__(self, request_id, vnfs, communication_demands, reward=config.DEFAULT_REQUEST_REWARD):
        self.id = request_id
        self.vnfs = vnfs # VNF 对象列表
        # {(vnf_id1, vnf_id2): demand}
        self.communication_demands = communication_demands
        self.reward = reward
        self.is_serviced = False

    def get_required_location_ids(self):
        """获取此请求需要的所有位置ID"""
        return sorted(list(set(vnf.location_id for vnf in self.vnfs)))

    def __repr__(self):
        loc_ids = self.get_required_location_ids()
        return f"Request(id={self.id}, locations={loc_ids}, serviced={self.is_serviced})"