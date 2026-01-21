# -*- coding: utf-8 -*-
"""
配置文件
参考论文中的 Table III，集中管理所有模拟参数
"""

# === 场景参数 ===
AREA_SIZE = 100.0  # 区域大小 (100m x 100m)
NUM_LOCATIONS = 10  # 监控点数量
BASE_STATION_LOCATION = (50.0, 50.0) # 基站位置

# === 时间参数 ===
NUM_TIME_SLOTS = 4  # 规划范围的时间槽总数
TIME_SLOT_DURATION = 1.0  # 每个时间槽的持续时间 (s)

# === 无人机 (UAV) 参数 ===
NUM_UAVS = 4  # UAV 的数量
UAV_MAX_SPEED = 15.0  # UAV 最大飞行速度 (m/s)
# 基于最大速度和区域对角线距离计算出的固定飞行时间
UAV_TRAVEL_TIME = (AREA_SIZE**2 + AREA_SIZE**2)**0.5 / UAV_MAX_SPEED
UAV_BATTERY_CAPACITY = 277200.0  # UAV 最大电池容量 (J)，参考大疆 Mavic 3
UAV_COMPUTATION_CAPACITY = 1.0 * 1e3  # UAV 计算能力 (MHz)
UAV_TRANSMIT_POWER = 1.0  # UAV 发射功率 (W)

# 悬停功率参数 (W)
UAV_BLADE_PROFILE_POWER = 79.86
UAV_INDUCED_POWER = 88.63

# 飞行能耗模型参数 (参考论文公式 18)
UAV_ROTOR_BLADE_TIP_SPEED = 120.0 #飞行时螺旋桨速度
UAV_MEAN_ROTOR_INDUCED_VELOCITY = 4.03  #悬停时速度
UAV_FUSELAGE_DRAG_RATIO = 0.6
AIR_DENSITY = 1.225
UAV_ROTOR_SOLIDITY = 0.05
UAV_ROTOR_DISC_AREA = 0.503

# === 虚拟网络功能 (VNF) 和请求参数 ===
NUM_REQUESTS = 20  # 总请求数量
VNFS_PER_REQUEST = 2  # 每个请求包含的 VNF 数量
VNF_CPU_FREQUENCY_RANGE = (0.1, 0.5)  # VNF CPU 频率需求范围 (GHz)
VNF_WORKLOAD_RANGE = (100, 500)  # VNF 计算工作量范围 (MCCs)
VNF_COMMUNICATION_DEMAND_RANGE = (1e5, 5e5)  # VNF 间通信需求范围 (bits)
DEFAULT_REQUEST_REWARD = 1000.0 # 默认的请求奖励

# === 通信模型参数 ===
PATH_LOSS_EXPONENT = 2.05  # 路径损耗指数
CHANNEL_BANDWIDTH = 200e3  # 信道带宽 (Hz)
NOISE_POWER = 10**(-95 / 10) / 1000  # 噪声功率 (W), 从 -95 dBm 转换

# 收发器电路参数 (参考论文公式 10)
TRANSCEIVER_CIRCUIT_POWER = 59.8e-3 # (W)
RECEIVER_SENSITIVITY = -85 # (dBm)
POWER_AMPLIFIER_DRAIN_EFFICIENCY = 0.05

# === 计算能耗模型参数 ===
# (参考论文 III-D2, a part of formula 19)
CHIP_ARCHITECTURE_CONSTANT = 1e-11