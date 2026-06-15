# -*- coding: utf-8 -*-
"""
配置文件
参考论文中的 Table III，集中管理所有模拟参数
"""

# === 场景参数 ===
AREA_SIZE = 1000.0  # 区域大小 (1000m x 1000m)
NUM_LOCATIONS = 300  # 监控点数量
BASE_STATION_LOCATION = (500.0, 500.0) # 基站位置

# === 时间参数 ===
NUM_TIME_SLOTS = 4  # 规划范围的时间槽总数
TIME_SLOT_DURATION = 1.0  # 每个时间槽的持续时间 (s)

# === 无人机 (UAV) 参数 ===
NUM_UAVS = 100  # 增加UAV数量确保有足够的资源处理所有SFC（200请求×2VNF=400UAV需求，4时间槽需要至少100UAV）
UAV_MAX_SPEED = 15.0  # UAV 最大飞行速度 (m/s)
# 基于最大速度和区域对角线距离计算出的固定飞行时间
UAV_TRAVEL_TIME = (AREA_SIZE**2 + AREA_SIZE**2)**0.5 / UAV_MAX_SPEED
# UAV 最大电池容量 (J)
# 计算依据: 区域对角线1414m往返飞行(约26125J) + 服务能耗(约170J) = 约26295J
# 考虑90%安全余量，设置为50000J，确保UAV可完成最远距离任务并有充足余量
UAV_BATTERY_CAPACITY = 50000.0
UAV_COMPUTATION_CAPACITY = 1.0 * 1e3  # UAV 计算能力 (MHz)
UAV_TRANSMIT_POWER = 10.0  # UAV 发射功率 (W)

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
NUM_REQUESTS = 300  # 总请求数量
VNFS_PER_REQUEST = 2  # 每个请求包含的 VNF 数量
VNF_CPU_FREQUENCY_RANGE = (0.1, 0.5)  # VNF CPU 频率需求范围 (GHz)
VNF_WORKLOAD_RANGE = (100, 500)  # VNF 计算工作量范围 (MCCs)
# VNF 间通信需求范围 (bits)
# 约束条件: 最远距离(1414m)下链路容量约309 Kbps，即309 Kbits/时间槽
# 设置为50~200 Kbits，确保通信需求不超过链路容量上限(余量35%)
VNF_COMMUNICATION_DEMAND_RANGE = (50000, 200000)  # 50 Kbits ~ 200 Kbits
DEFAULT_REQUEST_REWARD = 1000.0 # 默认的请求奖励

# === 简化对数通信模型参数 ===
# 基于香农公式简化: C = B * log2(1 + SNR)，其中 SNR ∝ 1/d^path_loss
# 简化为对数衰减模型:
#   capacity = C_MAX * max(0, 1 - log(1 + distance/D_REF) / log(1 + D_MAX/D_REF))
# 
# 参数说明:
#   C_MAX: 零距离时的最大容量 (bps)
#   D_MAX: 最大有效通信距离，超过此距离容量为0 (m)
#   D_REF: 参考距离，控制衰减速率 (m)
#   PATH_LOSS_EXPONENT: 路径损耗指数 (影响对数曲线的形状)

COMM_CAPACITY_MAX = 10e5           # 最大容量 10 Mbps
COMM_DISTANCE_MAX = 5000.0         # 最大有效距离 5000m，超过此距离容量为0
COMM_DISTANCE_REF = 100.0          # 参考距离 100m
COMM_PATH_LOSS_EXPONENT = 2.0      # 路径损耗指数 (通常2-4)

# 简化通信能耗参数
COMM_ENERGY_PER_TIMESLOT = 0.06  # 每个时间槽的通信能耗 (J)

# === 计算能耗模型参数 ===
# (参考论文 III-D2, a part of formula 19)
CHIP_ARCHITECTURE_CONSTANT = 1e-11