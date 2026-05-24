# -*- coding: utf-8 -*-
"""
包含所有计算逻辑的辅助函数
- 距离计算
- 通信容量计算
- 能耗计算 (飞行、悬停、计算、通信)
"""
import math
import numpy as np
import config

def calculate_distance(point1, point2):
    """计算两点间的欧氏距离 (论文公式 1)"""
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def calculate_travel_energy(distance):
    """计算飞行指定距离所需的能耗 (论文公式 18)"""
    if distance == 0:
        return 0.0

    velocity = distance / config.UAV_TRAVEL_TIME
    if velocity == 0:
        return 0.0


    # 公式分解
    term1 = config.UAV_BLADE_PROFILE_POWER * (1 / velocity + (3 * velocity) / config.UAV_ROTOR_BLADE_TIP_SPEED**2)
    term2 = config.UAV_INDUCED_POWER * ((np.sqrt(velocity**-4 + 1 / (4 * config.UAV_MEAN_ROTOR_INDUCED_VELOCITY**4)) - (1 / (2 * config.UAV_MEAN_ROTOR_INDUCED_VELOCITY**2)))**0.5)
    term3 = 0.5 * config.UAV_FUSELAGE_DRAG_RATIO * config.AIR_DENSITY * config.UAV_ROTOR_SOLIDITY * config.UAV_ROTOR_DISC_AREA * velocity**2
    
    # 每米能耗
    energy_per_meter = term1 + term2 + term3
    return energy_per_meter * distance


def calculate_link_capacity(distance):
    """根据距离计算两点间的信道容量（简化对数模型）
    
    基于香农公式的简化对数衰减模型:
    capacity = C_MAX * max(0, 1 - log(1 + distance/D_REF) / log(1 + D_MAX/D_REF))
    
    特性:
    - distance = 0 时, capacity = C_MAX (最大容量)
    - distance = D_MAX 时, capacity = 0
    - 中间区域呈对数衰减（类似香农公式特性）
    """
    if distance <= 0:
        return float('inf')  # 同一点通信能力无限大
    
    # 超过最大有效距离，容量为0
    if distance >= config.COMM_DISTANCE_MAX:
        return 0.0
    
    # 对数衰减模型
    # 归一化对数衰减因子 (0 到 1 之间)
    attenuation = math.log(1 + distance / config.COMM_DISTANCE_REF) / \
                  math.log(1 + config.COMM_DISTANCE_MAX / config.COMM_DISTANCE_REF)
    
    # 应用路径损耗指数调整曲线形状
    attenuation = attenuation ** (config.COMM_PATH_LOSS_EXPONENT / 2.0)
    
    capacity = config.COMM_CAPACITY_MAX * max(0.0, 1.0 - attenuation)
    return capacity

def calculate_transmission_energy_per_bit(distance):
    """计算每比特的传输能耗（简化模型）
    
    传输能耗与距离成正比，单位: J/bit
    """
    if distance == 0:
        return 1e-9  # 极小值，同位置传输能耗
    # 简化模型：每比特能耗与距离成正比
    return 1e-9 * distance


def calculate_servicing_energy(vnf, communication_partners, locations):
    """计算服务单个VNF所需的总能耗 (论文 Algorithm 1, line 21 / 论文公式 19)"""
    # 1. 悬停能耗
    hover_energy = (config.UAV_BLADE_PROFILE_POWER + config.UAV_INDUCED_POWER) * config.TIME_SLOT_DURATION

    # 2. 计算能耗
    h = config.CHIP_ARCHITECTURE_CONSTANT
    # q_aki (GHz to Hz), Psi_aki (MCCs to cycles)
    cpu_freq_hz = vnf.cpu_freq
    workload_cycles = vnf.workload
    computation_energy = h * (cpu_freq_hz**2) * workload_cycles

    # 3. 接收数据能耗 (假设只要激活就在接收)
    receive_energy = config.COMM_ENERGY_PER_TIMESLOT

    # 4. 发送数据能耗（简化模型）
    transmission_energy = 0.0
    vnf_location_coords = locations[vnf.location_id]
    
    for partner_vnf, demand in communication_partners.items():
        partner_location_coords = locations[partner_vnf.location_id]
        distance = calculate_distance(vnf_location_coords, partner_location_coords)
        energy_per_bit = calculate_transmission_energy_per_bit(distance)
        transmission_energy += demand * energy_per_bit

    total_energy = hover_energy + computation_energy + receive_energy + transmission_energy
    return total_energy
