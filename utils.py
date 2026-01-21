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
    """根据距离计算两点间的信道容量 (论文公式 8)"""
    if distance == 0:
        return float('inf') # 假设在同一点的两个UAV通信能力无限大

    # 路径损耗 (dB) (论文公式 7)
    path_loss_db = 10 * config.PATH_LOSS_EXPONENT * np.log10(distance)
    # 频道增益
    channel_gain = 10**(-path_loss_db / 10)

    # 香农-哈特利公式
    snr = (config.UAV_TRANSMIT_POWER * channel_gain) / config.NOISE_POWER
    capacity = config.CHANNEL_BANDWIDTH * np.log2(1 + snr)
    return capacity

def calculate_transmission_power_per_bit(channel_gain):
    """计算每比特的平均传输功率 (论文公式 10)"""
    eta = config.TRANSCEIVER_CIRCUIT_POWER
    epsilon = config.RECEIVER_SENSITIVITY
    gamma = config.POWER_AMPLIFIER_DRAIN_EFFICIENCY
    
    # rho(i,j)
    power = eta + (10**((epsilon - 3) / 10) * channel_gain) / gamma
    return power


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
    receive_energy = config.TRANSCEIVER_CIRCUIT_POWER * config.TIME_SLOT_DURATION

    # 4. 发送数据能耗
    transmission_energy = 0.0
    vnf_location_coords = locations[vnf.location_id]
    
    for partner_vnf, demand in communication_partners.items():
        partner_location_coords = locations[partner_vnf.location_id]
        distance = calculate_distance(vnf_location_coords, partner_location_coords)
        if distance > 0:
            path_loss_db = 10 * config.PATH_LOSS_EXPONENT * np.log10(distance)
            channel_gain = 10**(-path_loss_db / 10)
            power_per_bit = calculate_transmission_power_per_bit(channel_gain)
            transmission_energy += config.TIME_SLOT_DURATION * power_per_bit * demand

    total_energy = hover_energy + computation_energy + receive_energy + transmission_energy
    return total_energy
