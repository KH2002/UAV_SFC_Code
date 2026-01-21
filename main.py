# -*- coding: utf-8 -*-
"""
模拟程序入口
1. 设置模拟场景 (生成UAV, 位置, 请求)
2. 实例化并运行 MPopLocSolver
3. 打印最终结果
"""
import random
import numpy as np
import config
from entities import UAV, VNF, Request
from mpoploc import MPopLocSolver

def setup_scenario():
    """初始化模拟场景"""
    print("--- 正在设置模拟场景 ---")
    # 1. 生成监控点位置 (ID 0 是基站)
    locations = {0: config.BASE_STATION_LOCATION}
    for i in range(1, config.NUM_LOCATIONS + 1):
        x = random.uniform(0, config.AREA_SIZE)
        y = random.uniform(0, config.AREA_SIZE)
        locations[i] = (x, y)
        print(f"生成了位置 {i}: ({x}, {y})")
    print(f"生成了 {len(locations)} 个位置 (包括基站)。")

    # 2. 初始化 UAV
    uavs = [UAV(uav_id=i, location=config.BASE_STATION_LOCATION) for i in range(config.NUM_UAVS)]
    print(f"初始化了 {len(uavs)} 架 UAV, 初始位置在基站。")

    # 3. 生成请求和 VNF
    requests = []
    vnf_global_id = 0
    for i in range(config.NUM_REQUESTS):
        # 为每个请求随机选择不重复的位置
        required_loc_ids = random.sample(range(1, config.NUM_LOCATIONS + 1), config.VNFS_PER_REQUEST)
        
        vnfs = []
        for loc_id in required_loc_ids:
            cpu_freq = random.uniform(*config.VNF_CPU_FREQUENCY_RANGE)
            workload = random.uniform(*config.VNF_WORKLOAD_RANGE)
            vnfs.append(VNF(vnf_id=vnf_global_id, location_id=loc_id, cpu_freq=cpu_freq, workload=workload))
            vnf_global_id += 1
            
        # 为 VNF 对生成通信需求
        communication_demands = {}
        if len(vnfs) > 1:
            for j in range(len(vnfs)):
                for k in range(j + 1, len(vnfs)):
                    demand = random.uniform(*config.VNF_COMMUNICATION_DEMAND_RANGE)
                    communication_demands[(vnfs[j].id, vnfs[k].id)] = demand

        requests.append(Request(request_id=i, vnfs=vnfs, communication_demands=communication_demands))
    
    print(f"生成了 {len(requests)} 个请求，每个请求包含 {config.VNFS_PER_REQUEST} 个VNF。")
    print("--- 场景设置完毕 ---\n")
    return uavs, requests, locations

def main():
    """主函数"""
    uavs, requests, locations = setup_scenario()
    
    # 实例化求解器并运行
    solver = MPopLocSolver(uavs, requests, locations)
    serviced_requests_timeline = solver.solve()
    
    # 打印最终结果
    print("\n\n========== 最终模拟结果 ==========")
    total_serviced_count = 0
    total_reward = 0
    for t, req_ids in serviced_requests_timeline.items():
        if req_ids:
            print(f"时间槽 {t}: 服务了请求 {req_ids}")
            total_serviced_count += len(req_ids)
            total_reward += len(req_ids) * config.DEFAULT_REQUEST_REWARD
        else:
            print(f"时间槽 {t}: 没有服务新请求")
            
    print("\n------------------------------------")
    print(f"总服务请求数: {total_serviced_count} / {config.NUM_REQUESTS}")
    print(f"总获得奖励: ${total_reward}")
    print("====================================")


if __name__ == "__main__":
    main()
