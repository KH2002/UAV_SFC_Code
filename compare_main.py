# -*- coding: utf-8 -*-
"""
模拟主程序：
1. 初始化场景 (UAVs, VNF请求, 地理位置)
2. 运行 MPopLoc 启发式算法
3. (可选) 运行 MILP 精确/基准算法
4. 打印并对比结果
"""
import random
import copy
import config
from entities import UAV, VNF, Request
from mpoploc import MPopLocSolver
from MILP import MILPSolver
from gurobipy import GRB

# --- 开关 ---
RUN_MILP_SOLVER = True # 设置为 True 来运行MILP求解器作为对比

def setup_simulation():
    """创建模拟所需的UAV、位置和请求"""
    print("--- 初始化模拟场景 ---")
    
    # 1. 创建位置
    locations = {0: config.BASE_STATION_LOCATION} # location_id: (x, y)
    for i in range(1, config.NUM_LOCATIONS + 1):
        x = random.uniform(0, config.AREA_SIZE)
        y = random.uniform(0, config.AREA_SIZE)
        locations[i] = (x, y)

    # 2. 创建UAV
    uavs = [UAV(uav_id=i, location=config.BASE_STATION_LOCATION) for i in range(1, config.NUM_UAVS + 1)]

    # 3. 创建请求
    requests = []
    vnf_global_id = 1
    for i in range(1, config.NUM_REQUESTS + 1):
        num_vnfs_in_req = config.VNFS_PER_REQUEST
        
        # 确保VNF位置不重复
        required_loc_ids = random.sample(list(locations.keys())[1:], num_vnfs_in_req)
        
        vnfs = []
        for j in range(num_vnfs_in_req):
            vnf = VNF(
                vnf_id=vnf_global_id,
                location_id=required_loc_ids[j],
                cpu_freq=random.uniform(0.1, 0.5), # GHz
                workload=random.uniform(100, 500) # MCCs
            )
            vnfs.append(vnf)
            vnf_global_id += 1
        
        # 创建VNF间的通信需求 (简化为链式)
        communication_demands = {}
        for k in range(len(vnfs) - 1):
            demand = random.uniform(1e5, 5e5) # bits
            communication_demands[(vnfs[k].id, vnfs[k+1].id)] = demand

        req = Request(request_id=i, vnfs=vnfs, communication_demands=communication_demands)
        requests.append(req)

    print(f"创建了 {len(locations)} 个位置, {len(uavs)} 架UAV, {len(requests)} 个请求。")
    return uavs, requests, locations

def print_results(solver_name, timeline):
    """格式化打印结果"""
    print(f"\n--- {solver_name} 结果 ---")
    if not timeline:
        print("没有服务任何请求。")
        return
        
    total_serviced = 0
    for t, req_ids in timeline.items():
        if req_ids:
            print(f"时间槽 {t}: 服务了请求 {req_ids}")
            total_serviced += len(req_ids)
    
    print(f"\n{solver_name} 总计服务了 {total_serviced} 个请求。")

if __name__ == "__main__":
    # 初始化
    uavs, requests, locations = setup_simulation()

    # 为每个求解器创建独立的实例副本，防止交叉影响
    mpoploc_uavs = copy.deepcopy(uavs)
    mpoploc_requests = copy.deepcopy(requests)
    
    # 运行 MPopLoc
    mpoploc_solver = MPopLocSolver(mpoploc_uavs, mpoploc_requests, locations)
    mpoploc_timeline = mpoploc_solver.solve()
    print_results("MPopLoc", mpoploc_timeline)

    # 运行 MILP
    if RUN_MILP_SOLVER:
        try:
            milp_uavs = copy.deepcopy(uavs)
            milp_requests = copy.deepcopy(requests)
            
            milp_solver = MILPSolver(milp_uavs, milp_requests, locations)
            milp_timeline = milp_solver.solve(time_limit=600) # 可调整求解时间
        
        except ImportError:
            print("\n警告: 未找到 'gurobipy' 库。跳过 MILP 求解器。")
            print("请安装 Gurobi 并配置 Python 环境以运行 MILP 对比。")
        except Exception as e:
            print(f"\n运行 MILP 求解器时出错: {e}")

