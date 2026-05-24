# -*- coding: utf-8 -*-
"""
UAV数量扩展测试脚本
确保只有UAV数量变化，其他所有条件（监控点、请求、随机种子）完全相同
所有参数遵循config.py中的配置
"""
import random
import numpy as np
import pandas as pd
import time
import copy
from datetime import datetime
import os
import config
from entities import UAV, VNF, Request
from mpoploc import MPopLocSolver
from random_solver import RandomOrderSolver


def setup_scenario(num_uavs, locations=None, requests_data=None):
    """
    初始化模拟场景
    
    参数:
        num_uavs: UAV数量（唯一变量）
        locations: 如果提供，直接使用这些监控点位置（确保位置固定）
        requests_data: 如果提供，直接使用这些请求数据（确保请求固定）
    """
    base_station = config.BASE_STATION_LOCATION
    
    # 使用固定的locations或从config生成
    if locations is not None:
        locations_dict = locations
    else:
        locations_dict = {0: base_station}
        for i in range(1, config.NUM_LOCATIONS + 1):
            # 使用固定种子确保可重复性，但不在这里设置，由调用者控制
            x = random.uniform(0, config.AREA_SIZE)
            y = random.uniform(0, config.AREA_SIZE)
            locations_dict[i] = (x, y)
    
    # 生成UAV（这是唯一变化的）
    uavs = [UAV(uav_id=i, location=base_station) for i in range(num_uavs)]
    
    # 使用固定的requests或从config生成
    if requests_data is not None:
        requests = requests_data
    else:
        requests = []
        vnf_global_id = 0
        for i in range(config.NUM_REQUESTS):
            required_loc_ids = random.sample(range(1, config.NUM_LOCATIONS + 1), 
                                            min(config.VNFS_PER_REQUEST, config.NUM_LOCATIONS))
            
            vnfs = []
            for loc_id in required_loc_ids:
                cpu_freq = random.uniform(*config.VNF_CPU_FREQUENCY_RANGE)
                workload = random.uniform(*config.VNF_WORKLOAD_RANGE)
                vnfs.append(VNF(vnf_id=vnf_global_id, location_id=loc_id, 
                              cpu_freq=cpu_freq, workload=workload))
                vnf_global_id += 1
                
            communication_demands = {}
            if len(vnfs) > 1:
                for j in range(len(vnfs)):
                    for k in range(j + 1, len(vnfs)):
                        demand = random.uniform(*config.VNF_COMMUNICATION_DEMAND_RANGE)
                        communication_demands[(vnfs[j].id, vnfs[k].id)] = demand

            requests.append(Request(request_id=i, vnfs=vnfs, 
                                   communication_demands=communication_demands))
    
    return uavs, requests, locations_dict


def generate_fixed_scenario(seed=42):
    """
    生成固定的场景（监控点和请求），在所有UAV数量测试中复用
    所有参数来自config.py
    """
    random.seed(seed)
    np.random.seed(seed)
    
    base_station = config.BASE_STATION_LOCATION
    locations = {0: base_station}
    
    # 生成监控点位置
    for i in range(1, config.NUM_LOCATIONS + 1):
        x = random.uniform(0, config.AREA_SIZE)
        y = random.uniform(0, config.AREA_SIZE)
        locations[i] = (x, y)
    
    # 生成请求
    requests = []
    vnf_global_id = 0
    for i in range(config.NUM_REQUESTS):
        required_loc_ids = random.sample(range(1, config.NUM_LOCATIONS + 1), 
                                        min(config.VNFS_PER_REQUEST, config.NUM_LOCATIONS))
        
        vnfs = []
        for loc_id in required_loc_ids:
            cpu_freq = random.uniform(*config.VNF_CPU_FREQUENCY_RANGE)
            workload = random.uniform(*config.VNF_WORKLOAD_RANGE)
            vnfs.append(VNF(vnf_id=vnf_global_id, location_id=loc_id, 
                          cpu_freq=cpu_freq, workload=workload))
            vnf_global_id += 1
            
        communication_demands = {}
        if len(vnfs) > 1:
            for j in range(len(vnfs)):
                for k in range(j + 1, len(vnfs)):
                    demand = random.uniform(*config.VNF_COMMUNICATION_DEMAND_RANGE)
                    communication_demands[(vnfs[j].id, vnfs[k].id)] = demand

        requests.append(Request(request_id=i, vnfs=vnfs, 
                               communication_demands=communication_demands))
    
    return locations, requests


def run_mpoploc_test(uavs, requests, locations):
    """运行MPopLoc算法测试"""
    uavs_copy = copy.deepcopy(uavs)
    requests_copy = copy.deepcopy(requests)
    
    start_time = time.time()
    try:
        solver = MPopLocSolver(uavs_copy, requests_copy, locations)
        serviced_requests_timeline = solver.solve()
        end_time = time.time()
        
        total_serviced = sum(len(reqs) for reqs in serviced_requests_timeline.values())
        
        # 打印每个时间槽的详情
        print(f"    MPopLoc 时间槽详情: ", end="")
        for t, reqs in sorted(serviced_requests_timeline.items()):
            print(f"[T{t}:{len(reqs)}]", end=" ")
        print()
        
        return {
            'success': True,
            'total_serviced': total_serviced,
            'runtime': end_time - start_time,
            'timeline': serviced_requests_timeline
        }
    except Exception as e:
        end_time = time.time()
        print(f"    MPopLoc算法出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'total_serviced': 0,
            'runtime': end_time - start_time,
            'timeline': {}
        }


def run_random_test(uavs, requests, locations, num_runs=5):
    """运行RandomOrder算法测试（多次运行取平均）"""
    results = []
    
    print(f"    RandomOrder 运行 {num_runs} 次...", end=" ")
    
    for run in range(num_runs):
        # 每次运行使用不同的随机种子（但场景已经固定）
        random.seed(1000 + run)
        
        uavs_copy = copy.deepcopy(uavs)
        requests_copy = copy.deepcopy(requests)

        start_time = time.time()
        try:
            solver = RandomOrderSolver(uavs_copy, requests_copy, locations)
            serviced_requests_timeline = solver.solve()
            end_time = time.time()

            total_serviced = sum(len(reqs) for reqs in serviced_requests_timeline.values())

            results.append({
                'total_serviced': total_serviced,
                'runtime': end_time - start_time
            })
            print(f"[{run+1}:{total_serviced}]", end="")
        except Exception as e:
            end_time = time.time()
            print(f"[E]", end="")
            results.append({
                'total_serviced': 0,
                'runtime': end_time - start_time
            })
    
    print()  # 换行
    
    avg_serviced = sum(r['total_serviced'] for r in results) / len(results)
    avg_runtime = sum(r['runtime'] for r in results) / len(results)
    std_serviced = np.std([r['total_serviced'] for r in results])
    
    print(f"    RandomOrder 统计: 平均={avg_serviced:.2f}, 标准差={std_serviced:.2f}")
    
    return {
        'success': True,
        'total_serviced': avg_serviced,
        'runtime': avg_runtime,
        'serviced_std': std_serviced,
        'all_results': [r['total_serviced'] for r in results]
    }


def run_single_test(num_uavs, fixed_locations, fixed_requests, random_runs=5):
    """
    运行单个测试案例
    使用固定的locations和requests，只有UAV数量变化
    所有其他参数来自config.py
    """
    print(f"\n{'='*70}")
    print(f"测试: UAVs={num_uavs:2d}, Requests={config.NUM_REQUESTS}, "
          f"Area={config.AREA_SIZE}m, TimeSlots={config.NUM_TIME_SLOTS}")
    print(f"{'='*70}")
    
    # 保存原始UAV数量
    original_num_uavs = config.NUM_UAVS
    
    try:
        # 临时修改UAV数量（唯一变量）
        config.NUM_UAVS = num_uavs
        
        # 使用固定的场景，只改变UAV数量
        uavs, requests, locations = setup_scenario(
            num_uavs=num_uavs,
            locations=fixed_locations,      # 使用固定的监控点
            requests_data=fixed_requests    # 使用固定的请求
        )
        
        print(f"场景: {num_uavs}架UAV, {len(requests)}个请求, {len(locations)-1}个监控点")
        print(f"通信需求范围: {config.VNF_COMMUNICATION_DEMAND_RANGE}")
        print(f"注意: 使用固定场景（监控点位置、请求位置和属性完全相同）")
        
        # 运行MPopLoc
        print(f"  MPopLoc...")
        mpoploc_result = run_mpoploc_test(uavs, requests, locations)
        
        # 运行Random
        random_result = run_random_test(uavs, requests, locations, num_runs=random_runs)
        
        print(f"  结果对比: MPopLoc={mpoploc_result['total_serviced']}, "
              f"Random={random_result['total_serviced']:.2f}")
        
        return {
            'num_uavs': num_uavs,
            'num_requests': config.NUM_REQUESTS,
            'area_size': config.AREA_SIZE,
            'time_slots': config.NUM_TIME_SLOTS,
            'mpoploc_serviced': mpoploc_result['total_serviced'],
            'mpoploc_runtime': mpoploc_result['runtime'],
            'random_serviced': random_result['total_serviced'],
            'random_runtime': random_result['runtime'],
            'random_std': random_result.get('serviced_std', 0),
            'random_all': random_result.get('all_results', []),
        }
        
    finally:
        # 恢复原始UAV数量
        config.NUM_UAVS = original_num_uavs


def save_results(all_results, output_dir, timestamp):
    """保存结果到CSV"""
    # MPopLoc结果
    mpoploc_data = [{
        'num_uavs': r['num_uavs'],
        'num_requests': r['num_requests'],
        'area_size': r['area_size'],
        'time_slots': r['time_slots'],
        'serviced_sfc': r['mpoploc_serviced'],
        'runtime': r['mpoploc_runtime']
    } for r in all_results]
    pd.DataFrame(mpoploc_data).to_csv(
        os.path.join(output_dir, f'mpoploc_results.csv'), 
        index=False, encoding='utf-8-sig'
    )
    
    # Random结果
    random_data = [{
        'num_uavs': r['num_uavs'],
        'num_requests': r['num_requests'],
        'area_size': r['area_size'],
        'time_slots': r['time_slots'],
        'serviced_sfc_avg': r['random_serviced'],
        'serviced_std': r['random_std'],
        'runtime': r['random_runtime'],
        'all_runs': str(r['random_all'])
    } for r in all_results]
    pd.DataFrame(random_data).to_csv(
        os.path.join(output_dir, f'random_results.csv'), 
        index=False, encoding='utf-8-sig'
    )


def print_config_summary():
    """打印config参数摘要"""
    print("\n" + "="*70)
    print("当前配置 (来自config.py):")
    print("="*70)
    print(f"场景参数:")
    print(f"  区域大小: {config.AREA_SIZE}m x {config.AREA_SIZE}m")
    print(f"  监控点数量: {config.NUM_LOCATIONS}")
    print(f"  基站位置: {config.BASE_STATION_LOCATION}")
    print(f"\n时间参数:")
    print(f"  时间槽数: {config.NUM_TIME_SLOTS}")
    print(f"  时间槽时长: {config.TIME_SLOT_DURATION}s")
    print(f"\nUAV参数:")
    print(f"  UAV数量(默认): {config.NUM_UAVS}")
    print(f"  UAV计算能力: {config.UAV_COMPUTATION_CAPACITY} MHz")
    print(f"  UAV电池容量: {config.UAV_BATTERY_CAPACITY/1000:.0f} KJ")
    print(f"\nVNF参数:")
    print(f"  请求数量: {config.NUM_REQUESTS}")
    print(f"  每请求VNF数: {config.VNFS_PER_REQUEST}")
    print(f"  CPU频率范围: {config.VNF_CPU_FREQUENCY_RANGE} GHz")
    print(f"  工作量范围: {config.VNF_WORKLOAD_RANGE} MCCs")
    print(f"  通信需求范围: {config.VNF_COMMUNICATION_DEMAND_RANGE} bits")
    print(f"\n通信模型:")
    print(f"  最大容量: {config.COMM_CAPACITY_MAX/1e6:.1f} Mbps")
    print(f"  最大有效距离: {config.COMM_DISTANCE_MAX}m")
    print(f"  参考距离: {config.COMM_DISTANCE_REF}m")
    print("="*70)


def main():
    print("="*70)
    print("UAV数量扩展测试 - 控制变量版")
    print("="*70)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 打印当前配置
    print_config_summary()
    
    # UAV数量变化范围
    uav_counts = list(range(20, 81, 10))  # 20, 30, 40, 50, 60, 70, 80
    random_runs = 5  # Random算法运行次数
    
    print(f"\n测试设计:")
    print(f"  变化参数: UAV数量 {uav_counts}")
    print(f"  控制变量: 监控点位置、请求位置和属性完全相同")
    print(f"  Random算法运行: {random_runs}次取平均")
    print(f"\n重要: 所有测试使用完全相同的场景！")
    print(f"      只有UAV数量变化\n")
    
    # 先生成固定场景（只生成一次）
    print("生成固定场景...")
    fixed_locations, fixed_requests = generate_fixed_scenario(seed=42)
    print(f"  已生成 {len(fixed_locations)-1} 个监控点")
    print(f"  已生成 {len(fixed_requests)} 个请求")
    print(f"  场景将在所有UAV数量测试中复用\n")
    
    all_results = []
    
    for i, num_uavs in enumerate(uav_counts):
        result = run_single_test(num_uavs, fixed_locations, fixed_requests, 
                                random_runs=random_runs)
        result['test_id'] = i
        all_results.append(result)
    
    # 保存结果
    output_dir = 'results/uav_scaling'
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_results(all_results, output_dir, timestamp)
    
    # 打印汇总表格
    print("\n\n" + "="*80)
    print("测试结果汇总")
    print("="*80)
    print(f"{'UAV数':<8} {'MPopLoc':<10} {'Random':<12} {'Std':<8} {'提升':<10} {'提升率':<8}")
    print("-"*80)
    for r in all_results:
        diff = r['mpoploc_serviced'] - r['random_serviced']
        pct = (diff / r['random_serviced'] * 100) if r['random_serviced'] > 0 else 0
        print(f"{r['num_uavs']:<8} {r['mpoploc_serviced']:<10.0f} {r['random_serviced']:<12.2f} "
              f"{r['random_std']:<8.2f} {diff:<10.2f} {pct:<8.1f}%")
    print("="*80)
    
    print(f"\n结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"结果保存于: {output_dir}/")


if __name__ == "__main__":
    main()
