# -*- coding: utf-8 -*-
"""
测试脚本：对比MILP、MPopLoc、RandomOrder算法
可变参数：无人机数量、SFC数量、区域大小
将测试数据和结果输出到CSV文件
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
from MILP import MILPSolver
from mpoploc import MPopLocSolver
from random_solver import RandomOrderSolver


def setup_scenario(num_uavs, num_requests, area_size, num_locations=10, vnfs_per_request=2, seed=None):
    """
    初始化模拟场景
    
    参数:
        num_uavs: UAV数量
        num_requests: 请求数量
        area_size: 区域大小 (正方形区域的边长)
        num_locations: 监控点数量
        vnfs_per_request: 每个请求的VNF数量
        seed: 随机种子，确保可重复性
    
    返回:
        uavs, requests, locations
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # 1. 生成监控点位置 (ID 0 是基站)
    base_station = (area_size / 2, area_size / 2)
    locations = {0: base_station}
    for i in range(1, num_locations + 1):
        x = random.uniform(0, area_size)
        y = random.uniform(0, area_size)
        locations[i] = (x, y)

    # 2. 初始化 UAV
    uavs = [UAV(uav_id=i, location=base_station) for i in range(num_uavs)]

    # 3. 生成请求和 VNF
    requests = []
    vnf_global_id = 0
    for i in range(num_requests):
        # 为每个请求随机选择不重复的位置
        required_loc_ids = random.sample(range(1, num_locations + 1), 
                                        min(vnfs_per_request, num_locations))
        
        vnfs = []
        for loc_id in required_loc_ids:
            cpu_freq = random.uniform(*config.VNF_CPU_FREQUENCY_RANGE)
            workload = random.uniform(*config.VNF_WORKLOAD_RANGE)
            vnfs.append(VNF(vnf_id=vnf_global_id, location_id=loc_id, 
                          cpu_freq=cpu_freq, workload=workload))
            vnf_global_id += 1
            
        # 为 VNF 对生成通信需求
        communication_demands = {}
        if len(vnfs) > 1:
            for j in range(len(vnfs)):
                for k in range(j + 1, len(vnfs)):
                    demand = random.uniform(*config.VNF_COMMUNICATION_DEMAND_RANGE)
                    communication_demands[(vnfs[j].id, vnfs[k].id)] = demand

        requests.append(Request(request_id=i, vnfs=vnfs, 
                               communication_demands=communication_demands))
    
    return uavs, requests, locations


def run_milp_test(uavs, requests, locations, time_limit=60):
    """
    运行MILP算法测试
    
    返回:
        dict: 包含测试结果的字典
    """
    print("\n=== 运行MILP算法 ===")
    
    # 创建深拷贝，避免修改原始数据
    uavs_copy = copy.deepcopy(uavs)
    requests_copy = copy.deepcopy(requests)
    
    start_time = time.time()
    try:
        solver = MILPSolver(uavs_copy, requests_copy, locations)
        serviced_requests_timeline = solver.solve(time_limit=time_limit)
        end_time = time.time()
        
        if serviced_requests_timeline is None:
            return {
                'success': False,
                'total_serviced': 0,
                'total_reward': 0,
                'runtime': end_time - start_time,
                'status': 'infeasible'
            }
        
        # 统计结果
        total_serviced = sum(len(reqs) for reqs in serviced_requests_timeline.values())
        total_reward = solver.model.ObjVal if hasattr(solver.model, 'ObjVal') else 0
        
        return {
            'success': True,
            'total_serviced': total_serviced,
            'total_reward': total_reward,
            'runtime': end_time - start_time,
            'status': 'optimal',
            'serviced_timeline': serviced_requests_timeline
        }
    except Exception as e:
        end_time = time.time()
        print(f"MILP算法出错: {str(e)}")
        return {
            'success': False,
            'total_serviced': 0,
            'total_reward': 0,
            'runtime': end_time - start_time,
            'status': f'error: {str(e)}'
        }


def run_mpoploc_test(uavs, requests, locations):
    """
    运行MPopLoc算法测试
    
    返回:
        dict: 包含测试结果的字典
    """
    print("\n=== 运行MPopLoc算法 ===")
    
    # 创建深拷贝，避免修改原始数据
    uavs_copy = copy.deepcopy(uavs)
    requests_copy = copy.deepcopy(requests)
    
    start_time = time.time()
    try:
        solver = MPopLocSolver(uavs_copy, requests_copy, locations)
        serviced_requests_timeline = solver.solve()
        end_time = time.time()
        
        # 统计结果
        total_serviced = sum(len(reqs) for reqs in serviced_requests_timeline.values())
        total_reward = total_serviced * config.DEFAULT_REQUEST_REWARD
        
        return {
            'success': True,
            'total_serviced': total_serviced,
            'total_reward': total_reward,
            'runtime': end_time - start_time,
            'status': 'completed',
            'serviced_timeline': serviced_requests_timeline
        }
    except Exception as e:
        end_time = time.time()
        print(f"MPopLoc算法出错: {str(e)}")
        return {
            'success': False,
            'total_serviced': 0,
            'total_reward': 0,
            'runtime': end_time - start_time,
            'status': f'error: {str(e)}'
        }


def run_random_test(uavs, requests, locations):
    """
    运行RandomOrder算法测试
    
    返回:
        dict: 包含测试结果的字典
    """
    print("\n=== 运行RandomOrder算法 ===")
    
    uavs_copy = copy.deepcopy(uavs)
    requests_copy = copy.deepcopy(requests)

    start_time = time.time()
    try:
        solver = RandomOrderSolver(uavs_copy, requests_copy, locations)
        serviced_requests_timeline = solver.solve()
        end_time = time.time()

        total_serviced = sum(len(reqs) for reqs in serviced_requests_timeline.values())
        total_reward = total_serviced * config.DEFAULT_REQUEST_REWARD

        return {
            'success': True,
            'total_serviced': total_serviced,
            'total_reward': total_reward,
            'runtime': end_time - start_time,
            'status': 'completed',
            'serviced_timeline': serviced_requests_timeline
        }
    except Exception as e:
        end_time = time.time()
        print(f"RandomOrder算法出错: {str(e)}")
        return {
            'success': False,
            'total_serviced': 0,
            'total_reward': 0,
            'runtime': end_time - start_time,
            'status': f'error: {str(e)}'
        }


def run_single_test(test_params, seed=None):
    """
    运行单个测试案例
    
    参数:
        test_params: dict, 包含测试参数
            - num_uavs: UAV数量
            - num_requests: 请求数量
            - area_size: 区域大小
            - time_limit: MILP时间限制
        seed: 随机种子
    
    返回:
        dict: 包含测试数据和结果的字典
    """
    num_uavs = test_params['num_uavs']
    num_requests = test_params['num_requests']
    area_size = test_params['area_size']
    time_limit = test_params.get('time_limit', 60)
    
    print(f"\n{'='*60}")
    print(f"测试参数: UAVs={num_uavs}, Requests={num_requests}, Area={area_size}m")
    print(f"{'='*60}")
    
    # 临时修改config中的参数
    original_num_uavs = config.NUM_UAVS
    original_area_size = config.AREA_SIZE
    original_base_station = config.BASE_STATION_LOCATION
    
    config.NUM_UAVS = num_uavs
    config.AREA_SIZE = area_size
    config.BASE_STATION_LOCATION = (area_size / 2, area_size / 2)
    
    try:
        # 设置场景
        uavs, requests, locations = setup_scenario(
            num_uavs=num_uavs,
            num_requests=num_requests,
            area_size=area_size,
            seed=seed
        )
        
        # 运行三个算法
        milp_result = run_milp_test(uavs, requests, locations, time_limit=time_limit)
        mpoploc_result = run_mpoploc_test(uavs, requests, locations)
        random_result = run_random_test(uavs, requests, locations)
        
        # 组织结果
        result = {
            # 测试参数
            'num_uavs': num_uavs,
            'num_requests': num_requests,
            'area_size': area_size,
            'num_locations': len(locations) - 1,  # 不包括基站
            'vnfs_per_request': config.VNFS_PER_REQUEST,
            'time_slots': config.NUM_TIME_SLOTS,
            'seed': seed,
            
            # MILP结果
            'milp_success': milp_result['success'],
            'milp_serviced': milp_result['total_serviced'],
            'milp_reward': milp_result['total_reward'],
            'milp_runtime': milp_result['runtime'],
            'milp_status': milp_result['status'],
            
            # MPopLoc结果
            'mpoploc_success': mpoploc_result['success'],
            'mpoploc_serviced': mpoploc_result['total_serviced'],
            'mpoploc_reward': mpoploc_result['total_reward'],
            'mpoploc_runtime': mpoploc_result['runtime'],
            'mpoploc_status': mpoploc_result['status'],

            # RandomOrder结果
            'random_success': random_result['success'],
            'random_serviced': random_result['total_serviced'],
            'random_reward': random_result['total_reward'],
            'random_runtime': random_result['runtime'],
            'random_status': random_result['status'],
            
            # 性能比较
            'serviced_diff': milp_result['total_serviced'] - mpoploc_result['total_serviced'],
            'reward_diff': milp_result['total_reward'] - mpoploc_result['total_reward'],
            'runtime_ratio': mpoploc_result['runtime'] / milp_result['runtime'] if milp_result['runtime'] > 0 else 0,
            'milp_vs_random_serviced_diff': milp_result['total_serviced'] - random_result['total_serviced'],
            'mpoploc_vs_random_serviced_diff': mpoploc_result['total_serviced'] - random_result['total_serviced'],
            'milp_vs_random_reward_diff': milp_result['total_reward'] - random_result['total_reward'],
            'mpoploc_vs_random_reward_diff': mpoploc_result['total_reward'] - random_result['total_reward'],
            'random_vs_milp_runtime_ratio': random_result['runtime'] / milp_result['runtime'] if milp_result['runtime'] > 0 else 0,
            'random_vs_mpoploc_runtime_ratio': random_result['runtime'] / mpoploc_result['runtime'] if mpoploc_result['runtime'] > 0 else 0,
        }
        
        return result
        
    finally:
        # 恢复原始config值
        config.NUM_UAVS = original_num_uavs
        config.AREA_SIZE = original_area_size
        config.BASE_STATION_LOCATION = original_base_station


def run_batch_tests(test_cases, output_dir='results', num_runs_per_case=1):
    """
    批量运行测试案例
    
    参数:
        test_cases: list of dict, 每个dict包含测试参数
        output_dir: 结果输出目录
        num_runs_per_case: 每个测试案例运行的次数
    
    返回:
        pd.DataFrame: 测试结果
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成时间戳
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 存储所有结果
    all_results = []
    
    # 运行测试
    total_tests = len(test_cases) * num_runs_per_case
    current_test = 0
    
    for i, test_params in enumerate(test_cases):
        for run in range(num_runs_per_case):
            current_test += 1
            print(f"\n\n{'#'*60}")
            print(f"进度: {current_test}/{total_tests}")
            print(f"测试案例 {i+1}/{len(test_cases)}, 运行 {run+1}/{num_runs_per_case}")
            print(f"{'#'*60}")
            
            # 使用不同的种子确保多次运行的随机性
            seed = i * 1000 + run if num_runs_per_case > 1 else i * 1000
            
            result = run_single_test(test_params, seed=seed)
            result['test_case_id'] = i
            result['run_id'] = run
            all_results.append(result)
    
    # 转换为DataFrame
    df_results = pd.DataFrame(all_results)
    
    # 保存详细结果
    detailed_file = os.path.join(output_dir, f'detailed_results_{timestamp}.csv')
    df_results.to_csv(detailed_file, index=False, encoding='utf-8-sig')
    print(f"\n详细结果已保存到: {detailed_file}")
    
    # 如果有多次运行，计算统计结果
    if num_runs_per_case > 1:
        summary_results = []
        for i in range(len(test_cases)):
            case_data = df_results[df_results['test_case_id'] == i]
            summary = {
                'test_case_id': i,
                'num_uavs': case_data['num_uavs'].iloc[0],
                'num_requests': case_data['num_requests'].iloc[0],
                'area_size': case_data['area_size'].iloc[0],
                
                # MILP统计
                'milp_serviced_mean': case_data['milp_serviced'].mean(),
                'milp_serviced_std': case_data['milp_serviced'].std(),
                'milp_reward_mean': case_data['milp_reward'].mean(),
                'milp_reward_std': case_data['milp_reward'].std(),
                'milp_runtime_mean': case_data['milp_runtime'].mean(),
                'milp_runtime_std': case_data['milp_runtime'].std(),
                'milp_success_rate': case_data['milp_success'].mean(),
                
                # MPopLoc统计
                'mpoploc_serviced_mean': case_data['mpoploc_serviced'].mean(),
                'mpoploc_serviced_std': case_data['mpoploc_serviced'].std(),
                'mpoploc_reward_mean': case_data['mpoploc_reward'].mean(),
                'mpoploc_reward_std': case_data['mpoploc_reward'].std(),
                'mpoploc_runtime_mean': case_data['mpoploc_runtime'].mean(),
                'mpoploc_runtime_std': case_data['mpoploc_runtime'].std(),
                'mpoploc_success_rate': case_data['mpoploc_success'].mean(),
                
                # RandomOrder统计
                'random_serviced_mean': case_data['random_serviced'].mean(),
                'random_serviced_std': case_data['random_serviced'].std(),
                'random_reward_mean': case_data['random_reward'].mean(),
                'random_reward_std': case_data['random_reward'].std(),
                'random_runtime_mean': case_data['random_runtime'].mean(),
                'random_runtime_std': case_data['random_runtime'].std(),
                'random_success_rate': case_data['random_success'].mean(),
                
                # 性能比较统计
                'serviced_diff_mean': case_data['serviced_diff'].mean(),
                'reward_diff_mean': case_data['reward_diff'].mean(),
                'runtime_ratio_mean': case_data['runtime_ratio'].mean(),
                'milp_vs_random_serviced_diff_mean': case_data['milp_vs_random_serviced_diff'].mean(),
                'mpoploc_vs_random_serviced_diff_mean': case_data['mpoploc_vs_random_serviced_diff'].mean(),
                'milp_vs_random_reward_diff_mean': case_data['milp_vs_random_reward_diff'].mean(),
                'mpoploc_vs_random_reward_diff_mean': case_data['mpoploc_vs_random_reward_diff'].mean(),
                'random_vs_milp_runtime_ratio_mean': case_data['random_vs_milp_runtime_ratio'].mean(),
                'random_vs_mpoploc_runtime_ratio_mean': case_data['random_vs_mpoploc_runtime_ratio'].mean(),
            }
            summary_results.append(summary)
        
        df_summary = pd.DataFrame(summary_results)
        summary_file = os.path.join(output_dir, f'summary_results_{timestamp}.csv')
        df_summary.to_csv(summary_file, index=False, encoding='utf-8-sig')
        print(f"统计摘要已保存到: {summary_file}")
    
    return df_results


def generate_test_cases():
    """
    生成测试案例
    可以根据需要修改这个函数来定义测试参数
    
    返回:
        list of dict: 测试案例列表
    """
    test_cases = []
    
    # 测试1: 变化UAV数量 (固定请求数=20, 区域=100)
    for num_uavs in [2, 4, 6, 8, 10]:
        test_cases.append({
            'num_uavs': num_uavs,
            'num_requests': 20,
            'area_size': 100.0,
            'time_limit': 60
        })
    
    # 测试2: 变化请求数量 (固定UAV=4, 区域=100)
    for num_requests in [10, 15, 20, 25, 30]:
        test_cases.append({
            'num_uavs': 4,
            'num_requests': num_requests,
            'area_size': 100.0,
            'time_limit': 60
        })
    
    # 测试3: 变化区域大小 (固定UAV=4, 请求=20)
    for area_size in [50.0, 75.0, 100.0, 125.0, 150.0]:
        test_cases.append({
            'num_uavs': 4,
            'num_requests': 20,
            'area_size': area_size,
            'time_limit': 60
        })
    
    return test_cases


def main():
    """
    主函数
    """
    print("="*60)
    print("UAV-SFC算法测试脚本")
    print("="*60)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 生成测试案例
    test_cases = generate_test_cases()
    print(f"\n总共生成 {len(test_cases)} 个测试案例")
    
    # 运行批量测试
    # num_runs_per_case: 每个测试案例运行的次数，可以增加以获得更稳定的统计结果
    df_results = run_batch_tests(test_cases, output_dir='results', num_runs_per_case=1)
    
    print("\n"+"="*60)
    print("测试完成!")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    # 打印简要统计
    print("\n简要统计:")
    print(f"- MILP成功率: {df_results['milp_success'].mean():.2%}")
    print(f"- MPopLoc成功率: {df_results['mpoploc_success'].mean():.2%}")
    print(f"- RandomOrder成功率: {df_results['random_success'].mean():.2%}")
    print(f"- MILP平均服务请求数: {df_results['milp_serviced'].mean():.2f}")
    print(f"- MPopLoc平均服务请求数: {df_results['mpoploc_serviced'].mean():.2f}")
    print(f"- RandomOrder平均服务请求数: {df_results['random_serviced'].mean():.2f}")
    print(f"- MILP平均运行时间: {df_results['milp_runtime'].mean():.2f}秒")
    print(f"- MPopLoc平均运行时间: {df_results['mpoploc_runtime'].mean():.2f}秒")
    print(f"- RandomOrder平均运行时间: {df_results['random_runtime'].mean():.2f}秒")


if __name__ == "__main__":
    main()
