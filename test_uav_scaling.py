# -*- coding: utf-8 -*-
"""
UAV数量扩展测试脚本
测试UAV数量从20到80（步长10）时，MPopLoc和Random算法的SFC处理能力
每个算法的结果保存到独立的CSV文件
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


def setup_scenario(num_uavs, num_requests, area_size, num_locations=100, vnfs_per_request=2, seed=None):
    """
    初始化模拟场景
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    base_station = (area_size / 2, area_size / 2)
    locations = {0: base_station}
    for i in range(1, num_locations + 1):
        x = random.uniform(0, area_size)
        y = random.uniform(0, area_size)
        locations[i] = (x, y)

    uavs = [UAV(uav_id=i, location=base_station) for i in range(num_uavs)]

    requests = []
    vnf_global_id = 0
    for i in range(num_requests):
        required_loc_ids = random.sample(range(1, num_locations + 1), 
                                        min(vnfs_per_request, num_locations))
        
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
    
    return uavs, requests, locations


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
        
        return {
            'success': True,
            'total_serviced': total_serviced,
            'runtime': end_time - start_time,
            'status': 'completed'
        }
    except Exception as e:
        end_time = time.time()
        print(f"MPopLoc算法出错: {str(e)}")
        return {
            'success': False,
            'total_serviced': 0,
            'runtime': end_time - start_time,
            'status': f'error: {str(e)}'
        }


def run_random_test(uavs, requests, locations, num_runs=5):
    """运行RandomOrder算法测试（多次运行取平均）"""
    results = []
    
    for run in range(num_runs):
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
        except Exception as e:
            end_time = time.time()
            print(f"RandomOrder算法出错 (run {run+1}): {str(e)}")
            results.append({
                'total_serviced': 0,
                'runtime': end_time - start_time
            })
    
    avg_serviced = sum(r['total_serviced'] for r in results) / len(results)
    avg_runtime = sum(r['runtime'] for r in results) / len(results)
    std_serviced = np.std([r['total_serviced'] for r in results])
    
    return {
        'success': True,
        'total_serviced': avg_serviced,
        'runtime': avg_runtime,
        'status': 'completed',
        'serviced_std': std_serviced,
        'all_runs': [r['total_serviced'] for r in results]
    }


def run_single_test(num_uavs, fixed_params, seed=None):
    """运行单个测试案例"""
    num_requests = fixed_params['num_requests']
    area_size = fixed_params['area_size']
    random_runs = fixed_params.get('random_runs', 5)
    
    print(f"\n{'='*60}")
    print(f"测试参数: UAVs={num_uavs}, Requests={num_requests}, Area={area_size}m")
    print(f"{'='*60}")
    
    original_num_uavs = config.NUM_UAVS
    original_area_size = config.AREA_SIZE
    original_base_station = config.BASE_STATION_LOCATION
    
    config.NUM_UAVS = num_uavs
    config.AREA_SIZE = area_size
    config.BASE_STATION_LOCATION = (area_size / 2, area_size / 2)
    
    try:
        uavs, requests, locations = setup_scenario(
            num_uavs=num_uavs,
            num_requests=num_requests,
            area_size=area_size,
            seed=seed
        )
        
        print(f"场景设置完成: {num_uavs}架UAV, {num_requests}个请求")
        
        # 运行MPopLoc算法
        print("\n--- 运行 MPopLoc 算法 ---")
        mpoploc_result = run_mpoploc_test(uavs, requests, locations)
        print(f"MPopLoc 服务请求数: {mpoploc_result['total_serviced']}")
        
        # 运行Random算法
        print(f"\n--- 运行 RandomOrder 算法 ({random_runs}次取平均) ---")
        random_result = run_random_test(uavs, requests, locations, num_runs=random_runs)
        print(f"RandomOrder 平均服务请求数: {random_result['total_serviced']:.2f} (标准差: {random_result.get('serviced_std', 0):.2f})")
        
        return {
            'num_uavs': num_uavs,
            'num_requests': num_requests,
            'area_size': area_size,
            'seed': seed,
            'mpoploc_serviced': mpoploc_result['total_serviced'],
            'mpoploc_runtime': mpoploc_result['runtime'],
            'random_serviced': random_result['total_serviced'],
            'random_runtime': random_result['runtime'],
            'random_std': random_result.get('serviced_std', 0),
            'random_all_runs': random_result.get('all_runs', [])
        }
        
    finally:
        config.NUM_UAVS = original_num_uavs
        config.AREA_SIZE = original_area_size
        config.BASE_STATION_LOCATION = original_base_station


def save_algorithm_results(results, output_dir, timestamp):
    """
    为每个算法保存独立的CSV文件
    """
    # MPopLoc结果
    mpoploc_data = []
    for r in results:
        mpoploc_data.append({
            'num_uavs': r['num_uavs'],
            'num_requests': r['num_requests'],
            'area_size': r['area_size'],
            'seed': r['seed'],
            'serviced_sfc': r['mpoploc_serviced'],
            'runtime': r['mpoploc_runtime']
        })
    df_mpoploc = pd.DataFrame(mpoploc_data)
    mpoploc_path = os.path.join(output_dir, f'mpoploc_results_{timestamp}.csv')
    df_mpoploc.to_csv(mpoploc_path, index=False, encoding='utf-8-sig')
    print(f"MPopLoc结果已保存: {mpoploc_path}")
    
    # RandomOrder结果
    random_data = []
    for r in results:
        random_data.append({
            'num_uavs': r['num_uavs'],
            'num_requests': r['num_requests'],
            'area_size': r['area_size'],
            'seed': r['seed'],
            'serviced_sfc_avg': r['random_serviced'],
            'serviced_std': r['random_std'],
            'runtime': r['random_runtime'],
            'all_runs': str(r['random_all_runs'])
        })
    df_random = pd.DataFrame(random_data)
    random_path = os.path.join(output_dir, f'random_results_{timestamp}.csv')
    df_random.to_csv(random_path, index=False, encoding='utf-8-sig')
    print(f"RandomOrder结果已保存: {random_path}")
    
    # 对比汇总结果
    summary_data = []
    for r in results:
        summary_data.append({
            'num_uavs': r['num_uavs'],
            'mpoploc_serviced': r['mpoploc_serviced'],
            'random_serviced_avg': r['random_serviced'],
            'random_std': r['random_std'],
            'difference': r['mpoploc_serviced'] - r['random_serviced'],
            'improvement_percent': ((r['mpoploc_serviced'] - r['random_serviced']) / r['random_serviced'] * 100) if r['random_serviced'] > 0 else 0
        })
    df_summary = pd.DataFrame(summary_data)
    summary_path = os.path.join(output_dir, f'comparison_summary_{timestamp}.csv')
    df_summary.to_csv(summary_path, index=False, encoding='utf-8-sig')
    print(f"对比汇总已保存: {summary_path}")
    
    return mpoploc_path, random_path, summary_path


def main():
    """主函数"""
    print("="*70)
    print(" "*20 + "UAV数量扩展测试")
    print("="*70)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    fixed_params = {
        'num_requests': 40,
        'area_size': 1000.0,
        'random_runs': 5
    }
    
    uav_counts = list(range(20, 81, 10))
    
    print(f"\n固定参数:")
    print(f"  - 请求数量: {fixed_params['num_requests']}")
    print(f"  - 区域大小: {fixed_params['area_size']}m x {fixed_params['area_size']}m")
    print(f"  - 每请求VNF数: {config.VNFS_PER_REQUEST}")
    print(f"  - 时间槽数: {config.NUM_TIME_SLOTS}")
    print(f"\n测试UAV数量: {uav_counts}")
    print(f"RandomOrder算法将运行 {fixed_params['random_runs']} 次取平均值\n")
    
    all_results = []
    base_seed = 42
    
    for i, num_uavs in enumerate(uav_counts):
        print(f"\n{'#'*70}")
        print(f"进度: {i+1}/{len(uav_counts)} - UAV数量: {num_uavs}")
        print(f"{'#'*70}")
        
        seed = base_seed + i * 100
        result = run_single_test(num_uavs, fixed_params, seed=seed)
        result['test_id'] = i
        all_results.append(result)
    
    # 创建输出目录
    output_dir = 'results/uav_scaling'
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 保存各算法独立CSV
    save_algorithm_results(all_results, output_dir, timestamp)
    
    # 打印结果表格
    print("\n\n" + "="*70)
    print("测试结果汇总")
    print("="*70)
    print(f"{'UAV数量':<10} {'MPopLoc':<12} {'Random(Avg)':<14} {'提升':<12} {'提升率':<10}")
    print("-"*70)
    for r in all_results:
        print(f"{r['num_uavs']:<10} {r['mpoploc_serviced']:<12.1f} {r['random_serviced']:<14.2f} {r['mpoploc_serviced'] - r['random_serviced']:<12.2f} {((r['mpoploc_serviced'] - r['random_serviced']) / r['random_serviced'] * 100) if r['random_serviced'] > 0 else 0:<10.1f}%")
    print("="*70)
    
    print(f"\n结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    print(f"\n所有结果已保存到: {output_dir}/")
    print(f"运行 'python plot_uav_scaling_results.py' 可生成对比图表")


if __name__ == "__main__":
    main()
