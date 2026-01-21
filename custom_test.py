# -*- coding: utf-8 -*-
"""
自定义测试脚本
用户可以轻松自定义测试参数，比较 MILP、MPopLoc、RandomOrder 三种算法
"""
import os
import pandas as pd
from test_algorithms import run_batch_tests


def custom_test_cases():
    """
    在这里自定义你的测试案例
    每个测试分类包含若干测试案例，案例参数如下：
    - num_uavs: UAV数量
    - num_requests: SFC请求数量
    - area_size: 区域大小（米）
    - time_limit: MILP求解时间限制（秒）
    
    返回:
        list[dict]: 每个元素包含 name, description, cases
    """
    categories = []

    # 示例1: 测试UAV数量与SFC数量等比例变化的影响
    # 基准比例: 4 架 UAV 对应 40 个请求
    uav_cases = []
    for num_uavs in [4, 6, 8, 10]:
        proportional_requests = int(num_uavs * 8)  # 保持8:1的比例
        uav_cases.append({
            'num_uavs': num_uavs,
            'num_requests': proportional_requests,
            'area_size': 2500.0,
            'time_limit': 60
        })
    categories.append({
        'name': 'vary_uav_sfc',
        'description': 'UAV数量与SFC数量等比例变化',
        'cases': uav_cases
    })

    # 示例2: 测试不同SFC请求数量的影响
    # 固定UAV=8, 区域大小=2500m
    request_cases = []
    for num_requests in range(40, 81, 5):
        request_cases.append({
            'num_uavs': 8,
            'num_requests': num_requests,
            'area_size': 2500.0,
            'time_limit': 180
        })
    categories.append({
        'name': 'vary_requests',
        'description': '固定UAV数量，变化SFC请求数量',
        'cases': request_cases
    })

    # 示例3: 测试不同区域大小的影响
    # 固定UAV=8, 请求数=40
    area_cases = []
    for area_size in range(1000, 4001, 500):
        area_cases.append({
            'num_uavs': 8,
            'num_requests': 40,
            'area_size': area_size,
            'time_limit': 120
        })
    categories.append({
        'name': 'vary_area',
        'description': '固定规模下变化部署区域大小',
        'cases': area_cases
    })
    
    # 你可以添加更多自定义测试案例...
    # categories.append({
    #     'name': 'custom_case',
    #     'description': '自定义说明',
    #     'cases': [{
    #         'num_uavs': 10,
    #         'num_requests': 30,
    #         'area_size': 150.0,
    #         'time_limit': 120
    #     }]
    # })
    
    return categories


if __name__ == "__main__":
    print("="*70)
    print(" "*20 + "自定义测试脚本")
    print("="*70)
    
    # 生成测试案例分类
    categories = custom_test_cases()
    total_cases = sum(len(cat['cases']) for cat in categories)
    print(f"\n已定义 {total_cases} 个测试案例，分属 {len(categories)} 类\n")
    
    # 显示测试案例
    print("测试案例列表:")
    print("-"*70)
    idx = 1
    for category in categories:
        print(f"\n类别: {category['description']} ({category['name']})")
        for case in category['cases']:
            print(f"{idx:2d}. UAVs={case['num_uavs']:2d}, "
                  f"Requests={case['num_requests']:2d}, "
                  f"Area={case['area_size']:6.1f}m, "
                  f"TimeLimit={case['time_limit']:3d}s")
            idx += 1
    print("-"*70)
    
    # 询问用户是否继续
    print("\n提示: 测试结果将保存到 'results' 目录，文件包含三种算法的对比指标")
    print("每个测试案例默认运行1次，可以修改 num_runs_per_case 参数增加运行次数\n")
    
    response = input("是否开始测试? (y/n): ").lower().strip()
    
    if response == 'y' or response == 'yes':
        # 运行测试
        # num_runs_per_case: 每个案例运行次数，增加可以得到更稳定的统计结果
        category_results = []
        for category in categories:
            print("\n" + "#"*70)
            print(f"开始运行类别: {category['description']} ({category['name']})")
            print("#"*70)

            df_results = run_batch_tests(
                category['cases'], 
                output_dir=os.path.join('results', category['name']),
                num_runs_per_case=1  # 修改这里可以改变每个案例的运行次数
            )
            category_results.append((category, df_results))
        
        print("\n" + "="*70)
        print("全部分类测试完成! 结果已保存到各自的 results 子目录")
        print("="*70)

        # 打印分类摘要
        print("\n按分类的三种算法总体表现:")
        for category, df_results in category_results:
            print(f"\n[{category['description']}] -> CSV目录: results/{category['name']}")
            milp_success = df_results['milp_success'].mean()
            mpoploc_success = df_results['mpoploc_success'].mean()
            random_success = df_results['random_success'].mean()
            milp_service = df_results['milp_serviced'].mean()
            mpoploc_service = df_results['mpoploc_serviced'].mean()
            random_service = df_results['random_serviced'].mean()
            milp_runtime = df_results['milp_runtime'].mean()
            mpoploc_runtime = df_results['mpoploc_runtime'].mean()
            random_runtime = df_results['random_runtime'].mean()

            print(f"- MILP: 成功率 {milp_success:.2%}, 平均服务请求 {milp_service:.2f}, 平均用时 {milp_runtime:.2f}s")
            print(f"- MPopLoc: 成功率 {mpoploc_success:.2%}, 平均服务请求 {mpoploc_service:.2f}, 平均用时 {mpoploc_runtime:.2f}s")
            print(f"- RandomOrder: 成功率 {random_success:.2%}, 平均服务请求 {random_service:.2f}, 平均用时 {random_runtime:.2f}s")

        if category_results:
            combined_df = pd.concat([df for _, df in category_results], ignore_index=True)
            print("\n整体平均表现:")
            milp_success = combined_df['milp_success'].mean()
            mpoploc_success = combined_df['mpoploc_success'].mean()
            random_success = combined_df['random_success'].mean()
            milp_service = combined_df['milp_serviced'].mean()
            mpoploc_service = combined_df['mpoploc_serviced'].mean()
            random_service = combined_df['random_serviced'].mean()
            milp_runtime = combined_df['milp_runtime'].mean()
            mpoploc_runtime = combined_df['mpoploc_runtime'].mean()
            random_runtime = combined_df['random_runtime'].mean()

            print(f"- MILP: 成功率 {milp_success:.2%}, 平均服务请求 {milp_service:.2f}, 平均用时 {milp_runtime:.2f}s")
            print(f"- MPopLoc: 成功率 {mpoploc_success:.2%}, 平均服务请求 {mpoploc_service:.2f}, 平均用时 {mpoploc_runtime:.2f}s")
            print(f"- RandomOrder: 成功率 {random_success:.2%}, 平均服务请求 {random_service:.2f}, 平均用时 {random_runtime:.2f}s")

        print("\n详细对比请查看各分类目录下的 CSV 文件。")
    else:
        print("\n测试已取消。")
