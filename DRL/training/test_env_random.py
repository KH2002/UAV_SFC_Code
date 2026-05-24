#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DRL环境测试脚本 - 随机策略

用于测试环境是否正确运行，随机选择SFC和UAV分配。
输出每个时隙所有UAV的动作、充电情况以及每个SFC的详细信息。

使用方法:
    cd /mnt/sdb11/HK/UAV_SFC_code/DRL/training
    python test_env_random.py
    
可选参数:
    --num_locations: 监控点数量 (默认100)
    --num_uavs: UAV数量 (默认4)
    --num_requests: 请求数量 (默认12)
    --area_size: 区域大小 (默认1000)
    --seed: 随机种子 (默认42)
    --output: 输出日志文件路径
    --force: 强制重新生成数据集（修改位置数量时需要）
    
注意:
    修改 --num_locations 后需要使用 --force 参数重新生成数据集
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import argparse
import numpy as np
import random
from typing import Dict, List, Tuple
from datetime import datetime

from DRL.env import UAVSFCEnv
from DRL.training.dataset import create_or_load_dataset, generate_locations, generate_uavs, generate_requests, TrainingDataset
import config


# 设置输出重定向到文件和控制台
class TeeOutput:
    """同时将输出写入文件和控制台"""
    def __init__(self, filename):
        self.console = sys.stdout
        self.logfile = open(filename, 'w', encoding='utf-8')
        
    def write(self, message):
        self.console.write(message)
        self.logfile.write(message)
        self.logfile.flush()
        
    def flush(self):
        self.console.flush()
        self.logfile.flush()


class RandomPolicy:
    """随机策略：随机选择有效的请求和UAV"""
    
    def __init__(self, env: UAVSFCEnv):
        self.env = env
    
    def select_action(self, observation: Dict, mask: Dict) -> Dict:
        """
        随机选择动作，确保两个VNF选择不同的UAV
        """
        # 获取有效的请求掩码
        request_mask = mask['request']  # [max_pending]
        valid_requests = np.where(request_mask == 1)[0]
        
        if len(valid_requests) == 0:
            return {
                'request_idx': 0,
                'uav_for_vnf1': 0,
                'uav_for_vnf2': 0
            }
        
        # 随机选择一个有效请求
        request_idx = random.choice(valid_requests)
        
        # 获取有效的UAV掩码
        uav_mask = mask['uav']  # [num_uavs]
        valid_uavs = np.where(uav_mask == 1)[0]
        
        if len(valid_uavs) < 2:
            return {
                'request_idx': int(request_idx),
                'uav_for_vnf1': int(valid_uavs[0]) if len(valid_uavs) > 0 else 0,
                'uav_for_vnf2': int(valid_uavs[1]) if len(valid_uavs) > 1 else 0
            }
        
        # 随机选择两个**不同**的UAV
        uav_for_vnf1 = random.choice(valid_uavs)
        # 从剩余的UAV中选择
        remaining_uavs = [u for u in valid_uavs if u != uav_for_vnf1]
        uav_for_vnf2 = random.choice(remaining_uavs) if remaining_uavs else uav_for_vnf1
        
        return {
            'request_idx': int(request_idx),
            'uav_for_vnf1': int(uav_for_vnf1),
            'uav_for_vnf2': int(uav_for_vnf2)
        }


def print_uav_status(uavs: List, title: str = "UAV状态", uav_to_vnfs: dict = None):
    """打印所有UAV的详细状态，包括部署的VNF信息"""
    print(f"\n{'='*110}")
    print(f"{title}")
    print(f"{'='*110}")
    print(f"{'UAV ID':<8} {'位置ID':<10} {'位置(X,Y)':<22} {'电量(J)':<15} {'CPU容量':<12} {'忙碌':<8} {'部署的VNF':<20}")
    print(f"{'-'*105}")
    
    for uav in uavs:
        loc_str = f"({uav.location[0]:.1f}, {uav.location[1]:.1f})"
        # 获取该UAV部署的VNF信息
        if uav_to_vnfs and uav.id in uav_to_vnfs and uav_to_vnfs[uav.id]:
            vnf_list = uav_to_vnfs[uav.id]
            vnf_str = ", ".join([f"VNF{v['vnf_id']}(SFC{v['sfc_id']})" for v in vnf_list])
        else:
            vnf_str = "-"
        print(f"{uav.id:<8} {uav.location_id:<10} {loc_str:<22} {uav.energy:<15.1f} {uav.cpu_capacity:<12.1f} {str(uav.is_busy):<8} {vnf_str:<20}")
    
    # 统计信息
    total_energy = sum(u.energy for u in uavs)
    avg_energy = total_energy / len(uavs) if uavs else 0
    busy_count = sum(1 for u in uavs if u.is_busy)
    at_base = sum(1 for u in uavs if u.location_id == 0)
    can_charge_count = sum(1 for u in uavs if not u.is_busy and u.location_id != 0)
    active_vnfs = sum(len(vnfs) for vnfs in (uav_to_vnfs or {}).values())
    
    print(f"{'-'*105}")
    print(f"统计: 平均电量={avg_energy:.1f}J, 忙碌={busy_count}/{len(uavs)}, 在基站={at_base}/{len(uavs)}, "
          f"可返回充电={can_charge_count}/{len(uavs)}, 活跃VNF={active_vnfs}")


def print_sfc_details(env: UAVSFCEnv, request_id: int, action: Dict, success: bool, info: Dict):
    """打印SFC（请求）的简化信息，只显示VNF目标位置ID"""
    # 获取请求对象
    request = None
    for req in env.requests:
        if req.id == request_id:
            request = req
            break
    
    if request is None:
        return
    
    print(f"\n  {'='*80}")
    print(f"  SFC #{request_id} 信息")
    print(f"  {'='*80}")
    
    # 简化VNF信息：只显示目标位置ID
    vnf_locations = [vnf.location_id for vnf in request.vnfs]
    print(f"  VNF目标位置: {vnf_locations}")
    
    # UAV分配信息
    uav1_idx = action['uav_for_vnf1']
    uav2_idx = action['uav_for_vnf2']
    uav1 = env.uavs[uav1_idx]
    uav2 = env.uavs[uav2_idx]
    
    print(f"  UAV分配: VNF1->UAV{uav1_idx}(位置{uav1.location_id}), VNF2->UAV{uav2_idx}(位置{uav2.location_id})")
    
    # 执行结果
    if success and info.get('sfc_completed'):
        print(f"  结果: ✓ 成功 | 能耗:{info.get('energy_consumed', 0):.1f}J | CPU:{info.get('cpu_used', 0):.1f}")
    elif info.get('action_invalid'):
        print(f"  结果: ✗ 无效 | 错误:{info.get('error', 'unknown')}")
        
        # 简要分析失败原因
        if uav1_idx == uav2_idx:
            print(f"        原因: 两VNF分配至同一UAV {uav1_idx}")
    else:
        print(f"  结果: ✗ 失败 | {info.get('error', 'unknown')}")
    
    print(f"  {'='*80}")


def print_action_summary(action: Dict, env: UAVSFCEnv, success: bool, info: Dict):
    """简要打印动作执行信息"""
    req_idx = action['request_idx']
    uav1_idx = action['uav_for_vnf1']
    uav2_idx = action['uav_for_vnf2']
    
    if success and info.get('sfc_completed'):
        req_id = info.get('request_id', 'N/A')
        print(f"  ✓ 请求 {req_idx} (ID:{req_id}) -> UAV {uav1_idx} + UAV {uav2_idx} | 成功")
    elif info.get('action_invalid'):
        print(f"  ✗ 请求 {req_idx} -> UAV {uav1_idx} + UAV {uav2_idx} | 无效 ({info.get('error', 'unknown')})")


def test_single_episode(env: UAVSFCEnv, max_steps: int = 1000, verbose: bool = True, detailed_sfc: bool = True):
    """
    测试单个episode，使用随机策略
    
    Args:
        env: 环境实例
        max_steps: 最大步数
        verbose: 是否打印详细信息
        detailed_sfc: 是否打印每个SFC的详细信息
    
    Returns:
        episode_info: 包含成功率等信息的字典
    """
    policy = RandomPolicy(env)
    obs = env.reset()
    
    episode_reward = 0
    step_count = 0
    completed_requests = set()
    failed_requests = set()  # 记录失败的请求
    
    # 记录每个时隙的信息
    time_slot_info = {}
    current_slot = env.current_time_slot
    
    # 记录每个UAV在当前时隙部署了哪些VNF: {uav_id: [{'vnf_id': x, 'sfc_id': y}, ...]}
    uav_to_vnfs = {}
    
    if verbose:
        print(f"\n{'#'*100}")
        print(f"# Episode 开始")
        print(f"# UAV数量: {env.num_uavs}, 请求数量: {len(env.requests)}, 时间槽: {env.num_time_slots}")
        print(f"# 最大步数: {max_steps}, 每个时隙最大步数: {env.max_steps_per_slot}")
        print(f"{'#'*100}")
        print_uav_status(env.uavs, f"初始状态 (时间槽 {current_slot})", uav_to_vnfs)
    
    done = False
    while not done and step_count < max_steps:
        # 检查是否进入新的时间槽
        if env.current_time_slot != current_slot:
            if verbose:
                returned_count = sum(1 for u in env.uavs if u.location_id == 0 and u.energy == config.UAV_BATTERY_CAPACITY)
                print_uav_status(env.uavs, f"时间槽 {current_slot} 结束 - 充电后状态", uav_to_vnfs)
                print(f"\n  本时隙完成请求数: {len(time_slot_info.get(current_slot, []))}")
                print(f"  返回基站充电的UAV: {returned_count}")
            
            current_slot = env.current_time_slot
            # 清空UAV到VNF的映射（新时隙开始）
            uav_to_vnfs = {}
            
            if verbose:
                print(f"\n{'#'*100}")
                print(f"# 进入时间槽 {current_slot}")
                print(f"{'#'*100}")
                print_uav_status(env.uavs, f"时间槽 {current_slot} 开始", uav_to_vnfs)
        
        # 获取动作掩码
        mask = env.get_action_mask()
        
        # 选择动作
        action = policy.select_action(obs, mask)
        
        # 执行动作
        next_obs, reward, done, info = env.step(action)
        
        episode_reward += reward
        step_count += 1
        
        # 记录完成的请求
        if info.get('sfc_completed') and 'request_id' in info:
            req_id = info['request_id']
            if req_id not in completed_requests:
                completed_requests.add(req_id)
                if current_slot not in time_slot_info:
                    time_slot_info[current_slot] = []
                time_slot_info[current_slot].append(req_id)
                
                # 记录UAV到VNF的映射
                uav1_idx = action['uav_for_vnf1']
                uav2_idx = action['uav_for_vnf2']
                # 找到对应的VNF ID
                for req in env.requests:
                    if req.id == req_id:
                        vnf1_id = req.vnfs[0].id
                        vnf2_id = req.vnfs[1].id
                        
                        if uav1_idx not in uav_to_vnfs:
                            uav_to_vnfs[uav1_idx] = []
                        uav_to_vnfs[uav1_idx].append({'vnf_id': vnf1_id, 'sfc_id': req_id})
                        
                        if uav2_idx not in uav_to_vnfs:
                            uav_to_vnfs[uav2_idx] = []
                        uav_to_vnfs[uav2_idx].append({'vnf_id': vnf2_id, 'sfc_id': req_id})
                        break
                
                # 打印SFC详细信息
                if verbose:
                    if detailed_sfc:
                        print_sfc_details(env, req_id, action, True, info)
                    else:
                        print_action_summary(action, env, True, info)
        
        # 记录失败的请求（可选）
        if info.get('action_invalid') and verbose:
            req_idx = action['request_idx']
            if req_idx < len(env.pending_queue):
                req = env.pending_queue[req_idx]
                if req.id not in failed_requests:
                    failed_requests.add(req.id)
                    if detailed_sfc:
                        print_sfc_details(env, req.id, action, False, info)
                    else:
                        print_action_summary(action, env, False, info)
        
        obs = next_obs
    
    # 计算成功率
    total_requests = len(env.requests)
    success_rate = len(completed_requests) / total_requests if total_requests > 0 else 0
    
    if verbose:
        print(f"\n{'#'*100}")
        print(f"# Episode 结束")
        print(f"# 总步数: {step_count}")
        print(f"# 总奖励: {episode_reward:.2f}")
        print(f"# 完成请求: {len(completed_requests)}/{total_requests} ({success_rate:.1%})")
        print(f"# 各时隙完成情况:")
        for slot, reqs in sorted(time_slot_info.items()):
            print(f"#   时隙 {slot}: {len(reqs)} 个请求")
        print(f"{'#'*100}")
    
    return {
        'success_rate': success_rate,
        'completed_count': len(completed_requests),
        'total_requests': total_requests,
        'episode_reward': episode_reward,
        'step_count': step_count,
        'time_slot_info': time_slot_info
    }


def main():
    """主函数：运行环境测试"""
    parser = argparse.ArgumentParser(description='DRL环境测试 - 随机策略')
    parser.add_argument('--num_locations', type=int, default=12,
                        help='监控点数量 (默认12)。修改后需配合 --force 使用')
    parser.add_argument('--num_uavs', type=int, default=4,
                        help='UAV数量 (默认4)')
    parser.add_argument('--num_requests', type=int, default=12,
                        help='请求数量 (默认12)')
    parser.add_argument('--area_size', type=float, default=1000.0,
                        help='区域大小 (默认1000)')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子 (默认42)')
    parser.add_argument('--num_episodes', type=int, default=3,
                        help='测试回合数 (默认3)')
    parser.add_argument('--output', type=str, default=None,
                        help='输出日志文件路径 (默认test_random_<timestamp>.log)')
    parser.add_argument('--simple', action='store_true',
                        help='简化输出，不显示每个SFC的详细信息')
    parser.add_argument('--force', action='store_true',
                        help='强制重新生成数据集（当修改位置数量时使用）')
    
    args = parser.parse_args()
    
    # 设置日志输出
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = args.output if args.output else f'test_random_{timestamp}.log'
    sys.stdout = TeeOutput(log_filename)
    
    print("="*100)
    print("DRL环境测试 - 随机策略")
    print("="*100)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"输出同时保存到: {os.path.abspath(log_filename)}")
    print(f"\n配置参数:")
    print(f"  监控点数量: {args.num_locations}")
    print(f"  UAV数量: {args.num_uavs}")
    print(f"  请求数量: {args.num_requests}")
    print(f"  区域大小: {args.area_size}")
    print(f"  随机种子: {args.seed}")
    print(f"  详细SFC输出: {'否' if args.simple else '是'}")
    
    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # 创建或加载数据集
    print("\n准备数据集...")
    
    # 构建数据集文件名（包含位置数量以便区分）
    dataset_filename = f"dataset_seed{args.seed}_ep1_loc{args.num_locations}_uav{args.num_uavs}_req{args.num_requests}.pkl"
    dataset_path = os.path.join('./data', dataset_filename)
    
    if not args.force and os.path.exists(dataset_path):
        # 加载已有数据集
        print(f"从 {dataset_path} 加载数据集...")
        dataset = TrainingDataset.load(dataset_path)
    else:
        # 生成新数据集
        if args.force and os.path.exists(dataset_path):
            print("强制重新生成数据集...")
        dataset = TrainingDataset(
            base_seed=args.seed,
            num_episodes=1,
            num_locations=args.num_locations,
            area_size=args.area_size,
            num_uavs=args.num_uavs,
            num_requests=args.num_requests
        )
        dataset.generate()
        dataset.save(dataset_path)
    
    print(f"数据集准备完成: {len(dataset)} 个episode")
    
    # 使用第一个episode创建环境
    episode_data = dataset[0]
    
    # 创建环境
    env = UAVSFCEnv(
        uavs=None,
        requests=None,
        locations=None,
        max_pending=20,
        max_steps_per_episode=1000,
        episode_data=episode_data
    )
    
    print(f"\n环境创建成功:")
    print(f"  UAV数量: {env.num_uavs}")
    print(f"  请求数量: {len(env.requests)}")
    print(f"  时间槽数: {env.num_time_slots}")
    print(f"  最大待处理: {env.max_pending}")
    print(f"  每个时隙最大步数: {env.max_steps_per_slot}")
    
    # 打印所有请求（SFC）的信息
    print(f"\n{'='*100}")
    print("所有SFC（请求）信息:")
    print(f"{'='*100}")
    for req in env.requests:
        vnf_locs = [vnf.location_id for vnf in req.vnfs]
        print(f"  SFC #{req.id}: VNF位置={vnf_locs}")
    
    # 运行多个episode测试
    results = []
    
    for ep in range(args.num_episodes):
        print(f"\n\n{'='*100}")
        print(f"测试 Episode {ep + 1}/{args.num_episodes}")
        print(f"{'='*100}")
        
        # 重新加载数据
        env.reset()
        
        # 运行测试
        result = test_single_episode(env, max_steps=1000, verbose=True, detailed_sfc=not args.simple)
        results.append(result)
    
    # 汇总结果
    print(f"\n\n{'='*100}")
    print("测试汇总")
    print(f"{'='*100}")
    
    avg_success_rate = np.mean([r['success_rate'] for r in results])
    avg_completed = np.mean([r['completed_count'] for r in results])
    avg_reward = np.mean([r['episode_reward'] for r in results])
    
    print(f"平均成功率: {avg_success_rate:.1%}")
    print(f"平均完成请求: {avg_completed:.1f}")
    print(f"平均奖励: {avg_reward:.2f}")
    
    print("\n测试完成!")


if __name__ == '__main__':
    main()
