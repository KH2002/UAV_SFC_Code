#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DRL模型测试脚本 - 加载训练好的模型进行推理

使用方法:
    cd /mnt/sdb11/HK/UAV_SFC_code/DRL/training
    python test_model.py --model_path ./checkpoints/ppo_seedNone_20260406_001138/policy_episode_500.pt
    
可选参数:
    --config: 配置文件路径 (默认使用训练时配置)
    --num_episodes: 测试回合数 (默认10)
    --seed: 随机种子 (默认42)
    --device: 运行设备 cuda/cpu (默认auto)
    --verbose: 是否输出详细信息 (默认True)
    --output: 输出结果保存路径 (默认 test_logs/results_<timestamp>.json)
    --log_file: 日志文件路径 (默认 test_logs/test_<timestamp>.log)
    --output_dir: 输出目录 (默认 test_logs/)
    
输出:
    - 日志文件默认保存到 DRL/training/test_logs/test_<timestamp>.log
    - 结果文件默认保存到 DRL/training/test_logs/results_<timestamp>.json
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import argparse
import yaml
import json
import torch
import numpy as np
import random
from typing import Dict, List, Tuple, Optional
from datetime import datetime

from DRL.models import PolicyNetwork
from DRL.env import UAVSFCEnv
from DRL.training.dataset import create_or_load_dataset, generate_locations, generate_uavs, generate_requests, TrainingDataset
from DRL.training.train import load_config, create_env, create_policy
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


def load_model(model_path: str, policy: PolicyNetwork, device: str = 'cpu') -> bool:
    """
    加载训练好的模型
    
    Args:
        model_path: 模型文件路径
        policy: 策略网络实例
        device: 运行设备
    
    Returns:
        是否加载成功
    """
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在: {model_path}")
        return False
    
    try:
        state_dict = torch.load(model_path, map_location=device)
        policy.load_state_dict(state_dict)
        policy.to(device)
        policy.eval()  # 设置为评估模式
        print(f"模型加载成功: {model_path}")
        print(f"  设备: {device}")
        return True
    except Exception as e:
        print(f"模型加载失败: {e}")
        return False


def obs_to_tensor(obs: Dict, device: str) -> Dict[str, torch.Tensor]:
    """将观察字典转换为 tensor"""
    return {
        'uav_states': torch.tensor(obs['uav_states'], dtype=torch.float32).unsqueeze(0).to(device),
        'pending_requests': torch.tensor(obs['pending_requests'], dtype=torch.float32).unsqueeze(0).to(device),
        'global_features': torch.tensor(obs['global_features'], dtype=torch.float32).unsqueeze(0).to(device),
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


def test_single_episode(
    env: UAVSFCEnv, 
    policy: PolicyNetwork, 
    device: str,
    max_steps: int = 1000, 
    verbose: bool = True,
    detailed_sfc: bool = True
) -> Dict:
    """
    测试单个episode，使用训练好的模型
    
    Args:
        env: 环境实例
        policy: 策略网络
        device: 运行设备
        max_steps: 最大步数
        verbose: 是否打印详细信息
        detailed_sfc: 是否打印每个SFC的详细信息
    
    Returns:
        episode_info: 包含成功率等信息的字典
    """
    obs = env.reset()
    
    episode_reward = 0
    step_count = 0
    completed_requests = set()
    failed_requests = set()  # 记录失败的请求
    invalid_action_count = 0
    
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
                # 打印上一时隙结束时的UAV状态（充电后）
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
        mask_tensor = {
            'request': torch.tensor(mask['request'], dtype=torch.bool).unsqueeze(0).to(device),
            'uav': torch.tensor(mask['uav'], dtype=torch.bool).unsqueeze(0).to(device),
        }
        
        # 使用模型选择动作 (deterministic=True 用于测试)
        obs_tensor = obs_to_tensor(obs, device)
        with torch.no_grad():
            output = policy(obs_tensor, mask_tensor, deterministic=True)
        
        action_tensor = output['action']
        
        # 直接使用模型输出的动作
        action = {
            'request_idx': action_tensor['request_idx'].cpu().numpy()[0],
            'uav_for_vnf1': action_tensor['uav_for_vnf1'].cpu().numpy()[0],
            'uav_for_vnf2': action_tensor['uav_for_vnf2'].cpu().numpy()[0],
        }
        
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
        
        # 记录失败的请求
        if info.get('action_invalid'):
            invalid_action_count += 1
            if verbose:
                req_idx = action['request_idx']
                if req_idx < len(env.pending_queue):
                    req = env.pending_queue[req_idx]
                    if req.id not in failed_requests:
                        failed_requests.add(req.id)
                        if detailed_sfc:
                            print_sfc_details(env, req.id, action, False, info)
        
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
        print(f"# 无效动作次数: {invalid_action_count}")
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
        'invalid_action_count': invalid_action_count,
        'time_slot_info': time_slot_info
    }


def main():
    """主函数：运行模型测试"""
    parser = argparse.ArgumentParser(description='测试训练好的DRL模型')
    parser.add_argument('--model_path', type=str, required=True,
                        help='模型文件路径 (.pt文件)')
    parser.add_argument('--config', type=str, default=None,
                        help='配置文件路径 (YAML格式)，默认使用train.py中的默认配置')
    parser.add_argument('--num_episodes', type=int, default=10,
                        help='测试回合数 (默认10)')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子 (默认42)')
    parser.add_argument('--device', type=str, default='auto',
                        help='运行设备 (cuda/cpu/auto，默认auto)')
    parser.add_argument('--verbose', action='store_true', default=True,
                        help='是否输出详细信息 (默认True)')
    parser.add_argument('--quiet', action='store_true',
                        help='安静模式，不输出详细信息')
    parser.add_argument('--simple', action='store_true',
                        help='简化输出，不显示每个SFC的详细信息')
    parser.add_argument('--output', type=str, default=None,
                        help='测试结果保存路径 (JSON格式)，默认为test_logs/results_<timestamp>.json')
    parser.add_argument('--log_file', type=str, default=None,
                        help='日志文件路径，默认为test_logs/test_<timestamp>.log')
    parser.add_argument('--output_dir', type=str, default='./test_logs',
                        help='输出目录，用于保存日志和结果 (默认./test_logs)')
    
    args = parser.parse_args()
    
    # 设置verbose模式
    verbose = not args.quiet and args.verbose
    
    # 创建输出目录
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成时间戳
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 设置默认日志文件路径
    log_file = args.log_file
    if log_file is None:
        log_file = os.path.join(output_dir, f'test_{timestamp}.log')
    
    # 设置默认结果文件路径
    output_file = args.output
    if output_file is None:
        output_file = os.path.join(output_dir, f'results_{timestamp}.json')
    
    # 设置日志输出
    sys.stdout = TeeOutput(log_file)
    print(f"输出同时保存到: {os.path.abspath(log_file)}")
    
    print("="*100)
    print("DRL模型测试 - 训练好的策略")
    print("="*100)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"模型路径: {args.model_path}")
    print(f"测试回合数: {args.num_episodes}")
    print(f"随机种子: {args.seed}")
    
    # 设置设备
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    print(f"设备: {device}")
    
    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # 加载配置
    print("\n加载配置...")
    cfg = load_config(args.config)
    scene_cfg = cfg.get('scene', {})
    dataset_cfg = cfg.get('dataset', {})
    
    # 创建或加载数据集
    print("\n准备数据集...")
    dataset = create_or_load_dataset(
        base_seed=dataset_cfg.get('base_seed', 42),
        num_episodes=max(args.num_episodes, dataset_cfg.get('num_episodes', 100)),
        data_dir=dataset_cfg.get('data_dir', './data'),
        num_locations=scene_cfg.get('num_locations', config.NUM_LOCATIONS),
        area_size=scene_cfg.get('area_size', config.AREA_SIZE),
        num_uavs=scene_cfg.get('num_uavs', config.NUM_UAVS),
        num_requests=scene_cfg.get('num_requests', config.NUM_REQUESTS)
    )
    print(f"数据集准备完成: {len(dataset)} 个episode")
    
    # 创建策略网络
    print("\n创建策略网络...")
    policy = create_policy(cfg)
    total_params = sum(p.numel() for p in policy.parameters())
    print(f"  总参数量: {total_params:,}")
    
    # 加载模型
    print("\n加载模型...")
    if not load_model(args.model_path, policy, device):
        print("模型加载失败，退出测试")
        return
    
    # 运行测试
    print(f"\n{'='*100}")
    print(f"开始测试 {args.num_episodes} 个episode")
    print(f"{'='*100}")
    
    results = []
    for ep in range(args.num_episodes):
        print(f"\n\n{'='*100}")
        print(f"测试 Episode {ep + 1}/{args.num_episodes}")
        print(f"{'='*100}")
        
        # 使用数据集中的场景
        episode_data = dataset[ep % len(dataset)]
        env = create_env(cfg, episode_data=episode_data)
        
        # 运行测试
        result = test_single_episode(env, policy, device, max_steps=1000, verbose=verbose, detailed_sfc=not args.simple)
        results.append(result)
    
    # 汇总结果
    print(f"\n\n{'='*100}")
    print("测试汇总")
    print(f"{'='*100}")
    
    avg_success_rate = np.mean([r['success_rate'] for r in results])
    std_success_rate = np.std([r['success_rate'] for r in results])
    avg_completed = np.mean([r['completed_count'] for r in results])
    avg_reward = np.mean([r['episode_reward'] for r in results])
    avg_steps = np.mean([r['step_count'] for r in results])
    avg_invalid_actions = np.mean([r['invalid_action_count'] for r in results])
    
    print(f"平均成功率: {avg_success_rate:.1%} (±{std_success_rate:.1%})")
    print(f"平均完成请求: {avg_completed:.1f}")
    print(f"平均奖励: {avg_reward:.2f}")
    print(f"平均步数: {avg_steps:.1f}")
    print(f"平均无效动作: {avg_invalid_actions:.1f}")
    
    # 各episode详细结果
    print(f"\n各Episode结果:")
    print(f"{'Episode':<10} {'成功率':<10} {'完成数':<10} {'奖励':<12} {'步数':<10} {'无效动作':<10}")
    print("-" * 70)
    for i, r in enumerate(results):
        print(f"{i+1:<10} {r['success_rate']:<10.1%} {r['completed_count']:<10} {r['episode_reward']:<12.2f} {r['step_count']:<10} {r['invalid_action_count']:<10}")
    
    # 保存结果到文件
    if output_file:
        output_data = {
            'model_path': args.model_path,
            'config_path': args.config,
            'num_episodes': args.num_episodes,
            'seed': args.seed,
            'device': device,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'summary': {
                'avg_success_rate': float(avg_success_rate),
                'std_success_rate': float(std_success_rate),
                'avg_completed': float(avg_completed),
                'avg_reward': float(avg_reward),
                'avg_steps': float(avg_steps),
                'avg_invalid_actions': float(avg_invalid_actions),
            },
            'episodes': results
        }
        
        # 将numpy类型转换为普通Python类型以便JSON序列化
        for ep_result in output_data['episodes']:
            for key in ep_result:
                if isinstance(ep_result[key], (np.integer, np.floating)):
                    ep_result[key] = float(ep_result[key]) if isinstance(ep_result[key], np.floating) else int(ep_result[key])
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存到: {os.path.abspath(output_file)}")
    
    print("\n测试完成!")


if __name__ == '__main__':
    main()
