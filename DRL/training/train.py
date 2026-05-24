# -*- coding: utf-8 -*-
"""
PPO 训练入口脚本

使用方法:
    python train.py --config config.yaml
    
或直接使用默认配置:
    python train.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import argparse
import yaml
import torch
from datetime import datetime
from typing import Optional

# 导入自定义模块
from DRL.models import PolicyNetwork
from DRL.env import UAVSFCEnv
from DRL.training import PPOTrainer
from DRL.training.dataset import create_or_load_dataset, generate_uavs, generate_requests, generate_locations, EpisodeData
from DRL.training.logger import TrainingLogger
import config


def load_config(config_path: str = None) -> dict:
    """
    加载配置文件
    
    Args:
        config_path: 配置文件路径，如果为 None 则使用默认配置
    
    Returns:
        配置字典
    """
    # 默认配置（与 config.py 保持一致）
    default_config = {
        'scene': {
            'area_size': 1000.0,
            'num_locations': 200,
            'num_uavs': 80,
            'num_requests': 200,
            'num_time_slots': 4,
            'vnfs_per_request': 2,
        },
        'ppo': {
            'lr': 1e-4,
            'gamma': 0.995,  # 高折扣因子
            'lambda': 0.95,
            'epsilon': 0.2,
            'value_loss_coef': 0.5,
            'entropy_coef': 0.01,
            'max_grad_norm': 0.5,
            'ppo_epochs': 4,
            'batch_size': 512,  # 增大以充分利用GPU
            'num_episodes_per_update': 20,  # 增大以收集更多数据，提高GPU利用率
        },
        'network': {
            'hidden_dim': 256,
            'num_heads': 4,
            'num_encoder_layers': 2,
            'dropout': 0.1,
            'use_cross_attn': True,
        },
        'training': {
            'total_episodes': 3000,
            'max_steps_per_episode': 1000,
            'eval_interval': 100,
            'save_interval': 500,
            'use_reward_norm': True,
            'reward_scale': 1.0,
        },
        'env': {
            'max_pending': 200,
            'max_steps_per_slot': 250,
            'invalid_action_penalty': 0.1,
        },
        'dataset': {
            'base_seed': 42,
            'num_episodes': 100,
            'data_dir': './data',
        },
        'logging': {
            'log_dir': './logs',
            'checkpoint_dir': './checkpoints',
            'use_tensorboard': True,
            'verbose': True,
        }
    }
    
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            user_config = yaml.safe_load(f)
        
        # 合并配置
        for key, value in user_config.items():
            if key in default_config and isinstance(value, dict):
                default_config[key].update(value)
            else:
                default_config[key] = value
        
        print(f"已加载配置文件: {config_path}")
    else:
        print("使用默认配置")
    
    return default_config


def create_env(cfg: dict, episode_data: Optional[EpisodeData] = None) -> UAVSFCEnv:
    """
    创建环境
    
    Args:
        cfg: 配置字典
        episode_data: 回合数据（用于固定种子训练）
    
    Returns:
        UAVSFCEnv 实例
    """
    # 从配置文件获取参数
    scene_cfg = cfg.get('scene', {})
    env_cfg = cfg.get('env', {})
    training_cfg = cfg.get('training', {})
    
    # 场景参数（从 config.yaml 读取，如果没有则使用 config.py 的默认值）
    area_size = scene_cfg.get('area_size', config.AREA_SIZE)
    num_locations = scene_cfg.get('num_locations', config.NUM_LOCATIONS)
    num_uavs = scene_cfg.get('num_uavs', config.NUM_UAVS)
    num_requests = scene_cfg.get('num_requests', config.NUM_REQUESTS)
    num_time_slots = scene_cfg.get('num_time_slots', config.NUM_TIME_SLOTS)
    vnfs_per_request = scene_cfg.get('vnfs_per_request', config.VNFS_PER_REQUEST)
    
    # 环境参数
    max_pending = env_cfg.get('max_pending', 20)
    max_steps_per_episode = training_cfg.get('max_steps_per_episode', 600)
    max_steps_per_slot = env_cfg.get('max_steps_per_slot', 50)
    invalid_action_penalty = env_cfg.get('invalid_action_penalty', 0.1)
    
    if episode_data is not None:
        # 使用预生成的数据
        env = UAVSFCEnv(
            uavs=None,
            requests=None,
            locations=None,
            max_pending=max_pending,
            max_steps_per_episode=max_steps_per_episode,
            max_steps_per_slot=max_steps_per_slot,
            num_time_slots=num_time_slots,
            num_locations=num_locations,
            area_size=area_size,
            invalid_action_penalty=invalid_action_penalty,
            episode_data=episode_data
        )
    else:
        # 随机生成场景
        locations = generate_locations(num_locations, area_size)
        uavs = generate_uavs(num_uavs, locations)
        requests = generate_requests(num_requests, locations)
        
        env = UAVSFCEnv(
            uavs=uavs,
            requests=requests,
            locations=locations,
            max_pending=max_pending,
            max_steps_per_episode=max_steps_per_episode,
            max_steps_per_slot=max_steps_per_slot,
            num_time_slots=num_time_slots,
            num_locations=num_locations,
            area_size=area_size,
            invalid_action_penalty=invalid_action_penalty
        )
    
    return env


def create_policy(cfg: dict) -> PolicyNetwork:
    """
    创建策略网络
    
    Args:
        cfg: 配置字典
    
    Returns:
        PolicyNetwork 实例
    """
    network_cfg = cfg.get('network', {})
    env_cfg = cfg.get('env', {})
    scene_cfg = cfg.get('scene', {})
    
    # UAV特征维度: energy, cpu, pos_x, pos_y, is_busy, dist_to_base = 6
    # Request特征维度: num_vnfs, workload, comm, loc1, loc2 = 5
    policy = PolicyNetwork(
        uav_input_dim=6,
        request_input_dim=5,
        global_dim=4,
        hidden_dim=network_cfg.get('hidden_dim', 64),
        num_uavs=scene_cfg.get('num_uavs', config.NUM_UAVS),
        max_pending=env_cfg.get('max_pending', 20),
        num_heads=network_cfg.get('num_heads', 4),
        num_encoder_layers=network_cfg.get('num_encoder_layers', 2),
        use_cross_attn=network_cfg.get('use_cross_attn', True),
        dropout=network_cfg.get('dropout', 0.1),
    )
    
    return policy


def main():
    parser = argparse.ArgumentParser(description='PPO 训练 UAV-SFC 部署')
    parser.add_argument('--config', type=str, default=None,
                        help='配置文件路径 (YAML 格式)')
    parser.add_argument('--device', type=str, default='auto',
                        help='训练设备 (cuda/cpu/auto)')
    parser.add_argument('--seed', type=int, default=None,
                        help='随机种子（覆盖配置文件中的设置）')
    parser.add_argument('--num-episodes', type=int, default=None,
                        help='数据集包含的回合数（覆盖配置文件中的设置）')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='数据保存目录（覆盖配置文件中的设置）')
    parser.add_argument('--no-fixed-seed', action='store_true',
                        help='不使用固定种子（每次随机生成场景）')
    
    args = parser.parse_args()
    
    # 加载配置
    cfg = load_config(args.config)
    
    # 从配置读取参数（命令行参数覆盖配置文件）
    scene_cfg = cfg.get('scene', {})
    dataset_cfg = cfg.get('dataset', {})
    
    # 设置设备
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    print(f"使用设备: {device}")
    
    # 设置随机种子（命令行 > 配置文件 > 默认值）
    seed = args.seed if args.seed is not None else dataset_cfg.get('base_seed', 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # 数据集参数（命令行 > 配置文件）
    num_episodes = args.num_episodes if args.num_episodes is not None else dataset_cfg.get('num_episodes', 100)
    data_dir = args.data_dir if args.data_dir is not None else dataset_cfg.get('data_dir', './data')
    
    # 创建或加载数据集
    dataset = None
    if not args.no_fixed_seed:
        print("\n准备固定种子数据集...")
        dataset = create_or_load_dataset(
            base_seed=seed,
            num_episodes=num_episodes,
            data_dir=data_dir,
            num_locations=scene_cfg.get('num_locations'),
            area_size=scene_cfg.get('area_size'),
            num_uavs=scene_cfg.get('num_uavs'),
            num_requests=scene_cfg.get('num_requests')
        )
        print(f"  数据集大小: {len(dataset)} 个回合")
        print(f"  基础种子: {seed}")
        # 使用第一个episode初始化环境
        env = create_env(cfg, episode_data=dataset[0])
    else:
        print("\n使用随机生成场景（非固定种子）...")
        env = create_env(cfg)
    
    # 打印场景配置（从 cfg 读取）
    print(f"\n场景配置（来自 config.yaml）:")
    print(f"  UAV 数量: {scene_cfg.get('num_uavs', config.NUM_UAVS)}")
    print(f"  请求数量: {scene_cfg.get('num_requests', config.NUM_REQUESTS)}")
    print(f"  监控点数量: {scene_cfg.get('num_locations', config.NUM_LOCATIONS)}")
    print(f"  区域大小: {scene_cfg.get('area_size', config.AREA_SIZE)}m")
    print(f"  时间槽数: {scene_cfg.get('num_time_slots', config.NUM_TIME_SLOTS)}")
    
    # 创建策略网络
    print("\n创建策略网络...")
    policy = create_policy(cfg)
    total_params = sum(p.numel() for p in policy.parameters())
    print(f"  总参数量: {total_params:,}")
    
    # 创建日志记录器
    print("\n创建日志记录器...")
    logging_cfg = cfg.get('logging', {})
    log_dir = logging_cfg.get('log_dir', './logs')
    experiment_name = f"ppo_seed{args.seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger = TrainingLogger(log_dir=log_dir, experiment_name=experiment_name)
    
    # 添加 UAV 资源设置到配置
    import config as env_config
    cfg['uav_resources'] = {
        'battery_capacity': env_config.UAV_BATTERY_CAPACITY,
        'computation_capacity': env_config.UAV_COMPUTATION_CAPACITY,
        'max_speed': env_config.UAV_MAX_SPEED,
        'transmit_power': env_config.UAV_TRANSMIT_POWER,
        'travel_time': env_config.UAV_TRAVEL_TIME,
        'blade_profile_power': env_config.UAV_BLADE_PROFILE_POWER,
        'induced_power': env_config.UAV_INDUCED_POWER,
    }
    
    # 保存完整配置（包括环境、网络、训练等所有设置）
    logger.log_config(cfg)
    
    # 为当前实验创建独立的 checkpoint 目录
    checkpoint_base_dir = logging_cfg.get('checkpoint_dir', './checkpoints')
    experiment_checkpoint_dir = os.path.join(checkpoint_base_dir, experiment_name)
    os.makedirs(experiment_checkpoint_dir, exist_ok=True)
    print(f"  Checkpoint 目录: {experiment_checkpoint_dir}")
    
    # 创建训练器
    print("创建 PPO 训练器...")
    ppo_cfg = cfg.get('ppo', {})
    training_cfg = cfg.get('training', {})
    
    trainer = PPOTrainer(
        env=env,
        policy=policy,
        dataset=dataset,  # 传入数据集
        logger=logger,    # 传入日志记录器
        lr=float(ppo_cfg.get('lr', 3e-4)),
        gamma=float(ppo_cfg.get('gamma', 0.99)),
        lam=float(ppo_cfg.get('lambda', 0.95)),
        epsilon=float(ppo_cfg.get('epsilon', 0.2)),
        value_loss_coef=float(ppo_cfg.get('value_loss_coef', 0.5)),
        entropy_coef=float(ppo_cfg.get('entropy_coef', 0.01)),
        max_grad_norm=float(ppo_cfg.get('max_grad_norm', 0.5)),
        ppo_epochs=int(ppo_cfg.get('ppo_epochs', 4)),
        batch_size=int(ppo_cfg.get('batch_size', 64)),
        use_reward_norm=training_cfg.get('use_reward_norm', True),
        device=device,
        log_dir=log_dir,
    )
    
    # 开始训练
    print("\n" + "=" * 50)
    print("开始训练")
    print("=" * 50 + "\n")
    
    trainer.train(
        total_episodes=int(training_cfg.get('total_episodes', 10000)),
        num_episodes_per_update=int(ppo_cfg.get('num_episodes_per_update', 10)),
        eval_interval=int(training_cfg.get('eval_interval', 100)),
        save_interval=int(training_cfg.get('save_interval', 500)),
        save_dir=experiment_checkpoint_dir,
    )


if __name__ == '__main__':
    main()
