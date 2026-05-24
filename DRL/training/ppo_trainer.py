# -*- coding: utf-8 -*-
"""
PPO (Proximal Policy Optimization) 训练器

参考: docs/DRL_Solution_Design.md
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque
import yaml
import time
from datetime import datetime

# 导入自定义模块
from DRL.models import PolicyNetwork
from DRL.env import UAVSFCEnv
from DRL.training.dataset import TrainingDataset, EpisodeData
from DRL.training.logger import TrainingLogger
import config


class RolloutBuffer:
    """
    存储轨迹数据的缓冲区
    
    用于收集每个 episode 的数据，供 PPO 更新使用
    按 episode 存储，避免GAE跨 episode 传播
    支持 truncated episodes 的 bootstrap value
    """
    def __init__(self):
        self.episodes = []  # 每个 episode 是一个字典
        self.current_episode = {
            'observations': [],
            'actions': [],
            'log_probs': [],
            'rewards': [],
            'values': [],
            'dones': [],
            'masks': [],
            'last_value': 0.0,  # 用于 truncated episodes 的 bootstrap value
            'truncated': False,  # 是否是被截断的 episode
        }
        
    def add(self, obs, action, log_prob, reward, value, done, mask):
        """添加一个时间步的数据"""
        self.current_episode['observations'].append(obs)
        self.current_episode['actions'].append(action)
        self.current_episode['log_probs'].append(log_prob)
        self.current_episode['rewards'].append(reward)
        self.current_episode['values'].append(value)
        self.current_episode['dones'].append(done)
        self.current_episode['masks'].append(mask)
        
        if done:
            # 将完成的 episode 添加到列表
            self.episodes.append({
                'observations': self.current_episode['observations'][:],
                'actions': self.current_episode['actions'][:],
                'log_probs': self.current_episode['log_probs'][:],
                'rewards': self.current_episode['rewards'][:],
                'values': self.current_episode['values'][:],
                'dones': self.current_episode['dones'][:],
                'masks': self.current_episode['masks'][:],
                'last_value': self.current_episode['last_value'],
                'truncated': self.current_episode['truncated'],
            })
            # 清空当前 episode
            self.current_episode = {
                'observations': [],
                'actions': [],
                'log_probs': [],
                'rewards': [],
                'values': [],
                'dones': [],
                'masks': [],
                'last_value': 0.0,
                'truncated': False,
            }
    
    def finish_episode(self, last_value: float = 0.0, truncated: bool = False):
        """
        标记 episode 结束，设置 bootstrap value
        
        Args:
            last_value: 最终状态的 value（用于 truncated episodes）
            truncated: 是否是被截断的 episode
        """
        # 常规路径：若当前episode仍未入库，则在此写入元数据并入库
        if len(self.current_episode['observations']) > 0:
            self.current_episode['last_value'] = last_value
            self.current_episode['truncated'] = truncated
            self.episodes.append({
                'observations': self.current_episode['observations'][:],
                'actions': self.current_episode['actions'][:],
                'log_probs': self.current_episode['log_probs'][:],
                'rewards': self.current_episode['rewards'][:],
                'values': self.current_episode['values'][:],
                'dones': self.current_episode['dones'][:],
                'masks': self.current_episode['masks'][:],
                'last_value': self.current_episode['last_value'],
                'truncated': self.current_episode['truncated'],
            })
            self.current_episode = {
                'observations': [],
                'actions': [],
                'log_probs': [],
                'rewards': [],
                'values': [],
                'dones': [],
                'masks': [],
                'last_value': 0.0,
                'truncated': False,
            }
            return

        # 兼容当前收集流程：done时已入库，这里回填最后一个episode的元数据
        if len(self.episodes) > 0:
            self.episodes[-1]['last_value'] = last_value
            self.episodes[-1]['truncated'] = truncated
    
    def clear(self):
        """清空缓冲区"""
        self.episodes.clear()
        self.current_episode = {
            'observations': [],
            'actions': [],
            'log_probs': [],
            'rewards': [],
            'values': [],
            'dones': [],
            'masks': [],
            'last_value': 0.0,
            'truncated': False,
        }
    
    def get_batch(self) -> Dict:
        """获取整个批次的数据（展平所有 episodes）"""
        observations = []
        actions = []
        log_probs = []
        rewards = []
        values = []
        dones = []
        masks = []
        
        for ep in self.episodes:
            observations.extend(ep['observations'])
            actions.extend(ep['actions'])
            log_probs.extend(ep['log_probs'])
            rewards.extend(ep['rewards'])
            values.extend(ep['values'])
            dones.extend(ep['dones'])
            masks.extend(ep['masks'])
        
        return {
            'observations': observations,
            'actions': actions,
            'log_probs': log_probs,
            'rewards': rewards,
            'values': values,
            'dones': dones,
            'masks': masks,
            'episodes': self.episodes,  # 保留 episode 边界信息
        }
    
    def get_episodes(self) -> List[Dict]:
        """获取所有 episode 数据（用于按 episode 计算GAE）"""
        return self.episodes
    
    def __len__(self):
        """返回总步数"""
        return sum(len(ep['observations']) for ep in self.episodes)


class RewardNormalizer:
    """
    奖励归一化器（Running Mean & Std）- Welford 算法
    
    用于稳定训练，减少奖励的方差
    使用 Welford 在线算法计算 running mean 和 variance
    """
    def __init__(self, eps=1e-8):
        self.mean = 0.0
        self.M2 = 0.0  # 累积平方差 (sum of squares of differences)
        self.count = 0
        self.eps = eps
    
    def update(self, reward):
        """更新 running mean 和 M2（Welford 算法）"""
        self.count += 1
        delta = reward - self.mean
        self.mean += delta / self.count
        delta2 = reward - self.mean
        self.M2 += delta * delta2
    
    def normalize(self, reward):
        """归一化奖励"""
        if self.count < 1:
            return reward
        # 使用总体标准差 (M2 / count)
        std = np.sqrt(self.M2 / self.count) + self.eps
        return (reward - self.mean) / std
    
    def get_stats(self):
        """获取当前统计信息"""
        if self.count < 1:
            return 0.0, 1.0
        std = np.sqrt(self.M2 / self.count)
        return self.mean, std


class PPOTrainer:
    """
    PPO 训练器
    
    实现了 PPO-Clip 算法，支持：
    - GAE (Generalized Advantage Estimation)
    - 奖励归一化
    - 梯度裁剪
    - 学习率衰减（可选）
    """
    
    def __init__(self,
                 env: UAVSFCEnv,
                 policy: PolicyNetwork,
                 dataset: Optional[TrainingDataset] = None,
                 logger: Optional[TrainingLogger] = None,
                 lr: float = 3e-4,
                 gamma: float = 0.99,
                 lam: float = 0.95,
                 epsilon: float = 0.2,
                 value_loss_coef: float = 0.5,
                 entropy_coef: float = 0.01,
                 max_grad_norm: float = 0.5,
                 ppo_epochs: int = 4,
                 batch_size: int = 64,
                 use_reward_norm: bool = True,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 log_dir: str = './logs'):
        """
        初始化 PPO 训练器
        
        Args:
            env: UAV-SFC 环境
            policy: 策略网络（PolicyNetwork）
            dataset: 训练数据集（用于固定种子训练）
            logger: 训练日志记录器（如果为None则自动创建）
            lr: 学习率
            gamma: 折扣因子
            lam: GAE 参数
            epsilon: PPO 裁剪参数
            value_loss_coef: 价值损失系数
            entropy_coef: 熵奖励系数
            max_grad_norm: 梯度裁剪阈值
            ppo_epochs: 每次数据收集后的 PPO 更新次数
            batch_size: 每次更新的批次大小
            use_reward_norm: 是否使用奖励归一化
            device: 训练设备
            log_dir: 日志保存目录（仅在logger为None时使用）
        """
        self.env = env
        self.policy = policy.to(device)
        self.dataset = dataset
        self.device = device
        self.current_episode_idx = 0
        
        # 超参数
        self.lr = lr
        self.gamma = gamma
        self.lam = lam
        self.epsilon = epsilon
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.use_reward_norm = use_reward_norm
        
        # 日志记录器
        if logger is None:
            self.logger = TrainingLogger(log_dir=log_dir)
        else:
            self.logger = logger
        
        # 优化器
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # 奖励归一化器
        if use_reward_norm:
            self.reward_normalizer = RewardNormalizer()
        else:
            self.reward_normalizer = None
        
        # ε-贪心探索参数（用于缓解无效动作死锁）
        self.epsilon_start = 0.3  # 初始探索率
        self.epsilon_end = 0.05   # 最终探索率
        self.epsilon_decay = 0.995  # 每 episode 衰减
        
        # 缓冲区
        self.buffer = RolloutBuffer()
        
        # 训练统计
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.episode_success_rates = deque(maxlen=100)
    
    def collect_trajectories(self, num_episodes: int, start_episode: int = 0) -> Dict:
        """
        收集轨迹数据
        
        如果使用数据集，则从数据集中加载场景数据；
        否则使用环境的默认reset。
        
        Args:
            num_episodes: 要收集的 episode 数量
            start_episode: 起始episode编号（用于日志记录）
        
        Returns:
            包含收集数据的字典
        """
        all_episodes_data = []
        
        # PPO 采样必须保持 on-policy，这里禁用 ε-贪心随机行为
        current_epsilon = 0.0

        if start_episode == 0:
            print("  [On-Policy] Epsilon-greedy disabled during rollout collection")
        
        for i in range(num_episodes):
            episode_idx = start_episode + i
            
            # 如果使用数据集，加载对应回合的数据
            if self.dataset is not None:
                if self.current_episode_idx >= len(self.dataset):
                    # 数据集用完，循环使用
                    self.current_episode_idx = 0
                
                episode_data = self.dataset[self.current_episode_idx]
                self.env.set_episode_data(episode_data)
                self.current_episode_idx += 1
            
            obs = self.env.reset()
            
            # 调试信息：打印第一个 episode 的信息
            if episode_idx == 0 and start_episode == 0:
                print(f"\n[调试] Episode 0 初始状态:")
                print(f"  待处理请求数: {len(self.env.pending_queue)}")
                print(f"  UAV数量: {len(self.env.uavs)}")
                print(f"  位置数量: {len(self.env.locations)}")
                mask = self.env.get_action_mask()
                print(f"  有效请求数: {mask['request'].sum()}/{len(mask['request'])}")
                print(f"  有效UAV数: {mask['uav'].sum()}/{len(mask['uav'])}")
            
            # 记录初始状态
            initial_state = {
                'uavs': [(uav.id, uav.location, uav.energy, uav.cpu_capacity) for uav in self.env.uavs],
                'requests': [(req.id, req.get_required_location_ids()) for req in self.env.requests],
                'observation': obs,
            }
            self.logger.log_initial_state(episode_idx, initial_state)
            
            done = False
            episode_reward = 0
            episode_length = 0
            
            step_count = 0
            while not done:
                # 调试信息：每10步打印一次进度（仅第一个 episode）
                if episode_idx == 0 and start_episode == 0 and step_count % 10 == 0 and step_count > 0:
                    print(f"    Step {step_count}: pending={len(self.env.pending_queue)}, "
                          f"completed={len(self.env.completed_requests)}, "
                          f"slot={self.env.current_time_slot}")
                
                # 转换观察为 tensor
                obs_tensor = self._obs_to_tensor(obs)
                
                # 获取动作掩码
                mask = self.env.get_action_mask()
                mask_tensor = {
                    'request': torch.tensor(mask['request'], dtype=torch.bool).unsqueeze(0).to(self.device),
                    'uav': torch.tensor(mask['uav'], dtype=torch.bool).unsqueeze(0).to(self.device),
                    'uav_by_request_vnf1': torch.tensor(mask['uav_by_request_vnf1'], dtype=torch.bool).unsqueeze(0).to(self.device),
                    'uav_by_request_vnf2': torch.tensor(mask['uav_by_request_vnf2'], dtype=torch.bool).unsqueeze(0).to(self.device),
                    'uav_by_request': torch.tensor(mask['uav_by_request'], dtype=torch.bool).unsqueeze(0).to(self.device),
                }
                
                # 使用策略分布采样动作（on-policy）
                with torch.no_grad():
                    output = self.policy(obs_tensor, mask_tensor, deterministic=False, epsilon=current_epsilon)
                
                action = output['action']
                log_prob = output['log_prob']
                value = output['value']
                
                # 执行动作
                action_np = {
                    'request_idx': action['request_idx'].cpu().numpy()[0],
                    'uav_for_vnf1': action['uav_for_vnf1'].cpu().numpy()[0],
                    'uav_for_vnf2': action['uav_for_vnf2'].cpu().numpy()[0],
                }
                
                next_obs, reward, done, info = self.env.step(action_np)
                step_count += 1
                
                # 奖励归一化（如果启用）
                if self.use_reward_norm and self.reward_normalizer is not None:
                    self.reward_normalizer.update(reward)
                    reward = self.reward_normalizer.normalize(reward)
                
                # 存储数据
                self.buffer.add(
                    obs=obs,
                    action=action_np,
                    log_prob=log_prob.cpu().numpy()[0],
                    reward=reward,
                    value=value.cpu().numpy()[0, 0],
                    done=done,
                    mask=mask
                )
                
                episode_reward += reward
                episode_length += 1
                obs = next_obs
            
            # 判断是否是 truncated episode（被 max_steps 截断）
            # 自然结束条件：pending_queue 为空 或 current_time_slot > num_time_slots
            natural_done = (
                len(self.env.pending_queue) == 0 or
                self.env.current_time_slot > self.env.num_time_slots
            )
            is_truncated = episode_length >= self.env.max_steps_per_episode and not natural_done
            
            if is_truncated:
                # 对于被截断的 episode，使用 bootstrap value（最后一个状态的 V(s)）
                with torch.no_grad():
                    # 获取最后一个状态的 value
                    last_obs_tensor = self._obs_to_tensor(obs)
                    last_mask = self.env.get_action_mask()
                    last_mask_tensor = {
                        'request': torch.tensor(last_mask['request'], dtype=torch.bool).unsqueeze(0).to(self.device),
                        'uav': torch.tensor(last_mask['uav'], dtype=torch.bool).unsqueeze(0).to(self.device),
                        'uav_by_request_vnf1': torch.tensor(last_mask['uav_by_request_vnf1'], dtype=torch.bool).unsqueeze(0).to(self.device),
                        'uav_by_request_vnf2': torch.tensor(last_mask['uav_by_request_vnf2'], dtype=torch.bool).unsqueeze(0).to(self.device),
                        'uav_by_request': torch.tensor(last_mask['uav_by_request'], dtype=torch.bool).unsqueeze(0).to(self.device),
                    }
                    last_value = self.policy(last_obs_tensor, last_mask_tensor, deterministic=False)['value'].item()
                
                self.buffer.finish_episode(last_value=last_value, truncated=True)
                if episode_idx == 0 and start_episode == 0:
                    print(f"  [Truncated] Bootstrap value: {last_value:.4f}")
            else:
                self.buffer.finish_episode(last_value=0.0, truncated=False)
            
            # 记录 episode 统计
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            
            # 记录成功率
            if 'final_success_rate' in info:
                self.episode_success_rates.append(info['final_success_rate'])
            
            all_episodes_data.append({
                'reward': episode_reward,
                'length': episode_length,
                'success_rate': info.get('final_success_rate', 0),
                'truncated': is_truncated,
            })
            
            # 调试信息：打印第一个 episode 的结果
            if episode_idx == 0 and start_episode == 0:
                print(f"\n[调试] Episode 0 结束:")
                print(f"  总奖励: {episode_reward:.2f}")
                print(f"  步数: {episode_length}")
                print(f"  成功率: {info.get('final_success_rate', 0):.2%}")
                print(f"  完成请求数: {info.get('total_completed', 0)}/{len(self.env.requests)}")
                print(f"  结束原因: {'Max steps (truncated)' if is_truncated else 'Natural'}")
        
        return all_episodes_data
    
    def compute_gae_for_episode(self, rewards: np.ndarray, values: np.ndarray, 
                                last_value: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        为单个 episode 计算 GAE (Generalized Advantage Estimation)
        
        Args:
            rewards: 奖励序列 [T]
            values: 价值估计序列 [T]
            last_value: 最后一个状态的 bootstrap value（默认为0）
        
        Returns:
            advantages: 优势估计 [T]
            returns: 回报估计 [T]
        """
        advantages = np.zeros_like(rewards, dtype=np.float32)
        last_gae = 0
        
        # 构建完整的 value 序列，包含 bootstrap value
        values_extended = np.append(values, last_value)
        
        for t in reversed(range(len(rewards))):
            next_value = values_extended[t + 1]
            
            # TD 误差
            delta = rewards[t] + self.gamma * next_value - values[t]
            
            # GAE
            advantages[t] = last_gae = delta + self.gamma * self.lam * last_gae
        
        # 回报 = 优势 + 价值
        returns = advantages + values
        
        return advantages, returns
    
    def update(self) -> Dict:
        """
        使用收集的数据更新策略
        
        按 episode 计算 GAE，避免跨 episode 传播优势
        
        Returns:
            包含训练统计的字典
        """
        print(f"  [Update] 开始PPO更新，缓冲区大小: {len(self.buffer)} steps, {len(self.buffer.episodes)} episodes")
        
        # 获取所有 episodes
        episodes = self.buffer.get_episodes()
        
        # 准备数据
        all_observations = []
        all_actions = []
        all_log_probs_old = []
        all_advantages = []
        all_returns = []
        all_masks = []
        
        truncated_count = 0
        for ep in episodes:
            rewards = np.array(ep['rewards'])
            values = np.array(ep['values'])
            last_value = ep.get('last_value', 0.0)
            is_truncated = ep.get('truncated', False)
            
            if is_truncated:
                truncated_count += 1
            
            # 对每个 episode 计算 GAE（使用 episode 的 last_value）
            advantages, returns = self.compute_gae_for_episode(rewards, values, last_value=last_value)
            
            all_observations.extend(ep['observations'])
            all_actions.extend(ep['actions'])
            all_log_probs_old.extend(ep['log_probs'])
            all_advantages.extend(advantages)
            all_returns.extend(returns)
            all_masks.extend(ep['masks'])
        
        if truncated_count > 0:
            print(f"    Truncated episodes: {truncated_count}/{len(episodes)} (with bootstrap value)")
        
        # 转换为 numpy 数组
        all_advantages = np.array(all_advantages, dtype=np.float32)
        all_returns = np.array(all_returns, dtype=np.float32)
        all_log_probs_old = np.array(all_log_probs_old, dtype=np.float32)
        
        # 标准化优势（有助于训练稳定性）
        all_advantages = (all_advantages - all_advantages.mean()) / (all_advantages.std() + 1e-8)
        
        # 注意：不应对 returns 进行归一化，Value network 应该学习原始回报
        # 奖励归一化只用于计算 advantages，不用于 value target
        
        # 转换为 tensor
        obs_batch = [self._obs_to_tensor(o) for o in all_observations]
        masks_batch = [{
            'request': torch.tensor(m['request'], dtype=torch.bool).unsqueeze(0).to(self.device),
            'uav': torch.tensor(m['uav'], dtype=torch.bool).unsqueeze(0).to(self.device),
            'uav_by_request_vnf1': torch.tensor(m['uav_by_request_vnf1'], dtype=torch.bool).unsqueeze(0).to(self.device),
            'uav_by_request_vnf2': torch.tensor(m['uav_by_request_vnf2'], dtype=torch.bool).unsqueeze(0).to(self.device),
            'uav_by_request': torch.tensor(m['uav_by_request'], dtype=torch.bool).unsqueeze(0).to(self.device),
        } for m in all_masks]
        
        actions_batch = [{
            'request_idx': torch.tensor([a['request_idx']], dtype=torch.long).to(self.device),
            'uav_for_vnf1': torch.tensor([a['uav_for_vnf1']], dtype=torch.long).to(self.device),
            'uav_for_vnf2': torch.tensor([a['uav_for_vnf2']], dtype=torch.long).to(self.device),
        } for a in all_actions]
        
        log_probs_old_tensor = torch.tensor(all_log_probs_old, dtype=torch.float32).to(self.device)
        advantages_tensor = torch.tensor(all_advantages, dtype=torch.float32).to(self.device)
        returns_tensor = torch.tensor(all_returns, dtype=torch.float32).to(self.device)
        
        # PPO 更新
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        num_updates = 0
        
        print(f"    PPO Epochs: {self.ppo_epochs}, Batch size: {self.batch_size}, Data size: {len(obs_batch)}")
        
        for epoch in range(self.ppo_epochs):
            # 随机打乱数据
            indices = np.random.permutation(len(obs_batch))
            num_batches = (len(obs_batch) + self.batch_size - 1) // self.batch_size
            
            if epoch == 0:
                print(f"    Epoch {epoch+1}/{self.ppo_epochs}: {num_batches} batches")
            
            for batch_num, start in enumerate(range(0, len(obs_batch), self.batch_size)):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                
                # 获取批次数据
                obs_mini = [obs_batch[i] for i in batch_indices]
                masks_mini = [masks_batch[i] for i in batch_indices]
                actions_mini = [actions_batch[i] for i in batch_indices]
                
                # 合并批次数据
                obs_merged = self._merge_observations(obs_mini)
                masks_merged = {
                    'request': torch.cat([masks_mini[i]['request'] for i in range(len(masks_mini))]),
                    'uav': torch.cat([masks_mini[i]['uav'] for i in range(len(masks_mini))]),
                    'uav_by_request_vnf1': torch.cat([masks_mini[i]['uav_by_request_vnf1'] for i in range(len(masks_mini))]),
                    'uav_by_request_vnf2': torch.cat([masks_mini[i]['uav_by_request_vnf2'] for i in range(len(masks_mini))]),
                    'uav_by_request': torch.cat([masks_mini[i]['uav_by_request'] for i in range(len(masks_mini))]),
                }
                actions_merged = {
                    'request_idx': torch.cat([actions_mini[i]['request_idx'] for i in range(len(actions_mini))]),
                    'uav_for_vnf1': torch.cat([actions_mini[i]['uav_for_vnf1'] for i in range(len(actions_mini))]),
                    'uav_for_vnf2': torch.cat([actions_mini[i]['uav_for_vnf2'] for i in range(len(actions_mini))]),
                }
                
                # 前向传播
                output = self.policy.evaluate(obs_merged, actions_merged, masks_merged)
                
                log_probs_new = output['log_prob']
                values_new = output['value'].squeeze()
                entropy = output['entropy']
                
                # 计算比率
                ratio = torch.exp(log_probs_new - log_probs_old_tensor[batch_indices])
                
                # 计算裁剪后的策略损失
                surr1 = ratio * advantages_tensor[batch_indices]
                surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages_tensor[batch_indices]
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # 价值损失
                value_loss = nn.MSELoss()(values_new, returns_tensor[batch_indices])
                
                # 总损失
                loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                
                # 梯度裁剪
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                
                self.optimizer.step()
                
                if epoch == 0 and batch_num == 0:
                    print(f"      First batch updated, loss: {loss.item():.4f}")
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                num_updates += 1
        
        # 清空缓冲区
        self.buffer.clear()
        
        result = {
            'policy_loss': total_policy_loss / num_updates,
            'value_loss': total_value_loss / num_updates,
            'entropy': total_entropy / num_updates,
        }
        print(f"    Update finished: policy_loss={result['policy_loss']:.4f}, value_loss={result['value_loss']:.4f}")
        return result
    
    def _obs_to_tensor(self, obs: Dict) -> Dict[str, torch.Tensor]:
        """将观察字典转换为 tensor"""
        return {
            'uav_states': torch.tensor(obs['uav_states'], dtype=torch.float32).unsqueeze(0).to(self.device),
            'pending_requests': torch.tensor(obs['pending_requests'], dtype=torch.float32).unsqueeze(0).to(self.device),
            'global_features': torch.tensor(obs['global_features'], dtype=torch.float32).unsqueeze(0).to(self.device),
        }
    
    def _merge_observations(self, obs_list: List[Dict]) -> Dict[str, torch.Tensor]:
        """合并多个观察"""
        return {
            'uav_states': torch.cat([o['uav_states'] for o in obs_list]),
            'pending_requests': torch.cat([o['pending_requests'] for o in obs_list]),
            'global_features': torch.cat([o['global_features'] for o in obs_list]),
        }
    
    def train(self, total_episodes: int, num_episodes_per_update: int = 10, 
              eval_interval: int = 100, save_interval: int = 500,
              save_dir: str = './checkpoints'):
        """
        训练循环
        
        Args:
            total_episodes: 总训练回合数
            num_episodes_per_update: 每次更新收集的 episode 数量
            eval_interval: 评估间隔
            save_interval: 保存间隔
            save_dir: 模型保存目录
        """
        os.makedirs(save_dir, exist_ok=True)

        if num_episodes_per_update <= 0:
            raise ValueError("num_episodes_per_update must be > 0")
        
        print(f"开始训练，设备: {self.device}")
        print(f"总回合数: {total_episodes}")
        print(f"每次更新收集 {num_episodes_per_update} 个 episode")
        
        if self.dataset is not None:
            print(f"使用固定种子数据集: base_seed={self.dataset.base_seed}, episodes={len(self.dataset)}")
        
        print(f"环境: {self.env.num_uavs} UAVs, {len(self.env.requests)} Requests, {self.env.num_time_slots} Time Slots")
        print(f"最大步数限制: {self.env.max_steps_per_episode}")
        
        # 记录运行时配置（追加到已有的config.json中）
        runtime_config = {
            'runtime': {
                'total_episodes': total_episodes,
                'num_episodes_per_update': num_episodes_per_update,
                'save_interval': save_interval,
                'eval_interval': eval_interval,
                'device': str(self.device),
                'dataset_seed': self.dataset.base_seed if self.dataset else None,
                'num_uavs': self.env.num_uavs if self.env else None,
                'num_requests': len(self.env.requests) if self.env and hasattr(self.env, 'requests') else None,
            }
        }
        self.logger.log_config_append(runtime_config)
        
        start_time = time.time()
        update_stats = {}
        eval_every_updates = max(1, eval_interval // num_episodes_per_update)
        save_every_updates = max(1, save_interval // num_episodes_per_update)
        
        for episode in range(0, total_episodes, num_episodes_per_update):
            print(f"\n[Train] Episode {episode}/{total_episodes}")
            
            # 收集轨迹
            print(f"  Collecting {num_episodes_per_update} episodes...")
            episode_data = self.collect_trajectories(num_episodes_per_update, start_episode=episode)
            print(f"  Collection done, buffer size: {len(self.buffer)}")
            
            # 更新策略
            if len(self.buffer) > 0:
                print(f"  Updating policy...")
                update_stats = self.update()
                print(f"  Update done")
            
            # 记录训练日志
            avg_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0
            avg_length = np.mean(self.episode_lengths) if self.episode_lengths else 0
            avg_success_rate = np.mean(self.episode_success_rates) if self.episode_success_rates else 0
            
            self.logger.log_episode(
                episode=episode,
                metrics={
                    'reward': round(avg_reward, 4),
                    'success_rate': round(avg_success_rate, 4),
                    'episode_length': round(avg_length, 2),
                    'policy_loss': round(update_stats.get('policy_loss', 0), 6),
                    'value_loss': round(update_stats.get('value_loss', 0), 6),
                    'entropy': round(update_stats.get('entropy', 0), 6),
                }
            )
            
            # 打印训练进度
            if (episode // num_episodes_per_update) % 10 == 0:
                elapsed = time.time() - start_time
                print(f"Episode {episode}/{total_episodes} | "
                      f"Avg Reward: {avg_reward:.2f} | "
                      f"Avg Success Rate: {avg_success_rate:.2%} | "
                      f"Avg Length: {avg_length:.1f} | "
                      f"Time: {elapsed:.1f}s")
                
                if update_stats:
                    print(f"  Policy Loss: {update_stats['policy_loss']:.4f} | "
                          f"Value Loss: {update_stats['value_loss']:.4f} | "
                          f"Entropy: {update_stats['entropy']:.4f}")
            
            # 评估
            if (episode // num_episodes_per_update) % eval_every_updates == 0 and episode > 0:
                eval_reward, eval_success_rate = self.evaluate(num_episodes=10)
                print(f"[Eval] Avg Reward: {eval_reward:.2f} | Success Rate: {eval_success_rate:.2%}")
                
                # 记录评估结果
                self.logger.log_evaluation(
                    episode=episode,
                    eval_metrics={
                        'avg_reward': eval_reward,
                        'avg_success_rate': eval_success_rate,
                    }
                )
            
            # 保存模型
            if (episode // num_episodes_per_update) % save_every_updates == 0 and episode > 0:
                save_path = os.path.join(save_dir, f"policy_episode_{episode}.pt")
                torch.save(self.policy.state_dict(), save_path)
                self.logger.log_checkpoint(episode, save_path, {'success_rate': avg_success_rate})
                print(f"模型已保存到 {save_path}")
        
        # 保存最终模型
        final_save_path = os.path.join(save_dir, "policy_final.pt")
        torch.save(self.policy.state_dict(), final_save_path)
        self.logger.log_checkpoint(total_episodes, final_save_path)
        
        # 关闭日志
        self.logger.close()
        
        print(f"训练完成！最终模型已保存到 {final_save_path}")
    
    def evaluate(self, num_episodes: int = 10) -> Tuple[float, float]:
        """
        评估当前策略
        
        Args:
            num_episodes: 评估的 episode 数量
        
        Returns:
            (平均奖励, 平均成功率)
        """
        rewards = []
        success_rates = []
        
        for i in range(num_episodes):
            # 如果使用数据集，加载对应回合的数据
            if self.dataset is not None:
                episode_data = self.dataset[i % len(self.dataset)]
                self.env.set_episode_data(episode_data)
            
            obs = self.env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                obs_tensor = self._obs_to_tensor(obs)
                mask = self.env.get_action_mask()
                mask_tensor = {
                    'request': torch.tensor(mask['request'], dtype=torch.bool).unsqueeze(0).to(self.device),
                    'uav': torch.tensor(mask['uav'], dtype=torch.bool).unsqueeze(0).to(self.device),
                    'uav_by_request_vnf1': torch.tensor(mask['uav_by_request_vnf1'], dtype=torch.bool).unsqueeze(0).to(self.device),
                    'uav_by_request_vnf2': torch.tensor(mask['uav_by_request_vnf2'], dtype=torch.bool).unsqueeze(0).to(self.device),
                    'uav_by_request': torch.tensor(mask['uav_by_request'], dtype=torch.bool).unsqueeze(0).to(self.device),
                }
                
                with torch.no_grad():
                    output = self.policy(obs_tensor, mask_tensor, deterministic=True)
                
                action = output['action']
                action_np = {
                    'request_idx': action['request_idx'].cpu().numpy()[0],
                    'uav_for_vnf1': action['uav_for_vnf1'].cpu().numpy()[0],
                    'uav_for_vnf2': action['uav_for_vnf2'].cpu().numpy()[0],
                }
                
                obs, reward, done, info = self.env.step(action_np)
                episode_reward += reward
            
            rewards.append(episode_reward)
            if 'final_success_rate' in info:
                success_rates.append(info['final_success_rate'])
        
        return np.mean(rewards), np.mean(success_rates) if success_rates else 0
