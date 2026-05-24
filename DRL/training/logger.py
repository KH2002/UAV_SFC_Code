# -*- coding: utf-8 -*-
"""
训练日志模块

用于记录训练过程中的各种信息：
- 奖励（reward）
- 损失（loss）
- 训练配置（config）
- 每次环境reset后的初始状态
- 评估结果
"""
import os
import json
import csv
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
import pickle


class TrainingLogger:
    """
    训练日志记录器
    
    自动创建日志目录结构：
    log_dir/
    ├── config.json          # 训练配置
    ├── training_log.csv     # 训练过程记录（episode级别的reward、loss等）
    ├── step_log.csv         # 每一步的详细记录（可选）
    ├── initial_states/      # 每次reset后的初始状态
    │   ├── episode_0000.pkl
    │   ├── episode_0001.pkl
    │   └── ...
    ├── checkpoints/         # 模型检查点
    │   └── ...
    └── summary.json         # 训练总结
    """
    
    def __init__(self, log_dir: str = "./logs", experiment_name: str = None):
        """
        初始化日志记录器
        
        Args:
            log_dir: 日志保存根目录
            experiment_name: 实验名称（如果为None则使用时间戳）
        """
        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.log_dir = os.path.join(log_dir, experiment_name)
        self.experiment_name = experiment_name
        self.start_time = time.time()
        
        # 创建目录结构
        self._create_directories()
        
        # 初始化日志文件
        self.training_log_file = None
        self.training_log_writer = None
        self.step_log_file = None
        self.step_log_writer = None
        
        # 统计信息
        self.episode_count = 0
        self.step_count = 0
        self.best_success_rate = 0.0
        self.best_reward = float('-inf')
        
    def _create_directories(self):
        """创建日志目录结构"""
        subdirs = ['initial_states', 'checkpoints']
        for subdir in subdirs:
            os.makedirs(os.path.join(self.log_dir, subdir), exist_ok=True)
        print(f"日志目录: {self.log_dir}")
    
    def log_config(self, config: Dict[str, Any]):
        """
        记录训练配置
        
        Args:
            config: 配置字典
        """
        config_path = os.path.join(self.log_dir, "config.json")
        
        # 递归处理不可序列化的对象
        def serialize_value(v):
            if isinstance(v, dict):
                return {k: serialize_value(val) for k, val in v.items()}
            elif isinstance(v, (list, tuple)):
                return [serialize_value(item) for item in v]
            elif isinstance(v, (int, float, str, bool)):
                return v
            elif v is None:
                return None
            else:
                return str(v)
        
        serializable_config = serialize_value(config)
        
        # 添加时间戳和额外信息
        serializable_config['_meta'] = {
            'experiment_name': self.experiment_name,
            'log_dir': self.log_dir,
            'start_time': datetime.now().isoformat(),
        }
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_config, f, indent=2, ensure_ascii=False)
        
        print(f"配置已保存到: {config_path}")
    
    def log_config_append(self, additional_config: Dict[str, Any]):
        """
        追加配置到已有的 config.json 文件中
        
        Args:
            additional_config: 要追加的配置字典
        """
        config_path = os.path.join(self.log_dir, "config.json")
        
        # 递归处理不可序列化的对象
        def serialize_value(v):
            if isinstance(v, dict):
                return {k: serialize_value(val) for k, val in v.items()}
            elif isinstance(v, (list, tuple)):
                return [serialize_value(item) for item in v]
            elif isinstance(v, (int, float, str, bool)):
                return v
            elif v is None:
                return None
            else:
                return str(v)
        
        # 读取现有配置
        existing_config = {}
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    existing_config = json.load(f)
            except (json.JSONDecodeError, IOError):
                existing_config = {}
        
        # 合并配置
        serialized_additional = serialize_value(additional_config)
        existing_config.update(serialized_additional)
        
        # 写回文件
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(existing_config, f, indent=2, ensure_ascii=False)
    
    def init_training_log(self, fieldnames: List[str] = None):
        """
        初始化训练日志CSV文件
        
        Args:
            fieldnames: CSV列名（如果为None则使用默认列）
        """
        if fieldnames is None:
            fieldnames = [
                'episode', 'timestamp', 'total_steps',
                'reward', 'success_rate', 'episode_length',
                'policy_loss', 'value_loss', 'entropy',
                'current_time_slot', 'completed_count', 'pending_count'
            ]
        
        log_path = os.path.join(self.log_dir, "training_log.csv")
        self.training_log_file = open(log_path, 'w', newline='', encoding='utf-8')
        self.training_log_writer = csv.DictWriter(self.training_log_file, fieldnames=fieldnames)
        self.training_log_writer.writeheader()
        self.training_log_file.flush()
        
        print(f"训练日志: {log_path}")
    
    def log_episode(self, episode: int, metrics: Dict[str, Any]):
        """
        记录每个episode的训练信息
        
        Args:
            episode: episode编号
            metrics: 包含reward、loss等信息的字典
        """
        if self.training_log_writer is None:
            self.init_training_log()
        
        # 添加时间戳和episode信息
        row = {
            'episode': episode,
            'timestamp': datetime.now().isoformat(),
            'total_steps': self.step_count,
            **metrics
        }
        
        self.training_log_writer.writerow(row)
        self.training_log_file.flush()
        
        # 更新统计
        self.episode_count = episode
        if 'success_rate' in metrics:
            self.best_success_rate = max(self.best_success_rate, metrics['success_rate'])
        if 'reward' in metrics:
            self.best_reward = max(self.best_reward, metrics['reward'])
    
    def log_initial_state(self, episode: int, initial_state: Dict):
        """
        记录每次reset后的初始状态
        
        Args:
            episode: episode编号
            initial_state: 初始状态字典（包含uavs, requests, locations等）
        """
        filename = f"episode_{episode:04d}.pkl"
        filepath = os.path.join(self.log_dir, "initial_states", filename)
        
        with open(filepath, 'wb') as f:
            pickle.dump(initial_state, f)
    
    def log_step(self, episode: int, step: int, info: Dict[str, Any]):
        """
        记录每一步的详细信息（可选，用于调试）
        
        Args:
            episode: episode编号
            step: 步数
            info: 包含action、reward、state等信息
        """
        if self.step_log_writer is None:
            step_log_path = os.path.join(self.log_dir, "step_log.csv")
            self.step_log_file = open(step_log_path, 'w', newline='', encoding='utf-8')
            self.step_log_writer = csv.DictWriter(
                self.step_log_file,
                fieldnames=['episode', 'step', 'timestamp', 'action', 'reward', 'done', 'info']
            )
            self.step_log_writer.writeheader()
        
        row = {
            'episode': episode,
            'step': step,
            'timestamp': datetime.now().isoformat(),
            **info
        }
        self.step_log_writer.writerow(row)
        self.step_count += 1
    
    def log_checkpoint(self, episode: int, checkpoint_path: str, metrics: Dict[str, Any] = None):
        """
        记录检查点信息
        
        Args:
            episode: episode编号
            checkpoint_path: 检查点文件路径
            metrics: 保存时的性能指标
        """
        checkpoint_log_path = os.path.join(self.log_dir, "checkpoints_log.csv")
        
        # 如果文件不存在，创建并写入表头
        if not os.path.exists(checkpoint_log_path):
            with open(checkpoint_log_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['episode', 'timestamp', 'path', 'metrics'])
        
        with open(checkpoint_log_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                episode,
                datetime.now().isoformat(),
                checkpoint_path,
                json.dumps(metrics) if metrics else ""
            ])
    
    def log_evaluation(self, episode: int, eval_metrics: Dict[str, Any]):
        """
        记录评估结果
        
        Args:
            episode: 训练episode编号
            eval_metrics: 评估指标
        """
        eval_log_path = os.path.join(self.log_dir, "evaluation_log.csv")
        
        if not os.path.exists(eval_log_path):
            with open(eval_log_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['episode', 'timestamp', 'avg_reward', 'avg_success_rate', 'details'])
        
        with open(eval_log_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                episode,
                datetime.now().isoformat(),
                eval_metrics.get('avg_reward', 0),
                eval_metrics.get('avg_success_rate', 0),
                json.dumps(eval_metrics)
            ])
    
    def save_summary(self):
        """保存训练总结"""
        summary = {
            'experiment_name': self.experiment_name,
            'start_time': datetime.fromtimestamp(self.start_time).isoformat(),
            'end_time': datetime.now().isoformat(),
            'total_episodes': self.episode_count,
            'total_steps': self.step_count,
            'best_success_rate': self.best_success_rate,
            'best_reward': self.best_reward,
            'duration_seconds': time.time() - self.start_time,
        }
        
        summary_path = os.path.join(self.log_dir, "summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"\n训练总结已保存到: {summary_path}")
        print(f"总回合数: {self.episode_count}")
        print(f"总步数: {self.step_count}")
        print(f"最佳成功率: {self.best_success_rate:.2%}")
        print(f"训练时长: {summary['duration_seconds']:.1f}秒")
    
    def close(self):
        """关闭日志文件"""
        if self.training_log_file:
            self.training_log_file.close()
        if self.step_log_file:
            self.step_log_file.close()
        self.save_summary()


class LoggerMixin:
    """
    日志混入类，方便在其他类中使用日志功能
    
    使用方式：
        class MyTrainer(LoggerMixin):
            def __init__(self):
                super().__init__(log_dir="./logs")
    """
    
    def __init__(self, log_dir: str = "./logs", experiment_name: str = None):
        self.logger = TrainingLogger(log_dir, experiment_name)
    
    def log_config(self, config: Dict):
        self.logger.log_config(config)
    
    def log_episode(self, episode: int, metrics: Dict):
        self.logger.log_episode(episode, metrics)
    
    def log_initial_state(self, episode: int, initial_state: Dict):
        self.logger.log_initial_state(episode, initial_state)
    
    def log_evaluation(self, episode: int, eval_metrics: Dict):
        self.logger.log_evaluation(episode, eval_metrics)
    
    def close_logger(self):
        self.logger.close()
