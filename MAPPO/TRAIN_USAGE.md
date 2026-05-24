# MAPPO 训练脚本使用说明
课程学习脚本：
nohup python MAPPO/train_curriculum.py --config MAPPO/config_small_curriculum.yaml > output_curriculum.log 2>&1 &


绘图脚本：
python MAPPO/plot_logs.py --log-dir MAPPO/output/mappo_seed42_20260510_141101/log --smooth 15

python MAPPO/plot_logs.py \
  --log-dirs \
  MAPPO/output/atten_200wstep_data100_8uav/log \
  MAPPO/output/atten_200wstep_data100_8uav_layernorm/log \
  MAPPO/output/atten_200wstep_data100_8uav_layernorm_warmup/log \
  MAPPO/output/atten_200wstep_data100_8uav_layernorm_warmup_提高网络层数/log \
  --smooth 20 \
  --out-dir MAPPO/output/plots_compare
数据集测试脚本：
python run_mpoploc_eval.py --config MAPPO/config_small.yaml

checkpoint测试：
python MAPPO/test_checkpoint.py \
  --config MAPPO/config_small.yaml \
  --checkpoint MAPPO/output/atten_200wstep_data100_8uav_layernorm_warmup/checkpoint/policy_step800000.pt

本文档说明如何使用 `MAPPO/train.py` 启动训练。

## 1. 脚本位置

- 训练入口：`MAPPO/train.py`
- 默认配置：`MAPPO/config_small.yaml`

## 2. 快速开始

在仓库根目录执行：

```bash
python MAPPO/train.py
```

这会使用默认参数：
- `--config MAPPO/config_small.yaml`
- `--seed 42`
- 其余训练超参数来自配置文件

## 3. 常用启动命令

### 3.1 指定配置文件

```bash
python MAPPO/train.py --config MAPPO/config_small.yaml
CUDA_VISIBLE_DEVICES=1 nohup python MAPPO/train.py --config MAPPO/config_small.yaml > output.log 2>&1 &
```

### 3.2 指定随机种子

```bash
python MAPPO/train.py --config MAPPO/config_small.yaml --seed 123
```

### 3.3 临时覆盖训练步数/rollout

```bash
python MAPPO/train.py --config MAPPO/config_small.yaml --total-timesteps 200000 --rollout-steps 1024
```

### 3.4 指定设备

```bash
python MAPPO/train.py --config MAPPO/config_small.yaml --device cuda
```

如果你的环境没有可用 GPU，请改为：

```bash
python MAPPO/train.py --config MAPPO/config_small.yaml --device cpu
```

### 3.5 自定义 checkpoint 文件名

```bash
python MAPPO/train.py --config MAPPO/config_small.yaml --save-name exp1.pt
```

## 4. 命令行参数说明

`python MAPPO/train.py --help` 可查看完整帮助。

当前支持参数：
- `--config`：训练配置 YAML 路径（默认 `MAPPO/config_small.yaml`）
- `--seed`：随机种子（默认 `42`）
- `--device`：训练设备（如 `cpu` / `cuda`）
- `--total-timesteps`：覆盖总训练环境步数
- `--rollout-steps`：覆盖每次 rollout 的采样步数
- `--log-interval`：控制台打印间隔（按 update 次数）
- `--step-log`：实时输出每个环境 step 的日志（适合调试）
- `--save-name`：checkpoint 文件名（不填则自动按时间戳命名）

## 5. 配置文件结构

训练脚本会读取同一个 YAML 文件中的多个配置块：

- `scene`：场景规模（UAV 数、请求数、时隙数等）
- `env`：环境参数（`max_pending`、惩罚/奖励 shaping 等）
- `ppo`：PPO 超参数（学习率、clip、batch size 等）
- `network`：网络结构参数（`hidden_dim`、`num_heads`、`num_encoder_layers`、`dropout`）
- `training`：训练过程参数（如 `total_episodes`，用于估算总步数）
  - `training.max_step`：训练总环境步数上限（推荐）
  - `training.save_step`：每隔多少 `env_step` 周期保存 checkpoint
- `logging`：输出目录（日志和 checkpoint）
  - `logging.step_log: true/false`：是否默认输出每个 step 的实时控制台日志
  - `logging.record_step_metrics: true/false`：是否写入 `step_metrics.csv`

## 6. 输出结果位置

默认情况下（以 `config_small.yaml` 为例）：

- 运行根目录：`MAPPO/output/<experiment_name>/`
- 模型权重：`MAPPO/output/<experiment_name>/checkpoint/`
- 实验日志目录：`MAPPO/output/<experiment_name>/log/`

实验日志目录结构与 `DRL/training/logs/...` 风格一致，包含：
- `config.json`
- `training_log.csv`
- `checkpoints_log.csv`
- `summary.json`
- `train_update_log.yaml`（MAPPO 额外保存的 update 详细记录）
- `step_metrics.csv`（可选，每 step：`reward/invalid_count/action_mask_density/...`，由 `logging.record_step_metrics` 控制）
- `update_metrics.csv`（每 update：`loss/policy_loss/value_loss/entropy/kl_div/...`）
- `episode_metrics.csv`（每 episode：`success_rate/total_completed/avg_reward/...`）

训练完成后终端会打印两条路径：
- `checkpoint=...`
- `logs saved to ...`

## 7. 最小验证命令（快速 smoke test）

先用很小步数验证脚本可运行：

```bash
python MAPPO/train.py --config MAPPO/config_small.yaml --total-timesteps 8 --rollout-steps 4 --log-interval 1
```

## 8. 常见问题

### 8.1 看到 gym 警告是否影响训练？

如果仅出现 `Gym has been unmaintained...` 提示，通常不影响当前训练流程，可先忽略。

### 8.2 为什么训练很慢？

可先降低以下参数做调试：
- `scene.num_uavs`
- `scene.num_requests`
- `env.max_steps_per_slot`
- `--total-timesteps`

确认流程正确后再逐步恢复到正式规模。
