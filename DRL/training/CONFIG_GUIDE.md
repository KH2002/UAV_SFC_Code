# 训练配置指南

## 快速开始

### 小规模场景（推荐用于调试）

```bash
cd /mnt/sdb11/HK/UAV_SFC_code/DRL/training
python train.py --config config_small.yaml
```

**场景规模：**
- 区域：500m x 500m
- 10 UAVs，20 请求，20 监控点
- 训练时间：约 30-60 分钟
- 预期成功率：80%+

### 中等规模场景（平衡选择）

```bash
python train.py --config config_medium.yaml
```

**场景规模：**
- 区域：800m x 800m
- 30 UAVs，60 请求，50 监控点
- 训练时间：约 2-4 小时
- 预期成功率：70%+

### 大规模场景（默认配置）

```bash
python train.py
```

**场景规模：**
- 区域：1000m x 1000m
- 80 UAVs，200 请求，200 监控点
- 训练时间：约 6-12 小时
- 预期成功率：60%+

---

## 配置参数说明

### scene（场景参数）

| 参数 | 小规模 | 中等规模 | 大规模 | 说明 |
|------|--------|----------|--------|------|
| `area_size` | 500 | 800 | 1000 | 区域大小（米） |
| `num_locations` | 20 | 50 | 200 | 监控点数量 |
| `num_uavs` | 10 | 30 | 80 | UAV数量 |
| `num_requests` | 20 | 60 | 200 | 请求数量 |
| `num_time_slots` | 4 | 4 | 4 | 时间槽数 |

**建议：**
- 调试算法时先用小规模（10 UAVs, 20 请求）
- 验证想法时用中等规模
- 最终实验用大规模

### ppo（训练参数）

| 参数 | 小规模 | 说明 |
|------|--------|------|
| `lr` | 3e-4 | 学习率，小规模可以更高 |
| `entropy_coef` | 0.02 | 探索系数，小规模需要更多探索 |
| `batch_size` | 64 | 批次大小，匹配场景规模 |
| `num_episodes_per_update` | 5 | 更新频率，小规模可以更频繁 |

### training（训练控制）

| 参数 | 小规模 | 说明 |
|------|--------|------|
| `total_episodes` | 1000 | 总训练回合 |
| `max_steps_per_episode` | 200 | 每回合最大步数 |
| `eval_interval` | 50 | 评估间隔（回合数） |

### env（环境参数）

| 参数 | 小规模 | 说明 |
|------|--------|------|
| `max_steps_per_slot` | 50 | 每时隙最大步数 |
| `invalid_action_penalty` | 1.0 | 无效动作惩罚（重要！） |

---

## 自定义配置

你可以基于现有配置创建自己的配置：

```yaml
# my_config.yaml
scene:
  num_uavs: 15
  num_requests: 30
  num_locations: 25

ppo:
  lr: 5e-4
  total_episodes: 500
```

然后运行：

```bash
python train.py --config my_config.yaml
```

---

## 常见问题

### Q: 小规模场景成功率应该多少？

**A:** 小规模场景（10 UAVs, 20 请求）理论成功率应该达到 **80-90%**，如果低于 70% 说明算法有问题。

### Q: 为什么小规模场景也需要 1000 个 episode？

**A:** 虽然场景小，但策略网络仍有约 400 万参数需要学习。如果收敛快，可以提前停止（成功率 plateau）。

### Q: 如何判断是否需要调整参数？

**A:** 观察训练日志：
- `success_rate` 一直很低（<30%）：增大 `entropy_coef` 或减小 `invalid_action_penalty`
- `episode_length` 总是最大值：检查 `max_steps_per_slot` 是否合理
- `policy_loss` 震荡很大：减小 `lr`

### Q: 可以从头开始创建配置吗？

**A:** 可以！配置文件只需要包含你想覆盖的参数，其他会使用 `train.py` 中的默认值。

---

## 配置文件模板

```yaml
# 最小化配置示例
scene:
  num_uavs: 10
  num_requests: 20
  
ppo:
  lr: 3e-4
  
training:
  total_episodes: 500
```

这个配置会：
- 使用小规模场景
- 自定义学习率
- 减少训练回合
- 其他参数使用默认值
