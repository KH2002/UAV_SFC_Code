# MAPPO 两周工作总结（2026-05-06 ~ 2026-05-20）

## 1. 工作总览
本周期围绕 MAPPO 在 UAV-SFC 任务上的稳定性与泛化能力开展了以下工作：

1. MLP 与 Attention 在固定环境（data1）下对比训练  
2. MLP 与 Attention 在多环境（data100）下对比训练  
3. 引入 LayerNorm 并训练  
4. 引入 warmup+cosine 学习率调度并训练  
5. 提高网络层数并训练  
6. 引入课程学习（curriculum）并训练  
7. 额外完成：周期性 deterministic eval、`eval_metrics.csv`、多实验对比绘图脚本

---

## 2. 关键实验结果（统一口径）
统计口径：
- `mean`：全程 `episode_success_rate` 平均值
- `last100`：最后 100 个 episode 的平均成功率
- `best_eval`：`eval_metrics.csv` 中 `eval_avg_success_rate` 最大值（若有）

| 实验 | 主要配置 | mean | last100 | best_eval |
|---|---|---:|---:|---:|
| fixed_attn | data1, attn | 0.6775 | 0.7770 | - |
| fixed_mlp | data1, mlp | 0.5882 | 0.7180 | - |
| multi_attn | data100, attn | 0.4370 | 0.4517 | - |
| multi_mlp | data100, mlp | 0.2754 | 0.2813 | - |
| layernorm_attn | data100, attn+layernorm | 0.4299 | 0.4433 | - |
| warmup_layernorm_attn | data100, attn+layernorm+warmup | 0.4109 | 0.4427 | 0.4400 |
| deeper_layernorm_warmup_attn | data100, 3层+layernorm+warmup | 0.4036 | 0.4347 | 0.4333 |
| curriculum | data100, curriculum | 0.4594 | 0.4683 | 0.4700 |

说明：部分实验训练预算不同（例如 `max_step` 不一致），以上结果用于阶段性趋势判断，不完全是严格公平对比。

---

## 3. 分项总结（含图片）

### 3.1 固定环境：MLP vs Attention
结论：在固定环境中，Attention 明显优于 MLP，且更容易达到高平台。

- Attention 图：![fixed_attn](output/mappo_atten_data1/log/plots/1_success_rate_vs_episode.png)
- MLP 图：![fixed_mlp](output/mappo_MLP_data1_30uav/log/plots/1_success_rate_vs_episode.png)

---

### 3.2 多环境：MLP vs Attention
结论：切换到多环境后，两者都下降，且 MLP 降幅更大；Attention 仍保持明显优势。

- Attention 图：![multi_attn](output/atten_200wstep_data100_8uav/log/plots/1_success_rate_vs_episode.png)
- MLP 图：![multi_mlp](output/MLP_30wsteps_data100/log/plots/1_success_rate_vs_episode.png)

---

### 3.3 引入 LayerNorm
结论：LayerNorm 改善了训练稳定性，但最终成功率提升有限，主要收益偏向“稳”而非“高”。

- 图：![layernorm_attn](output/atten_200wstep_data100_8uav_layernorm/log/plots/1_success_rate_vs_episode.png)

---

### 3.4 引入 Warmup（含余弦退火）
结论：warmup+cosine 后曲线更平滑，eval 结果稳定在约 0.42~0.44 区间，未显著抬高上限。

- 图：![warmup_layernorm_attn](output/atten_200wstep_data100_8uav_layernorm_warmup/log/plots/1_success_rate_vs_episode.png)
- eval 指标文件：[`output/atten_200wstep_data100_8uav_layernorm_warmup/log/eval_metrics.csv`](output/atten_200wstep_data100_8uav_layernorm_warmup/log/eval_metrics.csv)

---

### 3.5 提高网络层数（2层 -> 3层）
结论：在当前设置下，增加层数未带来明显收益，反而略有下降（可能与训练预算/优化难度有关）。

- 图：![deeper_layernorm_warmup_attn](output/atten_200wstep_data100_8uav_layernorm_warmup_提高网络层数/log/plots/1_success_rate_vs_episode.png)
- 结果目录：[`output/atten_200wstep_data100_8uav_layernorm_warmup_提高网络层数/log`](output/atten_200wstep_data100_8uav_layernorm_warmup_提高网络层数/log)
- 评估文件：[`output/atten_200wstep_data100_8uav_layernorm_warmup_提高网络层数/log/eval_metrics.csv`](output/atten_200wstep_data100_8uav_layernorm_warmup_提高网络层数/log/eval_metrics.csv)

---

### 3.6 课程学习（Curriculum）
结论：课程学习是本周期最有效改进项之一，`last100` 与 `best_eval` 均高于非课程版 warmup/layernorm。

- 对比图（课程学习实验目录）：![curriculum_compare](output/mappo_curr_seed42_20260520_000254/log/plots_compare/1_success_rate_vs_episode.png)
- 评估文件：[`output/mappo_curr_seed42_20260520_000254/log/eval_metrics.csv`](output/mappo_curr_seed42_20260520_000254/log/eval_metrics.csv)

---

## 4. 阶段结论
1. 从固定环境到多环境，性能下降是主要矛盾，核心问题是泛化而不是单次收敛速度。  
2. Attention 架构在本任务下明显优于 MLP（无论固定环境还是多环境）。  
3. LayerNorm 和 warmup 更偏训练稳定性改进，对上限提升有限。  
4. 简单加深网络（3层）在当前配置下未见正收益。  
5. 课程学习目前是最有价值的方向，建议继续细化阶段划分和阶段步数分配。  

---

## 5. 下阶段建议
1. 做严格同预算 ablation（固定 `max_step`、`rollout_steps`、`batch_size`、seed）仅改单一变量。  
2. 对课程学习做网格：`stage_dataset_episodes` 与 `stage_step_ratios`。  
3. 统一使用 `eval_metrics.csv` 的 deterministic 指标作为主比较口径。  
4. 补充对“提高网络层数”实验的 loss/eval 曲线联动分析，定位是优化问题还是容量问题。  
