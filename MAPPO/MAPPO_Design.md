# UAV-SFC 部署问题的 MAPPO 方案设计

## 1. 问题建模

### 1.1 问题概述
- **智能体（Agents）**：每台 UAV 作为一个独立智能体，参数共享。
- **环境（Environment）**：多 UAV 协同服务 SFC 请求池，包含时隙推进、资源消耗与约束判定。
- **目标（Objective）**：在给定总时隙内，最大化 SFC 完成率，同时减少无效决策与资源冲突。

### 1.2 训练范式
采用 **CTDE（Centralized Training, Decentralized Execution）**：
- **Actor（共享）**：每个 UAV 基于本地观测输出动作分布。
- **Critic（中心化）**：训练时读取全局状态，估计联合策略价值。

### 1.3 决策流程抽象
一个 episode 被划分为多个时间槽（slot）。每个 slot 内进行多轮顺序决策：
1. 各 UAV 轮流决策（或并行后按规则解析冲突）。
2. 对合法动作更新“本时隙临时认领池”。
3. 达到轮次上限或所有 UAV 输出结束动作后，结算本时隙完成的 SFC。
4. 推进到下一个时隙并刷新时隙相关状态。

---

## 2. 配置驱动设计

所有规模参数和训练参数均从配置读取，不在代码中写死。

### 2.1 核心配置项

| 类别 | 配置项 | 记号 | 说明 |
|------|--------|------|------|
| 场景 | `num_uavs` | \(N_u\) | UAV 数量 |
| 场景 | `num_requests` | \(N_r\) | 请求总量 |
| 场景 | `num_time_slots` | \(T\) | 时间槽数量 |
| 场景 | `num_locations` | \(N_l\) | 监控点数量 |
| 场景 | `area_size` | \(L\) | 地图边长 |
| 环境 | `max_pending` | \(P\) | 观测窗口内最多保留的请求数 |
| 环境 | `max_rounds_per_slot` | \(R_{slot}\) | 每个 slot 最大子轮数 |
| 训练 | `total_episodes` | \(E\) | 训练回合数 |
| 训练 | `num_episodes_per_update` | \(E_u\) | 每次更新采样回合数 |

### 2.2 派生规模
- 每个请求固定 2 个 VNF，则可操作 VNF 数量上限：
  \[
  N_{vnf}=2\times P
  \]
- 单 agent 动作空间维度：
  \[
  |\mathcal{A}| = N_{vnf}+1
  \]
  其中 `+1` 为结束动作 `END_TOKEN`。

---

## 3. MDP / POSG 建模

该问题更准确属于参数共享多智能体 POSG。MAPPO 中按共享策略处理。

### 3.1 局部观测（Agent Observation）
UAV \(i\) 在轮次 \(r\) 的观测：
\[
 o_i^r = [\text{self}_i,\ \text{task\_pool},\ \text{context}]
\]

#### 3.1.1 自身特征 `self_i`
1. `agent_id_embedding`（或 one-hot）
2. `pos_x, pos_y`（归一化到 `[0,1]`）
3. `energy_ratio`
4. `cpu_ratio`
5. `is_busy`
6. `dist_to_base_ratio`

### 3.1.2 任务池特征 `task_pool`
对前 `P` 个待处理 SFC 构建矩阵，每条请求包含：
- `vnf1_workload_ratio`
- `vnf1_cpu_freq_ratio`
- `vnf1_state_code`（0=未认领，1=本时隙已认领，2=已完成）
- `vnf2_workload_ratio`
- `vnf2_cpu_freq_ratio`
- `vnf2_state_code`
- `loc1_x, loc1_y`（由 Location ID 映射坐标后归一化）
- `loc2_x, loc2_y`（同上）
- `comm_demand_ratio`

> 关键变更：位置特征不再使用 `Location ID`，统一改为 `坐标 (x,y)`。

### 3.1.3 上下文特征 `context`
- `current_slot_ratio = current_slot / T`
- `remaining_slot_ratio = (T-current_slot)/T`
- `completed_ratio = completed_count / N_r`

### 3.2 中心化状态（Critic State）
Critic 输入全局状态：
\[
 s^r=[\text{all\_uav\_states},\ \text{full\_task\_pool},\ \text{global\_context}]
\]
其中包含所有 UAV 特征、全请求状态矩阵和时隙上下文。

### 3.3 动作空间（保持不变）
每个 UAV 的离散动作：
\[
 a_i \in \{0,1,\dots,N_{vnf}-1, N_{vnf}\}
\]
- `0 ... N_vnf-1`：认领一个具体 VNF（通过扁平索引映射到 `(request_idx, vnf_idx)`）。
- `N_vnf`：`END_TOKEN`，表示本 slot 不再执行新任务。

动作映射示例：
```python
def decode_action(a, max_pending):
    # a in [0, 2*max_pending-1] => (req_idx, vnf_idx)
    req_idx = a // 2
    vnf_idx = a % 2   # 0 or 1
    return req_idx, vnf_idx
```

### 3.4 状态转移
执行动作后，环境执行：
1. 合法性检查（资源、位置、冲突、时隙约束）。
2. 合法则写入临时认领池并更新 UAV 资源与位置。
3. 非法则不更新任务池，仅记录无效动作。
4. 在 slot 结束阶段统一判断哪些 SFC 在本 slot 成功完成。

---

## 4. 约束建模

### 4.1 基础资源约束
对动作 `claim(vnf_k)`，要求：
- `cpu_remaining(uav_i) >= workload(vnf_k)`
- `energy_remaining(uav_i) >= travel + service + reserve_return`

### 4.2 位置与机动约束
- 若 `is_busy=True`，UAV 不能切换到不同位置执行新 VNF。
- 同一 UAV 在同一 slot 内不能同时认领不同地点的 VNF。

### 4.3 SFC 完成一致性约束
一个 SFC 仅在以下条件满足时计为完成：
1. 该 SFC 两个 VNF 均在本 slot 被认领；
2. 两个 VNF 的认领满足资源约束；

### 4.4 结束动作约束
- `END_TOKEN` 始终合法；
- 当 agent 在当前 slot 已无法合法认领任何 VNF 时，应优先输出 `END_TOKEN`。

---

## 5. 动作掩码设计

在 actor softmax 前对非法动作置 `-inf`。

### 5.1 掩码类型
1. **资源掩码**：CPU、电量、返航裕量不满足则屏蔽。
2. **状态掩码**：`已完成` / `本时隙已被他人认领` 的 VNF 屏蔽。
3. **同步掩码**：若当前 UAV 在本 slot 已锁定位置，则屏蔽其他位置 VNF。
4. **结构掩码**：越界请求索引（`req_idx >= pending_count`）屏蔽。
5. **结束动作位**：始终置 1。

### 5.2 掩码伪代码
```python
def build_action_mask(agent_i, slot_state, cfg):
    n_vnf = 2 * cfg.max_pending
    mask = np.zeros(n_vnf + 1, dtype=np.int32)

    for a in range(n_vnf):
        req_idx, vnf_idx = decode_action(a, cfg.max_pending)
        if req_idx >= slot_state.pending_count:
            continue
        if is_vnf_completed_or_claimed(req_idx, vnf_idx, slot_state):
            continue
        if not cpu_feasible(agent_i, req_idx, vnf_idx, slot_state):
            continue
        if not energy_feasible(agent_i, req_idx, vnf_idx, slot_state):
            continue
        if violates_same_slot_mobility(agent_i, req_idx, vnf_idx, slot_state):
            continue
        mask[a] = 1

    mask[n_vnf] = 1  # END_TOKEN
    return mask
```

---

## 6. 奖励函数设计

### 6.1 目标导向
奖励以“提升最终完成率”为主，避免稀疏到不可学习，也避免过度 shaping 偏离目标。

### 6.2 推荐奖励分解
每个 agent 每步奖励：
\[
 r_i = r_{valid} + r_{slot\_complete} + r_{episode\_success}
\]

- `r_valid`：非法动作惩罚（例如 `-c_invalid`）
- `r_slot_complete`：本 slot 新完成 SFC 的共享奖励（例如每完成 1 个 +1，团队均分或全体共享）
- `r_episode_success`：episode 结束时按最终成功率给额外奖励

示例：
```python
if action_invalid:
    r_i -= c_invalid

r_i += delta_completed_sfc_in_slot

if done:
    success_rate = completed_total / total_requests
    r_i += alpha * success_rate
```

### 6.3 团队奖励分配
建议使用**共享团队奖励**（所有 agent 接收同一团队项），减少多智能体信用冲突。

---

## 7. 神经网络架构

### 7.1 Actor（参数共享）
输入 `o_i`，输出 `N_vnf+1` logits。

1. `Self Encoder`：先用 MLP 编码 UAV 自身特征，再通过标准 Transformer block 融合群体信息：
   `Multi-Head Self-Attention -> Residual -> LayerNorm -> FFN -> Residual -> LayerNorm`。其中注意力计算使用 `agent_availability_mask`（或 padding mask），确保仅对当前有效 UAV 进行信息交互。
2. `Task Encoder`：对任务池逐条 MLP 编码。
3. `Cross-Attention`：`self_token` 查询任务池，建模“我与任务匹配度”。
4. `Policy Head`：输出离散动作 logits，并应用动作掩码。

### 7.2 Critic（中心化）
输入全局状态 `s`：
- 所有 UAV 特征编码后聚合（mean/attention pooling）
- 全任务池编码后聚合
- 上下文向量拼接

输出标量 `V(s)`。

### 7.3 参数化建议
- `hidden_dim`: 128 或 256
- `num_heads`: 4
- `encoder_layers`: 2
- `dropout`: 0.1

全部从配置读取。

---

## 8. MAPPO 训练方法

### 8.1 轨迹收集
1. 重置环境，读取配置规模 `N_u, T, R_slot`。
2. 按 slot / round / agent 循环采样动作。
3. 保存 `(o_i, a_i, logp_i, r_i, done, mask_i, s)`。

### 8.2 优势估计（GAE）
对每个 agent 轨迹计算：
\[
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
\]
\[
\hat A_t = \delta_t + \gamma\lambda\hat A_{t+1}
\]

### 8.3 PPO-Clip 更新
\[
L^{clip} = \mathbb{E}[\min(r_t(\theta)\hat A_t,\ clip(r_t(\theta),1-\epsilon,1+\epsilon)\hat A_t)]
\]
\[
L^{value}=\mathbb{E}[(V_\phi(s_t)-\hat R_t)^2],\quad
L^{entropy}=\mathbb{E}[\mathcal{H}(\pi_\theta(\cdot|o_t))]
\]
\[
L = -L^{clip} + c_v L^{value} - c_e L^{entropy}
\]

### 8.4 关键训练细节
- 参数共享 actor，按 agent 样本合并训练。
- 优势标准化。
- 梯度裁剪。
- 使用 mask 参与 `log_prob` 与 `entropy` 计算，避免策略梯度污染。

---

## 9. 环境执行逻辑（配置驱动）

### 9.1 时隙内循环
```python
for slot in range(cfg.num_time_slots):
    reset_slot_temp_pool()

    for rnd in range(cfg.max_rounds_per_slot):
        all_end = True
        for agent_id in range(cfg.num_uavs):
            obs_i = get_local_obs(agent_id)
            mask_i = build_action_mask(agent_id)
            action_i = actor.sample(obs_i, mask_i)

            if action_i != END_TOKEN:
                all_end = False
                apply_claim_if_valid(agent_id, action_i)

        if all_end:
            break

    settle_slot_completion()
    advance_time_slot()
```

### 9.2 Episode 终止条件
- `current_slot >= cfg.num_time_slots`
- 或 `pending_queue` 为空

---

## 10. 与原 DRL 方案的对齐关系

1. **动作空间**：保持离散 `N_vnf+1`，不改为 `(request, uav1, uav2)` 组合动作。
2. **位置表达**：由 Location ID 改为 `(x,y)` 坐标输入。
3. **规模与时隙**：`num_uavs / num_time_slots / max_pending / max_rounds_per_slot` 全部配置化。
4. **约束语义**：保持“资源可行 + 同时隙机动可行 + SFC双VNF完整性”的核心逻辑。

---

## 11. 实现建议（MAPPO 文件夹）

建议模块划分：
- `mappo_config.py`：配置加载与校验
- `mappo_env.py`：多智能体环境与掩码
- `mappo_models.py`：Actor/Critic 网络
- `mappo_buffer.py`：多智能体 rollout 缓冲区
- `mappo_trainer.py`：MAPPO 训练主循环
- `train.py`：训练入口

---

## 12. 默认超参数建议

| 参数 | 建议值 | 说明 |
|------|--------|------|
| `lr` | `1e-4` | Adam 学习率 |
| `gamma` | `0.99` | 折扣因子 |
| `gae_lambda` | `0.95` | GAE 系数 |
| `clip_eps` | `0.2` | PPO clip |
| `entropy_coef` | `0.01~0.02` | 探索强度 |
| `value_coef` | `0.5` | 价值损失权重 |
| `max_grad_norm` | `0.5` | 梯度裁剪 |
| `ppo_epochs` | `4` | 每轮更新次数 |
| `mini_batch_size` | `128~512` | 与样本量匹配 |

> 以上均应允许配置文件覆盖。

---

## 13. 验证与诊断指标

训练与评估阶段建议记录：
- `success_rate`（最终核心指标）
- `completed_sfc_count`
- `invalid_action_ratio`
- `avg_reward`
- `avg_energy_remaining`
- `slot_utilization`（每个 slot 实际使用轮数）

若出现“成功率长期停滞”：
1. 先检查动作掩码是否过严导致可行动作不足；
2. 再检查奖励是否被无效动作惩罚主导；
3. 最后检查配置规模是否超过当前资源可行上限。
