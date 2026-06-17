# MADRL2Stages 算法概述

## 1. 算法定位

`MADRL2Stages` 目录实现的是“多智能体强化学习 + 规则部署”的混合算法。整体仍采用参数共享 MAPPO / PPO 训练框架，但将 agent 的动作从“直接认领 VNF”改为“选择一个 location”。当所有 UAV 在当前 slot 完成 location 选择后，环境再使用规则或策略给出的 VNF 部署分数，自动完成可行 SFC 的 VNF 分配。

因此，该算法的核心思想是：强化学习负责学习 UAV 去哪些位置，规则模块负责在已选择的位置集合上决定部署哪些 VNF。

关键文件：

- `MADRL2Stages/envs/env.py`：location 选择环境、规则部署、动作掩码、奖励和时隙推进。
- `MADRL2Stages/models/policy.py`：策略网络，除 location 动作外，还可输出 VNF 部署分数。
- `MADRL2Stages/src/training/trainer.py`：rollout 收集、slot team reward 重分配、PPO 更新。
- `MADRL2Stages/src/training/train.py`：训练入口。
- `MADRL2Stages/config/config_small.yaml`：主要训练和环境配置。

## 2. 算法流程

1. 初始化场景数据，包括 UAV、SFC 请求、VNF、位置与通信需求。
2. 每个 episode 从第 1 个 time slot 开始，清空当前 slot 的 UAV location 选择和 VNF 部署记录。
3. 在每个 slot 内，UAV 按顺序各行动一次。
4. 当前 UAV 选择一个离散动作 `action`，环境将其解码为 `location_id = action + 1`。
5. 环境检查该 location 对当前 UAV 是否可行：该位置是否存在可服务 VNF，且 UAV 的 CPU、能量、busy 状态等约束满足。
6. 若 location 可行，则 UAV 飞到该位置，扣除飞行能耗，并锁定本 slot 的 location。
7. 当所有 UAV 都选择完 location 后，环境进入规则部署阶段：遍历当前可操作请求，寻找两个 VNF 分别可由两个不同 UAV 服务的候选 SFC。
8. 规则模块对候选进行排序：如果策略提供 `deploy_score_table`，则使用策略分数；否则使用能量、CPU、通信占用率构造的规则分数。
9. 选择最高分候选并调用认领逻辑部署两个 VNF，扣除服务能耗和 CPU；循环执行直到没有可行候选。
10. slot 结算时，双 VNF 均部署成功的请求被标记为完成。
11. 推进到下一时隙：空闲 UAV 返航，基站 UAV 充电，所有 UAV 恢复 CPU 并清除 busy 状态。
12. trainer 收集轨迹，并可在 slot 结束后把团队级完成/部署奖励重分配给该 slot 内的动作，再执行 PPO 更新。

## 3. MDP / POSG 建模

该算法同样是多智能体部分可观测建模，采用 CTDE 训练。与 `MAPPO` 的差异在于动作语义从 VNF 认领变成 location 选择，VNF 分配由环境规则完成。

### 3.1 智能体

- 每台 UAV 是一个智能体。
- 所有 UAV 共享策略参数。
- 每个 UAV 在一个 slot 内选择一个 location。

### 3.2 状态 / 观测空间

`MAPPOSFCEnv.get_obs()` 返回：

- `agent_self`：所有 UAV 的特征，包括位置、能量、CPU、busy 状态、到基站距离等。
- `task_matrix`：前 `max_pending` 个待处理 SFC 的 VNF 特征、位置、状态等。
- `vnf_location_ids`：每个可观察请求的两个 VNF 对应的 location action id，用于把 VNF 与 location 动作关联起来。
- `context`：当前时隙、轮次、turn、完成率、待处理比例等。
- `action_mask`：当前 UAV 的可行 location 掩码。
- `action_masks_all`：所有 UAV 的可行 location 掩码。
- `global_state`：中心化 Critic 使用的全局状态。

### 3.3 动作空间

单个 agent 的动作空间大小为：

```text
|A| = num_locations
```

动作解码为：

```text
location_id = action + 1
```

基站 `location_id = 0` 不作为部署动作；本模式不使用 `END_TOKEN`。

### 3.4 动作掩码与约束

`get_action_mask()` 会屏蔽不可行 location，主要规则包括：

- 已被其他 UAV 在当前 slot 选择的位置不可重复选择。
- 该 location 必须存在当前 UAV 可服务的 VNF。
- UAV 必须满足 CPU 约束。
- UAV 必须满足飞行、服务和返航预留能量约束。
- busy UAV 不能跨位置移动。
- 同一 SFC 的两个 VNF 最终必须由不同 UAV 服务。

当掩码全 0 时，代码会放宽为可选择未被其他 UAV 选择的位置，避免策略分布异常；这属于训练稳定性处理。

### 3.5 规则部署

规则部署由 `_deploy_slot_by_rule()` 完成：

- 在所有 UAV 的 selected location 上搜索可行候选。
- 一个候选由 `(request_idx, agent0, agent1)` 构成，其中 `agent0` 服务 VNF0，`agent1` 服务 VNF1，且二者不同。
- 若提供 `deploy_score_table`，用策略网络输出的两个 VNF 分数之和排序。
- 否则使用 `_rule_score()`，基于能量占用率、CPU 占用率和通信链路占用率计算分数。
- 每次选择最高分可行候选部署，并循环直到没有可行候选。

### 3.6 奖励函数

奖励由 `MAPPOSFCEnv.step()` 计算，包括：

- 可行 location 奖励：选择可用于部署的 location 得分。
- SFC 潜力奖励：所选 location 可参与形成完整 SFC 候选时给奖励。
- VNF 部署奖励：规则阶段成功部署 VNF 时给奖励。
- SFC 完成奖励：slot 结算完成请求时给较大奖励。
- 空 slot 惩罚：slot 结束但没有完成 SFC 时可扣分。
- 非法动作惩罚：选择不可行 location 时扣分。
- 终局成功率 bonus：episode 结束按成功率给奖励。

训练器还支持 slot team reward sharing，把 slot 级部署和完成奖励重分配给该 slot 中的相关决策，缓解多智能体延迟奖励问题。

### 3.7 终止条件

满足任一条件即结束：

- 当前时隙超过 `num_time_slots`。
- 所有请求均完成。
- `episode_step` 达到 `max_steps_per_episode`。

## 4. 与 MPopLoc 问题设定的一致性检查

### 4.1 一致部分

- 都将位置作为关键决策对象，强调 UAV 到监控位置部署 VNF。
- 都在部署前检查 CPU、能量、飞行、服务和返航预留约束。
- 都要求同一 SFC 的两个 VNF 由不同 UAV 服务。
- 都按时隙推进，并在新时隙恢复计算资源、处理返航和充电。
- 都使用规则排序思想决定哪些可行部署优先执行。

### 4.2 不一致或需注意部分

- `mpoploc.py` 先按请求的 location popularity 排序，再对每个请求执行资源承诺；`MADRL2Stages` 先由 agent 选择 location，再在这些 location 上搜索和排序可部署 SFC。
- `mpoploc.py` 的决策对象是请求，`MADRL2Stages` 的 RL 动作对象是 location，VNF/SFC 选择由规则或策略分数后处理完成。
- `MADRL2Stages` 使用 `max_pending` 观测窗口；`mpoploc.py` 每个时隙考虑全部未服务请求。
- `MADRL2Stages` 的规则分数使用能量、CPU 和通信占用率的 log margin；`mpoploc.py` 的优先级主要来自请求流行度和最小旅行成本。
- `MADRL2Stages` 的奖励包含多个训练塑形项，不是单纯最大化启发式完成数量。
- 动作掩码全 0 时会被放宽，这可能让策略在极端状态下选择理论上不可行的 location，然后通过非法惩罚修正；这与启发式算法的严格可行性筛选不同。
- 环境时隙推进中对 busy UAV 额外扣除估计悬停能耗，而 `mpoploc.py` 已在服务能耗中计入服务相关能耗，二者能耗统计口径需在实验对比中说明。

## 5. 小结

`MADRL2Stages` 是三种强化学习方案中与 `mpoploc.py` 最接近的一版，因为它保留了“位置优先 + 规则部署”的思想。但它不是 MPopLoc 的直接神经化复刻：RL 学习的是 UAV 选址策略，规则模块负责在已选位置集合上完成 VNF/SFC 部署。用于实验对比时，应强调它与启发式算法的问题约束一致度较高，但请求排序、候选选择和奖励目标存在实现差异。
