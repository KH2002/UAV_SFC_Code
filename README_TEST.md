# 算法测试脚本使用说明

## 概述

本测试脚本用于对比测试 MILP、MPopLoc、RandomOrder 三种算法，支持以下可变参数：
- **无人机(UAV)数量**
- **SFC请求数量**
- **区域大小**

测试结果会自动保存到 CSV 文件中，方便后续分析。

## 文件说明

### 核心文件
- `test_algorithms.py` - 主测试框架，包含所有测试逻辑
- `custom_test.py` - 自定义测试脚本，方便用户快速配置测试参数
- `README_TEST.md` - 本说明文档

## 快速开始

### 方法1: 使用自定义测试脚本（推荐）

1. 编辑 `custom_test.py` 文件中的 `custom_test_cases()` 函数
2. 运行测试：
```bash
python custom_test.py
```

3. 按提示确认并开始测试

### 方法2: 使用预定义测试

直接运行主测试脚本：
```bash
python test_algorithms.py
```

该脚本会自动运行预定义的测试案例集。

## 自定义测试参数

在 `custom_test.py` 中修改 `custom_test_cases()` 函数：

```python
def custom_test_cases():
    test_cases = []
    
    # 添加测试案例
    test_cases.append({
        'num_uavs': 5,          # UAV数量
        'num_requests': 20,      # SFC请求数量
        'area_size': 100.0,      # 区域大小（米）
        'time_limit': 60         # MILP时间限制（秒）
    })
    
    return test_cases
```

### 示例：测试UAV数量的影响

```python
# 固定请求数和区域大小，变化UAV数量
for num_uavs in [3, 5, 7, 9, 11]:
    test_cases.append({
        'num_uavs': num_uavs,
        'num_requests': 20,
        'area_size': 100.0,
        'time_limit': 60
    })
```

### 示例：测试请求数量的影响

```python
# 固定UAV数和区域大小，变化请求数量
for num_requests in [10, 15, 20, 25, 30]:
    test_cases.append({
        'num_uavs': 5,
        'num_requests': num_requests,
        'area_size': 100.0,
        'time_limit': 60
    })
```

### 示例：测试区域大小的影响

```python
# 固定UAV数和请求数，变化区域大小
for area_size in [60, 80, 100, 120, 140]:
    test_cases.append({
        'num_uavs': 5,
        'num_requests': 20,
        'area_size': area_size,
        'time_limit': 60
    })
```

## 输出结果

### 结果文件

所有结果保存在 `results/` 目录下：

1. **详细结果** - `detailed_results_YYYYMMDD_HHMMSS.csv`
   - 包含每次测试的完整数据
   - 适合详细分析

2. **统计摘要** - `summary_results_YYYYMMDD_HHMMSS.csv` (仅在多次运行时生成)
   - 包含多次运行的平均值和标准差
   - 适合统计分析

### 详细结果字段说明

| 字段名 | 说明 |
|--------|------|
| `test_case_id` | 测试案例编号 |
| `run_id` | 运行次数编号 |
| `num_uavs` | UAV数量 |
| `num_requests` | SFC请求数量 |
| `area_size` | 区域大小（米）|
| `num_locations` | 监控点数量 |
| `vnfs_per_request` | 每个请求的VNF数量 |
| `time_slots` | 时间槽数量 |
| `seed` | 随机种子 |
| **MILP结果** | |
| `milp_success` | MILP是否成功 (True/False) |
| `milp_serviced` | MILP服务的请求数 |
| `milp_reward` | MILP获得的总奖励 |
| `milp_runtime` | MILP运行时间（秒）|
| `milp_status` | MILP求解状态 |
| **MPopLoc结果** | |
| `mpoploc_success` | MPopLoc是否成功 |
| `mpoploc_serviced` | MPopLoc服务的请求数 |
| `mpoploc_reward` | MPopLoc获得的总奖励 |
| `mpoploc_runtime` | MPopLoc运行时间（秒）|
| `mpoploc_status` | MPopLoc求解状态 |
| **RandomOrder结果** | |
| `random_success` | RandomOrder是否成功 |
| `random_serviced` | RandomOrder服务的请求数 |
| `random_reward` | RandomOrder获得的总奖励 |
| `random_runtime` | RandomOrder运行时间（秒）|
| `random_status` | RandomOrder求解状态 |
| **性能比较** | |
| `serviced_diff` | 服务请求数差异 (MILP - MPopLoc) |
| `reward_diff` | 奖励差异 |
| `runtime_ratio` | 运行时间比率 (MPopLoc / MILP) |
| `milp_vs_random_serviced_diff` | 服务请求数差异 (MILP - RandomOrder) |
| `mpoploc_vs_random_serviced_diff` | 服务请求数差异 (MPopLoc - RandomOrder) |
| `milp_vs_random_reward_diff` | 奖励差异 (MILP - RandomOrder) |
| `mpoploc_vs_random_reward_diff` | 奖励差异 (MPopLoc - RandomOrder) |
| `random_vs_milp_runtime_ratio` | 运行时间比率 (RandomOrder / MILP) |
| `random_vs_mpoploc_runtime_ratio` | 运行时间比率 (RandomOrder / MPopLoc) |

## 高级配置

### 多次运行取平均

修改 `custom_test.py` 中的 `num_runs_per_case` 参数：

```python
df_results = run_batch_tests(
    test_cases, 
    output_dir='results',
    num_runs_per_case=3  # 每个测试案例运行3次
)
```

这样会生成统计摘要文件，包含平均值和标准差。

### 修改MILP求解时间限制

在测试案例中设置 `time_limit`：

```python
test_cases.append({
    'num_uavs': 10,
    'num_requests': 30,
    'area_size': 150.0,
    'time_limit': 120  # 120秒时间限制
})
```

### 修改其他配置参数

编辑 `config.py` 文件可以修改更多参数：
- `NUM_TIME_SLOTS` - 时间槽数量
- `VNFS_PER_REQUEST` - 每个请求的VNF数量
- `UAV_BATTERY_CAPACITY` - UAV电池容量
- 等等...

## 数据分析示例

使用 pandas 分析测试结果：

```python
import pandas as pd
import matplotlib.pyplot as plt

# 读取结果
df = pd.read_csv('results/detailed_results_20231119_120000.csv')

# 分析UAV数量对性能的影响
uav_analysis = df.groupby('num_uavs').agg({
    'milp_serviced': 'mean',
    'mpoploc_serviced': 'mean',
    'random_serviced': 'mean',
    'milp_runtime': 'mean',
    'mpoploc_runtime': 'mean',
    'random_runtime': 'mean'
})

# 绘图
uav_analysis[['milp_serviced', 'mpoploc_serviced', 'random_serviced']].plot(kind='bar')
plt.xlabel('UAV数量')
plt.ylabel('平均服务请求数')
plt.title('UAV数量对算法性能的影响')
plt.show()
```

## 注意事项

1. **运行时间**: 大规模测试可能需要较长时间，特别是MILP算法
2. **内存使用**: 确保系统有足够内存运行Gurobi求解器
3. **随机性**: 使用随机种子确保结果可重复
4. **Gurobi许可**: 确保Gurobi许可证有效

## 故障排除

### 问题：MILP求解失败

**可能原因**:
- 时间限制太短
- 问题规模太大
- 约束条件不可行

**解决方案**:
- 增加 `time_limit`
- 减少请求数量或UAV数量
- 检查参数配置是否合理

### 问题：内存不足

**解决方案**:
- 减少测试案例数量
- 减少 `num_runs_per_case`
- 分批次运行测试

### 问题：结果文件编码问题

所有CSV文件使用 `utf-8-sig` 编码，在Excel中可以正确显示中文。

## 联系与支持

如有问题，请检查：
1. 所有依赖包是否正确安装
2. Gurobi是否正确配置
3. config.py中的参数是否合理
