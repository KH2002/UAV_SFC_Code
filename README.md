# UAV-SFC 仿真流程概览

本项目实现了无人机 (UAV) 服务功能链 (SFC) 部署的仿真环境，支持精确求解 (MILP) 与启发式求解 (MPopLoc、RandomOrder) 的多算法对比。下面概述主要流程与关键脚本，帮助快速理解整体仿真。

## 1. 参数配置

所有基础参数集中在 [config.py](config.py)，包括：
- 场景尺度：监控点数量、区域边界、基站位置等；
- 无人机特性：数量、能量与计算能力、飞行模型参数；
- 请求与 VNF 属性：请求数量、VNF 计算/通信需求范围；
- 时间轴设置：时间槽长度与数量。

修改该文件即可统一调整仿真规模与物理模型假设。

## 2. 场景构建

仿真的基础实体定义在 [entities.py](entities.py)：
- `UAV`：记录位置、电量、计算容量等状态；
- `VNF`：描述目标部署位置、CPU 频率、工作量；
- `Request`：封装一组 VNF 及其通信需求。

场景生成函数 `setup_scenario` 位于 [main.py](main.py) 与 [test_algorithms.py](test_algorithms.py)，其步骤包括：
1. 在给定区域内随机生成监控点坐标；
2. 初始化全体 UAV 于基站位置；
3. 为每个请求随机挑选部署位置并生成 VNF 列表，同时随机化 VNF 之间的通信需求。

这些数据结构随后传递给各类求解器使用。

## 3. 算法模块

- [MILP.py](MILP.py)：基于 Gurobi 的混合整数线性规划模型，给出近似最优或最优解；
- [mpoploc.py](mpoploc.py)：实现 MPopLoc 启发式，按请求流行度迭代分配 UAV；
- [random_solver.py](random_solver.py)：按请求生成顺序逐一尝试部署，提供基线对比。

每个算法接收同一份 UAV、请求、位置数据，返回各时间槽被服务的请求 ID。

## 4. 核心入口

- [main.py](main.py)：默认调用 MPopLoc 算法完成单次仿真并输出服务结果；
- [test_algorithms.py](test_algorithms.py)：批量驱动三种算法，支持参数扫描与 CSV 输出；
- [custom_test.py](custom_test.py)：按分类批量测试并将结果划分到独立目录，便于多轮统计；
- [recompute_heuristic_averages.py](recompute_heuristic_averages.py)：在保留 MILP 结果的同时，对启发式算法重复随机试验取平均值；
- [plot_results.py](plot_results.py)：读取结果 CSV，绘制 “变量 vs. 部署成功请求数” 折线图。

## 5. 数据流与结果

1. 运行入口脚本生成/读取配置，随机化场景；
2. 将场景传给指定算法运行，得到各时间槽的服务情况；
3. 汇总统计（成功率、服务数量、运行时间等）写入 `results/` 目录下的 CSV；
4. 可选：使用平均化脚本对启发式算法做多次随机试验并输出 `_avg_*.csv`；
5. 使用绘图脚本在 `results/plots/` 内生成折线图，直观展示不同参数设置下的部署表现。

## 6. 快速开始

1. 安装依赖（含 Gurobi 与其许可）；
2. 按需修改 [config.py](config.py)；
3. 执行 `python test_algorithms.py` 生成默认测试数据，或 `python custom_test.py` 根据模板运行自定义批量测试；
4. 如需多次随机试验平均，先运行 `python recompute_heuristic_averages.py`；
5. 最后执行 `python plot_results.py` 生成图形化报告。

以上即整体仿真流程。可根据研究需求增删算法、扩展指标或调整参数设定。