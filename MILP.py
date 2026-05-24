# -*- coding: utf-8 -*-
"""
使用 Gurobi 实现论文中的 MILP (混合整数线性规划) 模型
"""
import gurobipy as gp
from gurobipy import GRB
import config
import utils

class MILPSolver:
    def __init__(self, uavs, requests, locations):
        self.uavs = uavs
        self.requests = requests
        self.locations = locations
        self.model = gp.Model("UAV_VNF_Deployment_MILP")

    def solve(self, time_limit=60):
        """构建并求解 MILP 模型"""
        print("\n--- MILP 求解器开始 ---")
        print(f"求解时限: {time_limit} 秒")
        self.model.setParam('TimeLimit', time_limit)
        self.model.setParam('MIPGap', 0.01)  # 设置1%的最优性间隙

        # 1. 预计算和创建索引
        self._prepare_data_and_indices()

        # 2. 创建 Gurobi 变量
        self._create_variables()

        # 3. 设置目标函数
        self._set_objective()

        # 4. 添加约束
        self._add_constraints()

        # 5. 求解模型
        self.model.optimize()

        # 6. 解析并返回结果
        return self._format_results()

    def _prepare_data_and_indices(self):
        # 索引
        self.uav_ids = [u.id for u in self.uavs]
        self.request_ids = [r.id for r in self.requests]
        self.location_ids = list(self.locations.keys())
        self.time_slots = range(1, config.NUM_TIME_SLOTS + 1)
        
        # VNF 相关
        self.all_vnfs = [vnf for r in self.requests for vnf in r.vnfs]
        self.vnf_ids = [v.id for v in self.all_vnfs]
        self.vnf_map = {v.id: v for v in self.all_vnfs}
        self.vnf_to_req_map = {vnf.id: req.id for req in self.requests for vnf in req.vnfs}

        # 提取所有虚拟链路及其通信需求
        self.virtual_links = []
        self.comm_demand_map = {}
        for r in self.requests:
            for vnf_pair, demand in r.communication_demands.items():
                # 将 frozenset 转换为排序后的元组，确保密钥的唯一性和顺序
                sorted_pair = tuple(sorted(list(vnf_pair)))
                if sorted_pair not in self.comm_demand_map:
                    self.virtual_links.append(sorted_pair)
                    self.comm_demand_map[sorted_pair] = demand

        # 预计算成本
        self.travel_energy = {
            (i, j): utils.calculate_travel_energy(utils.calculate_distance(self.locations[i], self.locations[j]))
            for i in self.location_ids for j in self.location_ids
        }
        self.link_capacity = {
            (i, j): utils.calculate_link_capacity(utils.calculate_distance(self.locations[i], self.locations[j]))
            for i in self.location_ids for j in self.location_ids if i != 0 and j != 0
        }

    def _create_variables(self):
        # omega_k^t: 请求k是否在时间t被服务
        self.omega = self.model.addVars(self.request_ids, self.time_slots, vtype=GRB.BINARY, name="omega")
        
        # b_u^t(alpha_ki): VNF是否在时间t被部署在UAV u上
        self.b = self.model.addVars(self.vnf_ids, self.uav_ids, self.time_slots, vtype=GRB.BINARY, name="b")
        
        # mu_u^t(i): UAV u在时间t是否位于位置i
        self.mu = self.model.addVars(self.uav_ids, self.location_ids, self.time_slots, vtype=GRB.BINARY, name="mu")
        
        # pi_u^{t,t+1}(i,j): UAV u是否在t和t+1之间从i移动到j
        self.pi = self.model.addVars(self.uav_ids, self.location_ids, self.location_ids, 
                                    range(1, config.NUM_TIME_SLOTS), vtype=GRB.BINARY, name="pi")
        
        # theta_u^t: UAV u在时间t是否活跃
        self.theta = self.model.addVars(self.uav_ids, self.time_slots, vtype=GRB.BINARY, name="theta")
        
        # E_u^t: UAV u在时间t开始时的能量
        self.E = self.model.addVars(self.uav_ids, self.time_slots, vtype=GRB.CONTINUOUS, 
                                   lb=0, ub=config.UAV_BATTERY_CAPACITY, name="E")
        
        # a_u^t: 在基站为UAV u补充的能量
        self.a = self.model.addVars(self.uav_ids, self.time_slots, vtype=GRB.CONTINUOUS, 
                                   lb=0, ub=config.UAV_BATTERY_CAPACITY, name="a")

        # xi_u^t: UAV u在时间t的服务能耗
        self.xi = self.model.addVars(self.uav_ids, self.time_slots, vtype=GRB.CONTINUOUS, lb=0, name="xi")
        
        # f_{(u,v)}^{(alpha_{ki},beta_{kj}),t}: 虚拟链路上的流量
        # 修复：正确创建流量变量索引
        self.f = {}
        for v_pair in self.virtual_links:
            for u in self.uav_ids:
                for v in self.uav_ids:
                    if u != v:  # 只为不同的UAV对创建流量变量
                        for t in self.time_slots:
                            self.f[v_pair, u, v, t] = self.model.addVar(
                                vtype=GRB.CONTINUOUS, lb=0, 
                                name=f"f_{v_pair}_{u}_{v}_{t}"
                            )
    
    def _set_objective(self):
        requests_map = {r.id: r for r in self.requests}
        self.model.setObjective(
            gp.quicksum(requests_map[k].reward * self.omega[k, t] 
                       for k in self.request_ids for t in self.time_slots),
            GRB.MAXIMIZE
        )

    def _add_constraints(self):
        print("添加MILP约束...")
        
        # 约束 (3) & (5): 如果请求被服务，其所有VNF必须被部署一次
        for r in self.requests:
            for vnf in r.vnfs:
                for t in self.time_slots:
                    self.model.addConstr(
                        gp.quicksum(self.b[vnf.id, u, t] for u in self.uav_ids) == self.omega[r.id, t], 
                        f"req_service_{r.id}_{vnf.id}_{t}"
                    )

        # 约束 (4): 每个请求最多被服务一次
        for r in self.requests:
            self.model.addConstr(
                gp.quicksum(self.omega[r.id, t] for t in self.time_slots) <= 1, 
                f"serve_once_{r.id}"
            )

        # 约束 (6): UAV活跃状态
        for u in self.uav_ids:
            for t in self.time_slots:
                self.model.addConstr(
                    self.theta[u, t] <= gp.quicksum(self.b[v_id, u, t] for v_id in self.vnf_ids), 
                    f"theta_active_upper_{u}_{t}"
                )
                for v_id in self.vnf_ids:
                    self.model.addConstr(
                        self.b[v_id, u, t] <= self.theta[u, t], 
                        f"theta_active_lower_{v_id}_{u}_{t}"
                    )
        
        # --- 通信约束 ---
        # 约束 (2) 的线性化: 定义流量f
        for t in self.time_slots:
            for v_pair in self.virtual_links:
                v1_id, v2_id = v_pair
                demand = self.comm_demand_map[v_pair]
                
                for u in self.uav_ids:
                    for v in self.uav_ids:
                        if u == v: 
                            continue
                        
                        # 检查流量变量是否存在
                        if (v_pair, u, v, t) in self.f:
                            # 线性化 f = demand * b1 * b2
                            self.model.addConstr(
                                self.f[v_pair, u, v, t] <= demand * self.b[v1_id, u, t], 
                                f"flow_lin1_{v1_id}_{v2_id}_{u}_{v}_{t}"
                            )
                            self.model.addConstr(
                                self.f[v_pair, u, v, t] <= demand * self.b[v2_id, v, t], 
                                f"flow_lin2_{v1_id}_{v2_id}_{u}_{v}_{t}"
                            )
                            self.model.addConstr(
                                self.f[v_pair, u, v, t] >= demand * (self.b[v1_id, u, t] + self.b[v2_id, v, t] - 1), 
                                f"flow_lin3_{v1_id}_{v2_id}_{u}_{v}_{t}"
                            )
        
        # 约束 (9): 链路容量约束 - 简化版本
        for t in self.time_slots:
            for u in self.uav_ids:
                for v in self.uav_ids:
                    if u >= v: 
                        continue
                    
                    # 计算UAV对之间的总流量
                    total_flow = gp.quicksum(
                        self.f.get((v_pair, u, v, t), 0) + self.f.get((v_pair, v, u, t), 0)
                        for v_pair in self.virtual_links
                    )
                    
                    # 对每个位置对创建容量约束
                    for i in self.location_ids:
                        if i == 0: 
                            continue
                        for j in self.location_ids:
                            if j == 0 or i == j: 
                                continue
                            
                            capacity = self.link_capacity.get((i, j), 0)
                            if capacity > 0:
                                # 使用 Big-M 方法
                                M = 1e6  # Big-M 常数
                                
                                # 如果UAV u在位置i且UAV v在位置j，则流量受容量限制
                                self.model.addConstr(
                                    total_flow <= capacity + M * (2 - self.mu[u, i, t] - self.mu[v, j, t]),
                                    f"cap_{u}_{v}_{i}_{j}_{t}"
                                )

        # 约束 (11): VNF部署位置
        for vnf in self.all_vnfs:
            for u in self.uav_ids:
                for t in self.time_slots:
                    self.model.addConstr(
                        self.b[vnf.id, u, t] <= self.mu[u, vnf.location_id, t], 
                        f"vnf_location_{vnf.id}_{u}_{t}"
                    )

        # 约束 (12): 每个UAV在每个时间点只能在一个位置
        for u in self.uav_ids:
            for t in self.time_slots:
                self.model.addConstr(
                    gp.quicksum(self.mu[u, i, t] for i in self.location_ids) == 1, 
                    f"uav_one_loc_{u}_{t}"
                )

        # 约束 (13): 每个监控点最多一个UAV
        for i in self.location_ids:
            if i != 0:  # 基站除外
                for t in self.time_slots:
                    self.model.addConstr(
                        gp.quicksum(self.mu[u, i, t] for u in self.uav_ids) <= 1, 
                        f"loc_one_uav_{i}_{t}"
                    )

        # 约束 (14-16): UAV移动逻辑
        for t in range(1, config.NUM_TIME_SLOTS):
            for u in self.uav_ids:
                for i in self.location_ids:
                    for j in self.location_ids:
                        self.model.addConstr(
                            self.pi[u, i, j, t] >= self.mu[u, i, t] + self.mu[u, j, t+1] - 1, 
                            f"pi_logic1_{u}_{i}_{j}_{t}"
                        )
                        self.model.addConstr(
                            self.pi[u, i, j, t] <= self.mu[u, i, t], 
                            f"pi_logic2_{u}_{i}_{j}_{t}"
                        )
                        self.model.addConstr(
                            self.pi[u, i, j, t] <= self.mu[u, j, t+1], 
                            f"pi_logic3_{u}_{i}_{j}_{t}"
                        )

        # 约束 (27): 计算能力约束
        for u in self.uav_ids:
            for t in self.time_slots:
                self.model.addConstr(
                    gp.quicksum(self.vnf_map[v_id].workload * self.b[v_id, u, t] for v_id in self.vnf_ids) 
                    <= config.UAV_COMPUTATION_CAPACITY, 
                    f"comp_cap_{u}_{t}"
                )
        
        # 能量相关约束
        # 约束 (19): 服务能耗
        hover_energy = config.TIME_SLOT_DURATION * (config.UAV_BLADE_PROFILE_POWER + config.UAV_INDUCED_POWER)
        for u in self.uav_ids:
            for t in self.time_slots:
                comp_energy = gp.quicksum(
                    config.CHIP_ARCHITECTURE_CONSTANT * (self.vnf_map[v_id].cpu_freq)**2 * 
                    self.vnf_map[v_id].workload * self.b[v_id, u, t] 
                    for v_id in self.vnf_ids
                )
                # 简化通信能耗
                comm_energy = config.COMM_ENERGY_PER_TIMESLOT * self.theta[u, t]
                
                self.model.addConstr(
                    self.xi[u, t] == hover_energy * self.theta[u, t] + comp_energy + comm_energy, 
                    f"servicing_energy_{u}_{t}"
                )
        
        # 能量演进
        for u in self.uav_ids:
            # t=1: 初始能量
            initial_travel_energy = gp.quicksum(
                self.mu[u, i, 1] * self.travel_energy[0, i] 
                for i in self.location_ids if i != 0
            )
            self.model.addConstr(
                self.E[u, 1] == config.UAV_BATTERY_CAPACITY - initial_travel_energy, 
                f"energy_init_{u}"
            )
            
            # t > 1: 能量演进
            for t in range(2, config.NUM_TIME_SLOTS + 1):
                travel_energy_prev = gp.quicksum(
                    self.pi[u, i, j, t-1] * self.travel_energy[i, j] 
                    for i in self.location_ids for j in self.location_ids
                )
                self.model.addConstr(
                    self.E[u, t] == self.E[u, t-1] + self.a[u, t-1] - self.xi[u, t-1] - travel_energy_prev, 
                    f"energy_evol_{u}_{t}"
                )

            # 充电逻辑
            for t in self.time_slots:
                M = config.UAV_BATTERY_CAPACITY  # Big-M 常数
                
                self.model.addConstr(
                    self.a[u, t] <= config.UAV_BATTERY_CAPACITY - self.E[u, t], 
                    f"charge_logic1_{u}_{t}"
                )
                self.model.addConstr(
                    self.a[u, t] >= config.UAV_BATTERY_CAPACITY - self.E[u, t] - (1 - self.mu[u, 0, t]) * M, 
                    f"charge_logic2_{u}_{t}"
                )
                self.model.addConstr(
                    self.a[u, t] <= self.mu[u, 0, t] * M, 
                    f"charge_logic3_{u}_{t}"
                )

            # 能耗不能超过当前电量
            for t in range(1, config.NUM_TIME_SLOTS):
                travel_energy_curr = gp.quicksum(
                    self.pi[u, i, j, t] * self.travel_energy[i, j] 
                    for i in self.location_ids for j in self.location_ids
                )
                self.model.addConstr(
                    self.xi[u, t] + travel_energy_curr <= self.E[u, t] + self.a[u, t], 
                    f"energy_pos_inter_{u}_{t}"
                )
            
            # 最后时刻的能量约束
            final_travel_energy = gp.quicksum(
                self.mu[u, i, config.NUM_TIME_SLOTS] * self.travel_energy[i, 0] 
                for i in self.location_ids if i != 0
            )
            self.model.addConstr(
                self.xi[u, config.NUM_TIME_SLOTS] + final_travel_energy 
                <= self.E[u, config.NUM_TIME_SLOTS] + self.a[u, config.NUM_TIME_SLOTS], 
                f"energy_pos_final_{u}"
            )
            
        print("所有约束添加完毕。")

    def _format_results(self):
        if self.model.status == GRB.OPTIMAL:
            print("\n--- MILP 求解结果 ---")
            print(f"求解状态: 最优解")
            print(f"最优性间隙: {self.model.MIPGap:.2%}")
        elif self.model.status == GRB.TIME_LIMIT:
            print("\n--- MILP 求解结果 ---")
            print(f"求解状态: 达到时间限制")
            print(f"当前最优性间隙: {self.model.MIPGap:.2%}")
        elif self.model.status == GRB.INFEASIBLE:
            print("\n--- MILP 求解结果 ---")
            print("模型不可行。正在计算导致冲突的最小约束集(IIS)...")
            self.model.computeIIS()
            print("以下约束导致了不可行性:")
            for c in self.model.getConstrs():
                if c.IISConstr:
                    print(f"\t{c.ConstrName}")
            return None
        else:
            print(f"\n--- MILP 求解结果 ---")
            print(f"求解失败，状态码: {self.model.status}")
            return None

        # 检查是否找到了解
        if self.model.SolCount == 0:
            print("没有找到可行解。")
            return {t: [] for t in self.time_slots}

        # 提取结果
        if self.model.ObjVal < 1e-6:
            print(f"总奖励: 0 (没有服务任何请求)")
            self._diagnose_zero_result()
            return {t: [] for t in self.time_slots}
        
        serviced_requests_timeline = {t: [] for t in self.time_slots}
        
        for t in self.time_slots:
            for r_id in self.request_ids:
                if self.omega[r_id, t].X > 0.5:
                    serviced_requests_timeline[t].append(r_id)
        
        print(f"总奖励: {self.model.ObjVal:.2f}")
        total_requests_serviced = sum(len(reqs) for reqs in serviced_requests_timeline.values())
        print(f"服务的请求总数: {total_requests_serviced}/{len(self.requests)}")
        
        # 打印每个时间槽的服务情况
        for t in self.time_slots:
            if serviced_requests_timeline[t]:
                print(f"  时间槽 {t}: 服务请求 {serviced_requests_timeline[t]}")
        
        return serviced_requests_timeline

    def _diagnose_zero_result(self):
        """
        当最优解为0时，尝试找出可能的原因。
        """
        print("\n诊断: 检查各项约束是否存在明显的不可行性...")
        
        if not self.uavs or not self.all_vnfs:
            print("没有无人机或VNF可供分析。")
            return

        # 1. 能量约束诊断
        print("\n--- 1. 能量约束诊断 ---")
        if self.all_vnfs:
            uav = self.uavs[0]
            vnf = self.all_vnfs[0]
            initial_travel_energy = self.travel_energy.get((0, vnf.location_id), float('inf'))
            return_travel_energy = self.travel_energy.get((vnf.location_id, 0), float('inf'))
            remaining_energy = config.UAV_BATTERY_CAPACITY - initial_travel_energy
            
            print(f"UAV {uav.id} 部署 VNF {vnf.id} (位于位置 {vnf.location_id}):")
            print(f"  - 电池容量: {config.UAV_BATTERY_CAPACITY:.2f} J")
            print(f"  - 初始飞行能耗: {initial_travel_energy:.2f} J")
            print(f"  - 返回飞行能耗: {return_travel_energy:.2f} J")
            print(f"  - 到达后剩余能量: {remaining_energy:.2f} J")
            
            if remaining_energy < return_travel_energy:
                print(">> [问题发现] 能量不足以支持往返飞行")

        # 2. 计算能力约束诊断
        print("\n--- 2. 计算能力约束诊断 ---")
        if self.all_vnfs:
            max_workload_vnf = max(self.all_vnfs, key=lambda v: v.workload)
            print(f"  - UAV 计算能力: {config.UAV_COMPUTATION_CAPACITY:.2f}")
            print(f"  - 最高工作负载 VNF ({max_workload_vnf.id}): {max_workload_vnf.workload:.2f}")
            
            if max_workload_vnf.workload > config.UAV_COMPUTATION_CAPACITY:
                print(">> [问题发现] 存在单个VNF超过UAV计算能力")

        # 3. 通信容量约束诊断
        print("\n--- 3. 通信容量约束诊断 ---")
        if self.comm_demand_map:
            max_demand_pair = max(self.comm_demand_map.items(), key=lambda item: item[1])
            max_demand = max_demand_pair[1]
            
            valid_links = {k: v for k, v in self.link_capacity.items() if k[0] != 0 and k[1] != 0}
            if valid_links:
                max_cap_pair = max(valid_links.items(), key=lambda item: item[1])
                max_capacity = max_cap_pair[1]
                
                print(f"  - 最高通信需求: {max_demand:.2f} bits/s")
                print(f"  - 最高链路容量: {max_capacity:.2f} bits/s")
                
                if max_demand > max_capacity:
                    print(">> [问题发现] 通信需求超过最大链路容量")

        print("\n--- 建议 ---")
        print("1. 增加UAV数量或电池容量")
        print("2. 减少VNF工作负载或通信需求")
        print("3. 增加时间槽数量")
        print("4. 检查位置分布是否合理")