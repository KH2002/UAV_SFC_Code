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
        # for r in self.requests:
        #     for vnf_pair, demand in r.communication_demands.items():
        #         self.virtual_links.append(vnf_pair)
        #         self.comm_demand_map[vnf_pair] = demand
        for r in self.requests:
            for vnf_pair, demand in r.communication_demands.items():
                # 将 frozenset 转换为排序后的元组，确保密钥的唯一性和顺序
                sorted_pair = tuple(sorted(list(vnf_pair)))
                if sorted_pair not in self.comm_demand_map:
                    self.virtual_links.append(sorted_pair)
                self.comm_demand_map[sorted_pair] = demand
        # [新增] 为能耗计算预处理VNF的总通信需求
        self.vnf_total_demand = {vnf_id: 0 for vnf_id in self.vnf_ids}
        for (v1_id, v2_id), demand in self.comm_demand_map.items():
            self.vnf_total_demand[v1_id] += demand
            self.vnf_total_demand[v2_id] += demand

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
        self.pi = self.model.addVars(self.uav_ids, self.location_ids, self.location_ids, range(1, config.NUM_TIME_SLOTS), vtype=GRB.BINARY, name="pi")
        
        # theta_u^t: UAV u在时间t是否活跃
        self.theta = self.model.addVars(self.uav_ids, self.time_slots, vtype=GRB.BINARY, name="theta")
        
        # E_u^t: UAV u在时间t开始时的能量
        self.E = self.model.addVars(self.uav_ids, self.time_slots, vtype=GRB.CONTINUOUS, name="E")
        
        # a_u^t: 在基站为UAV u补充的能量
        self.a = self.model.addVars(self.uav_ids, self.time_slots, vtype=GRB.CONTINUOUS, name="a")

        # xi_u^t: UAV u在时间t的服务能耗
        self.xi = self.model.addVars(self.uav_ids, self.time_slots, vtype=GRB.CONTINUOUS, name="xi")
        
        # f_{(u,v)}^{(\alpha_{ki},\beta_{kj}),t}: 虚拟链路上的流量
        self.f = self.model.addVars([(v_pair,) for v_pair in self.virtual_links], self.uav_ids, self.uav_ids, self.time_slots, vtype=GRB.CONTINUOUS, name="f")

    
    def _set_objective(self):
        requests_map = {r.id: r for r in self.requests}
        self.model.setObjective(
            gp.quicksum(requests_map[k].reward * self.omega[k, t] for k, t in self.omega),
            GRB.MAXIMIZE
        )

    def _add_constraints(self):
        print("添加MILP约束...")
        # 约束 (3) & (5): 如果请求被服务，其所有VNF必须被部署一次
        for r in self.requests:
            for vnf in r.vnfs:
                for t in self.time_slots:
                    self.model.addConstr(gp.quicksum(self.b[vnf.id, u, t] for u in self.uav_ids) == self.omega[r.id, t], f"req_service_{r.id}_{vnf.id}_{t}")

        # 约束 (4): 每个请求最多被服务一次
        for r in self.requests:
            self.model.addConstr(gp.quicksum(self.omega[r.id, t] for t in self.time_slots) <= 1, f"serve_once_{r.id}")

        # 约束 (6): UAV活跃状态
        for u in self.uav_ids:
            for t in self.time_slots:
                self.model.addConstr(self.theta[u, t] <= gp.quicksum(self.b[v_id, u, t] for v_id in self.vnf_ids), f"theta_active_upper_{u}_{t}")
                for v_id in self.vnf_ids:
                    self.model.addConstr(self.b[v_id, u, t] <= self.theta[u, t], f"theta_active_lower_{v_id}_{u}_{t}")
        # --- 新增：通信约束 ---
        # 约束 (2) 的线性化: 定义流量f
        for t in self.time_slots:
            for v_pair in self.virtual_links:
                v1_id, v2_id = v_pair
                demand = self.comm_demand_map[v_pair]
                for u in self.uav_ids:
                    for v in self.uav_ids:
                        if u == v: continue
                        # 线性化 f = demand * b1 * b2
                        b1 = self.b[v1_id, u, t]
                        b2 = self.b[v2_id, v, t]    
                        self.model.addConstr(self.f[v_pair, u, v, t] <= demand * self.b[v1_id, u, t], f"flow_lin1_{v1_id}_{v2_id}_{u}_{v}_{t}")
                        self.model.addConstr(self.f[v_pair, u, v, t] <= demand * self.b[v2_id, v, t], f"flow_lin2_{v1_id}_{v2_id}_{u}_{v}_{t}")
                        self.model.addConstr(self.f[v_pair, u, v, t] >= demand * (self.b[v1_id, u, t] + self.b[v2_id, v, t] - 1), f"flow_lin3_{v1_id}_{v2_id}_{u}_{v}_{t}")
                        # try:
                        #     # 尝试添加这三个约束
                        #     self.model.addConstr(self.f[v_pair, u, v, t] <= demand * self.b[v1_id, u, t], f"flow_lin1_{v1_id}_{v2_id}_{u}_{v}_{t}")
                        #     self.model.addConstr(self.f[v_pair, u, v, t] <= demand * self.b[v2_id, v, t], f"flow_lin2_{v1_id}_{v2_id}_{u}_{v}_{t}")
                        #     self.model.addConstr(self.f[v_pair, u, v, t] >= demand * (self.b[v1_id, u, t] + self.b[v2_id, v, t] - 1), f"flow_lin3_{v1_id}_{v2_id}_{u}_{v}_{t}")
                        
                        # except KeyError as e:
                        #     print("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                        #     print("!!!程序崩溃：找到 KeyError!!!")
                        #     print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
                        #     print(f"导致错误的变量键是: {e}")
                        #     print("\n崩溃时循环变量的值是:")
                        #     print(f"  t = {t}")
                        #     print(f"  v_pair = {v_pair}  (v1_id={v1_id}, v2_id={v2_id})")
                        #     print(f"  u = {u}")
                        #     print(f"  v = {v}")

                        #     # 检查是哪个变量出了问题
                        #     print("\n请检查以下变量是否存在：")
                        #     print(f"  - 键 (v_pair, u, v, t): {(v_pair, u, v, t)} in self.f = {(v_pair, u, v, t) in self.f}")
                        #     print("\n请检查定义 self.b 变量时使用的索引集，是否包含了上面检查结果为 False 的键。")
                            
                        #     # 抛出异常，中断程序
                        #     raise e
                        # # --- 结束调试代码 ---
        # 约束 (9): 链路容量约束
        # 使用 Big-M 方法处理 mu_u * mu_v * C_ij 的非线性项
        for t in self.time_slots:
            for u in self.uav_ids:
                for v in self.uav_ids:
                    if u >= v: continue 
                    
                    total_flow = gp.quicksum(self.f[v_pair, u, v, t] + self.f[v_pair, v, u, t] for v_pair in self.virtual_links)
                    
                    for i in self.location_ids:
                        if i == 0: continue
                        for j in self.location_ids:
                            if j == 0 or i == j: continue
                            
                            capacity = self.link_capacity.get((i, j), 0)
                            # 指示性约束: IF mu[u,i,t]=1 AND mu[v,j,t]=1 THEN total_flow <= capacity
                            # Gurobi需要辅助二进制变量来表示AND条件
                            z = self.model.addVar(vtype=GRB.BINARY, name=f"z_and_{u}_{v}_{i}_{j}_{t}")
                            self.model.addConstr(z >= self.mu[u, i, t] + self.mu[v, j, t] - 1, name=f"z_and_1_{u}_{v}_{i}_{j}_{t}")
                            self.model.addConstr(z <= self.mu[u, i, t], name=f"z_and_2_{u}_{v}_{i}_{j}_{t}")
                            self.model.addConstr(z <= self.mu[v, j, t], name=f"z_and_3_{u}_{v}_{i}_{j}_{t}")

                            self.model.addConstr((z == 1) >> (total_flow <= capacity), name=f"cap_indicator_{u}_{v}_{i}_{j}_{t}")

                    # [修正] 使用 .get() 来安全地构建求和表达式
                    # total_flow = gp.quicksum(
                    #     self.f.get((v_pair, u, v, t), 0) + self.f.get((v_pair, v, u, t), 0)
                    #     for v_pair in self.virtual_links
                    # )
                    
                    # for i in self.location_ids:
                    #     if i == 0: continue
                    #     for j in self.location_ids:
                    #         if j == 0 or i == j: continue
                            
                    #         capacity = self.link_capacity.get((i, j), 0)
                    #         # Gurobi需要辅助二进制变量来表示AND条件
                    #         z = self.model.addVar(vtype=GRB.BINARY, name=f"z_and_{u}_{v}_{i}_{j}_{t}")
                    #         self.model.addConstr(z >= self.mu[u, i, t] + self.mu[v, j, t] - 1, name=f"z_and_1_{u}_{v}_{i}_{j}_{t}")
                    #         self.model.addConstr(z <= self.mu[u, i, t], name=f"z_and_2_{u}_{v}_{i}_{j}_{t}")
                    #         self.model.addConstr(z <= self.mu[v, j, t], name=f"z_and_3_{u}_{v}_{i}_{j}_{t}")
                            
                    #         self.model.addConstr((z == 1) >> (total_flow <= capacity), name=f"cap_indicator_{u}_{v}_{i}_{j}_{t}")

        # 约束 (11): VNF部署位置
        for vnf in self.all_vnfs:
            for u in self.uav_ids:
                for t in self.time_slots:
                    self.model.addConstr(self.b[vnf.id, u, t] <= self.mu[u, vnf.location_id, t], f"vnf_location_{vnf.id}_{u}_{t}")

        # 约束 (12): 每个UAV在每个时间点只能在一个位置
        for u in self.uav_ids:
            for t in self.time_slots:
                self.model.addConstr(gp.quicksum(self.mu[u, i, t] for i in self.location_ids) == 1, f"uav_one_loc_{u}_{t}")

        # 约束 (13): 每个监控点最多一个UAV
        for i in self.location_ids:
            if i != 0: # 基站除外
                for t in self.time_slots:
                    self.model.addConstr(gp.quicksum(self.mu[u, i, t] for u in self.uav_ids) <= 1, f"loc_one_uav_{i}_{t}")

        # 约束 (14-16): UAV移动逻辑
        for t in range(1, config.NUM_TIME_SLOTS):
            for u in self.uav_ids:
                for i in self.location_ids:
                    for j in self.location_ids:
                        self.model.addConstr(self.pi[u, i, j, t] >= self.mu[u, i, t] + self.mu[u, j, t+1] - 1, f"pi_logic1_{u}_{i}_{j}_{t}")
                        self.model.addConstr(self.pi[u, i, j, t] <= self.mu[u, i, t], f"pi_logic2_{u}_{i}_{j}_{t}")
                        self.model.addConstr(self.pi[u, i, j, t] <= self.mu[u, j, t+1], f"pi_logic3_{u}_{i}_{j}_{t}")

        # 约束 (27): 计算能力约束
        for u in self.uav_ids:
            for t in self.time_slots:
                self.model.addConstr(gp.quicksum(self.vnf_map[v_id].workload  * self.b[v_id, u, t] for v_id in self.vnf_ids) <= config.UAV_COMPUTATION_CAPACITY, f"comp_cap_{u}_{t}")
        
        # 能量相关约束 (20-26)
        # 约束 (19): 服务能耗. NOTE: 论文中的通信能耗部分是二次的，这里进行线性化简化，
        # 假设能耗基于VNF的服务需求，而不是实际的流量伙伴。这是一个常见的简化。
        hover_energy = config.TIME_SLOT_DURATION * (config.UAV_BLADE_PROFILE_POWER + config.UAV_INDUCED_POWER)
        for u in self.uav_ids:
            for t in self.time_slots:
                comp_energy = gp.quicksum(config.CHIP_ARCHITECTURE_CONSTANT * (self.vnf_map[v_id].cpu_freq)**2 * self.vnf_map[v_id].workload * self.b[v_id, u, t] for v_id in self.vnf_ids)
                # 简化通信能耗，仅考虑一个基础的收发能耗
                comm_energy = config.TIME_SLOT_DURATION * config.TRANSCEIVER_CIRCUIT_POWER * self.theta[u, t]
                self.model.addConstr(self.xi[u, t] == hover_energy * self.theta[u, t] + comp_energy + comm_energy, f"servicing_energy_{u}_{t}")
        
        # 能量演进
        for u in self.uav_ids:
            # t=1
            initial_travel_energy = gp.quicksum(self.mu[u, i, 1] * self.travel_energy[0, i] for i in self.location_ids if i != 0)
            self.model.addConstr(self.E[u, 1] == config.UAV_BATTERY_CAPACITY - initial_travel_energy, f"energy_init_{u}")
            # t > 1
            for t in range(2, config.NUM_TIME_SLOTS + 1):
                travel_energy_prev = gp.quicksum(self.pi[u, i, j, t-1] * self.travel_energy[i, j] for i in self.location_ids for j in self.location_ids)
                self.model.addConstr(self.E[u, t] == self.E[u, t-1] + self.a[u, t-1] - self.xi[u, t-1] - travel_energy_prev, f"energy_evol_{u}_{t}")

            # 充电逻辑 (23-24)
            for t in self.time_slots:
                self.model.addConstr(self.a[u, t] <= config.UAV_BATTERY_CAPACITY - self.E[u, t], f"charge_logic1_{u}_{t}")
                self.model.addConstr(self.a[u, t] >= config.UAV_BATTERY_CAPACITY - self.E[u, t] - (1 - self.mu[u, 0, t]) * config.UAV_BATTERY_CAPACITY, f"charge_logic2_{u}_{t}")
                self.model.addConstr(self.a[u, t] <= self.mu[u, 0, t] * config.UAV_BATTERY_CAPACITY, f"charge_logic3_{u}_{t}")
                self.model.addConstr(self.a[u, t] >= 0, f"charge_logic4_{u}_{t}")

            # 能耗不能超过当前电量
            for t in range(1, config.NUM_TIME_SLOTS):
                travel_energy_curr = gp.quicksum(self.pi[u, i, j, t] * self.travel_energy[i, j] for i in self.location_ids for j in self.location_ids)
                self.model.addConstr(self.xi[u, t] + travel_energy_curr <= self.E[u, t] + self.a[u, t], f"energy_pos_inter_{u}_{t}")
            
            # T
            final_travel_energy = gp.quicksum(self.mu[u, i, config.NUM_TIME_SLOTS] * self.travel_energy[i, 0] for i in self.location_ids if i != 0)
            self.model.addConstr(self.xi[u, config.NUM_TIME_SLOTS] + final_travel_energy <= self.E[u, config.NUM_TIME_SLOTS] + self.a[u, config.NUM_TIME_SLOTS], f"energy_pos_final_{u}")
            
        print("所有约束添加完毕。")

    def _format_results(self):
        if self.model.status == GRB.OPTIMAL or self.model.status == GRB.TIME_LIMIT:
            # 检查最优解是否为0
            if self.model.ObjVal < 1e-6:
                print("\n--- MILP 求解结果 ---")
                print("模型找到最优解，但总奖励为 0。这意味着没有服务任何请求。")
                print("正在进行诊断分析...")
                self._diagnose_zero_result()
                return {t: [] for t in self.time_slots}

            print("\n--- MILP 求解结果 ---")
            if self.model.status == GRB.TIME_LIMIT:
                print("警告: 已达到时间限制，结果可能是次优的。")
            
            serviced_requests_timeline = {t: [] for t in self.time_slots}
            
            for t in self.time_slots:
                for r_id in self.request_ids:
                    if self.omega[r_id, t].X > 0.5:
                        serviced_requests_timeline[t].append(r_id)
            
            print(f"总奖励: {self.model.ObjVal:.2f}")
            return serviced_requests_timeline
        elif self.model.status == GRB.INFEASIBLE:
            print("模型不可行。正在计算导致冲突的最小约束集(IIS)...")
            self.model.computeIIS()
            print("以下约束导致了不可行性:")
            for c in self.model.getConstrs():
                if c.IISConstr:
                    print(f"\t{c.ConstrName}")
            return None
        else:
            print(f"求解失败，状态码: {self.model.status}")
            return None

    def _diagnose_zero_result(self):
        """
        当最优解为0时，尝试找出可能的原因。
        检查能量、计算和通信约束是否过于严格。
        """
        print("诊断: 检查各项约束是否存在明显的不可行性...")
        
        if not self.uavs or not self.all_vnfs:
            print("没有无人机或VNF可供分析。")
            return

        # --- 1. 能量约束诊断 (Energy Constraint Diagnosis) ---
        print("\n--- 1. 能量约束诊断 ---")
        uav = self.uavs[0]
        vnf = self.all_vnfs[0]
        initial_travel_energy = self.travel_energy.get((0, vnf.location_id), float('inf'))
        return_travel_energy = self.travel_energy.get((vnf.location_id, 0), float('inf'))
        remaining_energy_after_initial_travel = config.UAV_BATTERY_CAPACITY - initial_travel_energy
        
        print(f"以 UAV {uav.id} 部署 VNF {vnf.id} (位于位置 {vnf.location_id}) 为例:")
        print(f"  - 总电池容量: {config.UAV_BATTERY_CAPACITY:.2f} J")
        print(f"  - 初始飞行能耗: {initial_travel_energy:.2f} J")
        print(f"  - 到达后剩余能量: {remaining_energy_after_initial_travel:.2f} J")
        
        if remaining_energy_after_initial_travel < return_travel_energy:
             print(">> 能量诊断结论: [高可能性] 初始飞行后，剩余能量甚至不足以支付返航的能耗，更不用说服务能耗。")
        else:
            print(">> 能量诊断结论: 单次飞行的能量约束似乎可以通过。")

        # --- 2. 计算能力约束诊断 (Computation Capacity Diagnosis) ---
        print("\n--- 2. 计算能力约束诊断 ---")
        max_workload_vnf = max(self.all_vnfs, key=lambda v: v.workload)
        print(f"  - UAV 计算能力上限: {config.UAV_COMPUTATION_CAPACITY:.2f} GCC")
        print(f"  - 最高工作负载的VNF ({max_workload_vnf.id}): {max_workload_vnf.workload:.2f} MCC")
        
        if max_workload_vnf.workload > config.UAV_COMPUTATION_CAPACITY:
            print(">> 计算诊断结论: [高可能性] 存在单个VNF的工作负载就超过了UAV的处理能力上限。")
        else:
            print(">> 计算诊断结论: 单个VNF的计算需求可以满足。请注意，在单个UAV上部署多个VNF的组合仍可能超限。")

        # --- 3. 通信容量约束诊断 (Communication Capacity Diagnosis) ---
        print("\n--- 3. 通信容量约束诊断 ---")
        if not self.comm_demand_map:
            print(">> 通信诊断结论: 模型中没有通信需求，跳过此项检查。")
        else:
            max_demand_pair = max(self.comm_demand_map.items(), key=lambda item: item[1])
            max_demand = max_demand_pair[1]
            
            valid_links = {k: v for k, v in self.link_capacity.items() if k[0] != 0 and k[1] != 0}
            if not valid_links:
                 print(">> 通信诊断结论: 没有有效的物理通信链路可供分析。")
            else:
                max_cap_pair = max(valid_links.items(), key=lambda item: item[1])
                max_capacity = max_cap_pair[1]
                min_cap_pair = min(valid_links.items(), key=lambda item: item[1])
                min_capacity = min_cap_pair[1]

                print(f"  - 物理链路最高容量 (位置 {max_cap_pair[0]}): {max_capacity:.2f} bits/s")
                print(f"  - 物理链路最低容量 (位置 {min_cap_pair[0]}): {min_capacity:.2f} bits/s")
                print(f"  - 最高虚拟链路通信需求 (VNF对 {max_demand_pair[0]}): {max_demand:.2f} bits/s")
            
                if max_demand > max_capacity:
                    print(">> 通信诊断结论: [高可能性] 存在单个虚拟链路的需求就超过了最强物理链路的容量，因此永远无法满足。")
                elif max_demand > min_capacity:
                    print(">> 通信诊断结论: [可能性] 最高通信需求超过了最弱物理链路的容量。如果求解器必须使用弱链路，则可能导致不可行。")
                else:
                    print(">> 通信诊断结论: 单个虚拟链路的需求似乎可以满足。请注意，多条虚拟链路在同一物理链路上汇合仍可能超限。")

        print("\n--- 最终建议 ---")
        print("如果以上诊断均未发现明显问题，请考虑：")
        print("1. 多时间槽的能量演进：检查是否在某个后续时间槽，所有无人机的能量都耗尽且无法返回充电。")
        print("2. 资源组合冲突：可能不存在任何一个能同时满足能量、计算和通信约束的VNF部署组合。")
        print("3. 尝试进一步放宽约束：例如，临时性地极大增加电池/计算/通信容量，看是否能得到非零解，以定位瓶颈。")
