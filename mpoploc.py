# -*- coding: utf-8 -*-
"""
MPopLoc 启发式算法实现
"""
import copy
import config
import utils

class MPopLocSolver:
    def __init__(self, uavs, requests, locations):
        self.uavs = uavs
        self.requests = requests
        self.locations = locations # {location_id: (x, y)}
        self.serviced_requests_timeline = {} # {time_slot: [request_ids]}

    def solve(self):
        """主求解函数，实现 Algorithm 1"""
        print("--- MPopLoc 算法开始 ---")
        total_serviced_requests = []

        for t in range(1, config.NUM_TIME_SLOTS + 1):
            print(f"\n--- 时间槽 {t} ---")
            self.serviced_requests_timeline[t] = []
            uavs_to_charge_this_timeslot = set() # 主充电列表，使用set避免重复

            # 1. 重置当前时间槽UAV状态
            for uav in self.uavs:
                uav.is_busy = False
            # 2. 获取未被服务的请求 (line 8)
            unserviced_requests = [r for r in self.requests if not r.is_serviced]
            if not unserviced_requests:
                print("所有请求都已服务完毕。")
                break

            # 3. 计算请求流行度并排序 (line 9)
            sorted_requests = self._sort_requests_by_popularity(unserviced_requests)
            
            # 4. 遍历排序后的请求 (line 10)
            for req in sorted_requests:
                # 5. 资源承诺阶段 (line 12)

                # 创建当前状态的快照，以便在分配失败时回滚
                uavs_snapshot = copy.deepcopy(self.uavs)
                
                s_flag, assigned_uav_ids, uavs_found_low_on_energy = self._res_commit(req, uavs_snapshot)
                #不论成功与否，均更新电量少的uav
                uavs_to_charge_this_timeslot.update(uavs_found_low_on_energy)
                # 6. 如果承诺成功，则进入资源分配阶段 (line 13-25)
                if s_flag:
                    print(f"请求 {req.id} 资源承诺成功。开始分配...")
                    
                    # 更新真实UAV对象的状态
                    self._allocate_resources(req, assigned_uav_ids)
                    
                    req.is_serviced = True
                    self.serviced_requests_timeline[t].append(req.id)
                    total_serviced_requests.append(req.id)
                    print(f"请求 {req.id} 分配成功！")
                # else:
                #     print(f"请求 {req.id} 资源承诺失败。")

            # 7. 更新需要充电的UAV的状态 (line 27-30)
            for uav_id in uavs_to_charge_this_timeslot:
                uav = next(u for u in self.uavs if u.id == uav_id)
                if not uav.is_busy:
                    uav.location = config.BASE_STATION_LOCATION
                    uav.location_id = 0
                    uav.energy = config.UAV_BATTERY_CAPACITY # 充满
            #所有VNF均结束，恢复无人机的计算资源
            for u in self.uavs:
                u.cpu_capacity = config.UAV_COMPUTATION_CAPACITY
            # 此处可以打印UAV的最终状态
            print(f"时间槽 {t} 结束时UAV状态:")
            for uav in self.uavs:
                print(f"  {uav}")


        print("\n--- MPopLoc 算法结束 ---")
        return self.serviced_requests_timeline


    def _sort_requests_by_popularity(self, requests):
        """计算每个位置的流行度，并据此对请求进行降序排序"""
        location_popularity = {loc_id: 0 for loc_id in self.locations}
        for req in requests:
            for loc_id in req.get_required_location_ids():
                if loc_id in location_popularity:
                    location_popularity[loc_id] += 1
        
        # 计算每个请求的流行度 (论文公式 29)
        request_popularity = {}
        for req in requests:
            popularity = sum(location_popularity.get(loc_id, 0) for loc_id in req.get_required_location_ids())
            request_popularity[req.id] = popularity

        return sorted(requests, key=lambda r: request_popularity[r.id], reverse=True)


    def _res_commit(self, req, uavs_snapshot):
        """资源承诺函数, 实现 Algorithm 2"""
        u_prime = {} # {location_id: uav_id}
        uavs_to_charge = []
        
        # 1. 检查通信容量 (line 3-9)
        required_locations = req.get_required_location_ids()
        for i in range(len(required_locations)):
            for j in range(i + 1, len(required_locations)):
                loc_id1 = required_locations[i]
                loc_id2 = required_locations[j]
                
                # 找到这两个位置对应的VNF
                vnf1 = next((v for v in req.vnfs if v.location_id == loc_id1), None)
                vnf2 = next((v for v in req.vnfs if v.location_id == loc_id2), None)

                comm_demand = req.communication_demands.get((vnf1.id, vnf2.id), 0) or \
                              req.communication_demands.get((vnf2.id, vnf1.id), 0)

                if comm_demand > 0:
                    distance = utils.calculate_distance(self.locations[loc_id1], self.locations[loc_id2])
                    capacity = utils.calculate_link_capacity(distance)
                    if comm_demand > capacity:
                        # print(f"通信容量不足: loc {loc_id1}-{loc_id2} demand {comm_demand:.2f} > capacity {capacity:.2f}")
                        return False, {}, []
        
        # 2. Phase 1: 优先选择已在目标位置的UAV (line 12-24)
        for loc_id in required_locations:
            vnf = next(v for v in req.vnfs if v.location_id == loc_id)
            for uav in uavs_snapshot:
                if uav.location_id == loc_id:
                    servicing_energy = self._get_total_energy_for_vnf(vnf, req)
                    return_energy = utils.calculate_travel_energy(
                        utils.calculate_distance(self.locations[loc_id], config.BASE_STATION_LOCATION)
                    )                    
                    if uav.energy < servicing_energy + return_energy:
                        uavs_to_charge.append(uav.id)
                    elif uav.cpu_capacity >= vnf.workload: # workload from MCC to GCC
                        u_prime[loc_id] = uav.id
                        uav.is_busy = True # 标记为已预定
                        break # 已为该位置找到UAV，继续下一个位置
        
        # 3. Phase 2: 为剩余位置选择成本最低的UAV (line 26-40)
        remaining_locs = [loc_id for loc_id in required_locations if loc_id not in u_prime]
        
        for loc_id in remaining_locs:
            vnf = next(v for v in req.vnfs if v.location_id == loc_id)
            
            available_uavs = [u for u in uavs_snapshot if (not u.is_busy and u.id not in uavs_to_charge)]
            
            # 计算并排序旅行成本
            sorted_uavs = sorted(
                available_uavs,
                key=lambda u: utils.calculate_travel_energy(
                    utils.calculate_distance(u.location, self.locations[loc_id])
                )
            )

            assigned = False
            for uav in sorted_uavs:
                travel_energy = utils.calculate_travel_energy(
                    utils.calculate_distance(uav.location, self.locations[loc_id])
                )
                servicing_energy = self._get_total_energy_for_vnf(vnf, req)
                return_energy = utils.calculate_travel_energy(
                    utils.calculate_distance(self.locations[loc_id], config.BASE_STATION_LOCATION)
                )
                if uav.energy < travel_energy + servicing_energy + return_energy :
                    if uav.id not in uavs_to_charge:
                         uavs_to_charge.append(uav.id)
                elif uav.cpu_capacity >= vnf.workload:
                    u_prime[loc_id] = uav.id
                    uav.is_busy = True
                    assigned = True
                    break # 已为该位置找到UAV
            
            if not assigned:
                # print(f"无法为位置 {loc_id} 找到合适的UAV。")
                return False, {}, []


        # 4. 最终检查 (line 42)
        if len(u_prime) == len(required_locations):
            return True, u_prime, list(set(uavs_to_charge))
        else:
            return False, {}, list(set(uavs_to_charge))

    def _allocate_resources(self, req, assigned_uav_ids):
        """根据承诺结果，实际更新UAV的状态"""
        for loc_id, uav_id in assigned_uav_ids.items():
            uav = next(u for u in self.uavs if u.id == uav_id)
            vnf = next(v for v in req.vnfs if v.location_id == loc_id)
            print(f"分配给UAV {uav_id} 的VNF: {vnf}")

            # 1. 扣除旅行能耗
            travel_dist = utils.calculate_distance(uav.location, self.locations[loc_id])
            travel_energy = utils.calculate_travel_energy(travel_dist)
            uav.energy -= travel_energy

            # 2. 扣除服务能耗
            servicing_energy = self._get_total_energy_for_vnf(vnf, req)
            uav.energy -= servicing_energy

            # 3. 扣除计算资源
            uav.cpu_capacity -= vnf.workload # MCC to GCC

            # 4. 更新UAV位置
            uav.location_id = loc_id
            uav.location = self.locations[loc_id]
            uav.is_busy = True # 在当前时间槽标记为忙碌



    def _get_total_energy_for_vnf(self, vnf, req):
        """获取服务单个VNF所需的能耗"""
        communication_partners = {}
        for v1_id, v2_id in req.communication_demands:
            demand = req.communication_demands[(v1_id, v2_id)]
            if vnf.id == v1_id:
                partner_vnf = next(v for v in req.vnfs if v.id == v2_id)
                communication_partners[partner_vnf] = demand
            elif vnf.id == v2_id:
                partner_vnf = next(v for v in req.vnfs if v.id == v1_id)
                communication_partners[partner_vnf] = demand
        
        return utils.calculate_servicing_energy(vnf, communication_partners, self.locations)
