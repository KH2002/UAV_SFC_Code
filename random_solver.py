# -*- coding: utf-8 -*-
"""
RandomOrder 算法：按照请求生成顺序处理 SFC 部署
"""
import random
import config
import utils


class RandomOrderSolver:
    def __init__(self, uavs, requests, locations):
        self.uavs = uavs
        self.requests = requests
        self.locations = locations
        self.serviced_requests_timeline = {}

    def solve(self):
        print("--- RandomOrder 算法开始 ---")

        for t in range(1, config.NUM_TIME_SLOTS + 1):
            print(f"\n--- 时间槽 {t} ---")
            self.serviced_requests_timeline[t] = []
            uavs_to_charge = set()

            # 清理状态，新的时间槽从空闲状态开始
            for uav in self.uavs:
                uav.is_busy = False

            for req in self.requests:
                if req.is_serviced:
                    continue

                success, recharge_candidates = self._handle_request(req)
                uavs_to_charge.update(recharge_candidates)

                if success:
                    req.is_serviced = True
                    self.serviced_requests_timeline[t].append(req.id)
                    print(f"请求 {req.id} 分配成功。")
                else:
                    print(f"请求 {req.id} 分配失败。")

            # 处理需要回到基站充电的 UAV
            for uav_id in uavs_to_charge:
                uav = next(u for u in self.uavs if u.id == uav_id)
                if not uav.is_busy:
                    uav.location = config.BASE_STATION_LOCATION
                    uav.location_id = 0
                    uav.energy = config.UAV_BATTERY_CAPACITY

            # 恢复计算资源，为下一时间槽做准备
            for uav in self.uavs:
                uav.cpu_capacity = config.UAV_COMPUTATION_CAPACITY

            # 所有请求均被服务则提前结束
            if all(r.is_serviced for r in self.requests):
                print("所有请求均已完成。")
                break

        print("\n--- RandomOrder 算法结束 ---")
        return self.serviced_requests_timeline

    def _handle_request(self, req):
        if not self._check_link_capacity(req):
            return False, []

        snapshot = self._snapshot_uavs()
        uavs_to_charge = set()

        # 逐个 VNF 分配
        for vnf in req.vnfs:
            candidates = self._collect_candidates(vnf, req)
            if not candidates:
                self._restore_uavs(snapshot)
                return False, []

            chosen = random.choice(candidates)
            self._deploy_vnf(chosen, vnf)

            # 检查是否需要在时间槽末回到基站充电
            target_coords = self.locations[vnf.location_id]
            return_energy = utils.calculate_travel_energy(
                utils.calculate_distance(target_coords, config.BASE_STATION_LOCATION)
            )
            if chosen['uav'].energy < return_energy:
                uavs_to_charge.add(chosen['uav'].id)

        return True, list(uavs_to_charge)

    def _collect_candidates(self, vnf, req):
        candidates = []
        communication_partners = self._get_communication_partners(vnf, req)
        target_coords = self.locations[vnf.location_id]

        for uav in self.uavs:
            if uav.is_busy:
                continue
            if uav.cpu_capacity < vnf.workload:
                continue

            travel_energy = utils.calculate_travel_energy(
                utils.calculate_distance(uav.location, target_coords)
            )
            servicing_energy = utils.calculate_servicing_energy(vnf, communication_partners, self.locations)
            return_energy = utils.calculate_travel_energy(
                utils.calculate_distance(target_coords, config.BASE_STATION_LOCATION)
            )

            if uav.energy >= travel_energy + servicing_energy + return_energy:
                candidates.append({
                    'uav': uav,
                    'travel_energy': travel_energy,
                    'servicing_energy': servicing_energy
                })

        return candidates

    def _deploy_vnf(self, candidate, vnf):
        uav = candidate['uav']
        travel_energy = candidate['travel_energy']
        servicing_energy = candidate['servicing_energy']

        uav.energy -= (travel_energy + servicing_energy)
        uav.cpu_capacity -= vnf.workload
        uav.location_id = vnf.location_id
        uav.location = self.locations[vnf.location_id]
        uav.is_busy = True

    def _get_communication_partners(self, vnf, req):
        partners = {}
        for (v1_id, v2_id), demand in req.communication_demands.items():
            if vnf.id == v1_id:
                partner = next(v for v in req.vnfs if v.id == v2_id)
                partners[partner] = demand
            elif vnf.id == v2_id:
                partner = next(v for v in req.vnfs if v.id == v1_id)
                partners[partner] = demand
        return partners

    def _check_link_capacity(self, req):
        required_locations = req.get_required_location_ids()
        for i in range(len(required_locations)):
            for j in range(i + 1, len(required_locations)):
                loc_id1 = required_locations[i]
                loc_id2 = required_locations[j]

                vnf1 = next(v for v in req.vnfs if v.location_id == loc_id1)
                vnf2 = next(v for v in req.vnfs if v.location_id == loc_id2)

                demand = req.communication_demands.get((vnf1.id, vnf2.id)) or \
                    req.communication_demands.get((vnf2.id, vnf1.id), 0)

                if demand > 0:
                    distance = utils.calculate_distance(self.locations[loc_id1], self.locations[loc_id2])
                    capacity = utils.calculate_link_capacity(distance)
                    if demand > capacity:
                        return False
        return True

    def _snapshot_uavs(self):
        return {
            uav.id: {
                'energy': uav.energy,
                'cpu_capacity': uav.cpu_capacity,
                'location': uav.location,
                'location_id': uav.location_id,
                'is_busy': uav.is_busy
            }
            for uav in self.uavs
        }

    def _restore_uavs(self, snapshot):
        for uav in self.uavs:
            state = snapshot[uav.id]
            uav.energy = state['energy']
            uav.cpu_capacity = state['cpu_capacity']
            uav.location = state['location']
            uav.location_id = state['location_id']
            uav.is_busy = state['is_busy']
