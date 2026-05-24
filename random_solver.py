# -*- coding: utf-8 -*-
"""
RandomOrder 算法：按照请求生成顺序处理 SFC 部署
"""
import random
import config
import utils


class RandomOrderSolver:
    def __init__(self, uavs, requests, locations, num_time_slots=None):
        self.uavs = uavs
        self.requests = requests
        self.locations = locations
        self.num_time_slots = int(num_time_slots) if num_time_slots is not None else int(config.NUM_TIME_SLOTS)
        self.serviced_requests_timeline = {}
        self.total_energy_consumed = 0.0

    def solve(self):
        print("--- RandomOrder 算法开始 ---")
        self.total_energy_consumed = 0.0

        for t in range(1, self.num_time_slots + 1):
            self.serviced_requests_timeline[t] = []
            # 新时隙开始：UAV 恢复为空闲，可参与本时隙调度
            for uav in self.uavs:
                uav.is_busy = False

            # 在当前时隙内持续尝试，直到没有任何请求可被成功部署
            while True:
                progress_made = False

                # 打乱未服务请求，保持随机策略
                unserviced_requests = [r for r in self.requests if not r.is_serviced]
                random.shuffle(unserviced_requests)

                for req in unserviced_requests:
                    success = self._handle_request(req)
                    if success:
                        req.is_serviced = True
                        self.serviced_requests_timeline[t].append(req.id)
                        progress_made = True

                # 当前时隙已经无法继续完成任何请求，推进到下一时隙
                if not progress_made:
                    break

                # 所有请求已服务则提前结束
                if all(r.is_serviced for r in self.requests):
                    break

            # 时隙推进：空闲 UAV 返回基站并充电；重置计算资源和忙碌状态
            returned_to_base = self._advance_time_slot()

            # 输出每个时间槽的关键信息
            num_sfc = len(self.serviced_requests_timeline[t])
            print(f"时间槽 {t}: 返回基站UAV={returned_to_base}, 完成SFC={num_sfc}")

            # 所有请求均被服务则提前结束
            if all(r.is_serviced for r in self.requests):
                break

        total_sfc = sum(len(reqs) for reqs in self.serviced_requests_timeline.values())
        print(f"--- RandomOrder 算法结束 (总计完成SFC: {total_sfc}) ---")
        return self.serviced_requests_timeline

    def _handle_request(self, req):
        if not self._check_link_capacity(req):
            return False

        snapshot = self._snapshot_uavs()
        request_energy_consumed = 0.0

        # 逐个 VNF 分配
        for vnf in req.vnfs:
            candidates = self._collect_candidates(vnf, req)
            if not candidates:
                self._restore_uavs(snapshot)
                return False

            chosen = random.choice(candidates)
            request_energy_consumed += self._deploy_vnf(chosen, vnf)

        self.total_energy_consumed += request_energy_consumed
        return True

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

        consumed = float(travel_energy + servicing_energy)
        uav.energy -= consumed
        uav.cpu_capacity -= vnf.workload
        uav.location_id = vnf.location_id
        uav.location = self.locations[vnf.location_id]
        uav.is_busy = True
        return consumed

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

    def _advance_time_slot(self):
        """推进到下一时隙：空闲 UAV 返回基站并充电，重置资源。"""
        returned_to_base = 0

        for uav in self.uavs:
            # 空闲且不在基站：飞回基站（扣返回能耗）
            if not uav.is_busy and uav.location_id != 0:
                return_distance = utils.calculate_distance(uav.location, config.BASE_STATION_LOCATION)
                return_energy = utils.calculate_travel_energy(return_distance)
                self._consume_energy(uav, return_energy)
                uav.location = config.BASE_STATION_LOCATION
                uav.location_id = 0
                returned_to_base += 1

            # 在基站的 UAV 充满电
            if uav.location_id == 0:
                uav.energy = config.UAV_BATTERY_CAPACITY

            # 下一时隙恢复状态
            uav.cpu_capacity = config.UAV_COMPUTATION_CAPACITY
            uav.is_busy = False

        return returned_to_base

    def _consume_energy(self, uav, amount):
        """统一扣能耗并累计总能耗。"""
        if amount <= 0:
            return
        actual = min(float(uav.energy), float(amount))
        uav.energy = max(0.0, float(uav.energy) - float(amount))
        self.total_energy_consumed += actual
