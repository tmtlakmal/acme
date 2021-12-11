from typing import List


class VehicleExternal:
    def __init__(self, vid, headPosition, headPositionFromEnd, speed, edgeId, done, timeRemain, is_success, gap,
                 frontVehicleSpeed, frontVehicleTimeRemain, frontVehicleDistance, crashed, externalControl, isVirtual, isCrashRisk,
                 backvGap, backvSpeed, backvTimeRemain, backvDistance, backvVirtual, backvCrashed):
        self.vid = vid
        self.headPosition = headPosition
        self.headPositionFromEnd = headPositionFromEnd
        self.speed = speed
        self.edgeId = edgeId
        self.done = done
        self.timeRemain = timeRemain
        self.is_success = is_success
        self.gap = gap
        self.frontVehicleSpeed = frontVehicleSpeed
        self.frontVehicleTimeRemain = frontVehicleTimeRemain
        self.frontVehicleDistance = frontVehicleDistance
        self.crashed = crashed
        self.externalControl = externalControl
        self.isVirtual = isVirtual
        self.isCrashRisk = isCrashRisk

        self.backvGap = backvGap
        self.backvSpeed = backvSpeed
        self.backvTimeRemain = backvTimeRemain
        self.backvDistance = backvDistance
        self.backvVirtual = backvVirtual
        self.backvCrashed = backvCrashed


class TrafficData:
    def __init__(self, vehicles: List[VehicleExternal]):
        # self.edges: List[VehicleExternal] = []
        self.vehicles: List[VehicleExternal] = vehicles

    def get_vehicle_with_id(self, id):
        for v in self.vehicles:
            if v["vid"] == id:
                return VehicleExternal(**v)