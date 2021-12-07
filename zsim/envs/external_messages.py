from typing import List


class VehicleExternal:
    def __init__(self, vid, headPosition, headPositionFromEnd, speed, edgeId, done, timeRemain, is_success, gap,
                 frontVehicleSpeed, frontVehicleTimeRemain, frontVehicleDistance, crashed, externalControl, isVirtual, isCrashRisk):
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


class TrafficData:
    def __init__(self, vehicles: List[VehicleExternal]):
        # self.edges: List[VehicleExternal] = []
        self.vehicles: List[VehicleExternal] = vehicles
