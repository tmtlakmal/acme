from typing import List


class VehicleControl:
    def __init__(self, index, paddle_command):
        self.index = index
        self.paddle_command = paddle_command


class SimulatorExternalControlObjects:
    def __init__(self, vehicles: List[VehicleControl]):
        self.vehicles: List[VehicleControl] = vehicles
