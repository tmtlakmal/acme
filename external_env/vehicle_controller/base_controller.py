import abc
import numpy as np
from external_env.vehicle_controller.vehicle_obj import Vehicle

class BaseVehicleController(abc.ABC):

    def __init__(self, step_size):
        self.step_size = step_size

    @abc.abstractmethod
    def compute_acceleration(self, vehicle : Vehicle) -> int:
        """ compute the acceleration give vehicle positions"""


class HumanDriven(BaseVehicleController):

    def __init__(self, step_size, max_acceleration, max_deacceleration):
        super().__init__(step_size)
        self.max_acc = max_acceleration
        self.max_deacc = max_deacceleration


    def compute_acceleration(self, vehicle : Vehicle) -> int:

        if vehicle.time_to_reach < 3 and vehicle.location > 0:
            return 2

        if not self.is_possible_to_stop(vehicle.speed, vehicle.location):
            return 0

        return 2

    def is_possible_to_stop(self, speed, distance_remain):
        dist_to_stop = (speed**2)/(2*self.max_deacc)
        if (dist_to_stop < 0.25*distance_remain):
            return True
        else:
            return False

class HumanDriven_1(BaseVehicleController):

    def __init__(self, step_size, max_acceleration, max_deacceleration):
        super().__init__(step_size)
        self.max_acc = max_acceleration
        self.max_deacc = max_deacceleration
        self.step = 0


    def compute_acceleration(self, vehicle : Vehicle) -> int:
        self.step += 1
        if self.step < 12:
            return  2 if vehicle.speed < 4 else 1
        else:
            return 2

class HumanDriven_2(BaseVehicleController):

    def __init__(self, step_size, max_acceleration, max_deacceleration):
        super().__init__(step_size)
        self.max_acc = max_acceleration
        self.max_deacc = max_deacceleration
        self.step = 0


    def compute_acceleration(self, vehicle : Vehicle) -> int:
        self.step += 1
        if self.step < 10:
            return  2
        elif self.step < 20:
            return 0
        else:
            return 2

class Random(BaseVehicleController):

    def __init__(self, step_size, max_acceleration, max_deacceleration):
        super().__init__(step_size)
        self.max_acc = max_acceleration
        self.max_deacc = max_deacceleration
        self.step = 0
        self.last_action = 0

    def compute_acceleration(self, vehicle : Vehicle) -> int:
        self.step += 1
        if self.step % 3:
            self.last_action = np.random.randint(3)
        return self.last_action



def map_to_paddle_command(action):
    if action == 0:
        paddleCommand = -1
    elif action == 1:
        paddleCommand = 0
    else:
        paddleCommand = 1
    return paddleCommand


if __name__== '__main__':
    v = Vehicle()

    controller = HumanDriven_2(1, v.max_acc, v.max_acc)

    for i in range(1):
        v.reset()
        steps = 0
        action = map_to_paddle_command(v)
        states, _, done, _ = v.step(action)
        while (states[2] > 0) and (steps < 100):
            action = map_to_paddle_command(controller.compute_acceleration(v))
            states, _, done, info = v.step(action)
            steps += 1
            print(states)
        print("Vehicle data: ", states)
        if info['is_success']:
            print("## Correctly Ended ##")
        else:
            print(" ---- ")