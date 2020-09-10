import abc
class BaseVehicleController(abc.ABC):

    def __init__(self, step_size):
        self.step_size = step_size

    @abc.abstractmethod
    def compute_acceleration(self, time_remain, speed, distance_remain) -> int:
        """ compute the acceleration give vehicle positions"""


class HumanDriven(BaseVehicleController):

    def __init__(self, step_size, max_acceleration, max_deacceleration):
        super().__init__(step_size)
        self.max_acc = max_acceleration
        self.max_deacc = max_deacceleration


    def compute_acceleration(self, time_remain, speed, distance_remain) -> int:

        if time_remain < 3 and distance_remain > 0:
            return 2

        if not self.is_possible_to_stop(speed, distance_remain):
            return 0

        return 2

    def is_possible_to_stop(self, speed, distance_remain):
        dist_to_stop = (speed**2)/(2*self.max_deacc)
        if (dist_to_stop < 0.25*distance_remain):
            return True
        else:
            return False

def map_to_paddle_command(action):
    if action == 0:
        paddleCommand = -1
    elif action == 1:
        paddleCommand = 0
    else:
        paddleCommand = 1
    return paddleCommand

from external_env.vehicle_controller.vehicle_obj import Vehicle
if __name__== '__main__':
    v = Vehicle()

    controller = HumanDriven(1, v.max_acc, v.max_acc)

    for i in range(10):
        v.reset()
        action = map_to_paddle_command(controller.compute_acceleration(v.time_to_reach, v.speed, v.location))
        states, _, done, _ = v.step(action)
        while not states[2] < 0:
            action = map_to_paddle_command(controller.compute_acceleration(v.time_to_reach, v.speed, v.location))
            states, _, done, info = v.step(action)
        print("Vehicle data: ", states)
        if info['is_success']:
            print("## Correctly Ended ##")
        else:
            print(" ---- ")