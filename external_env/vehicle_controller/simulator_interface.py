import abc
from gym import spaces
from external_env.vehicle_controller.base_controller import *
from external_interface.zeromq_client import ZeroMqClient

class BaseSimulatorInterface(abc.ABC):

    def __init__(self, num_actions, max_speed=22.0, time_to_reach=45.0, distance=500.0,
                 front_vehicle=False):
        self.num_actions = num_actions
        self.max_speed = max_speed
        self.time_to_reach = time_to_reach
        self.distance = distance

        self.is_front_vehicle = front_vehicle

    def get_state_space(self) -> spaces.Box:
        if not self.is_front_vehicle:
            observation_space = spaces.Box(low=np.array([0.0,0.0,0.0]),
                                                high=np.array([self.max_speed, self.time_to_reach, self.distance]), dtype=np.float32)
        else:
            observation_space = spaces.Box(low=np.array([0.0, 0.0, 0.0, 0.0, 0.0, -1.0]),
                                                high=np.array([self.max_speed, self.time_to_reach, self.distance, self.distance,
                                                               self.max_speed, 1.0]), dtype=np.float32)

        return observation_space

    def map_to_paddle_command(self, action):
        if action == 0:
            paddleCommand = -1
        elif action == 1:
            paddleCommand = 0
        else:
            paddleCommand = 1
        return paddleCommand

    @abc.abstractmethod
    def get_step_data(self, action):
        """return next step data using the message"""

    @abc.abstractmethod
    def reset(self):
        """return initial state after resetting"""

class PythonSimulator(BaseSimulatorInterface):

    def __init__(self, num_actions, step_size=1, max_speed=22.0, time_to_reach=45.0, distance=500.0,
                 front_vehicle=False, driver_type=HumanDriven):
        super().__init__(num_actions, max_speed=max_speed, time_to_reach=time_to_reach, distance=distance,
                 front_vehicle=front_vehicle)

        self.vehicle = Vehicle(max_speed=max_speed, max_acc=2)
        self.front_vehicle = Vehicle(max_speed=max_speed, max_acc=2)
        self.vehicle_controller = driver_type(step_size, self.front_vehicle.max_acc, self.front_vehicle.max_acc)
        self.step_size = step_size
        self.is_front_vehicle = front_vehicle

    def get_step_data(self, action):

        acc_front = self.vehicle_controller.compute_acceleration(vehicle=self.front_vehicle)

        paddle_command = map_to_paddle_command(action)
        paddle_command_front = map_to_paddle_command(acc_front)
        obs, _, done, info = self.vehicle.step(paddle_command, self.step_size)
        front_obs, _, _, _ = self.front_vehicle.step(paddle_command_front, self.step_size)
        obs.extend([front_obs[0], front_obs[0], acc_front])
        return obs, done, info


    def reset(self):
        l = self.vehicle.reset()
        time_to_reach = int(l[1]) - 1
        self.front_vehicle.assign_data()
        l.extend([self.vehicle.location - self.front_vehicle.location,
                  self.front_vehicle.speed, 0])

        info = {'is_success': False, 'is_virtual': False, 'is_crashed': False}
        return l, False, info

class SmartsSimulator(BaseSimulatorInterface):

    def __init__(self, vehicle_id=1, num_actions=3, front_vehicle=False):
        super().__init__(num_actions=num_actions)

        self.id = vehicle_id
        self.sim_client = ZeroMqClient()

        self.is_front_vehicle = front_vehicle
        self.front_last_speed = 0

    def get_step_data(self, action):

        paddle_command = map_to_paddle_command(action)
        message_send = {'edges': [], 'vehicles':[{"index":self.id, "paddleCommand": paddle_command}]}
        message_rvcd = self.sim_client.send_message(message_send)
        return self.decode_message(message_rvcd)

    def decode_message(self, message_rvcd):


        obs = [0, 0, 0, 0, 0, 0]
        info = {'is_success': False, 'is_virtual': False, 'is_crashed': False}
        done = False

        for vehicle in message_rvcd["vehicles"]:
            if vehicle["vid"] == self.id:

                # State information
                speed = vehicle["speed"]
                time = vehicle["timeRemain"]
                distance = vehicle["headPositionFromEnd"]
                done = vehicle["done"]
                gap = vehicle['gap']
                front_vehicle_speed = vehicle['frontVehicleSpeed']

                acc = 1 if front_vehicle_speed > self.front_last_speed else\
                     -1 if front_vehicle_speed == self.front_last_speed \
                     else -1
                self.front_last_speed = front_vehicle_speed

                obs = [speed, time, distance, gap, front_vehicle_speed, acc]

                # Additional info
                info["is_success"] = vehicle["is_success"]
                info["is_virtual"] = vehicle["isVirtual"]
                info["is_crashed"] = vehicle['crashed']

        return obs, done, info

    def reset(self):
        message_send = {'edges': [], 'vehicles': [{"index": self.id, "paddleCommand": 0}]}
        self.distance = 380
        self.front_vehicle_end_time = 0
        message_rvcd = self.sim_client.send_message(message_send)
        return self.decode_message(message_rvcd)
