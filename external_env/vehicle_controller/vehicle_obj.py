import numpy as np

class Vehicle:

    def __init__(self):
        self.speed = 0
        self.location = 400
        self.time_to_reach = 0
        self.max_speed = 22
        self.max_acc = 2
        self.last_command = 0

    def step(self, acceleration):
        done = False
        reward = 0
        info = {'is_success':False}
        self.last_command = acceleration

        if acceleration >= 0:
            self.speed += acceleration*self.max_acc*(1 - (self.speed/self.max_speed)**4)
        else:
            self.speed += acceleration*self.max_acc

        if self.speed < 0:
            self.speed = 0
        if self.speed > self.max_speed:
            self.speed = self.max_speed

        self.location -= self.speed
        self.time_to_reach -= 1

        if (self.location <= 0 or self.time_to_reach == 0):
            done = True

        if done:
            if (self.location < 10 and self.time_to_reach < 2):
                reward = 10 + self.speed
                info = {'is_success':True}
            else:
                reward = -10.0
        else:
            reward = -(self.location/400)


        return [int(self.speed), self.time_to_reach, self.location], reward, done, info

    def reset(self):
        self.time_to_reach = float(np.random.randint(30,45))
        self.speed = 0.0
        self.location = 400.0
        return [self.speed, self.time_to_reach, self.location]

    def assign_data(self, time_to_reach=45):
        self.time_to_reach = float(np.random.randint(30,time_to_reach))
        self.speed = 0.0
        self.location = 380.0
        return [self.speed, self.time_to_reach, self.location]

