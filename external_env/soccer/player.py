import numpy as np
import enum
class Actions(enum.Enum):
    Stand = 0
    Up = 1
    Down = 2
    Right = 3
    Left = 4

class Movable():
    def __init__(self, name : str):
        self.coord_x = 0
        self.coord_y = 0
        self.name = name

    def move(self, coord_x, coord_y):
        self.coord_x = coord_x
        self.coord_y = coord_y

    def reset(self, coord_x, coord_y):
        self.coord_x = coord_x
        self.coord_y = coord_y
        self.last_coord_x = coord_x
        self.last_coord_y = coord_y

    def is_same_position(self, obj1):
        if (obj1.coord_y == self.coord_y and obj1.coord_x == self.coord_x):
            return True
        else:
            return False

    def get_coordinates(self):
        return self.coord_x, self.coord_y


class Player(Movable):
    def __int__(self, name):
        super().__init__(name)
        self.is_ball = False


    def acquire_ball(self):
        #print("Ball acquired: ", self.name)
        self.is_ball = True

    def give_ball(self):
        self.is_ball = False

    def has_ball(self):
        return self.is_ball







