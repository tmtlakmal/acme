from external_env.soccer.player import Actions
import abc
import numpy as np

class BaseController(abc.ABC):
    def __init__(self, player_name : str):
        assert player_name == 'A' or player_name == 'B'

        self.player_name = player_name

        # player data
        self.my_pos_x = 0
        self.my_pos_y = 0
        self.opp_pos_x = 0
        self.opp_pos_y = 0
        self.has_ball = False

        self.my_goal_x = 0
        self.my_goal_y_bottom = 0
        self.my_goal_y_up = 0
        self.opp_goal_x = 0
        self.opp_goal_y_bottom = 0
        self.opp_goal_y_up = 0

        # player stats
        self.last_move : Actions = Actions.Stand
        self.moves_freq = np.zeros(shape=4, dtype=np.float32)
        self.last_coord_x = 0
        self.last_coord_y = 0


    def decode_states(self, states):
        self.last_coord_x = self.my_pos_x
        self.last_coord_y = self.my_pos_y

        if self.player_name == 'A':
            self.my_pos_x, self.my_pos_y = states['player A']
            self.opp_pos_x, self.opp_pos_y = states['player B']
            self.has_ball = True if states['has ball'] == 1 else False
            self.my_goal_y_bottom, self.my_goal_x, self.my_goal_y_up = states['goal A']
            self.opp_goal_y_bottom, self.opp_goal_x, self.opp_goal_y_up= states['goal B']
        else:
            self.my_pos_x, self.my_pos_y = states['player B']
            self.opp_pos_x, self.opp_pos_y= states['player A']
            self.has_ball = True if states['has ball'] == 0 else False
            self.my_goal_y_bottom, self.my_goal_x,  self.my_goal_y_up = states['goal B']
            self.opp_goal_y_bottom, self.opp_goal_x,  self.opp_goal_y_up= states['goal A']



    def is_towards(self, coord_x, coord_y, coord_y2 = None):
        if abs(coord_x - self.my_pos_x) < abs(coord_x - self.last_coord_x):
            return True
        if abs(coord_y - self.my_pos_y) < abs(coord_y - self.last_coord_y):
            if coord_y2:
                if not abs(coord_y2 - self.my_pos_y) < abs(coord_y2 - self.last_coord_y):
                    return False
            return True
        return False

    def update_move_freq(self):
        alpha = 0.8
        self.moves_freq[0] *= alpha * (1 if self.is_towards(self.opp_pos_x, self.opp_pos_y) else 0) # toward opponent
        self.moves_freq[1] *= alpha * (0 if self.is_towards(self.opp_pos_x, self.opp_pos_y) else 1) # avoiding opponent
        # towards goal
        self.moves_freq[2] *= alpha * (1 if self.is_towards(self.my_goal_x, self.my_goal_y_up, self.my_goal_y_bottom)  else 0)
        # towards opponents goal
        self.moves_freq[3] *= alpha * (1 if self.is_towards(self.opp_goal_x, self.opp_goal_y_up, self.opp_goal_y_bottom) else 0)

    def get_history(self) -> np.array:
        return np.append(self.moves_freq, self.last_move.value)

    @abc.abstractmethod
    def select_action(self, states):
        "select an action given states"

class RandomController(BaseController):
    def select_action(self, states):
        self.last_move = Actions(np.random.randint(0, 5))
        return self.last_move

class OffensiveController(BaseController):

    def __init__(self, player_name):
        super().__init__(player_name)
        self.stats = []

    def select_action(self, states):
        self.decode_states(states)
        action = Actions.Stand
        if self.has_ball:
            if self.opp_goal_y_up < self.my_pos_y:
                action = Actions.Down
            elif self.opp_goal_y_bottom > self.my_pos_y:
                action = Actions.Up
            else:
                if self.opp_goal_x > self.my_pos_x:
                    action = Actions.Right
                else:
                    action = Actions.Left
        else:
            if self.opp_pos_y < self.my_pos_y:
                action = Actions.Down
            elif self.opp_pos_y > self.my_pos_y:
                action = Actions.Up
            else:
                if self.opp_pos_x > self.my_pos_x:
                    action = Actions.Right
                else:
                    action = Actions.Left

        self.last_move = action
        return action

class DefensiveController(BaseController):

    def select_action(self, states):
        self.decode_states(states)
        action = Actions.Stand
        if self.has_ball:
            # move away from the opponent
            if self.opp_pos_x > self.my_pos_x:
                action = Actions.Left
            elif self.opp_pos_x < self.my_pos_x:
                action = Actions.Right
            elif self.opp_pos_y > self.my_pos_y:
                action = Actions.Down
            else:
                action = Actions.Up
        else:
            # if within goals x range
            if self.my_pos_y <= self.my_goal_y_up and self.my_pos_y >= self.my_goal_y_bottom:
                if self.my_goal_x - 1 > self.my_pos_x:
                    action = Actions.Right
                elif self.my_pos_y == self.my_goal_y_up:
                    action = Actions.Down
                elif self.my_pos_y == self.my_goal_y_bottom:
                    action = Actions.Up
                else:
                    action = Actions.Left
        self.last_move = action
        return action



