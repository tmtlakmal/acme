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

    def decode_states(self, states):
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


    @abc.abstractmethod
    def select_action(self, states):
        "select an action given states"

class RandomController(BaseController):
    def select_action(self, states):
        return Actions(np.random.randint(0,5))

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

        return action