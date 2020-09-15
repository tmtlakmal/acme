from gym import spaces
from external_env.soccer.soccer_game import SoccerGame
from external_env.soccer.player import Actions
from external_env.soccer.controller import RandomController, OffensiveController, DefensiveController, BaseController
import gym
import numpy as np


class SoccerGym(gym.Env):

    def __init__(self, grid_X, grid_Y):
        super(SoccerGym, self).__init__()

        self.action_space = spaces.Discrete(len(Actions))
        self.game : SoccerGame = SoccerGame(grid_X, grid_Y)
        self.observation_space = spaces.Box(low=np.array([0, 0, 0, 0, # player positions
                                                 0, 0, grid_X-1, grid_Y-1, # grid sizes
                                                 (grid_Y-1) // 2, 0, (grid_Y-1) // 2+1, # goal positions
                                                 (grid_Y-1) // 2, grid_X-1, (grid_Y-1) // 2+1, # goal positions
                                                 0]), # has ball
                                             high=np.array([grid_X-1, grid_Y-1, grid_X-1, grid_Y-1,
                                                   0, 0, grid_X-1, grid_Y-1,
                                                   (grid_Y - 1) // 2, 0, (grid_Y - 1) // 2 + 1,
                                                   (grid_Y - 1) // 2, grid_X - 1, (grid_Y - 1) // 2 + 1,
                                                   1])
                                                   ,dtype=np.float32)
        self.state_size = 15
        self.opponent : BaseController = None
        self.step_count = 0

    def reset(self):
        #self.opponent = OffensiveController('B') if np.random.randint(0,2) == 1 else DefensiveController('B')
        self.opponent = np.random.choice([DefensiveController('B')], 1)[0]
        self.game.reset()
        self.step_count = 0
        return self.game.get_serialized_states()

    def step(self, action_A):
        action_B = self.opponent.select_action(self.game.get_states())
        outcome = self.game.step(action_A, action_B)

        done = True if self.step_count == 100 or outcome != 0 else False
        self.step_count += 1

        return self.game.get_serialized_states(), float(self.game.check_for_winner()),  done, {}

    def render(self, mode='human'):
        pass

    def close(self):
        pass
