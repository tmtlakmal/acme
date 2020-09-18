from external_env.soccer.grid import Grid
from external_env.soccer.player import Player, Movable, Actions
from external_env.soccer.controller import OffensiveController, RandomController, DefensiveController
import  numpy as np

class SoccerGame():

    def __init__(self, grid_X=9, grid_Y=6, max_duration=100):
        self.player_A = Player('A')
        self.player_B = Player('B')

        self.grid_X = grid_X
        self.grid_Y = grid_Y

        self.grid = Grid(grid_X=self.grid_X, grid_Y=self.grid_Y)

        self.stats = np.zeros(shape=3, dtype=np.int32)
        self.steps = 0
        self.max_steps = max_duration

    def step(self, action_A, action_B):
        # Move player based on the actions
        next_x_A, next_y_A = self.player_next_pos(self.player_A, action_A)
        next_x_B, next_y_B = self.player_next_pos(self.player_B, action_B)

        if (next_x_A == next_x_B and next_y_A == next_y_B):
            self.update_ball_acquisition()
        else:
            self.player_A.move(next_x_A, next_y_A)
            self.player_B.move(next_x_B, next_y_B)
        self.steps += 1
        return self.check_for_winner()

    def get_states(self):
        dict = { 'player A' : self.player_A.get_coordinates(),
                 'player B' : self.player_B.get_coordinates(),
                 'play area' : self.grid.play_area.get_coordinates(),
                 'goal A' : self.grid.goal_A.get_coordinates()[1:4], # Goal y coordinates are same hence remove
                 'goal B' : self.grid.goal_B.get_coordinates()[1:4], # y from upper right coordinates
                 'has ball': 1 if self.player_A.has_ball() else 0}
        return dict

    def get_serialized_states(self, extra=False):
        return np.array(list(self.player_A.get_coordinates() + \
                    self.player_B.get_coordinates() + \
                    self.grid.play_area.get_coordinates() + \
                    self.grid.goal_A.get_coordinates()[1:4] + \
                    self.grid.goal_B.get_coordinates()[1:4] + \
                    (1 if self.player_A.has_ball() else 0,)), dtype=np.float32 )


    def check_for_winner(self):
        x, y = self.player_B.get_coordinates()
        if self.grid.is_in_goal('A', x, y) and self.player_B.has_ball():
            self.stats[0] += 1
            return -1

        x, y = self.player_A.get_coordinates()
        if self.grid.is_in_goal('B', x, y) and self.player_A.has_ball():
            self.stats[1] += 1
            return 1

        if self.max_steps == self.steps:
            self.stats[2] += 1

        return 0

    def update_ball_acquisition(self):
        if self.player_A.has_ball():
            self.player_A.give_ball()
            self.player_B.acquire_ball()
        else:
            self.player_B.give_ball()
            self.player_A.acquire_ball()

    def player_next_pos(self, player : Movable, action):
        x, y = player.get_coordinates()
        if Actions(action) == Actions.Up:
            if self.grid.is_coordinates_allowed(x, y+1):
                y += 1
        elif Actions(action) == Actions.Down:
            if self.grid.is_coordinates_allowed(x, y-1):
                y -= 1
        elif Actions(action) == Actions.Left:
            if self.grid.is_coordinates_allowed(x-1, y):
                x -= 1
        elif Actions(action) == Actions.Right:
            if self.grid.is_coordinates_allowed(x+1, y):
                x += 1
        return x, y

    def reset(self):
        self.steps = 0
        x = np.random.randint(1, self.grid_X//2)
        y = np.random.randint(0, self.grid_Y-1)
        self.player_A.move(x,y)

        x = np.random.randint(self.grid_X//2+1, self.grid_X-2)
        y = np.random.randint(0, self.grid_Y-1)
        self.player_B.move(x,y)

        if np.random.randint(0,2) == 1:
            self.player_A.acquire_ball()
            self.player_B.give_ball()
        else:
            self.player_B.acquire_ball()
            self.player_A.give_ball()

if __name__=='__main__':
    game = SoccerGame()
    game.reset()
    done = False
    playerA = RandomController('A')
    playerB = DefensiveController('B')

    num_win = 0
    num_tie = 0
    num_loose = 0

    for i in range(5000):
        steps = 0
        while not done:
            states = game.get_states()
            d = game.get_serialized_states()
            a = playerA.select_action(states).value
            b = playerB.select_action(states).value
            outcome = game.step(a,b)
            steps += 1
            #print("Player Positions:", Actions(a), game.player_A.get_coordinates(), Actions(b), game.player_B.get_coordinates())

            if outcome != 0 or steps == 100:
                if outcome == 1:
                    num_win += 1
                elif outcome == -1:
                    num_loose += 1
                else:
                    num_tie +=1
                done = True
        done = False
        game.reset()

    print("Num wins: ", num_win, " Num loose: ", num_loose, " Num tie", num_tie)