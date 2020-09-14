class Box():
    def __init__(self, left_bottom_x, left_bottom_y, right_up_x, right_up_y):
        self.left_bottom_x = left_bottom_x
        self.left_bottom_y = left_bottom_y
        self.right_up_x = right_up_x
        self.right_up_y = right_up_y

    def is_within_box(self, x, y):

        if (x >= self.left_bottom_x and x <= self.right_up_x and
            y >= self.left_bottom_y and y <= self.right_up_y):
            return True
        else:
            return False

    def get_coordinates(self):
        return self.left_bottom_x, self.left_bottom_y, self.right_up_x, self.right_up_y


class Grid():

    def __init__(self, grid_X, grid_Y):
        self.grid_X = grid_X
        self.grid_Y = grid_Y

        #Goal Positions
        self.goal_A = Box(0, (grid_Y-1)//2, 0, (grid_Y-1)//2 + 1)
        self.goal_B = Box(grid_X-1, (grid_Y-1)//2, grid_X-1, (grid_Y-1)//2 + 1)

        self.play_area = Box(1, 0, grid_X-2, grid_Y-1)

    def is_coordinates_allowed(self, x, y):
        if ( self.goal_A.is_within_box(x,y) or
             self.goal_B.is_within_box(x,y) or
             self.play_area.is_within_box(x,y)):
            return True
        else:
            return False

    def is_in_goal(self, goal_name : str, x, y):
        assert goal_name == 'A' or goal_name == 'B'

        if goal_name == 'A':
            return self.goal_A.is_within_box(x,y)
        else:
            return self.goal_B.is_within_box(x,y)



