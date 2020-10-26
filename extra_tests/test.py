import time, sys


def print2DGrid(X,Y, loc_x, loc_y):
    for x in range(X):
        for y in range(Y):
            if (loc_x == x and loc_y == y):
                print('O', end='\t')
            else:
                print(0, end='\t')
        print()

for i in range(5):
    print2DGrid(5, 5, i, i)
    time.sleep(2)
    print("\033[H\033[J")