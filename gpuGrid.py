from math import ceil, floor


class GRID(object):
    def __init__(self):
        self.threads_x = 32
        self.threads_y = 16
        self.arrangement = 1

        self.block_dict = {
            1: (1, 16),
            2: (5, 16),
            3: (10, 16),
            4: (32, 16),
            5: (64, 16),
            6: (125, 16)}

    def __str__(self):
        return 'Grid object has {} blocks and ({}, {}) threads per block'.format(self.block_dict[self.arrangement], self.threads_x, self.threads_y)

    def blockAlloc(self, n, multiplier):
        if n <= 350:
            self.arrangement = 1
        elif n <= 600:
            self.arrangement = 1
        elif n <= 800:
            self.arrangement = 3
        elif n > 800:
            self.arrangement = 3 

        return self.block_dict[self.arrangement][0], self.block_dict[self.arrangement][1]


if __name__ == '__main__':
    grid = GRID()
    print(grid.blockAlloc(4, 4))
