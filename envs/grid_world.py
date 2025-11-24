from typing import Literal

import numpy as np

Direction = Literal["up", "down", "left", "right"]


class GridWorld:
    def __init__(self):
        self.grid_size = 10
        self.name = self.__class__.__name__
        self.reset()

    def reset(self):
        self.state = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        self.current_pos = (0, 0)
        self.state[0, 0] = 1.0
        return self.state

    def step(self, action):
        x, y = self.current_pos
        nx, ny = x, y

        if action == "up":
            ny -= 1
        if action == "down":
            ny += 1
        if action == "left":
            nx -= 1
        if action == "right":
            nx += 1

        if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
            self.state[y, x] = 0.0
            self.state[ny, nx] = 1.0
            self.current_pos = (nx, ny)

        return self.state

    def observe(self):
        return self.state
