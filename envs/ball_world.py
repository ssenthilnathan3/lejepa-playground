import random
from typing import Literal

import numpy as np

Direction = Literal["up", "down", "left", "right"]


class BallWorld:
    def __init__(self, x, y, vx, vy):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.state = np.array([x, y, vx, vy])
        self.name = self.__class__.__name__

    def step(self):
        self.x += self.vx
        self.y += self.vy

        if self.x < 0 or self.x > 1:
            self.vx = -self.vx

        if self.y < 0 or self.y > 1:
            self.vy = -self.vy

        self.vx = max(-0.05, min(self.vx, 0.05))
        self.vy = max(-0.05, min(self.vy, 0.05))

        self.state = np.array([self.x, self.y, self.vx, self.vy])
        return self.state

    def reset(self):
        self.x = random.randint(0, 1)
        self.y = random.randint(0, 1)

        self.vx = random.uniform(-0.05, 0.05)
        self.vy = random.uniform(-0.05, 0.05)

        self.state = np.array([self.x, self.y, self.vx, self.vy])
        return self.state

    def observe(self):
        return self.state
