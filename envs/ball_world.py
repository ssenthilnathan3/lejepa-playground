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
        self.dt = 5.0
        self.state = np.array([x, y, vx, vy], dtype=np.float32)
        self.name = self.__class__.__name__

    def step(self):
        self.x += self.vx * self.dt
        self.y += self.vy * self.dt

        if self.x < 0 or self.x > 1:
            self.vx = -self.vx

        if self.y < 0 or self.y > 1:
            self.vy = -self.vy

        self.vx = max(-0.4, min(self.vx, 0.4))
        self.vy = max(-0.4, min(self.vy, 0.4))

        self.state = np.array([self.x, self.y, self.vx, self.vy], dtype=np.float32)
        return self.state

    def reset(self):
        self.x = random.randint(0, 1)
        self.y = random.randint(0, 1)

        self.vx = random.uniform(-0.4, 0.4)
        self.vy = random.uniform(-0.4, 0.4)

        self.state = np.array([self.x, self.y, self.vx, self.vy], dtype=np.float32)
        return self.state

    def observe(self):
        return self.state
