import math
import random


class KineticArm:
    def __init__(self, theta_1, theta_2, dtheta_1, dtheta_2):
        self.theta_1 = theta_1
        self.theta_2 = theta_2
        self.dtheta_1 = dtheta_1
        self.dtheta_2 = dtheta_2
        self.state = (theta_1, theta_2, dtheta_1, dtheta_2)
        self.name = self.__class__.__name__

    def step(self, dtheta_1, dtheta_2):
        self.dtheta_1 += dtheta_1
        self.dtheta_2 += dtheta_2

        self.theta_1 = (self.theta_1 + self.dtheta_1) % (2 * math.pi)
        self.theta_2 = (self.theta_2 + self.dtheta_2) % (2 * math.pi)

        self.state = (self.theta_1, self.theta_2, self.dtheta_1, self.dtheta_2)

        return self.state

    def reset(self):
        self.theta_1 = random.uniform(0, 2 * math.pi)
        self.theta_2 = random.uniform(0, 2 * math.pi)

        self.dtheta_1 = 0
        self.dtheta_2 = 0

        self.state = (self.theta_1, self.theta_2, self.dtheta_1, self.dtheta_2)
        return self.state

    def observe(self):
        return self.state
