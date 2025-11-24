import random
from typing import List

import numpy as np
import torch.nn as nn


class RolloutBuffer:
    def __init__(self):
        self.buf = []

    def add(self, obs_t, action_t, obs_t_1):
        self.buf.append((obs_t, action_t, obs_t_1))

    def sample(self, batch_size):
        buf_sample = random.sample(self.buf, batch_size)

        obs_t_batch = np.array([s[0] for s in buf_sample])
        action_batch = np.array([s[1] for s in buf_sample])
        obs_t1_batch = np.array([s[2] for s in buf_sample])

        return obs_t_batch, action_batch, obs_t1_batch

    def populate(self, env, n):
        if env.name == "GridWorld":
            obs_t = env.reset()
            actions = ["up", "down", "right", "left"]
            action_map = {"up": 0, "down": 1, "right": 2, "left": 3}
            for _ in range(n):
                action_t = random.choice(actions)
                obs_t_1 = env.step(action_t)

                # Flatten current observation
                obs_t_flattened = np.array(obs_t).flatten().astype(np.float32)
                # Flatten next observation
                obs_t_1_flattened = np.array(obs_t_1).flatten().astype(np.float32)
                action_onehot = np.eye(4)[action_map[action_t]].astype(np.float32)

                self.buf.append((obs_t_flattened, action_onehot, obs_t_1_flattened))

                obs_t = obs_t_1
        elif env.name == "BallWorld":
            obs_t = env.reset()
            for _ in range(n):
                obs_t_1 = env.step()

                obs_t_arr = np.array(obs_t, dtype=np.float32)
                obs_t_1_arr = np.array(obs_t_1, dtype=np.float32)
                action_t = np.array([0.0], dtype=np.float32)
                self.buf.append((obs_t_arr, action_t, obs_t_1_arr))
                obs_t = obs_t_1
        else:
            obs_t = env.reset()
            for _ in range(n):
                dtheta_1 = random.uniform(-0.05, 0.05)
                dtheta_2 = random.uniform(-0.05, 0.05)
                action_t = [dtheta_1, dtheta_2]
                obs_t_1 = env.step(dtheta_1, dtheta_2)
                self.buf.append((obs_t, action_t, obs_t_1))
                obs_t = obs_t_1

    def debug_shapes(self):
        for i, (o, a, o1) in enumerate(self.buf[:20]):
            print(i, np.array(o).shape, np.array(a).shape, np.array(o1).shape)
