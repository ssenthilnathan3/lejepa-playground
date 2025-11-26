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

    def sample_with_k(self, batch_size, K=10):
        # Randomly choose dt from {1, 2, ..., K}
        time_gap = random.choice([1, 3, 5, 10])

        # Sample with random time gap Δt
        # Only sample from indices where we can look ahead 'time_gap' steps
        max_start_index = len(self.buf) - time_gap
        if max_start_index < batch_size:
            # If not enough samples, reduce time_gap
            return self.sample_with_k(batch_size, K=time_gap - 1)

        # Sample starting indices
        start_indices = random.sample(range(max_start_index), batch_size)

        obs_t_batch = np.array([self.buf[i][0] for i in start_indices])
        action_batch = np.array([self.buf[i][1] for i in start_indices])
        obs_t_gap_batch = np.array([self.buf[i + time_gap][0] for i in start_indices])

        return obs_t_batch, action_batch, obs_t_gap_batch, time_gap

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
