import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from envs import GridWorld
from lejepa import RolloutBuffer
from models import Dynamics, Encoder

rollout_steps = 5000
batch_size = 64
iterations = 3000
lr = 0.001
latent_dim = 32
action_dim = 4

env = GridWorld()

buffer = RolloutBuffer()
buffer.populate(env, rollout_steps)

# print(buffer.sample(5))
encoder = Encoder()
dynamics = Dynamics(action_dim)

all_parameters = list(encoder.parameters()) + list(dynamics.parameters())
optimizer = Adam(all_parameters, lr)

for it in range(iterations):
    obs_t_batch, action_t_batch, obs_t1_batch = buffer.sample(batch_size)

    obs_t_batch = torch.tensor(obs_t_batch)
    action_t_batch = torch.tensor(action_t_batch)
    obs_t1_batch = torch.tensor(obs_t1_batch)

    z_t = encoder(obs_t_batch)
    z_t1 = encoder(obs_t1_batch)

    dinput = torch.cat([z_t, action_t_batch], dim=1)

    z_pred = dynamics(dinput)

    loss = F.mse_loss(z_pred, z_t1)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if it % 100 == 0:
        print("z_t[0]: ", z_t[0])
        print("z_t1[0]:", z_t1[0])
        print(f"Iteration {it}, Loss: {loss.item():.6f}")
