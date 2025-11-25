import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import stack
from torch.optim import Adam

from envs import BallWorld, GridWorld
from lejepa import RolloutBuffer
from models import Dynamics, Encoder

rollout_steps = 20_000
batch_size = 256
iterations = 5000
lr = 0.0005
latent_dim = 32
action_dim = 1

x_init = random.uniform(-1, 1)
y_init = random.uniform(-1, 1)
vx_init = random.uniform(-0.4, 0.4)
vy_init = random.uniform(-0.4, 0.4)
env = BallWorld(x_init, y_init, vx_init, vy_init)

buffer = RolloutBuffer()
buffer.populate(env, rollout_steps)

# print(buffer.debug_shapes())
encoder = Encoder(input_dim=4)
dynamics = Dynamics(action_dim)

all_parameters = list(encoder.parameters()) + list(dynamics.parameters())
optimizer = Adam(all_parameters, lr)


def normalize(batch):
    x = (batch[:, 0] - 0.5) * 2
    y = (batch[:, 1] - 0.5) * 2
    vx = batch[:, 2] * 50
    vy = batch[:, 3] * 50

    return stack([x, y, vx, vy], dim=1)


def plot_latent_trajectories(encoder, dynamics, buffer, num_trajectories=5, steps=10):
    """Plot real vs predicted latent trajectories"""
    _, axes = plt.subplots(1, num_trajectories, figsize=(15, 3))
    if num_trajectories == 1:
        axes = [axes]

    for i in range(num_trajectories):
        obs_t, action_t, obs_t1 = buffer.buf[
            random.randint(0, len(buffer.buf) - steps - 1)
        ]

        obs_sequence = [obs_t]
        actions_sequence = []

        # Get real trajectory
        for j in range(steps):
            idx = random.randint(0, len(buffer.buf) - 1)
            _, action, next_obs = buffer.buf[idx]
            obs_sequence.append(next_obs)
            actions_sequence.append(action)

        # Convert to tensors
        obs_tensor = torch.tensor(np.array(obs_sequence), dtype=torch.float32)
        actions_tensor = torch.tensor(np.array(actions_sequence), dtype=torch.float32)

        with torch.no_grad():
            z_real = encoder(obs_tensor)

            z_pred = [z_real[0:1]]
            for t in range(steps):
                dinput = torch.cat([z_pred[-1], actions_tensor[t : t + 1]], dim=1)
                z_next = dynamics(dinput)
                z_pred.append(z_next)

            z_pred = torch.cat(z_pred)

        axes[i].plot(
            z_real[:, 0].numpy(),
            z_real[:, 1].numpy(),
            "bo-",
            label="Real",
            linewidth=2,
            markersize=4,
        )
        axes[i].plot(
            z_pred[:, 0].numpy(),
            z_pred[:, 1].numpy(),
            "ro--",
            label="Predicted",
            linewidth=2,
            markersize=4,
        )
        axes[i].set_title(f"Trajectory {i + 1}")
        axes[i].set_xlabel("Latent dim 1")
        axes[i].set_ylabel("Latent dim 2")
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# Add this to your training loop to visualize every 500 iterations
for it in range(iterations):
    obs_t_batch, action_t_batch, obs_t1_batch = buffer.sample(batch_size)

    obs_t_batch = torch.tensor(obs_t_batch, dtype=torch.float32)
    action_t_batch = torch.tensor(action_t_batch, dtype=torch.float32)
    obs_t1_batch = torch.tensor(obs_t1_batch, dtype=torch.float32)

    obs_t_batch_norm = normalize(obs_t_batch)
    obs_t1_batch_norm = normalize(obs_t1_batch)

    z_t = encoder(obs_t_batch_norm)
    z_t1 = encoder(obs_t1_batch_norm)

    dinput = torch.cat([z_t, action_t_batch], dim=1)
    z_pred = dynamics(dinput)

    loss = F.mse_loss(z_pred, z_t1)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if it % 100 == 0:
        print(f"Iteration {it}, Loss: {loss.item():.6f}")

    # Plot every 500 iterations
    if it % 500 == 0 and it > 0:
        print(f"Plotting trajectories at iteration {it}")
        plot_latent_trajectories(encoder, dynamics, buffer)
