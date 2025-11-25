import random

import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_latent_trajectories(encoder, dynamics, buffer, num_trajectories=5, steps=10):
    """Plot real vs predicted latent trajectories"""
    _, axes = plt.subplots(1, num_trajectories, figsize=(15, 3))
    if num_trajectories == 1:
        axes = [axes]

    for i in range(num_trajectories):
        # Get a random starting point
        start_idx = random.randint(0, len(buffer.buf) - steps - 1)

        # Get real trajectory starting from this point
        obs_sequence = []
        actions_sequence = []

        for j in range(steps + 1):
            obs, action, _ = buffer.buf[start_idx + j]
            obs_sequence.append(obs)
            if j < steps:  # actions are one shorter than observations
                actions_sequence.append(action)

        # Convert to tensors
        obs_tensor = torch.tensor(np.array(obs_sequence), dtype=torch.float32)
        actions_tensor = torch.tensor(np.array(actions_sequence), dtype=torch.float32)

        with torch.no_grad():
            # Get real latent trajectory
            z_pos_real, z_vel_real, z_real = encoder(obs_tensor)

            # Predict latent trajectory step by step
            z_pos_pred = [z_pos_real[0:1]]  # Start with first real position
            z_vel_pred = [z_vel_real[0:1]]  # Start with first real velocity

            for t in range(steps):
                # Predict next latent state using dynamics
                z_pos_next, z_vel_next, _ = dynamics(z_pos_pred[-1], z_vel_pred[-1])
                z_pos_pred.append(z_pos_next)
                z_vel_pred.append(z_vel_next)

            # Concatenate predictions
            z_pos_pred = torch.cat(z_pos_pred)
            z_vel_pred = torch.cat(z_vel_pred)

            # Combine position and velocity for plotting (you might want to adjust this)
            z_pred = torch.cat([z_pos_pred, z_vel_pred], dim=1)

        # Plot first two dimensions of latent space
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
