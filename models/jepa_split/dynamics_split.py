import torch
import torch.nn as nn


class DynamicsSplit(nn.Module):
    def __init__(self, dt=4.0, pos_dim=16, vel_dim=16):
        super().__init__()

        self.dt = dt

        self.vel_mlp = nn.Sequential(
            nn.Linear(pos_dim + vel_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, vel_dim),
        )

    def forward(self, z_pos, z_vel):
        # Concat entire latent as input
        z = torch.cat([z_pos, z_vel], dim=1)

        z_vel_next = self.vel_mlp(z)

        z_pos_next = z_pos + z_vel * self.dt

        z_next = torch.cat([z_pos_next, z_vel_next], dim=1)
        return z_pos_next, z_vel_next, z_next
