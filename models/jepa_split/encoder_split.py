import torch
import torch.nn as nn


class EncoderSplit(nn.Module):
    def __init__(self, pos_dim=16, vel_dim=16):
        nn.Module.__init__(self)

        self.pos_mlp = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, pos_dim),
        )

        self.vel_mlp = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, vel_dim),
        )

    def forward(self, obs):
        # obs shape: (batch, 4) = [x, y, vx, vy]
        pos = obs[:, :2]
        vel = obs[:, 2:]

        z_pos = self.pos_mlp(pos)
        z_vel = self.vel_mlp(vel)

        z = torch.cat([z_pos, z_vel], dim=1)
        return z_pos, z_vel, z
