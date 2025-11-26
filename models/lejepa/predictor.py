import torch
import torch.nn as nn


class Predictor(nn.Module):
    def __init__(self, pos_dim=16, vel_dim=16, embed_dim=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.pos_dim = pos_dim
        self.vel_dim = vel_dim

        self.time_encoder = nn.Linear(1, self.embed_dim)

        pos_input_dim = self.pos_dim + self.embed_dim
        vel_input_dim = self.vel_dim + self.embed_dim

        self.pos_mlp = nn.Sequential(
            nn.Linear(pos_input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.pos_dim),
        )

        self.vel_mlp = nn.Sequential(
            nn.Linear(vel_input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.vel_dim),
        )

    def forward(self, z_pos_t, z_vel_t, time_gap) -> torch.Tensor:
        time_embedding = self.time_encoder(time_gap)

        pos_in = torch.cat([z_pos_t, time_embedding], dim=1)

        vel_in = torch.cat([z_vel_t, time_embedding], dim=1)

        z_pos_t_k = self.pos_mlp(pos_in)

        z_vel_t_k = self.vel_mlp(vel_in)

        z_hat_t_k = torch.cat([z_pos_t_k, z_vel_t_k], dim=1)

        return z_hat_t_k
