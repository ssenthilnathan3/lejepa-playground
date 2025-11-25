import torch
import torch.nn as nn


class Predictor(nn.Module):
    def __init__(self, latent_dim=32, embed_dim=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim

        # Temporal encoding from the scalar time gap. Input is (B, 1).
        self.time_encoder = nn.Linear(1, self.embed_dim)

        input_dim = self.latent_dim + self.embed_dim

        self.layers = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.latent_dim),
        )

    def forward(self, present_latent, time_gap):
        time_embedding = self.time_encoder(time_gap)

        z_in = torch.cat([present_latent, time_embedding], dim=1)

        return self.layers(z_in)
