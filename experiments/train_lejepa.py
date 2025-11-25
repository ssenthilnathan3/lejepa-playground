import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import stack
from torch.optim import Adam

from envs import BallWorld
from lejepa import RolloutBuffer
from models import LeJEPAEncoder, Predictor
from utils import plot_pca

rollout_steps = 20_000
batch_size = 256
iterations = 5000
lr = 0.0001

# how quickly the EMA responds to a change in the input data.
decay_rate = 0.00001

latent_dim = 32
action_dim = 1

# for lejepa: to select from buffer
time_gap = 10

x_init = random.uniform(-1, 1)
y_init = random.uniform(-1, 1)
vx_init = random.uniform(-0.4, 0.4)
vy_init = random.uniform(-0.4, 0.4)
env = BallWorld(x_init, y_init, vx_init, vy_init)

buffer = RolloutBuffer()
buffer.populate(env, rollout_steps)

context_encoder = LeJEPAEncoder()
target_encoder = LeJEPAEncoder()
predictor = Predictor()

target_encoder.load_state_dict(context_encoder.state_dict())

all_parameters = list(context_encoder.parameters()) + list(predictor.parameters())
optimizer = Adam(all_parameters, lr)


for it in range(iterations):
    obs_t_batch, _, obs_t_gap_batch, time_gap = buffer.sample_with_k(batch_size, 10)

    obs_t_batch = torch.tensor(obs_t_batch, dtype=torch.float32)
    obs_t_gap_batch = torch.tensor(obs_t_gap_batch, dtype=torch.float32)

    _, _, z = context_encoder(obs_t_batch)
    z = F.normalize(z, dim=1)

    with torch.no_grad():
        _, _, zt_1 = target_encoder(obs_t_gap_batch)

        zt_1 = F.normalize(zt_1, dim=1)

    time_gap_vec = torch.randint(1, 11, size=(batch_size, 1)).float()

    z_pred_t_t1 = predictor(z, time_gap_vec)

    z_pred_t_t1 = F.normalize(z_pred_t_t1, dim=1)

    loss = F.mse_loss(z_pred_t_t1, zt_1)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    for target_param, context_param in zip(
        target_encoder.parameters(), context_encoder.parameters()
    ):
        target_param.data.copy_(
            (1.0 - decay_rate) * target_param.data + decay_rate * context_param.data
        )

    if it % 100 == 0:
        print(f"Iteration {it}, Loss: {loss.item():.6f}")

    if it % 1000 == 0 and it > 0:
        print("\n--- Generating Latent Trajectory Plot ---")
        plot_pca(
            env=env,
            context_encoder=context_encoder,
            predictor=predictor,
            num_steps=500,
            time_gap_k=5,
            it=it,
        )
        print("------------------------------------------\n")
