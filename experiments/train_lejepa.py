import random

import torch
import torch.nn.functional as F
from torch.optim import Adam

from envs import BallWorld
from lejepa import RolloutBuffer
from models import LeJEPAEncoder, Predictor, TinyDynamic
from utils import mask_obs, plot_pca

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

# env related params
x_init = random.uniform(-1, 1)
y_init = random.uniform(-1, 1)
vx_init = random.uniform(-0.4, 0.4)
vy_init = random.uniform(-0.4, 0.4)
env = BallWorld(x_init, y_init, vx_init, vy_init)

# initializing the rollout buffer
buffer = RolloutBuffer()
buffer.populate(env, rollout_steps)

context_encoder = LeJEPAEncoder()
target_encoder = LeJEPAEncoder()
tiny_dyn = TinyDynamic()
predictor = Predictor()

target_encoder.load_state_dict(context_encoder.state_dict())

all_parameters = list(context_encoder.parameters()) + list(predictor.parameters())
optimizer = Adam(all_parameters, lr)


for it in range(iterations):
    k = [1, 2, 4, 8, 10]
    obs_t_batch, _, obs_t_k_batch, time_gap = buffer.sample_with_k(
        batch_size, random.choice(k)
    )

    obs_t_batch = torch.tensor(obs_t_batch, dtype=torch.float32)
    obs_t_k_batch = torch.tensor(obs_t_k_batch, dtype=torch.float32)

    masked_obs_t_batch = mask_obs(obs_t_batch)

    zt_pos, zt_vel, zt = context_encoder(masked_obs_t_batch)

    zt = F.normalize(zt, dim=1)

    with torch.no_grad():
        _, _, zt_k = target_encoder(obs_t_k_batch)

        zt_k = F.normalize(zt_k, dim=1)

    time_gap_vec = torch.randint(1, 11, size=(batch_size, 1)).float()

    latent_delta_true = zt_k - zt

    # adding dynamics to introduce prediction of default motion of the latent state.
    baseline_delta = tiny_dyn(zt)

    residual_delta = predictor(z_pos_t=zt_pos, z_vel_t=zt_vel, time_gap=time_gap_vec)

    latent_delta_pred = baseline_delta + residual_delta

    z_hat_t_k = zt + latent_delta_pred

    z_hat_t_k = F.normalize(z_hat_t_k, dim=1)

    L_delta = F.mse_loss(latent_delta_pred, latent_delta_true)
    # or
    # loss = F.mse_loss(z_hat_t_k, zt_k)

    L_prior_reg = torch.norm(residual_delta, p=2) ** 2

    loss = L_delta + 0.1 * L_prior_reg

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

    if it % 500 == 0 and it > 0:
        print("\n--- Generating Latent Trajectory Plot ---")
        plot_pca(
            env=env,
            context_encoder=context_encoder,
            predictor=predictor,
            tiny_dyn=tiny_dyn,
            num_steps=500,
            time_gap_k=5,
            it=it,
        )
        print("------------------------------------------\n")
