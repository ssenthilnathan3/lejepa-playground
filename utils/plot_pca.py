import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA

from envs import BallWorld
from models import LeJEPAEncoder, Predictor, TinyDynamic


# --- REVISED PLOT FUNCTION FOR RESIDUAL PREDICTION COMPARISON ---
def plot_pca(
    env: BallWorld,
    context_encoder: LeJEPAEncoder,
    predictor: Predictor,
    tiny_dyn: nn.Module,
    num_steps: int = 500,
    time_gap_k: int = 5,
    it: int = 0,
):
    """
    Generates a new trajectory, embeds it, and plots the TRUE latent trajectory
    vs. the PREDICTED latent points based on the residual delta model.
    """
    # Set models to evaluation mode
    context_encoder.eval()
    predictor.eval()
    tiny_dyn.eval()  # Set TinyDynamic to eval mode

    # 1. Generate a new, continuous trajectory
    env.reset()

    obs_sequence = []
    # Shape: (num_steps, 1)
    time_gap_vec = torch.full((num_steps, 1), float(time_gap_k), dtype=torch.float32)

    current_obs = env.observe()
    for _ in range(num_steps):
        obs_sequence.append(current_obs)
        current_obs = env.step()

    obs_batch = torch.tensor(np.array(obs_sequence), dtype=torch.float32)

    # 2. Embed the entire sequence to get the TRUE latent trajectory
    with torch.no_grad():
        zt_pos_all, zt_vel_all, latents_true = context_encoder(obs_batch)
        latents_true = F.normalize(latents_true, dim=1)
        # Apply normalization to components if done during training (safer to assume component normalization if pos/vel are used for input)
        # zt_pos_all = F.normalize(zt_pos_all, dim=1)
        # zt_vel_all = F.normalize(zt_vel_all, dim=1)

    # 3. Create the PREDICTED latent trajectory (z_hat)

    # Present latents (z_t) are latents_true up to the point of prediction
    z_present = latents_true[:-time_gap_k]
    z_pos_t = zt_pos_all[:-time_gap_k]
    z_vel_t = zt_vel_all[:-time_gap_k]

    # Time vector for prediction (must match length of z_present)
    time_vec_pred = time_gap_vec[:-time_gap_k]

    with torch.no_grad():
        # Predict Residual Delta (d_res)
        residual_delta = predictor(
            z_pos_t=z_pos_t, z_vel_t=z_vel_t, time_gap=time_vec_pred
        )

        # Predict Baseline Delta (d_base)
        baseline_delta = tiny_dyn(z_present)

        # Predicted Total Delta (d_hat)
        latent_delta_pred = baseline_delta + residual_delta

        # Predicted Future State (z_hat_t+k = zt + d_hat)
        z_hat_t_k = z_present + latent_delta_pred

        # Apply normalization to the final predicted state, consistent with training
        z_predicted = F.normalize(z_hat_t_k, dim=1)

    # 4. Prepare data for PCA
    # Latents used for prediction comparison
    latents_all = latents_true.detach().cpu().numpy()

    # 5. PCA for Visualization
    try:
        pca = PCA(n_components=2)
        latents_2d_all = pca.fit_transform(latents_all)

        # Split the projected latents
        true_2d = latents_2d_all[time_gap_k:]
        predicted_2d = pca.transform(z_predicted.detach().cpu().numpy())

        # 6. Plotting
        plt.figure(figsize=(10, 8))

        # ... (Plotting code remains the same as provided in your prompt) ...
        # Plot 1: The True Latent Trajectory (The baseline)
        plt.plot(
            latents_2d_all[:, 0],
            latents_2d_all[:, 1],
            label="True Latent Path ($z_t$)",
            color="gray",
            alpha=0.4,
            linestyle="--",
            linewidth=1,
        )

        # Plot 2: Scatter the actual target points
        plt.scatter(
            true_2d[:, 0],
            true_2d[:, 1],
            label="True Future Points ($z_{t+k}$)",
            s=20,
            color="blue",
            alpha=0.6,
        )

        # Plot 3: Plot the predictions (The "anticipated" points)
        plt.scatter(
            predicted_2d[:, 0],
            predicted_2d[:, 1],
            label="Predicted Future Points ($\hat{z}_{t+k}$)",
            s=20,
            color="red",
            marker="x",
        )

        # Plot 4: Prediction Error Vectors (Show the miss)
        for i in range(len(true_2d)):
            plt.plot(
                [predicted_2d[i, 0], true_2d[i, 0]],
                [predicted_2d[i, 1], true_2d[i, 1]],
                color="k",
                linewidth=0.5,
                alpha=0.1,
            )

        plt.title(f"JEPA Residual Prediction (k={time_gap_k}, Iteration {it})")
        plt.xlabel(f"PCA Component 1 (EVR: {pca.explained_variance_ratio_[0]:.2f})")
        plt.ylabel(f"PCA Component 2 (EVR: {pca.explained_variance_ratio_[1]:.2f})")
        plt.legend()
        plt.grid(True, linestyle=":", alpha=0.7)

        plt.savefig(f"latent_prediction_it_{it}.png")
        plt.close()
        print(f"Plot saved to latent_prediction_it_{it}.png")

    except Exception as e:
        print(f"Error during plotting: {e}")
        pass

    # Restore models to training mode
    context_encoder.train()
    predictor.train()
    tiny_dyn.train()
