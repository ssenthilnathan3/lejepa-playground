import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA

from envs import BallWorld
from models import LeJEPAEncoder, Predictor


# --- REVISED PLOT FUNCTION FOR PREDICTION COMPARISON ---
def plot_pca(
    env: BallWorld,
    context_encoder: LeJEPAEncoder,
    predictor: Predictor,  # Pass the predictor model
    num_steps: int = 500,
    time_gap_k: int = 10,  # Example constant time gap for prediction
    it: int = 0,
):
    """
    Generates a new trajectory, embeds it, and plots the
    TRUE latent trajectory vs. the PREDICTED latent points.
    """
    # Set models to evaluation mode
    context_encoder.eval()
    predictor.eval()

    # 1. Generate a new, continuous trajectory (Context and Target points)
    env.reset()

    obs_sequence = []
    # Create the time_gap vector (all steps predict k steps ahead)
    # Shape: (num_steps, 1)
    time_gap_vec = torch.full((num_steps, 1), float(time_gap_k), dtype=torch.float32)

    current_obs = env.observe()
    for _ in range(num_steps):
        obs_sequence.append(current_obs)
        # Assuming the env.step() advances the state without arguments (as per our last fix)
        current_obs = env.step()

    obs_batch = torch.tensor(np.array(obs_sequence), dtype=torch.float32)

    # 2. Embed the entire sequence to get the TRUE latent trajectory (z)
    with torch.no_grad():
        _, _, latents_true = context_encoder(obs_batch)
        # Apply normalization if you used it in training
        latents_true = F.normalize(latents_true, dim=1)

    # 3. Create the PREDICTED latent trajectory (z_pred)
    # Present latents (z_t) are latents_true up to the point of prediction
    z_present = latents_true[:-time_gap_k]

    # Predicted latents (z_hat_{t+k})
    z_pred_raw = predictor(z_present, time_gap_vec[:-time_gap_k])

    # Apply normalization (since prediction target z_t+k was normalized)
    z_predicted = F.normalize(z_pred_raw, dim=1)

    # Target latents (z_{t+k}) for comparison
    z_target = latents_true[time_gap_k:]

    # 4. Concatenate all latents for consistent PCA fitting
    # We only fit PCA once on the actual latent path to ensure consistency
    latents_all = latents_true.detach().cpu().numpy()

    # 5. PCA for Visualization
    try:
        pca = PCA(n_components=2)
        # Fit PCA on the full true trajectory
        latents_2d_all = pca.fit_transform(latents_all)

        # Split the projected latents
        true_2d = latents_2d_all[time_gap_k:]  # The path where we have targets
        target_2d = latents_2d_all[
            :-time_gap_k
        ]  # The path used as prediction starting points

        # Transform the predicted points using the fitted PCA
        predicted_2d = pca.transform(z_predicted.detach().cpu().numpy())

        # 6. Plotting
        plt.figure(figsize=(10, 8))

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

        # Plot 2: Start and End points for the prediction segments
        # Scatter the actual target points
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
            # Draw a line from the predicted point to the true point
            plt.plot(
                [predicted_2d[i, 0], true_2d[i, 0]],
                [predicted_2d[i, 1], true_2d[i, 1]],
                color="k",
                linewidth=0.5,
                alpha=0.1,
            )

        plt.title(f"JEPA Latent Prediction Comparison (k={time_gap_k}, Iteration {it})")
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
