import random

import torch
import torch.nn.functional as F

from .plot_pca import plot_pca


def mask_obs(obs_t_batch: torch.Tensor) -> torch.Tensor:
    """
    Applies the weighted JEPA masking schedule independently to each sample in the batch.

    Args:
        obs_t_batch: The input observation batch of shape (B, 4).

    Returns:
        A new tensor of shape (B, 4) with elements masked (set to 0) according
        to the JEPA schedule.
    """
    B, D = obs_t_batch.shape
    device = obs_t_batch.device

    # 1. Define the schedule, weights, and index tensors
    # Mapping: 0='coord', 1='vel', 2='rand_2dims', 3='rand_1dim', 4='rand_all'
    weights = torch.tensor(
        [0.3, 0.3, 0.2, 0.1, 0.1], dtype=torch.float32, device=device
    )

    pos_dims = torch.tensor([0, 1], device=device)
    vel_dims = torch.tensor([2, 3], device=device)

    mask_tensor = torch.ones_like(obs_t_batch)

    mask_choices = torch.multinomial(weights, num_samples=B, replacement=True)

    coord_mask_indices = (mask_choices == 0).nonzero(as_tuple=True)[0]
    if coord_mask_indices.numel() > 0:
        mask_tensor[coord_mask_indices[:, None], pos_dims] = 0.0

    vel_mask_indices = (mask_choices == 1).nonzero(as_tuple=True)[0]
    if vel_mask_indices.numel() > 0:
        mask_tensor[vel_mask_indices[:, None], vel_dims] = 0.0

    all_mask_indices = (mask_choices == 4).nonzero(as_tuple=True)[0]
    if all_mask_indices.numel() > 0:
        mask_tensor[all_mask_indices, :] = 0.0

    random_1_indices = (mask_choices == 3).nonzero(as_tuple=True)[0]
    random_2_indices = (mask_choices == 2).nonzero(as_tuple=True)[0]

    for idx in random_1_indices:
        cols_to_mask = random.sample(range(D), k=1)
        mask_tensor[idx.item(), cols_to_mask] = 0.0

    for idx in random_2_indices:
        cols_to_mask = random.sample(range(D), k=2)
        mask_tensor[idx.item(), cols_to_mask] = 0.0

    masked_obs_batch = obs_t_batch * mask_tensor

    return masked_obs_batch


__all__ = ["plot_pca", "mask_obs"]
