import numpy as np


def MSE(pred, obs):
    return np.mean((pred - obs) ** 2)
