from typing import OrderedDict

import numpy as np
import torch.nn as nn


class Dynamics(nn.Module):
    def __init__(self, action_t):
        super().__init__()
        self.input_dim = action_t + 32
        self.layers = nn.Sequential(
            OrderedDict(
                [
                    ("input", nn.Linear(self.input_dim, 64)),
                    ("act1", nn.ReLU()),
                    ("hidden", nn.Linear(64, 64)),
                    ("act2", nn.ReLU()),
                    ("output", nn.Linear(64, 32)),
                ]
            )
        )

    def forward(self, x):
        return self.layers(x)
