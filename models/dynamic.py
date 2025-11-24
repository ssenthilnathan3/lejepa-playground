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
                    ("input", nn.Linear(self.input_dim, 256)),
                    ("act1", nn.ReLU()),
                    ("hidden", nn.Linear(256, 256)),
                    ("act2", nn.ReLU()),
                    ("hidden", nn.Linear(256, 256)),
                    ("act3", nn.ReLU()),
                    ("output", nn.Linear(256, 32)),
                ]
            )
        )

    def forward(self, x):
        return self.layers(x)
