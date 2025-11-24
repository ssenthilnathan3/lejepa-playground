from typing import OrderedDict

import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            OrderedDict(
                [
                    ("input", nn.Linear(100, 64)),
                    ("act1", nn.ReLU()),
                    ("hidden1", nn.Linear(64, 64)),
                    ("act2", nn.ReLU()),
                    ("output", nn.Linear(64, 32)),
                ]
            )
        )

    def forward(self, x):
        return self.layers(x)
