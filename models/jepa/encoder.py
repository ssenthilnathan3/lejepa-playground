from typing import OrderedDict

import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_dim, output_dim=None):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim or 32
        self.layers = nn.Sequential(
            OrderedDict(
                [
                    ("input", nn.Linear(self.input_dim, 128)),
                    ("layernorm1", nn.LayerNorm(128)),
                    ("act1", nn.ReLU()),
                    ("hidden1", nn.Linear(128, 128)),
                    ("layernorm2", nn.LayerNorm(128)),
                    ("act2", nn.ReLU()),
                    ("hidden2", nn.Linear(128, 128)),
                    ("layernorm3", nn.LayerNorm(128)),
                    ("act3", nn.ReLU()),
                    ("output", nn.Linear(128, self.output_dim)),
                ]
            )
        )

    def forward(self, x):
        return self.layers(x)
