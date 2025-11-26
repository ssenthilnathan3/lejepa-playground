import torch.nn as nn


class TinyDynamic(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(32, 128), nn.ReLU(), nn.Linear(128, 32))

    def forward(self, x):
        return self.layers(x)
