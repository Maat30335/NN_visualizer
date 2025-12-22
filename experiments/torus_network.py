import torch
import torch.nn as nn


class TorusModel(nn.Module):
    def __init__(self, num_layers=4, hidden_dim=128):
        super().__init__()
        layers = []
        # Input is 3 (x, y, z)
        layers.append(nn.Linear(3, hidden_dim))
        layers.append(nn.ReLU())

        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        # Output is 4 (3 RGB + 1 Density)
        layers.append(nn.Linear(hidden_dim, 4))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """
        Input: (N, 3)
        Output: rgb (N, 3), density (N, 1)
        """
        out = self.net(x)

        # Sigmoid for color (0-1)
        rgb = torch.sigmoid(out[:, :3])

        # Sigmoid for density (0-1)
        # We don't use ReLU/Exp here to keep it stable for your visualizer
        density = torch.sigmoid(out[:, 3:])

        return rgb, density