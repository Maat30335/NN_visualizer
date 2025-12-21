import torch
import torch.nn as nn

class HousingModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 8 Inputs -> 1 Output (Price)
        self.net = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)