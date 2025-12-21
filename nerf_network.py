import torch
import torch.nn as nn


# This is the file you will edit when you change your model architecture
class NeRFModel(nn.Module):
    def __init__(self):
        super().__init__()
        # MOCK PARAMETERS (Replace with your real layers)
        self.center = torch.tensor([0.0, 0.0, 0.0])

    def forward(self, x):
        """
        Input: (N, 3)
        Output: rgb (N, 3), density (N, 1)
        """
        # --- MOCK LOGIC START (Replace with your real forward pass) ---
        dist = torch.norm(x - self.center, dim=1, keepdim=True)
        density = 15.0 * torch.exp(-2.0 * dist ** 2)
        rgb = (torch.sin(x * 2.0) + 1.0) / 2.0
        # --- MOCK LOGIC END ---

        return rgb, density