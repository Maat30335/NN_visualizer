import torch
import torch.nn as nn
from .housing_network import HousingModel
from visualizer import start_visualizer


class HousingNeRFAdapter(nn.Module):
    def __init__(self, housing_model):
        super().__init__()
        self.housing_model = housing_model

    def forward(self, x_3d):
        """
        Input: x_3d (N, 3)
        Output: rgb (N, 3), density (N, 1)
        """
        device = x_3d.device
        N = x_3d.shape[0]

        # 1. Map Viser inputs to Model Inputs
        inputs_8d = torch.zeros((N, 8), device=device)
        inputs_8d[:, 6] = x_3d[:, 0]  # Lat
        inputs_8d[:, 7] = x_3d[:, 1]  # Long
        inputs_8d[:, 0] = x_3d[:, 2]  # Income

        # 2. Get Price Prediction
        price = self.housing_model(inputs_8d)

        # --- FIX 1: BETTER COLOR MAPPING ---
        # Sigmoid maps (-inf, inf) -> (0, 1)
        # 0.0 = Blue (Cheap), 1.0 = Red (Expensive)
        val = torch.sigmoid(price)

        # Create a "Cool-to-Warm" gradient
        # Blue channel is high when val is low
        red = val
        green = torch.zeros_like(val)  # Keep it simple
        blue = 1.0 - val
        rgb = torch.cat([red, green, blue], dim=1)

        # --- FIX 2: BETTER DENSITY/SIZE MAPPING ---
        # Old way: density = torch.exp(price)  <-- Explodes too fast!

        # New way: Map price to a 0.1 to 1.0 range
        # We use sigmoid again so it stays controlled between 0 and 1
        # We add +0.1 so even "cheap" (0.0) houses have 0.1 density (visible!)
        density = (torch.sigmoid(price) * 0.9) + 0.1

        return rgb, density


if __name__ == "__main__":
    # Load and Run
    real_model = HousingModel()
    real_model.load_state_dict(torch.load("housing_model.pth"))

    adapter = HousingNeRFAdapter(real_model)

    # We increase the bounds slightly to see more context
    start_visualizer(adapter, grid_size=40, scene_bounds=2.5)