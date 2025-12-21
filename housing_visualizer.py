import torch
import torch.nn as nn
from housing_network import HousingModel

# Import the generic tool you created in the previous step
from visualizer import start_visualizer


class HousingNeRFAdapter(nn.Module):
    """
    Adapts the 8D Housing Model to look like a 3D NeRF Model.
    """

    def __init__(self, housing_model):
        super().__init__()
        self.housing_model = housing_model

    def forward(self, x_3d):
        """
        Input: x_3d (N, 3)  <-- From Viser
        Output: rgb (N, 3), density (N, 1)
        """
        device = x_3d.device
        N = x_3d.shape[0]

        # 1. CONSTRUCT THE 8-DIMENSIONAL INPUT
        # We initialize with 0.0. Since the model was trained on StandardScaled data,
        # 0.0 equals the "Average" for that feature.
        inputs_8d = torch.zeros((N, 8), device=device)

        # 2. MAP VISER AXES TO MODEL FEATURES
        # Viser X -> Latitude (Feature 6)
        inputs_8d[:, 6] = x_3d[:, 0]
        # Viser Y -> Longitude (Feature 7)
        inputs_8d[:, 7] = x_3d[:, 1]
        # Viser Z -> Median Income (Feature 0)
        inputs_8d[:, 0] = x_3d[:, 2]

        # 3. GET PREDICTION
        price = self.housing_model(inputs_8d)  # Output shape (N, 1)

        # 4. MAP PRICE TO DENSITY
        # Use exponential to ensure density is positive and sharp
        density = torch.exp(price)

        # 5. MAP PRICE TO COLOR
        # Sigmoid maps prediction to 0..1 range
        # High Price = Red, Low Price = Blue
        val = torch.sigmoid(price)
        red = val
        green = torch.zeros_like(val)
        blue = 1.0 - val

        rgb = torch.cat([red, green, blue], dim=1)

        return rgb, density


if __name__ == "__main__":
    # 1. Load the Real Model
    real_model = HousingModel()
    real_model.load_state_dict(torch.load("housing_model.pth"))

    # 2. Wrap it in the Adapter
    adapter = HousingNeRFAdapter(real_model)

    # 3. Launch the Generic Visualizer
    # We assume 'visualizer.py' is in the same folder
    start_visualizer(adapter, grid_size=40)