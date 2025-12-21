import time
import numpy as np
import torch
import torch.nn as nn
import viser


# --- 1. DEFINE A MOCK NERF MODEL (Replace with your actual network) ---
class MockNeRF(nn.Module):
    def __init__(self):
        super().__init__()
        # This is just to simulate a learned representation
        self.center = torch.tensor([0.0, 0.0, 0.0])

    def forward(self, x):
        """
        Input: x (N, 3) coordinates
        Output: rgb (N, 3), density (N, 1)
        """
        # Distance from center
        dist = torch.norm(x - self.center, dim=1, keepdim=True)

        # SIMULATE DENSITY: Higher density near the center (like a sphere)
        # We use a gaussian-like falloff for density
        density = 15.0 * torch.exp(-2.0 * dist ** 2)

        # SIMULATE COLOR: Color changes based on position (just for visualization)
        # Normalize positions to 0-1 range for RGB
        rgb = (torch.sin(x * 2.0) + 1.0) / 2.0

        return rgb, density


# --- 2. CONFIGURATION ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GRID_SIZE = 30  # How many points per axis (30x30x30 = 27,000 points)
SCENE_BOUNDS = 2.0  # The box size to sample points in (-2.0 to 2.0)
BASE_POINT_SIZE = 0.05


def main():
    # Initialize the server
    server = viser.ViserServer()
    print("Viser server started at http://localhost:8080")

    # Initialize model
    model = MockNeRF().to(DEVICE)
    model.eval()

    # --- 3. GENERATE QUERY POSITIONS (Input bunch of 3D positions) ---
    print("Generating input positions...")

    # Create a 3D grid of coordinates
    linspace = torch.linspace(-SCENE_BOUNDS, SCENE_BOUNDS, GRID_SIZE)
    grid_x, grid_y, grid_z = torch.meshgrid(linspace, linspace, linspace, indexing='ij')

    # Flatten to shape (N, 3)
    positions_tensor = torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(-1, 3).to(DEVICE)

    # --- 4. QUERY THE NETWORK ---
    print("Querying the network...")
    with torch.no_grad():
        # Pass positions into the function/network
        # If you have your own NeRF, call it here: rgb, sigma = my_nerf(positions_tensor)
        predicted_rgb, predicted_density = model(positions_tensor)

    # Convert to numpy for Viser
    points_np = positions_tensor.cpu().numpy()
    colors_np = predicted_rgb.cpu().numpy()
    density_np = predicted_density.cpu().numpy().flatten()

    # --- 5. VISUALIZATION LOOP ---
    # We add a slider to filter low-density noise, common in NeRFs
    density_threshold_handle = server.gui.add_slider(
        "Density Threshold", min=0.0, max=5.0, step=0.1, initial_value=0.5
    )

    size_multiplier_handle = server.gui.add_slider(
        "Size Scale", min=0.1, max=5.0, step=0.1, initial_value=1.0
    )

    print("Visualizing... Check the browser!")

    while True:
        threshold = density_threshold_handle.value
        size_mult = size_multiplier_handle.value

        # Filter: Only show points where density > threshold
        mask = density_np > threshold

        if np.sum(mask) > 0:
            active_points = points_np[mask]
            active_colors = colors_np[mask]
            active_densities = density_np[mask]

            # MAPPING: Map Density -> Point Size
            # We scale the density to a reasonable visual size
            point_sizes = active_densities * 0.01 * size_mult

            server.scene.add_point_cloud(
                name="/nerf_cloud",
                points=active_points,
                colors=active_colors,
                point_sizes=point_sizes,
            )
        else:
            # If everything is filtered out, clear the scene
            server.scene.remove("/nerf_cloud")

        time.sleep(0.05)  # Refresh rate


if __name__ == "__main__":
    main()