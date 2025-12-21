import time
import numpy as np
import torch
import torch.nn as nn
import viser


# --- 1. DEFINE A MOCK NERF MODEL ---
class MockNeRF(nn.Module):
    def __init__(self):
        super().__init__()
        self.center = torch.tensor([0.0, 0.0, 0.0])

    def forward(self, x):
        dist = torch.norm(x - self.center, dim=1, keepdim=True)
        # Density falls off from center
        density = 15.0 * torch.exp(-2.0 * dist ** 2)
        # Color pattern
        rgb = (torch.sin(x * 2.0) + 1.0) / 2.0
        return rgb, density


# --- 2. CONFIGURATION ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GRID_SIZE = 30
SCENE_BOUNDS = 2.0


def main():
    server = viser.ViserServer()
    print("Viser server started at http://localhost:8080")

    model = MockNeRF().to(DEVICE)
    model.eval()

    # --- 3. GENERATE QUERY POSITIONS ---
    print("Generating input positions...")
    linspace = torch.linspace(-SCENE_BOUNDS, SCENE_BOUNDS, GRID_SIZE)
    grid_x, grid_y, grid_z = torch.meshgrid(linspace, linspace, linspace, indexing='ij')
    positions_tensor = torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(-1, 3).to(DEVICE)

    # --- 4. QUERY THE NETWORK ---
    print("Querying the network...")
    with torch.no_grad():
        predicted_rgb, predicted_density = model(positions_tensor)

    # Convert to numpy
    points_np = positions_tensor.cpu().numpy()
    colors_np = predicted_rgb.cpu().numpy()
    density_np = predicted_density.cpu().numpy().flatten()

    # --- 5. VISUALIZATION LOOP ---
    density_threshold_handle = server.gui.add_slider(
        "Density Threshold", min=0.0, max=5.0, step=0.1, initial_value=0.5
    )

    size_multiplier_handle = server.gui.add_slider(
        "Size Scale", min=0.01, max=1.0, step=0.01, initial_value=0.1
    )

    print("Visualizing... Check the browser!")

    while True:
        threshold = density_threshold_handle.value
        size_mult = size_multiplier_handle.value

        mask = density_np > threshold

        if np.sum(mask) > 0:
            active_points = points_np[mask]
            active_colors = colors_np[mask]
            active_densities = density_np[mask]

            # Scale: (N, 1) array where higher density = bigger splat
            # We reshape to (N, 1) to match Viser expectations
            active_scales = (active_densities * size_mult).reshape(-1, 1)

            # Opacity: (N, 1) array
            # You can set this to 1.0 if you want them solid, or map it to density
            active_opacities = np.ones((len(active_points), 1))

            # Quaternions: (N, 4)
            # Gaussian splats have rotation. We just want spheres, so we use identity rotation
            # Identity quaternion is [w, x, y, z] = [1, 0, 0, 0]
            active_quats = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (len(active_points), 1))

            server.scene.add_gaussian_splats(
                name="/nerf_splats",
                centers=active_points,
                rgbs=active_colors,
                opacities=active_opacities,
                scales=active_scales,
                quaternions=active_quats
            )
        else:
            server.scene.remove("/nerf_splats")

        time.sleep(0.05)


if __name__ == "__main__":
    main()