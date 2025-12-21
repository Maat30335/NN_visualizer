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
        density = 15.0 * torch.exp(-2.0 * dist ** 2)
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

    # --- 3. PRE-COMPUTE DATA ---
    print("Generating input positions...")
    linspace = torch.linspace(-SCENE_BOUNDS, SCENE_BOUNDS, GRID_SIZE)
    grid_x, grid_y, grid_z = torch.meshgrid(linspace, linspace, linspace, indexing='ij')
    positions_tensor = torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(-1, 3).to(DEVICE)

    print("Querying the network...")
    with torch.no_grad():
        predicted_rgb, predicted_density = model(positions_tensor)

    # Move to CPU/Numpy once (optimization)
    all_points = positions_tensor.cpu().numpy()
    all_colors = predicted_rgb.cpu().numpy()
    all_densities = predicted_density.cpu().numpy().flatten()

    # --- 4. GUI SETUP ---
    density_threshold_handle = server.gui.add_slider(
        "Density Threshold", min=0.0, max=5.0, step=0.1, initial_value=0.5
    )

    size_multiplier_handle = server.gui.add_slider(
        "Size Scale", min=0.001, max=0.1, step=0.001, initial_value=0.01
    )

    # --- 5. THE UPDATE FUNCTION ---
    # This function runs ONLY when a slider moves
    def update_scene(_):
        threshold = density_threshold_handle.value
        size_mult = size_multiplier_handle.value

        mask = all_densities > threshold

        if np.sum(mask) > 0:
            active_points = all_points[mask]
            active_colors = all_colors[mask]
            active_densities = all_densities[mask]

            # Math: radii = density * scaler
            radii = active_densities * size_mult
            variances = radii ** 2

            num_points = len(active_points)
            covariances = np.zeros((num_points, 3, 3), dtype=np.float32)
            covariances[:, 0, 0] = variances
            covariances[:, 1, 1] = variances
            covariances[:, 2, 2] = variances

            active_opacities = np.ones((num_points, 1), dtype=np.float32)

            server.scene.add_gaussian_splats(
                name="/nerf_splats",
                centers=active_points,
                rgbs=active_colors,
                opacities=active_opacities,
                covariances=covariances
            )
        else:
            # If no points meet the threshold, remove the object
            server.scene.remove("/nerf_splats")

    # --- 6. ATTACH LISTENERS ---
    # "When this slider changes, run update_scene"
    density_threshold_handle.on_update(update_scene)
    size_multiplier_handle.on_update(update_scene)

    # Run once at startup to show the initial state
    update_scene(None)

    # Keep the script alive
    print("Visualizer ready. Ctrl+C to exit.")
    while True:
        time.sleep(10.0)


if __name__ == "__main__":
    main()