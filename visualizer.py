import time
import numpy as np
import torch
import viser


# HELPER: Auto-detect the best available device
def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"  # Apple Silicon GPU
    else:
        return "cpu"


def start_visualizer(model, device=None, grid_size=30, scene_bounds=2.0):
    """
    Generic function to visualize ANY NeRF-like model.
    """
    # 1. AUTO-DETECT DEVICE IF NOT PROVIDED
    if device is None:
        device = get_device()

    print(f"Viser server started at http://localhost:8080")
    print(f"Running on device: {device}")

    server = viser.ViserServer()

    model = model.to(device)
    model.eval()

    # --- 2. PRE-COMPUTE DATA ---
    print(f"Sampling {grid_size}x{grid_size}x{grid_size} grid...")
    linspace = torch.linspace(-scene_bounds, scene_bounds, grid_size)
    grid_x, grid_y, grid_z = torch.meshgrid(linspace, linspace, linspace, indexing='ij')
    positions_tensor = torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(-1, 3).to(device)

    print("Querying the network...")
    with torch.no_grad():
        predicted_rgb, predicted_density = model(positions_tensor)

    # Move to CPU/Numpy for Viser
    all_points = positions_tensor.cpu().numpy()
    all_colors = predicted_rgb.cpu().numpy()
    all_densities = predicted_density.cpu().numpy().flatten()

    # --- 3. GUI SETUP ---
    density_threshold_handle = server.gui.add_slider(
        "Density Threshold", min=0.0, max=5.0, step=0.1, initial_value=0.5
    )

    size_multiplier_handle = server.gui.add_slider(
        "Size Scale", min=0.001, max=0.1, step=0.001, initial_value=0.01
    )

    # --- 4. THE UPDATE FUNCTION ---
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
            # Create (N, 3, 3) diagonal covariance matrices
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
            server.scene.remove("/nerf_splats")

    # --- 5. ATTACH LISTENERS ---
    density_threshold_handle.on_update(update_scene)
    size_multiplier_handle.on_update(update_scene)

    # Init
    update_scene(None)

    print("Visualizer ready. Ctrl+C to exit.")
    while True:
        time.sleep(10.0)


# --- MAIN ENTRY POINT (For testing only) ---
if __name__ == "__main__":
    from experiments.nerf_network import NeRFModel

    model = NeRFModel()
    # No need to specify device anymore, it will auto-detect
    start_visualizer(model, grid_size=30)