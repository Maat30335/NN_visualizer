import time
import numpy as np
import torch
import viser

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def start_visualizer(model, trainer=None, device=None, grid_size=30, scene_bounds=2.0):
    if device is None: device = get_device()
    print(f"Viser started on {device}")

    server = viser.ViserServer()
    model = model.to(device)

    # --- 1. SETUP DATA GRID ---
    linspace = torch.linspace(-scene_bounds, scene_bounds, grid_size)
    grid_x, grid_y, grid_z = torch.meshgrid(linspace, linspace, linspace, indexing='ij')
    positions_tensor = torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(-1, 3).to(device)

    # --- 2. GUI ELEMENTS ---
    with server.gui.add_folder("Visualization"):
        density_slider = server.gui.add_slider("Threshold", min=0.0, max=5.0, step=0.1, initial_value=0.5)
        size_slider = server.gui.add_slider("Size Scale", min=0.001, max=0.1, step=0.001, initial_value=0.01)

    train_checkbox = None
    if trainer is not None:
        with server.gui.add_folder("Training Controls"):
            train_checkbox = server.gui.add_checkbox("Train Active", initial_value=False)
            reset_button = server.gui.add_button("Reset Weights")
            loss_handle = server.gui.add_text("Loss", initial_value="0.0000")

            @reset_button.on_click
            def _(_):
                trainer.reset()
                train_checkbox.value = False  # Pause training on reset
                update_scene(None)  # Force a redraw

    # --- 3. STATE TRACKING ---
    # We use a mutable dictionary to store the handle across function calls
    state = {"splat_handle": None}

    # --- 4. THE UPDATE LOOP ---
    def update_scene(_):
        # 1. Query the model
        with torch.no_grad():
            was_training = model.training
            model.eval()
            predicted_rgb, predicted_density = model(positions_tensor)
            model.train(was_training)

        # 2. Filter and Process (CPU side)
        all_points = positions_tensor.cpu().numpy()
        all_colors = predicted_rgb.cpu().numpy()
        all_densities = predicted_density.cpu().numpy().flatten()

        threshold = density_slider.value
        mask = all_densities > threshold

        if np.sum(mask) > 0:
            active_points = all_points[mask]
            active_colors = all_colors[mask]
            active_densities = all_densities[mask]

            radii = active_densities * size_slider.value
            variances = radii ** 2

            num_p = len(active_points)
            covariances = np.zeros((num_p, 3, 3), dtype=np.float32)
            covariances[:, 0, 0] = variances
            covariances[:, 1, 1] = variances
            covariances[:, 2, 2] = variances

            # UPDATE / CREATE: Viser automatically updates if name matches
            state["splat_handle"] = server.scene.add_gaussian_splats(
                "/nerf_splats",
                centers=active_points,
                rgbs=active_colors,
                opacities=np.ones((num_p, 1), dtype=np.float32),
                covariances=covariances
            )
        else:
            # REMOVE: Use the stored handle
            if state["splat_handle"] is not None:
                state["splat_handle"].remove()
                state["splat_handle"] = None

    density_slider.on_update(update_scene)
    size_slider.on_update(update_scene)

    update_scene(None)

    while True:
        if train_checkbox and train_checkbox.value:
            loss = trainer.step()
            loss_handle.value = f"{loss:.5f}"
            update_scene(None)
            time.sleep(0.01)
        else:
            time.sleep(0.1)

if __name__ == "__main__":
    pass