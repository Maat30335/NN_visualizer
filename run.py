import argparse
import torch
from visualizer import start_visualizer

# Import your experiments
# Note: You might need to move your adapter classes into these files
from experiments.nerf_network import NeRFModel
from experiments.housing_network import HousingModel
from experiments.housing_visualizer import HousingNeRFAdapter


def run_nerf():
    print("Loading NeRF Toy...")
    model = NeRFModel()
    # If you had weights: model.load_state_dict(...)

    # NeRF is already 3D, so maybe it doesn't need an adapter?
    # Or you can wrap it if you want to normalize inputs.
    # For now, let's assume NeRFModel works directly with visualizer.
    start_visualizer(model, scene_bounds=2.0)


def run_housing():
    print("Loading Housing Manifold...")
    # 1. Load Real Model
    real_model = HousingModel()
    try:
        real_model.load_state_dict(torch.load("experiments/housing_model.pth"))
    except FileNotFoundError:
        print("Error: 'housing_model.pth' not found. Run housing_train.py first!")
        return

    # 2. Wrap in Adapter
    adapter = HousingNeRFAdapter(real_model)

    # 3. Run
    start_visualizer(adapter, grid_size=40, scene_bounds=2.5)


def main():
    parser = argparse.ArgumentParser(description="Neural Network Visualizer")
    parser.add_argument("mode", choices=["nerf", "housing"], help="Which model to visualize")

    args = parser.parse_args()

    if args.mode == "nerf":
        run_nerf()
    elif args.mode == "housing":
        run_housing()


if __name__ == "__main__":
    main()