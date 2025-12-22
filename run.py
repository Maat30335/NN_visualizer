import argparse
import torch
from visualizer import start_visualizer

# Experiments
from experiments.nerf_network import NeRFModel
from experiments.housing_visualizer import HousingNeRFAdapter
from experiments.housing_trainer import HousingTrainer

# NEW IMPORTS
from experiments.torus_trainer import TorusTrainer


def run_nerf():
    print("Loading NeRF Toy...")
    model = NeRFModel()
    start_visualizer(model, scene_bounds=2.0)


def run_housing():
    print("Initializing Housing Trainer...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available(): device = "mps"

    trainer = HousingTrainer(device=device)
    adapter = HousingNeRFAdapter(trainer.model)

    start_visualizer(adapter, trainer=trainer, device=device, grid_size=30, scene_bounds=2.5)


# NEW FUNCTION
def run_torus():
    print("Initializing Torus Trainer...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available(): device = "mps"

    trainer = TorusTrainer(device=device)

    # TorusModel outputs (rgb, density) directly, so no Adapter needed!
    # We just pass the model straight to the visualizer.
    start_visualizer(trainer.model, trainer=trainer, device=device, grid_size=35, scene_bounds=2.0)


def main():
    parser = argparse.ArgumentParser()
    # Add "torus" to choices
    parser.add_argument("mode", choices=["nerf", "housing", "torus"], help="Which experiment to run")
    args = parser.parse_args()

    if args.mode == "nerf":
        run_nerf()
    elif args.mode == "housing":
        run_housing()
    elif args.mode == "torus":
        run_torus()


if __name__ == "__main__":
    main()