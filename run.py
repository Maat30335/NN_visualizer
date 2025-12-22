import argparse
import torch
from visualizer import start_visualizer

# Import Experiments
from experiments.nerf_network import NeRFModel
from experiments.housing_visualizer import HousingNeRFAdapter
# Import our new Trainer
from experiments.housing_trainer import HousingTrainer


def run_nerf():
    print("Loading NeRF Toy...")
    model = NeRFModel()
    start_visualizer(model, scene_bounds=2.0)


def run_housing():
    print("Initializing Housing Trainer...")
    # 1. Create the Trainer (This loads data and creates the 'real' model)
    #    We let the visualizer detect the device, or we pass one explicitly.
    #    Let's rely on standard PyTorch auto-detection inside the classes.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available(): device = "mps"

    trainer = HousingTrainer(device=device)

    # 2. Create the Adapter
    #    The adapter needs the trainer's model to forward-pass correctly.
    #    Important: We pass the reference 'trainer.model'.
    #    When trainer.step() updates the model, the adapter sees the changes instantly.
    adapter = HousingNeRFAdapter(trainer.model)

    # 3. Start Visualizer
    #    We pass 'adapter' for drawing, and 'trainer' for the GUI buttons.
    start_visualizer(adapter, trainer=trainer, device=device, grid_size=30, scene_bounds=2.5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["nerf", "housing"], help="Which experiment to run")
    args = parser.parse_args()

    if args.mode == "nerf":
        run_nerf()
    elif args.mode == "housing":
        run_housing()


if __name__ == "__main__":
    main()