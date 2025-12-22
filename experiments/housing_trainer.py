import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler

# Import the model definition
from .housing_network import HousingModel


class HousingTrainer:
    def __init__(self, device="cpu"):
        self.device = device

        # 1. Load and Prep Data (Once)
        print("Loading Data...")
        data = fetch_california_housing()
        X, y = data.data, data.target

        # Normalize
        self.X_scaled = StandardScaler().fit_transform(X)
        self.y_scaled = StandardScaler().fit_transform(y.reshape(-1, 1))

        # Keep tensors ready on the GPU/Device
        self.X_train = torch.tensor(self.X_scaled, dtype=torch.float32).to(device)
        self.y_train = torch.tensor(self.y_scaled, dtype=torch.float32).to(device)

        # 2. Initialize Model
        self.model = HousingModel().to(device)
        self.reset()  # Sets up optimizer and weights

    def reset(self):
        """Re-initializes weights in place, keeping the same object."""
        print("Resetting Model Weights...")

        # Helper function to reset a single layer
        def init_weights(m):
            if isinstance(m, nn.Linear):
                # Standard Xavier initialization for better training
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

        # Apply this to every layer in the EXISTING model
        self.model.apply(init_weights)

        # Re-create optimizer (it needs to know the parameters have changed)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        # Reset loss function just in case
        self.loss_fn = nn.MSELoss()

    def step(self):
        """Runs one epoch of training."""
        self.model.train()  # Set to train mode

        # Forward pass
        preds = self.model(self.X_train)
        loss = self.loss_fn(preds, self.y_train)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()