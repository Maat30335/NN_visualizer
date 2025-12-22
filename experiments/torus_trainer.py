import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from .torus_network import TorusModel


class TorusTrainer:
    def __init__(self, device="cpu"):
        self.device = device
        self.model = TorusModel().to(device)
        self.reset()

    def reset(self):
        print("Resetting Torus Model...")

        # Re-init weights
        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.0)

        self.model.apply(init_weights)

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        self.loss_fn = nn.MSELoss()

    def get_ground_truth(self, x):
        """
        Calculates the mathematical definition of a Torus.
        x: (N, 3) tensor
        """
        # Torus Parameters
        R = 1.2  # Major radius (distance from center of hole to center of tube)
        r = 0.5  # Minor radius (radius of the tube itself)

        # 1. Calculate Density (Shape)
        # Distance from Z-axis
        d_xy = torch.norm(x[:, :2], dim=1)
        # Distance from the center of the tube
        dist_tube = torch.sqrt((d_xy - R) ** 2 + x[:, 2] ** 2)

        # Inside the tube?
        # We create a soft boundary for easier learning: 1.0 inside, 0.0 outside
        target_density = (dist_tube < r).float().unsqueeze(1)

        # 2. Calculate Color (Pattern)
        # We make a rainbow pattern based on the angle around the Z-axis
        angle = torch.atan2(x[:, 1], x[:, 0])  # -pi to pi

        # Normalize angle to 0-1 for RGB math
        norm_angle = (angle + torch.pi) / (2 * torch.pi)

        red = norm_angle
        green = 1.0 - norm_angle
        blue = torch.abs(torch.sin(angle * 3.0))  # Stripes

        target_rgb = torch.stack([red, green, blue], dim=1)

        return target_rgb, target_density

    def step(self):
        self.model.train()

        # 1. Generate Random Training Points in range [-2, 2]
        batch_size = 4096
        # rand is [0, 1], so *4 -2 gives [-2, 2]
        x = (torch.rand(batch_size, 3, device=self.device) * 4.0) - 2.0

        # 2. Compute "Real" Answer (Ground Truth)
        target_rgb, target_density = self.get_ground_truth(x)

        # 3. Compute Network Prediction
        pred_rgb, pred_density = self.model(x)

        # 4. Calc Loss
        # We care more about density shape than color, so weight it higher
        loss_density = self.loss_fn(pred_density, target_density)
        loss_rgb = self.loss_fn(pred_rgb, target_rgb)

        # Only train color where density is > 0 (inside the object)
        # This prevents the model from wasting effort learning "invisible" colors
        mask = target_density > 0.5
        if mask.sum() > 0:
            loss_rgb = self.loss_fn(pred_rgb[mask.squeeze()], target_rgb[mask.squeeze()])
        else:
            loss_rgb = 0.0

        total_loss = loss_density + (loss_rgb * 0.1)

        # 5. Backprop
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return total_loss.item()