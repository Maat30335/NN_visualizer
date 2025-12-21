import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from housing_network import HousingModel


def train():
    print("Fetching data...")
    data = fetch_california_housing()
    X, y = data.data, data.target

    # Normalize data (Crucial!)
    # We save the scalers implicitly by training on scaled data
    X_scaled = StandardScaler().fit_transform(X)
    y_scaled = StandardScaler().fit_transform(y.reshape(-1, 1))

    X_train = torch.tensor(X_scaled, dtype=torch.float32)
    y_train = torch.tensor(y_scaled, dtype=torch.float32)

    # Setup Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HousingModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    print("Training (200 epochs)...")
    for epoch in range(200):
        preds = model(X_train.to(device))
        loss = loss_fn(preds, y_train.to(device))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            print(f"Epoch {epoch}: Loss {loss.item():.4f}")

    torch.save(model.state_dict(), "housing_model.pth")
    print("Saved 'housing_model.pth'")


if __name__ == "__main__":
    train()