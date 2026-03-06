import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# ---------------------------------------------------------------------------
# 1. Data Loading and Preprocessing
# ---------------------------------------------------------------------------
try:
    df = pd.read_csv("lss14.csv")
    print("Successfully loaded lss14.csv")
    df = df[(df["colon10"] >= 0) & (df["colon10"] <= 1000)]
except FileNotFoundError:
    print("Error: lss14.csv not found. Using dummy data for demonstration.")
    np.random.seed(42)
    df = pd.DataFrame(
        {
            "sex": np.random.randint(1, 3, 1000),  # 1 (Male) or 2 (Female)
            "agex": np.random.uniform(0, 80, 1000),
            "colon10": np.random.uniform(0, 1000, 1000),
            "solid": np.random.randint(0, 10, 1000),
            "pyr": np.random.uniform(10, 10000, 1000),
        }
    )

features = ["sex", "agex", "colon10"]
target = "solid"
exposure = "pyr"

X_data = torch.tensor(df[features].values, dtype=torch.float32)
y_data = torch.tensor(df[target].values, dtype=torch.float32).view(-1, 1)
pyr_data = torch.tensor(df[exposure].values, dtype=torch.float32).view(-1, 1)

# Standardization parameters (CRITICAL: We must save these for the interface)
age_mean = X_data[:, 1].mean().item()
age_std = X_data[:, 1].std().item()
dose_mean = X_data[:, 2].mean().item()
dose_std = X_data[:, 2].std().item()

X_data[:, 1] = (X_data[:, 1] - age_mean) / age_std
X_data[:, 2] = (X_data[:, 2] - dose_mean) / dose_std

dataset = TensorDataset(X_data, y_data, pyr_data)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)


# ---------------------------------------------------------------------------
# 2. Fully Connected Bayesian Network
# ---------------------------------------------------------------------------
class BayesianNetwork(nn.Module):
    def __init__(self, hidden_dim=64, dropout_p=0.05):
        super(BayesianNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.SiLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_dim, 1),
        )
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        linear_predictor = self.bias + self.net(x)
        rate = torch.exp(linear_predictor)
        return rate


def poisson_loss(rate, expected_deaths, pyr):
    lambda_val = rate * pyr
    epsilon = 1e-8
    nll = lambda_val - expected_deaths * torch.log(lambda_val + epsilon)
    return nll.mean()


# ---------------------------------------------------------------------------
# 3. Model Training
# ---------------------------------------------------------------------------
model = BayesianNetwork()
optimizer = optim.Adam(model.parameters(), lr=0.002)
epochs = 200

print("Training underlying predictive model...")
model.train()
for epoch in tqdm(range(epochs), desc="Training"):
    for batch_X, batch_y, batch_pyr in dataloader:
        optimizer.zero_grad()
        rate_pred = model(batch_X)
        loss = poisson_loss(rate_pred, batch_y, batch_pyr)
        loss.backward()
        optimizer.step()

print("\nModel trained successfully.")

# ---------------------------------------------------------------------------
# 4. Save Model and Parameters
# ---------------------------------------------------------------------------
# We save a dictionary containing the network weights AND the scaling variables
save_dict = {
    "model_state_dict": model.state_dict(),
    "age_mean": age_mean,
    "age_std": age_std,
    "dose_mean": dose_mean,
    "dose_std": dose_std,
}

save_path = "safe_hours_model.pth"
torch.save(save_dict, save_path)
print(f"Model weights and scaling parameters safely exported to '{save_path}'.")
