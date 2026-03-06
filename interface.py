import numpy as np
import torch
import torch.nn as nn
import sys


# ---------------------------------------------------------------------------
# 1. Re-Define the Network Architecture (Must match training exactly)
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


# ---------------------------------------------------------------------------
# 2. Load Model and Parameters
# ---------------------------------------------------------------------------
print("Loading saved model...")
try:
    # Use weights_only=False because we are loading Python floats (the means/stds) along with tensors
    checkpoint = torch.load("safe_hours_model.pth", weights_only=False)
except FileNotFoundError:
    print("\n[ERROR] 'safe_hours_model.pth' not found.")
    print("Please run 'train_and_save_model.py' first to generate the model file.")
    sys.exit()

model = BayesianNetwork()
model.load_state_dict(checkpoint["model_state_dict"])

age_mean = checkpoint["age_mean"]
age_std = checkpoint["age_std"]
dose_mean = checkpoint["dose_mean"]
dose_std = checkpoint["dose_std"]

print("Model loaded successfully.")


# ---------------------------------------------------------------------------
# 3. Calculator Logic
# ---------------------------------------------------------------------------
def calculate_safe_hours(user_age, user_sex, hourly_mGy):
    if hourly_mGy <= 0:
        print("Hourly exposure must be greater than 0 mGy/hr.")
        return

    scaled_age = (user_age - age_mean) / age_std

    # We scan doses from 0 to 65 mGy to find the threshold
    synthetic_doses_raw = np.linspace(0, 65, 1000)
    synthetic_doses_scaled = (synthetic_doses_raw - dose_mean) / dose_std

    profile_X = np.zeros((1000, 3))
    profile_X[:, 0] = user_sex
    profile_X[:, 1] = scaled_age
    profile_X[:, 2] = synthetic_doses_scaled
    profile_tensor = torch.tensor(profile_X, dtype=torch.float32)

    # Run Monte Carlo sampling
    model.train()  # Keep dropout active to extract statistical variance
    n_samples = 150
    mc_predictions = []

    with torch.no_grad():
        for _ in range(n_samples):
            rate_effect = model(profile_tensor).numpy()
            mc_predictions.append(rate_effect.flatten())

    mc_predictions = np.array(mc_predictions)

    mean_risk = mc_predictions.mean(axis=0)
    lower_bound = np.percentile(mc_predictions, 2.5, axis=0)

    baseline_risk = mean_risk[0]
    threshold_mGy = None

    for i in range(1, len(synthetic_doses_raw)):
        if lower_bound[i] > baseline_risk:
            threshold_mGy = synthetic_doses_raw[i]
            break

    sex_str = "Male" if user_sex == 1.0 else "Female"
    print("\n" + "=" * 50)
    print(f"SAFE WORKING HOURS REPORT")
    print(f"Profile: {int(user_age)}-year-old {sex_str}")
    print(f"Workspace Environment: {hourly_mGy} mGy/hr")
    print("-" * 50)

    if threshold_mGy is not None:
        safe_hours = threshold_mGy / hourly_mGy
        safe_days = safe_hours / 24

        print(f"Calculated Biological Threshold: {threshold_mGy:.2f} mGy")
        print(f"MAXIMUM SAFE WORKING TIME:")
        print(f"  -> {safe_hours:.2f} Hours")
        print(f"  -> ({safe_days:.2f} Days of continuous exposure)")
        print("\n*Note: Exceeding this time pushes the 95% statistical lower bound")
        print(" into the zone of definitive excess solid cancer risk.")
    else:
        print("Calculated Biological Threshold: >65 mGy")
        print("Based on the model, no statistically significant harm threshold")
        print("was detected for this profile under 65 mGy of total exposure.")
        print(
            f"You can safely work here for {65 / hourly_mGy:.2f} hours for the current year."
        )
    print("=" * 50)


# ---------------------------------------------------------------------------
# 4. Interactive Input Loop for Windows CLI
# ---------------------------------------------------------------------------
def main():
    print("\n" + "=" * 50)
    print(" RADIATION SAFE HOURS CALCULATOR INTERFACE")
    print("=" * 50)
    print("Type 'q' at any prompt to exit the program.")

    while True:
        try:
            age_input = input("\nEnter Employee Age: ")
            if age_input.lower() == "q":
                break
            user_age = float(age_input)

            sex_input = input("Enter Employee Sex (1 for Male, 2 for Female): ")
            if sex_input.lower() == "q":
                break
            user_sex = float(sex_input)

            if user_sex not in [1.0, 2.0]:
                print("Invalid input. Please enter exactly 1 or 2 for Sex.")
                continue

            dose_input = input("Enter expected hourly workspace radiation (mGy/hr): ")
            if dose_input.lower() == "q":
                break
            hourly_mGy = float(dose_input)

            calculate_safe_hours(user_age, user_sex, hourly_mGy)

        except ValueError:
            print("[ERROR] Invalid input. Please enter numerical values only.")


if __name__ == "__main__":
    main()
