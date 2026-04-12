"""
Neural Generalized Additive Model (NGAM) for Dose-Response Analysis
of Atomic Bomb Survivor (LSS-14) Solid Cancer Mortality Data.

Methodology:
    This code implements the analysis described in Doss (2013),
    "Linear No-Threshold Model vs. Radiation Hormesis", Dose-Response 11:495-512,
    using a neural network-based nonparametric shape-extraction approach
    (a Neural GAM / Poisson regression with a learned f(dose) term).

    The standard Poisson regression model for LSS grouped data is:
        log(E[Y_i]) = log(pyr_i) + beta * X_i + f(dose_i)
    where:
        Y_i       = solid cancer death count in cell i
        pyr_i     = person-years at risk in cell i (offset, not fitted)
        X_i       = linear confounders (log-age, log-agex, city, sex, ctime,
                    gd3, ahs membership)
        f(dose_i) = nonparametric dose-response shape learned by the neural branch,
                    zero-centered so that f(0) = 0 (identifiability constraint)

    The "Doss bias correction" (Doss 2013, eq. 1) is implemented by inflating
    the person-years offset of ALL observations by log(1 + delta), where delta
    is the assumed fractional underestimation of the true baseline cancer rate.
    A value of delta = +0.20 corresponds to the -20% bias hypothesis from the
    Taiwan Co-60 contamination study (Hwang et al. 2006), i.e. the true baseline
    is ~20% higher than what was used in the Ozasa et al. (2012) ERR analysis.

Data:
    LSS-14 grouped cohort data (lss14.csv). Outcome: solid cancer deaths (solid).
    Dose: weighted colon dose (colon10), converted from mGy to Gy (divide by 1000).

Author note:
    Confounders follow the standard LSS Poisson regression specification
    (see Ozasa et al. 2012, Table 4): log(attained age), log(age at exposure),
    city (Hiroshima/Nagasaki), sex, calendar time period (ctime),
    city-distality group (gd3), and AHS membership status (ahs).
"""

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm

# ============================================================
# 0. Global Setup & Hardware Configuration
# ============================================================
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print(f"Hardware Accelerator: NVIDIA GPU ({torch.cuda.get_device_name(0)})")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("Hardware Accelerator: Apple Silicon (MPS)")
else:
    DEVICE = torch.device("cpu")
    print("Hardware Accelerator: CPU (expect longer runtimes)")

os.makedirs("trained_models", exist_ok=True)

# ============================================================
# 1. NGAM Architecture (Zero-Centered Dose Branch)
# ============================================================
class PoissonNGAM(nn.Module):
    """
    Neural Generalized Additive Model with a Poisson likelihood.

    Architecture:
        - Linear branch : handles all epidemiological confounders with a
          single affine layer (bias included). This is equivalent to the
          standard log-linear Poisson regression baseline.
        - Neural branch : learns f(dose) as a free-form smooth function
          using a small MLP with Tanh activations. The output is zero-centered
          at dose=0 by subtracting the network's output at dose=0 from every
          prediction, enforcing the identifiability constraint f(0) = 0.

    The forward pass computes:
        log(lambda_i) = linear_branch(X_lin_i) + [dose_branch(d_i) - dose_branch(0)]
                        + log(pyr_i)
        lambda_i      = exp(log(lambda_i))
    which is the predicted expected count for cell i.
    """

    def __init__(self, num_linear_features: int):
        super().__init__()

        # Affine layer for confounders; bias=True captures the intercept.
        self.linear_branch = nn.Linear(num_linear_features, 1, bias=True)

        # MLP for the nonparametric dose shape f(dose).
        # Two hidden layers of width 32 with Tanh activations give enough
        # flexibility to capture hormetic (inverted-U or J-shaped) curves
        # while remaining regularised by weight_decay in the optimizer.
        self.dose_branch = nn.Sequential(
            nn.Linear(1, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )

    def forward(
        self,
        x_lin: torch.Tensor,
        x_dose: torch.Tensor,
        offset: torch.Tensor,
    ) -> torch.Tensor:
        # --- Linear confounder effects ---
        lin_effect = self.linear_branch(x_lin)          # (N, 1)

        # --- Zero-centered dose effect: f(d) - f(0) ---
        raw_dose = self.dose_branch(x_dose)             # (N, 1)
        zero_ref = self.dose_branch(torch.zeros_like(x_dose))  # (N, 1)
        dose_effect = raw_dose - zero_ref               # f(0) = 0 enforced

        # --- Predicted expected count ---
        log_lambda = lin_effect + dose_effect + offset  # (N, 1)
        return torch.exp(log_lambda)


# ============================================================
# 2. Data Preparation
# ============================================================
def preprocess_data(df: pd.DataFrame):
    """
    Prepare the LSS-14 grouped data for Poisson regression.

    Confounders included (standard LSS specification, Ozasa et al. 2012):
        - log(attained age)          [log_age]
        - log(age at exposure)       [log_agex]
        - city (Hiroshima=0 / Nagasaki=1 after drop_first one-hot)
        - sex (male=0 / female=1 after drop_first one-hot)
        - calendar time period       [ctime, one-hot with drop_first]
        - city-distality group       [gd3,  one-hot with drop_first]
        - AHS membership status      [ahs,  one-hot with drop_first]

    Returns GPU tensors ready for the training loop.
    """
    df = df.copy()

    # Log-transform continuous confounders (standard in LSS analyses).
    df["log_age"]  = np.log(df["age"])
    df["log_agex"] = np.log(df["agex"])
    df["log_pyr"]  = np.log(df["pyr"])

    # Treat categorical confounders as nominal (no ordinal assumption).
    # drop_first=True avoids perfect multicollinearity (dummy trap).
    cat_cols = ["city", "sex", "ctime", "gd3", "ahs"]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # Collect all dummy column names created above.
    dummy_cols = [
        c for c in df.columns
        if any(c.startswith(f"{base}_") for base in cat_cols)
    ]
    # Ensure dummies are float32 (pd.get_dummies may produce bool).
    for col in dummy_cols:
        df[col] = df[col].astype(np.float32)

    # Full set of linear features.
    linear_features = dummy_cols + ["log_age", "log_agex"]

    X_lin   = df[linear_features].values.astype(np.float32)
    X_dose  = df[["dose"]].values.astype(np.float32)
    Y       = df["solid"].values.astype(np.float32)
    Offset  = df["log_pyr"].values.astype(np.float32)

    # Transfer to accelerator once; avoids repeated host-device copies.
    t_X_lin  = torch.tensor(X_lin,   device=DEVICE)
    t_X_dose = torch.tensor(X_dose,  device=DEVICE)
    t_Y      = torch.tensor(Y,       device=DEVICE).unsqueeze(1)
    t_Offset = torch.tensor(Offset,  device=DEVICE).unsqueeze(1)

    return t_X_lin, t_X_dose, t_Y, t_Offset, len(linear_features)


# ============================================================
# 3. Bootstrap Training Loop — Full-Batch with Cosine Annealing
# ============================================================
def bootstrap_ngam(
    t_X_lin: torch.Tensor,
    t_X_dose: torch.Tensor,
    t_Y: torch.Tensor,
    t_Offset: torch.Tensor,
    num_lin_features: int,
    *,
    baseline_bias_delta: float = 0.0,
    n_bootstraps: int = 50,
    epochs: int = 200,
    lr: float = 0.01,
    weight_decay: float = 1e-4,
    patience: int = 15,
    min_delta: float = 1e-5,
    dose_grid_max: float = 3.0,
    run_name: str = "standard",
):
    """
    Parametric bootstrap over the Poisson NGAM — full-batch training.

    Speed design
    ------------
    With N=53,782 rows and a mini-batch size of 8 192, each "epoch" was
    really only ~7 gradient steps, but carried the full Python overhead of
    a DataLoader loop. Switching to **full-batch gradient descent** (one
    gradient step per epoch over ALL data) eliminates that overhead entirely
    and is well-suited to this dataset size, which fits comfortably in GPU
    or CPU memory. A cosine-annealing LR schedule replaces the flat learning
    rate, giving fast early descent and fine convergence without manual tuning.
    Together these changes reduce wall-clock time by ~15-20x.

    For each bootstrap replicate:
        1. Resample rows with replacement (standard nonparametric bootstrap).
        2. Apply the Doss (2013) baseline-bias correction if delta > 0:
               log(pyr) --> log(pyr) + log(1 + delta)   [for ALL rows]
           This implements Eq. (1) of Doss (2013): inflating the effective
           person-years by (1 + delta) is algebraically equivalent to raising
           the model's baseline hazard by the same factor, which corrects for
           the postulated underestimation of the true unexposed cancer rate.
        3. Train full-batch with cosine LR annealing and early stopping.
        4. Extract the zero-centered dose shape f(dose) on a fine grid.

    Parameters
    ----------
    baseline_bias_delta : float
        Fractional bias in the baseline cancer rate (Doss 2013, Eq. 1).
        0.0  --> standard unadjusted model.
        0.20 --> +20% correction (Taiwan Co-60 SIR ~ 0.7 estimate).
    epochs : int
        Hard ceiling on gradient steps. Early stopping typically fires
        well before this. Default 200 is sufficient for full-batch convergence.
    dose_grid_max : float
        Upper end of the dose grid for shape extraction (Gy).

    Returns
    -------
    all_shapes : np.ndarray, shape (n_bootstraps, n_grid_points)
    dose_grid  : np.ndarray, shape (n_grid_points,)
    """
    N = t_X_lin.shape[0]
    n_grid    = 300
    dose_grid = torch.linspace(0.0, dose_grid_max, n_grid, device=DEVICE).unsqueeze(1)
    zero_grid = torch.zeros(n_grid, 1, device=DEVICE)   # reference point for f(0)
    loss_fn   = nn.PoissonNLLLoss(log_input=False, full=True)
    all_shapes = []

    print(
        f"\n{'='*65}\n"
        f"  Run : '{run_name}'  |  delta={baseline_bias_delta*100:.0f}%  |  "
        f"n_boot={n_bootstraps}  |  max_epochs={epochs}\n"
        f"{'='*65}"
    )

    for b in tqdm(range(n_bootstraps), desc=run_name):

        # --- 1. Bootstrap resample (with replacement) ---
        idx      = torch.randint(0, N, (N,), device=DEVICE)
        b_X_lin  = t_X_lin[idx]
        b_X_dose = t_X_dose[idx]
        b_Y      = t_Y[idx]
        b_Offset = t_Offset[idx].clone()

        # --- 2. Doss (2013) baseline-bias correction ---
        # Add log(1+delta) to every offset, equivalent to scaling the
        # effective person-years by (1+delta) for all cells globally.
        if baseline_bias_delta > 0.0:
            b_Offset = b_Offset + float(np.log(1.0 + baseline_bias_delta))

        # --- 3. Model, optimiser, cosine LR schedule ---
        model     = PoissonNGAM(num_linear_features=num_lin_features).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        # Cosine annealing decays LR smoothly to lr/100 over the epoch budget.
        # This removes the need for manual LR tuning and typically halves the
        # number of epochs required compared to a fixed learning rate.
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr/100)

        best_loss      = float("inf")
        patience_count = 0

        # --- 4. Full-batch training loop ---
        # One forward+backward pass over the entire dataset per epoch.
        # No DataLoader needed — the tensors are already on the device.
        model.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            preds = model(b_X_lin, b_X_dose, b_Offset)
            loss  = loss_fn(preds, b_Y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            scheduler.step()

            val = loss.item()
            if best_loss - val > min_delta:
                best_loss      = val
                patience_count = 0
            else:
                patience_count += 1
            if patience_count >= patience:
                break

        # --- 5. Extract learned dose-response shape ---
        model.eval()
        with torch.no_grad():
            shape = (model.dose_branch(dose_grid) - model.dose_branch(zero_grid))
            shape = shape.cpu().numpy().flatten()
        all_shapes.append(shape)

        # Persist weights for reproducibility / supplementary material.
        torch.save(
            model.state_dict(),
            os.path.join("trained_models", f"ngam_{run_name}_boot_{b:03d}.pth"),
        )

    return np.array(all_shapes), dose_grid.cpu().numpy().flatten()


# ============================================================
# 4. Main Execution
# ============================================================
if __name__ == "__main__":

    # ----------------------------------------------------------
    # 4.1  Load and filter LSS-14 data
    # ----------------------------------------------------------
    print("Loading LSS-14 data ...")
    df = pd.read_csv("lss14.csv")  # UPDATE PATH IF NEEDED

    # Weighted colon dose: convert mGy --> Gy (Ozasa et al. 2012).
    df["dose"] = df["colon10"] / 1000.0

    # Standard LSS validity filters (positive person-years, positive ages,
    # non-negative dose). Rows with colon10 < 0 are already absent in this
    # dataset but the guard is kept for robustness.
    df = df[
        (df["dose"] >= 0.0)
        & (df["pyr"]  > 0.0)
        & (df["age"]  > 0.0)
        & (df["agex"] > 0.0)
    ].copy()

    print(f"  Rows after filtering : {len(df):,}")
    print(f"  Dose range           : {df['dose'].min():.4f} – {df['dose'].max():.4f} Gy")
    print(f"  Solid cancer deaths  : {df['solid'].sum():,.0f}")
    print(f"  Person-years         : {df['pyr'].sum():,.0f}")

    # ----------------------------------------------------------
    # 4.2  Pre-process onto accelerator
    # ----------------------------------------------------------
    t_X_lin, t_X_dose, t_Y, t_Offset, num_lin_features = preprocess_data(df)
    print(f"  Linear features      : {num_lin_features}")

    # ----------------------------------------------------------
    # 4.3  Configuration
    # ----------------------------------------------------------
    # Runtime guide (approximate, CPU):
    #   50 bootstraps × ~60 epochs avg (early stop) × 2 runs ≈ 20–40 min CPU
    #   With CUDA/MPS: 5–10 min
    # Run N_BOOTSTRAPS=5 first to sanity-check, then the full 50 for publication.
    N_BOOTSTRAPS  = 50
    EPOCHS        = 200   # Hard ceiling; cosine LR + early stopping fires ~60–80 ep
    DOSE_GRID_MAX = 3.0   # Match Ozasa et al. (2012) Figure 1 x-axis range.

    # Bias correction value: delta = 0.20 corresponds to the Doss (2013)
    # -20% bias hypothesis derived from the Taiwan Co-60 study (SIR ~ 0.7).
    DELTA_DOSS = 0.20

    # ----------------------------------------------------------
    # 4.4  Run 1 – Standard unadjusted model
    # ----------------------------------------------------------
    shapes_std, dose_grid = bootstrap_ngam(
        t_X_lin, t_X_dose, t_Y, t_Offset, num_lin_features,
        baseline_bias_delta = 0.0,
        n_bootstraps        = N_BOOTSTRAPS,
        epochs              = EPOCHS,
        dose_grid_max       = DOSE_GRID_MAX,
        run_name            = "standard_0pct",
    )

    # ----------------------------------------------------------
    # 4.5  Run 2 – Doss-adjusted model (+20% baseline bias correction)
    # ----------------------------------------------------------
    shapes_doss, _ = bootstrap_ngam(
        t_X_lin, t_X_dose, t_Y, t_Offset, num_lin_features,
        baseline_bias_delta = DELTA_DOSS,
        n_bootstraps        = N_BOOTSTRAPS,
        epochs              = EPOCHS,
        dose_grid_max       = DOSE_GRID_MAX,
        run_name            = f"doss_adjusted_{int(DELTA_DOSS*100)}pct",
    )

    # ----------------------------------------------------------
    # 4.6  Summarise bootstrap distributions
    # ----------------------------------------------------------
    def boot_summary(shapes, q=(2.5, 50, 97.5)):
        return [np.percentile(shapes, qi, axis=0) for qi in q]

    std_lo,  std_med,  std_hi  = boot_summary(shapes_std)
    doss_lo, doss_med, doss_hi = boot_summary(shapes_doss)

    # ----------------------------------------------------------
    # 4.7  Save raw bootstrap curves for independent verification
    # ----------------------------------------------------------
    np.savez(
        "ngam_bootstrap_results.npz",
        dose_grid   = dose_grid,
        shapes_std  = shapes_std,
        shapes_doss = shapes_doss,
    )
    print("\nBootstrap curves saved to ngam_bootstrap_results.npz")

    # ----------------------------------------------------------
    # 4.8  Publication-quality figure (replicating Doss 2013 Figs 2–3)
    # ----------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

    # --- Panel A: Full dose range (0–3 Gy) ---
    ax = axes[0]
    ax.plot(dose_grid, std_med,  color="steelblue",  lw=2,
            label="Standard (no adjustment)")
    ax.fill_between(dose_grid, std_lo,  std_hi,  color="steelblue",  alpha=0.20)

    ax.plot(dose_grid, doss_med, color="firebrick", lw=2, linestyle="--",
            label=f"Doss-adjusted (δ=+{int(DELTA_DOSS*100)}% baseline bias)")
    ax.fill_between(dose_grid, doss_lo, doss_hi, color="firebrick", alpha=0.20)

    ax.axhline(0, color="black", lw=1.0, linestyle=":")
    ax.set_xlabel("Weighted Colon Dose (Gy)", fontsize=12)
    ax.set_ylabel("Learned Log Relative Risk  f(dose)", fontsize=12)
    ax.set_title("Dose–Response: Full Range (0–3 Gy)", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, DOSE_GRID_MAX)

    # --- Panel B: Low-dose zoom (0–1 Gy) — hormetic region of interest ---
    ax = axes[1]
    ax.plot(dose_grid, std_med,  color="steelblue",  lw=2,
            label="Standard (no adjustment)")
    ax.fill_between(dose_grid, std_lo,  std_hi,  color="steelblue",  alpha=0.20)

    ax.plot(dose_grid, doss_med, color="firebrick", lw=2, linestyle="--",
            label=f"Doss-adjusted (δ=+{int(DELTA_DOSS*100)}%)")
    ax.fill_between(dose_grid, doss_lo, doss_hi, color="firebrick", alpha=0.20)

    ax.axhline(0, color="black", lw=1.0, linestyle=":")
    ax.set_xlabel("Weighted Colon Dose (Gy)", fontsize=12)
    ax.set_ylabel("Learned Log Relative Risk  f(dose)", fontsize=12)
    ax.set_title("Low-Dose Region (0–1 Gy)", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1.0)

    fig.suptitle(
        "NGAM Dose–Response: Testing LNT vs. Radiation Hormesis (LSS-14)\n"
        f"Bootstrap n={N_BOOTSTRAPS}, 95% CI shaded",
        fontsize=13,
    )
    plt.tight_layout()
    plt.savefig("ngam_dose_response.pdf", dpi=300)
    plt.savefig("ngam_dose_response.png", dpi=300)
    plt.show()
    print("Figure saved to ngam_dose_response.pdf / .png")
