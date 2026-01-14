# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import gc
import os
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"





# -------------------------
# Config
# -------------------------
BATCH_SIZE = 100
NUM_EPOCHS = 1000
LR = 1e-3
EVAL_EVERY = 50          # evaluate every N epochs
ITER_STEPS = 1000        # number of iterations x_{t+1} = AE(x_t)
EVAL_BATCHES = 1         # how many test batches to evaluate each time (1 is usually enough)
SEED = 123
ASYMP_TAIL=200

torch.manual_seed(SEED)

device = "cuda:0"

pct_belows=[]
OUT_DIR = "zncc_plots"
eps=10**-8

THRESH = 0
LAG_TO_CHECK = ITER_STEPS   # or choose some other lag

# history for the %<0.5 curve
eval_epochs = []
eval_pct_below = []
import matplotlib as mpl

mpl.rcParams.update({
    "font.size": 16,          # base font
    "axes.labelsize": 24,   # x/y label
    "axes.titlesize": 24,     # title
    "xtick.labelsize": 20,    # x tick labels
    "ytick.labelsize": 20,    # y tick labels
    "legend.fontsize": 20,    # legend
    "figure.titlesize": 24,   # suptitle
})
# -------------------------
# Data: Fashion-MNIST
# -------------------------
transform = transforms.ToTensor()  # outputs in [0, 1]

train_ds = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
test_ds  = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=False)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=False)

# -------------------------
# MLP Autoencoder
# -------------------------


def iterate_to_asymptotic_state(step_fn, x0, steps: int):
    """
    Runs x_{t+1} = step_fn(x_t) for `steps` steps and returns x_* (late-time state).
    NOTE: step_fn must be differentiable if you want Jacobians.
    """
    x = x0
    for _ in range(steps):
        x = step_fn(x)
    return x

@torch.no_grad()
def estimate_jac_frob_sq_over_n(step_fn, x_star, K: int = 4, eps: float = 1e-12, rademacher: bool = True):
    """
    Estimates (||J||_F^2)/N at x_star, where J = d step_fn(x)/dx at x_star,
    using Hutchinson: E||Jv||^2 = tr(J^T J) = ||J||_F^2 for Var(v_i)=1 probes.

    Returns: scalar tensor (on CPU) for this x_star sample.
    """
    x_star = x_star.detach().requires_grad_(True)
    N = x_star.numel()

    acc = 0.0
    for _ in range(K):
        if rademacher:
            v = torch.empty_like(x_star).bernoulli_(0.5).mul_(2).sub_(1)  # ±1
        else:
            v = torch.randn_like(x_star)  # N(0,1)

        # JVP gives (y, Jv)
        _, Jv = torch.autograd.functional.jvp(step_fn, (x_star,), (v,), create_graph=False, strict=False)  # :contentReference[oaicite:2]{index=2}

        acc = acc + (Jv.pow(2).sum() / (N + eps))

    return (acc / K).detach().cpu()


def estimate_spectral_norm_jt_j(step_fn, x_star, n_iters: int = 20, eps: float = 1e-12):
    """
    Estimates ||J||_2 (largest singular value) via power iteration on J^T J.

    Uses:
      w = J v   (JVP)
      u = J^T w (VJP with v=w)
    """
    x_star = x_star.detach().requires_grad_(True)

    v = torch.randn_like(x_star)
    v = v / v.norm().clamp_min(eps)

    sigma_est = None

    for _ in range(n_iters):
        # w = J v
        _, w = torch.autograd.functional.jvp(
            step_fn, (x_star,), (v,), create_graph=False, strict=False
        )

        # u = J^T w  (must pass v=w because output is not scalar)
        _, u = torch.autograd.functional.vjp(
            step_fn, x_star, v=w, create_graph=False, strict=False
        )

        # power iteration on J^T J
        u_norm = u.norm().clamp_min(eps)
        v = (u / u_norm).detach()

        # if ||v||=1, then ||J v|| approximates top singular value
        sigma_est = w.norm().detach()

    return sigma_est

def estimate_spectral_radius_jvp(step_fn, x_star, n_iters: int = 20, eps: float = 1e-12):
    """
    Estimates spectral radius rho(J) of J = d step_fn(x)/dx at x = x_star
    using power iteration on J via JVPs.

    Returns:
      rho_est: estimate of dominant |eigenvalue|
      jac_action_norm: ||J v|| / ||v|| at final iterate (often tracks rho for normal-ish J)
    """
    # Ensure we can differentiate w.r.t x_star
    x_star = x_star.detach().requires_grad_(True)

    # Random init direction
    v = torch.randn_like(x_star)
    v = v / (v.norm().clamp_min(eps))

    rho_est = None
    jac_action_norm = None

    for _ in range(n_iters):
        # JVP: gives (step_fn(x_star), J v)
        _, Jv = torch.autograd.functional.jvp(step_fn, (x_star,), (v,), create_graph=False, strict=False)

        # Use norm growth as a robust magnitude proxy
        Jv_norm = Jv.norm().clamp_min(eps)
        v_norm = v.norm().clamp_min(eps)
        jac_action_norm = (Jv_norm / v_norm).detach()

        # Rayleigh quotient (real-valued approximation); take abs for magnitude
        rq = (v * Jv).sum() / (v * v).sum().clamp_min(eps)
        rho_est = rq.abs().detach()

        # Power iteration update
        v = (Jv / Jv_norm).detach()

    return rho_est, jac_action_norm


def save_per_image_curve_png(curve_1d: torch.Tensor, epoch: int, out_dir: str, title_prefix: str = ""):
    """
    curve_1d: (L,) tensor on CPU or GPU
    Saves: per_image_curve_epochXXXX.png
    """
    os.makedirs(out_dir, exist_ok=True)
    y = curve_1d.detach().cpu().numpy()
    x = list(range(len(y)))

    fig = plt.figure()
    plt.plot(x, y)
    plt.xlabel("Lag time τ (iterations)")
    plt.ylabel(" ⟨ZNCC⟩ ")
    plt.title(f"{title_prefix}Per-image correlation curve (epoch {epoch})")
    plt.yscale("log")
    plt.tight_layout()

    fname = os.path.join(out_dir, f"per_image_curve_epoch{epoch:04d}.png")
    plt.savefig(fname, dpi=200, bbox_inches="tight")  # savefig docs :contentReference[oaicite:1]{index=1}
    plt.close(fig)  # helps in Spyder/interactive mode :contentReference[oaicite:2]{index=2}


def save_pct_below_curve_png(epochs: list[int], pct_below: list[float], out_dir: str, thresh: float, lag: int):
    """
    Saves: pct_below_curve.png
    """
    os.makedirs(out_dir, exist_ok=True)

    fig = plt.figure()
    plt.plot(epochs, pct_below, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel(f"% images with \n C_i(τ={lag}) < {thresh}")
    plt.title("% of images vs epoch")
    plt.tight_layout()

    fname = os.path.join(out_dir, "pct_below_curve.png")
    plt.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)

class MLPAutoencoder(nn.Module):
    def __init__(self, input_dim=28*28, latent_dim=10, hidden_dims=(512, 256)):
        super().__init__()
        h1, h2 = hidden_dims

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            #nn.Relu(inplace=True),
            nn.Tanh()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, input_dim),
            nn.Tanh(),
 
        )

    def forward(self, x):
        # x: (B, 1, 28, 28) or (B, 784)
        if x.dim() == 4:
            x = x.view(x.size(0), -1)
        z = self.encoder(x)
        out = self.decoder(z)
        return out  # (B, 784)

model = MLPAutoencoder(latent_dim=10).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# -------------------------
# ZMNCC / ZNCC
# -------------------------


def aging_coeff(curves: torch.Tensor,
                tau1: int,
                tau2: int) -> torch.Tensor:
    """
    curves: (B, T+1)
    Returns: aging coeff per image (B,)
    """
    B, T1 = curves.shape
    taus = torch.arange(T1, device=curves.device)

    # select tau window
    mask = (taus >= tau1) & (taus <= tau2)
    taus = taus[mask].float()   # (K,)
    C = curves[:, mask]         # (B, K)

    # avoid zeros or negatives
    C = torch.clamp(C, min=1e-12)

    logC = torch.log(C)         # (B, K)
    logT = torch.log(taus)      # (K,)

    # fit slope via least squares for each image:
    # slope = cov(logT, logC) / var(logT)
    logT_centered = logT - logT.mean()
    varT = (logT_centered**2).sum()

    slopes = (logC - logC.mean(dim=1, keepdim=True))
    slopes = (slopes * logT_centered).sum(dim=1) / varT  # (B,)

    return slopes




def zncc_anyshape(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    assert a.shape == b.shape
    D = a.shape[-1]
    a2 = a.reshape(-1, D)
    b2 = b.reshape(-1, D)

    a0 = a2 - a2.mean(dim=1, keepdim=True)
    b0 = b2 - b2.mean(dim=1, keepdim=True)

    num = (a0 * b0).sum(dim=1)
    var_a = (a0 * a0).sum(dim=1)
    var_b = (b0 * b0).sum(dim=1)

    # Stabilize denominator to avoid divide-by-zero / inf
    den = torch.sqrt(var_a * var_b).clamp_min(eps)  # clamp_min uses torch.clamp semantics :contentReference[oaicite:2]{index=2}
    corr = num / den

    # If either variance is ~0, correlation is undefined -> set to 0 (or choose another convention)
    #d = (var_a < eps) | (var_b < eps)
    #corr[bad] = 0.0

    # If BOTH are constant, you can optionally define corr=1 only when they are (almost) identical
    #both_const = (var_a < eps) & (var_b < eps)
    #same = (a2 - b2).abs().amax(dim=1) < 1e-6
    #corr[both_const & same] = 1.0

    #corr = corr.clamp(-1.0, 1.0)
    #corr = torch.nan_to_num(corr, nan=0.0, posinf=1.0, neginf=-1.0)  # :contentReference[oaicite:3]{index=3}

    return corr.reshape(a.shape[:-1])


def per_image_curve_avg_over_tw(
    model: nn.Module,
    x0_img: torch.Tensor,
    total_steps: int,
    max_lag: int | None = None,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        x = x0_img.to(device).view(x0_img.size(0), -1)  # (B, D)

        traj = [x]
        for _ in range(total_steps):
            x = model(x)
            traj.append(x)

        traj = torch.stack(traj, dim=0)  # (T+1, B, D)
        T = traj.size(0) - 1

        if max_lag is None:
            max_lag = T
        max_lag = min(max_lag, T)

        B = traj.size(1)

        # IMPORTANT: don't use torch.empty here unless you fill every entry
        curves = torch.zeros((B, max_lag + 1), device=traj.device)

        # Include tau = max_lag so the last column is valid
        for tau in range(max_lag + 1):
            a = traj[: (T + 1 - tau)]  # (num_tw, B, D)
            b = traj[tau:]             # (num_tw, B, D)

            corr_tw_b = zncc_anyshape(a, b)   # (num_tw, B)
            curves[:, tau] = corr_tw_b.mean(dim=0)

        return curves


# -------------------------
# Training + periodic evaluation every 50 epochs
# -------------------------
def main():
    # Where to save plots
    OUT_DIR = "zncc_plots"
   
    # Threshold metric settings
    THRESH = 0
    LAG_TO_CHECK = ITER_STEPS  # e.g., check the largest lag (same as “after 1000 iters”)

    model = MLPAutoencoder(latent_dim=10).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    eval_epochs: list[int] = []
    eval_pct_below: list[float] = []
    eval_tvd:list[float] = []
    loss_train: list[float] = []
    loss_test_epoch: list[float] = []      # test every epoch
    loss_test_eval: list[float] = []  
    eval_asymp:list[float]=[]
    eval_lyap:list[float]=[]     # test every eval_every
    eval_aging:list[float]=[]
    eval_specrad : list[float]=[]
    eval_jac_action :list[float]=[]
    eval_sigma :list [float]=[] 
    eval_jacfrob_over_n :list [float]=[]# optional
    for epoch in range(1, NUM_EPOCHS + 1):
        # ===== TRAIN =====
        model.train()
        running_loss = 0.0

        for imgs, _ in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            x = imgs.view(imgs.size(0), -1)

            optimizer.zero_grad(set_to_none=True)
            recon = model(x)
            batch_loss = criterion(recon, x)
            batch_loss.backward()
            optimizer.step()

            running_loss += batch_loss.item() * imgs.size(0)

        epoch_train_loss = running_loss / len(train_loader.dataset)
        loss_train.append(epoch_train_loss)
        print(f"[{epoch}] train loss = {epoch_train_loss:.6f}")

    # ===== TEST LOSS EVERY EPOCH =====
        model.eval()
        test_running = 0.0
        with torch.no_grad():
            for imgs, _ in test_loader:
                imgs = imgs.to(device, non_blocking=True)
                x = imgs.view(imgs.size(0), -1)
                recon = model(x)
                t_loss = criterion(recon, x)
                test_running += t_loss.item() * imgs.size(0)

        epoch_test_loss = test_running / len(test_loader.dataset)
        loss_test_epoch.append(epoch_test_loss)
        print(f"[{epoch}] test loss (epoch) = {epoch_test_loss:.6f}")

        # ---- Evaluation every EVAL_EVERY epochs (e.g., 50) ----
        # ---- Evaluation every EVAL_EVERY epochs (e.g., 50) ----
# ---- Evaluation every EVAL_EVERY epochs (e.g., 50) ----
        if epoch % EVAL_EVERY == 0 or epoch == 1:
           all_corr_at_lag = []
           example_curve = None
           all_tvd = []
           all_asymp_dist = []
           all_lyap_rates = []
           all_specrad = []
           all_jac_action = []
           all_sigma = []
           all_jacfrob_over_n = []

           print("here")
           for bi, (test_imgs, _) in enumerate(test_loader):
                def step_fn(z):
                    return model(z)
                x_star = iterate_to_asymptotic_state(step_fn, test_imgs.to(device), steps=ITER_STEPS)
                rho_list = []
                jacact_list = []
                sigma_list = []
                for i in range(x_star.size(0)):
                    rho_i, jacact_i = estimate_spectral_radius_jvp(
                        step_fn=step_fn,
                        x_star=x_star[i:i+1],   # keep batch dim for jvp
                        n_iters=15,
                        eps=eps,
                        )
                    rho_list.append(rho_i.cpu())
                    jacact_list.append(jacact_i.cpu())
                     # Optional: spectral norm upper bound
                    sigma_i = estimate_spectral_norm_jt_j(step_fn, x_star[i:i+1], n_iters=10, eps=eps)
                    sigma_list.append(sigma_i.cpu())

                    all_specrad.append(torch.stack(rho_list, dim=0))
                    all_jac_action.append(torch.stack(jacact_list, dim=0))
                    all_sigma.append(torch.stack(sigma_list, dim=0))
        # Get late-time state x_* for each image (use ITER_STEPS or a smaller tail if too expensive)
        # If ITER_STEPS is huge, you can use fewer steps here, e.g. steps=ASYMP_TAIL or 200.

                jacfrob_list = []
                for i in range(x_star.size(0)):
                    jacfrob_i = estimate_jac_frob_sq_over_n(
                        step_fn=step_fn,
                        x_star=x_star[i:i+1],   # keep batch dim
                        K=4,                    # probes; increase for lower variance
                        eps=eps,
                        rademacher=True,
                        )
                    jacfrob_list.append(jacfrob_i)

                all_jacfrob_over_n.append(torch.stack(jacfrob_list, dim=0))

                if bi >= EVAL_BATCHES:
                    break

        # === 1) Noiseless curves ===
                curves = per_image_curve_avg_over_tw(
                    model,
                    test_imgs,
                    total_steps=ITER_STEPS,
                    max_lag=ITER_STEPS,
                    device=device,
                    )  # (B, ITER_STEPS+1)

                if example_curve is None and curves.size(0) > 0:
                    example_curve = curves[0]
                all_corr_at_lag.append(curves[:, LAG_TO_CHECK].detach().cpu())

        # === 2) Noisy curves (1e-6 Gaussian) ===
                noise = torch.randn_like(test_imgs) * 1e-6
                test_imgs_noisy = (test_imgs + noise).clamp(0.0, 1.0)
                
                curves_noisy = per_image_curve_avg_over_tw(
                        model,
                        test_imgs_noisy,
                        total_steps=ITER_STEPS,
                        max_lag=ITER_STEPS,
                    device=device,
                    )

        # === 3) Total variation difference (per image) ===

                abs_diff = (curves - curves_noisy).abs() + eps  # ensure positivity

    # geometric mean along dim=1
                log_mean = abs_diff.log().mean(dim=1)
                tvd_geo = log_mean.exp()  # (B,)

                all_tvd.append(tvd_geo.detach().cpu())

        # === 4) Asymptotic distance (per image) ===
                tail_diff = (curves[:, -ASYMP_TAIL:] - curves_noisy[:, -ASYMP_TAIL:]).abs()  # (B, ASYMP_TAIL)

# geometric mean with small epsilon to avoid log(0)
            
                log_diff = torch.log(tail_diff + eps)  # (B, ASYMP_TAIL)

                geom_mean = torch.exp(log_diff.mean(dim=1)) - eps  # (B,)
                asymp_dist = geom_mean.clamp(min=0.0)  # ensure non-negative
                all_asymp_dist.append(asymp_dist.detach().cpu())

        # === 5) Lyapunov-style growth rate ===
        # measure relative growth of divergence over time
                diff_curve = (curves - curves_noisy).abs()  # (B, T+1)

        # Compute approximate per-step ratio and log rate
        # Avoid divide by zero with EPS
                ratios = (diff_curve[:, 1:] + eps) / (diff_curve[:, :-1] + eps)
                log_ratios = torch.log(ratios)  # natural log
                lyap_rate = log_ratios.mean(dim=1)  # average over time
                all_lyap_rates.append(lyap_rate.detach().cpu())

    # ---- Aggregate metrics ----
           corr_at_lag = torch.cat(all_corr_at_lag, dim=0)
           pct_below = (corr_at_lag < THRESH).float().mean().item() * 100.0

           tvd_all = torch.cat(all_tvd, dim=0)
           mean_tvd = tvd_all.mean().item()

           asymp_all = torch.cat(all_asymp_dist, dim=0)
           mean_asymp = asymp_all.mean().item()
        
           lyap_all = torch.cat(all_lyap_rates, dim=0)
           mean_lyap_rate = lyap_all.mean().item()

           eval_epochs.append(epoch)
           eval_pct_below.append(pct_below)
           eval_tvd.append(mean_tvd)
           eval_asymp.append(mean_asymp)
           eval_lyap.append(mean_lyap_rate)
           loss_test_eval.append(epoch_test_loss)
           aging = aging_coeff(curves, tau1=200, tau2=1000)  # pick a late window
           mean_aging = aging.mean().item()
           eval_aging.append(mean_aging)
           specrad_all = torch.cat(all_specrad, dim=0)
           mean_specrad = specrad_all.mean().item()

           jacact_all = torch.cat(all_jac_action, dim=0)
           mean_jac_action = jacact_all.mean().item()

           sigma_all = torch.cat(all_sigma, dim=0)
           mean_sigma = sigma_all.mean().item()

           eval_specrad.append(mean_specrad)
           eval_jac_action.append(mean_jac_action)
           eval_sigma.append(mean_sigma)
           jacfrob_all = torch.cat(all_jacfrob_over_n, dim=0)
           mean_jacfrob_over_n = jacfrob_all.mean().item()
           eval_jacfrob_over_n.append(mean_jacfrob_over_n)

           print(
                f"[{epoch}] test loss (eval) = {epoch_test_loss:.6f} | "
                f"% C_i(tau={LAG_TO_CHECK}) < {THRESH} = {pct_below:.2f}% | "
                f"mean TVD = {mean_tvd:.6e} | "
                f"mean asymptotic dist = {mean_asymp:.6e} | "
                f"mean lyap rate = {mean_lyap_rate:.6e} | "
                f"Aging coeff = {mean_aging:.4f} | "
                f"mean specrad(J@x*) = {mean_specrad:.6e} | "
                f"mean ||J v||/||v|| = {mean_jac_action:.6e} | "
                f" | mean ||J||_F^2/N = {mean_jacfrob_over_n:.6e}"
                )   

    # === Save Noiseless ZNCC curve ===
           if example_curve is not None:
              p1 = save_per_image_curve_png(example_curve, epoch=epoch, out_dir=OUT_DIR)
              print(f"Saved per-image ZNCC curve → {p1}")
            
        # === Save threshold curve ===
              p2 = save_pct_below_curve_png(
                    epochs=eval_epochs,
                    pct_below=eval_pct_below,
                    out_dir=OUT_DIR,
                    thresh=THRESH,
                    lag=LAG_TO_CHECK,
                    )
              print(f"Saved %<threshold curve → {p2}")

        # === Plot Total Variation Difference (log scale) ===
              plt.figure()
              plt.plot(eval_epochs, eval_tvd, marker="o")
              plt.yscale("log")
              plt.xlabel("Epoch")
              plt.ylabel("Mean TVD ")
              plt.title("Mean Total Variation Difference vs Epoch ")
              plt.grid(True, which="both", ls="--", lw=0.5)
                
              tvd_path = os.path.join(OUT_DIR, "tvd_curve_log.png")
              plt.savefig(tvd_path, dpi=200, bbox_inches="tight")
              plt.close()
              print(f"Saved TVD (log) curve → {tvd_path}")
            
            # === Plot Asymptotic Distance (log scale) ===
              plt.figure()
              plt.plot(eval_epochs, eval_asymp, marker="o")
              plt.yscale("log")
              plt.xlabel("Epoch")
              plt.ylabel(f"Asymptotic Distance\n (last {ASYMP_TAIL} steps)")
              plt.title("Asymptotic Distance vs Epoch ")
              plt.grid(True, which="both", ls="--", lw=0.5)
            
              asymp_path = os.path.join(OUT_DIR, "asymp_curve_log.png")
              plt.savefig(asymp_path, dpi=200, bbox_inches="tight")
              plt.close()
              print(f"Saved asymptotic distance curve → {asymp_path}")
            
            # === Plot Lyapunov Growth Rate ===
              plt.figure()
              plt.plot(eval_epochs, eval_lyap, marker="o")
              plt.xlabel("Epoch")
              plt.ylabel("Lyapunov Exponent")
              plt.title("Lyapunov exponent vs Epoch")
              plt.grid(True)
            
              lyap_path = os.path.join(OUT_DIR, "lyap_rate_curve.png")
              plt.savefig(lyap_path, dpi=200, bbox_inches="tight")
              plt.close()
              print(f"Saved lyapunov growth rate curve → {lyap_path}")
              
              plt.figure(figsize=(6,4))
              plt.plot(eval_epochs, eval_aging, marker='o')
              plt.xlabel("Epoch")
              plt.ylabel("Aging coefficient ")
              plt.title(f"Aging coefficient vs epoch ")
              plt.grid(True)
              
              save_path = f"figures/aging_vs_epoch.png"
              plt.savefig(save_path, dpi=200, bbox_inches='tight')
              plt.close()

              print(f"Saved aging coefficient curve -> {save_path}")
              # === Plot Spectral Radius (log scale) ===
              plt.figure()
              plt.plot(eval_epochs, eval_specrad, marker="o")
              plt.yscale("log")
              plt.xlabel("Epoch")
              plt.ylabel("spectral radius estimate")
              plt.title("spectral radius vs Epoch ")
              plt.grid(True, which="both", ls="--", lw=0.5)
              specrad_path = os.path.join(OUT_DIR, "specrad_curve_log.png")
              plt.savefig(specrad_path, dpi=200, bbox_inches="tight")
              plt.close()
              print(f"Saved spectral radius curve → {specrad_path}")

# === Plot Jacobian action norm (log scale) ===
              plt.figure()
              plt.plot(eval_epochs, eval_jac_action, marker="o")
              plt.yscale("log")
              plt.xlabel("Epoch")
              plt.ylabel("Mean ||Jv||/||v||")
              plt.title("Asymptotic Jacobian vs Epoch")
              plt.grid(True, which="both", ls="--", lw=0.5)
              jacact_path = os.path.join(OUT_DIR, "jac_action_curve_log.png")
              plt.savefig(jacact_path, dpi=200, bbox_inches="tight")
              plt.close()
              print(f"Saved Jacobian action curve → {jacact_path}")

              # === Plot spectral norm upper bound (log scale) ===
              plt.figure()
              plt.plot(eval_epochs, eval_jacfrob_over_n, marker="o")
              plt.yscale("log")
              plt.xlabel("Epoch")
              plt.ylabel("Mean ||J||_F^2 / N")
              plt.title("Asymptotic Jacobian per dimension vs Epoch )")
              plt.grid(True, which="both", ls="--", lw=0.5)

              jacfrob_path = os.path.join(OUT_DIR, "jac_frob_sq_over_n_curve_log.png")
              plt.savefig(jacfrob_path, dpi=200, bbox_inches="tight")
              plt.close()
              print(f"Saved ||J||_F^2/N curve → {jacfrob_path}")
           else:
              print("No example curve to save.")

    # === Always save loss curve ===
           plt.figure(figsize=(6,4))
           plt.plot(eval_epochs, loss_test_eval, marker='o')
           plt.xlabel("Epoch")
           plt.ylabel("Test Loss (MSE)")
           plt.title(f"Test Loss vs Epoch (")
           plt.yscale("log")  
           plt.grid(True)
            
           save_path = f"figures/test_loss_eval_every_{EVAL_EVERY}.png"
           plt.savefig(save_path, dpi=200, bbox_inches='tight')
           plt.close()
           print(f"Saved test loss figure to: {save_path}")
        


if __name__ == "__main__":
    # If you're on Windows and still get DataLoader worker spawn errors,
    # try setting DataLoader(num_workers=0) when constructing your loaders.
    main()