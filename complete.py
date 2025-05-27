"""
Reproduce the Fig. 3 flux‑noise simulations from Koch–DiVincenzo–Clarke,
Phys. Rev. Lett. 98 (2007) 267003.

Parameters are taken directly from the paper:
    • Film thickness              t   = 0.10 µm   (100 nm)  fileciteturn2file11
    • SQUID plane height          z0  = +1.0 µm   (loop at z = 1 µm)  fileciteturn2file11
    • Test‑spin plane             z_s = 0 µm      (substrate surface)
    • London penetration depth    λ   = 0.001 µm  (≈ ideal screening)
    • Spin areal density          n   = 5 × 10¹⁷ m⁻²  fileciteturn2file2
    • Spin loop area              A_s = 0.10 µm²   fileciteturn2file16
    • Integration extends         L   = 100 µm beyond the washer edge  fileciteturn2file10
    • Frequency band              f₁  = 10⁻⁴ Hz, f₂ = 10⁹ Hz  (ln‑factor ≈ 30)  fileciteturn2file10

Two geometry sweeps are implemented, matching Fig. 3:
    1. **Fixed aspect ratio** 2d / W = 4  ⇒  D = 3W, d = 2W.
       W is varied logarithmically from 2 µm to 200 µm, giving
       mean loop sizes (D+d) from 10 µm to 1000 µm.

    2. **Fixed linewidth** W = 20 µm, d varied logarithmically from
       5 µm to 500 µm ⇒ mean loop sizes from 45 µm to 1020 µm.

The script uses **Superscreen ≥ 0.6**.  Typical runtime for the full
sweep (~20 devices) is several minutes on a modern laptop.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import superscreen as sc
from superscreen.geometry import box
from tqdm import tqdm
from datetime import datetime

# ───────────────────────────────────────────────────────────────
# Physical constants
MU_B  = 9.274e-24         # J/T
PHI_0 = 2.067833848e-15   # Wb
LN_BW = np.log(1e9 / 1e-4)  # ln(f₂ / f₁)

# Simulation‑wide parameters (from paper)
T_FILM   = 0.10      # µm
LAMBDA   = 0.001     # µm  (≈ ideal)
Z_LOOP   = 1.0       # µm  (washer plane)
Z_SPIN   = 0.0       # µm  (substrate)
A_SPIN   = 0.10      # µm²
N_SPIN   = 5e17      # m⁻²
PAD_L    = 100.0     # µm  (integration cutoff beyond edge)

# Meshing parameters
MIN_POINTS = 10_000   # rough target for mesh resolution
SMOOTH_ITR = 10       # Lloyd smoothing passes

# ───────────────────────────────────────────────────────────────
# Geometry builder

def washer_device(D: float, d: float) -> sc.Device:
    """Return a *square* washer SQUID with outer half‑side *D* (µm) and
    inner half‑side *d* (µm). All lengths in µm."""
    layer = sc.Layer("Nb", london_lambda=LAMBDA, thickness=T_FILM, z0=Z_LOOP)

    outer = box(2*D, 2*D, points=200)   # full side = 2D (µm)
    inner = box(2*d, 2*d, points=200)   # full side = 2d (µm)

    film = sc.Polygon("film", layer="Nb", points=outer)
    hole = sc.Polygon("hole", layer="Nb", points=inner)

    dev = sc.Device("washer", layers=[layer], films=[film], holes=[hole],
                    length_units="um", solve_dtype="float32")
    # fig, ax = dev.draw(figsize=(10,5))
    # _ = dev.plot_polygons(ax=ax, legend=True)

    # Mesh density scales (very roughly) with size to keep point count sane
    max_el = 0.2 + 0.8 * (D / 1000.0)   # µm
    dev.make_mesh( min_points=MIN_POINTS,
                  smooth=SMOOTH_ITR)
    
    return dev

# ───────────────────────────────────────────────────────────────
# Flux‑noise calculator

def flux_noise_rms(device: sc.Device, n=N_SPIN, A_s=A_SPIN,
                   pad=PAD_L, grid_N=300) -> float:
    """Return √S_Φ(1 Hz) in µΦ₀/√Hz for isotropic spins."""
    # Reciprocity: 1 A circulating current around the hole
    model = sc.factorize_model(device=device, current_units="A")
    model.set_circulating_currents({"hole": 1.0})
    solution = sc.solve(model=model)[-1]

    # Quadrant grid (x,y ≥ 0) out to outer‑edge + PAD_L
    outer = max(abs(device.films["film"].points).flatten())
    Rmax  = outer + pad

    xs = np.linspace(0.0, Rmax, grid_N)
    ys = np.linspace(0.0, Rmax, grid_N)
    XY = [(x, y) for x in xs for y in ys]

    # B_z field of the 1‑A SQUID at spin plane
    Bz = solution.field_at_position(XY, zs=Z_SPIN, units="T").magnitude
    Bz = Bz.reshape((grid_N, grid_N))

    # Mutual‑inductance density M(x,y) = Bz * A_s
    rng   = np.random.default_rng(seed=42)          # reproducible Monte-Carlo
    n_z   = rng.uniform(-1.0, 1.0, size=Bz.shape)   # cosθ ~ U[−1,1]
                       
    Mp = Bz * n_z * A_s * 1e-12           # convert µm² → m² (H = Wb/A)
    dA = (xs[1] - xs[0]) * (ys[1] - ys[0]) * 1e-12  # m² per pixel

    integral = np.sum((Mp / (A_s * 1e-12))**2) * dA   # ∫(M/A)² over quadrant

    ms_flux = 8.0 * n * MU_B**2 * integral           # factor 8: 4 quadrants × 2 spins
    alpha   = ms_flux / LN_BW                        # 1/f prefactor

    return np.sqrt(alpha) / PHI_0 * 1e6              # µΦ₀ / √Hz

# ───────────────────────────────────────────────────────────────
# Sweep definitions matching Fig. 3

def sweep_fixed_aspect_ratio():
    """Sweep W so that 2d/W = 4 (=> D=3W, d=2W)."""
    W_vals = np.geomspace(2.0, 200.0, 10)   # µm
    mean_sizes = []
    noises      = []

    for W in tqdm(W_vals, desc="Aspect‑ratio 2d/W=4"):
        d = 2.0 * W
        D = 3.0 * W
        dev = washer_device(D, d)
        mean_sizes.append(D + d)
        noises.append(flux_noise_rms(dev))
    return np.array(mean_sizes), np.array(noises)


def sweep_fixed_width(W_fixed: float = 20.0):
    """Sweep inner half‑side *d* with fixed linewidth W = 20 µm."""
    d_vals = np.geomspace(5.0, 500.0, 10)   # µm
    mean_sizes = []
    noises      = []

    for d in tqdm(d_vals, desc="Fixed W = 20 µm"):
        D = d + W_fixed
        dev = washer_device(D, d)
        mean_sizes.append(D + d)
        noises.append(flux_noise_rms(dev))
    return np.array(mean_sizes), np.array(noises)


# ───────────────────────────────────────────────────────────────
# Helper: colour plot of Bz on the spin plane (z = Z_SPIN)

def plot_Bz_color(device: sc.Device,
                  pad: float = PAD_L,
                  grid_N: int = 300,
                  cmap: str = "plasma"):
    """
    Plot Bz(x, y) [Tesla per 1 A circulating current] on a quadrant grid.

    Parameters
    ----------
    device   : superscreen.Device
        A meshed SQUID device (same object you pass to flux_noise_rms).
    pad      : float
        Extra margin beyond the outer washer edge, in µm
        (should match *pad* in `flux_noise_rms`).
    grid_N   : int
        Number of grid points along x and y (same resolution as the noise calc).
    cmap     : str
        Matplotlib colormap.
    """
    # ----- 1 A circulating current around the hole  ---------------------------
    model = sc.factorize_model(device=device, current_units="A")
    model.set_circulating_currents({"hole": 1.0})
    solution = sc.solve(model=model)[-1]

    # grid covering the +x,+y quadrant  ----------------------------------------
    outer = max(abs(device.films["film"].points).flatten())  # µm
    Rmax  = outer*1.3
    xs    = np.linspace(-Rmax, Rmax, grid_N)
    ys    = np.linspace(-Rmax, Rmax, grid_N)
    XY    = [(x, y) for x in xs for y in ys]

    Bz = solution.field_at_position(XY, zs=Z_SPIN, units="T").magnitude.reshape((grid_N, grid_N))          

    # ----- plotting  ----------------------------------------------------------
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.pcolormesh(X, Y, Bz,
                       shading="auto", cmap=cmap)
    
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("$B_z$  [T]", rotation=270, labelpad=12)

    ax.set_xlabel("$x$ (µm)")
    ax.set_ylabel("$y$ (µm)")
    ax.set_aspect("equal")
    _ = device.plot_polygons(ax=ax,color="black")
    plt.tight_layout()
    plt.show()

# ───────────────────────────────────────────────────────────────
# Main & plotting

if __name__ == "__main__":
    sizes1, noises1 = sweep_fixed_aspect_ratio()
    sizes2, noises2 = sweep_fixed_width()

    plt.figure(figsize=(6.5, 4.2))
    plt.loglog(sizes1, noises1, "o-", label=r"Aspect ratio $2d/W = 4$")
    plt.loglog(sizes2, noises2, "s-", label=r"Fixed $W = 20\,\mu$m")

    plt.xlabel(r"Mean loop size $(D+d)$ (µm)")
    plt.ylabel(r"$\sqrt{S_\Phi(1\,\mathrm{Hz})}$  (µ$\Phi_0$/√Hz)")
    plt.grid(True, which="both", ls=":", lw=0.5)
    plt.legend(frameon=False)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.tight_layout()
    plt.savefig(f"plots/flux_noise_fig3_{ts}.png", dpi=300)
    plt.show()

