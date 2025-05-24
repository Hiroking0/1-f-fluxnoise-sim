"""
Batch recreation of Fig. 3 from Koch–DiVincenzo–Clarke PRL 98 (2007)
using SuperScreen **without a live Matplotlib window** and **with multiprocessing**.

▪ Circular & square washers, two linewidths each
▪ √SΦ(1 Hz) vs outer size
▪ Global progress tracked via a single `tqdm` bar
"""

from __future__ import annotations

import os
from datetime import datetime
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Tuple, List, Dict

import numpy as np
import matplotlib.pyplot as plt
import superscreen as sc
from superscreen.geometry import circle, box
from tqdm.auto import tqdm

# ─── sweep specification ─────────────────────────────────────────
outer_sizes: List[int] = list(range(10, 100, 3))          # µm
cases = [  # label, shape, linewidth, marker, Npts
    ("Circ 0.3 µm", "circle", 0.30, "o", 40),
    ("Circ 1.0 µm", "circle", 1.00, "s", 40),
    ("Square 0.3 µm", "square", 0.30, "D", 100),
    ("Square 1.0 µm", "square", 1.00, "v", 100),
]
min_points = 30_000   # mesh target for SuperScreen

# ─── helper functions ────────────────────────────────────────────

def edge_length(size_um: float, h_min=0.15, h_max=1.0,
                s_min=5, s_max=100, scale="linear") -> float:
    """Size‑dependent maximum triangle edge length for meshing."""
    t = np.clip((size_um - s_min) / (s_max - s_min), 0.0, 1.0)
    return h_min * (h_max / h_min) ** t if scale == "log" else h_min + t * (h_max - h_min)


def build_device(shape: str, size: float, linewidth: float,
                 lam=0.1, t=0.025, Npts=40) -> sc.Device:
    """Return a circular or square washer SuperScreen device."""
    layer = sc.Layer("Nb", london_lambda=lam, thickness=t, z0=0)
    if shape == "circle":
        film = sc.Polygon("film", layer="Nb", points=circle(size)).resample(Npts)
        hole = sc.Polygon("hole", layer="Nb", points=circle(size - linewidth)).resample(Npts)
    elif shape == "square":
        film = sc.Polygon("film", layer="Nb", points=box(size, size, points=Npts))
        hole = sc.Polygon("hole", layer="Nb", points=box(size - 2 * linewidth,
                                                         size - 2 * linewidth,
                                                         points=Npts))
    else:
        raise ValueError("shape must be 'circle' or 'square'")

    dev = sc.Device(f"{shape}_{size:.1f}µm", layers=[layer],
                    films=[film], holes=[hole],
                    length_units="um", solve_dtype="float32")
    dev.make_mesh(min_points=min_points, smooth=6)
    return dev


def flux_noise_rms(device: sc.Device, z_spin=0.02, A_s=0.10, n=5e17,
                   pad=50.0, N=150) -> float:
    """Return √SΦ(1 Hz) in µΦ₀ / √Hz for *device*."""
    model = sc.factorize_model(device=device, current_units="A")
    model.set_circulating_currents({"hole": 1.0})
    sol = sc.solve(model=model)[-1]  # keyword avoids superscreen bug

    # Set up evaluation grid
    pts = np.asarray(device.films["film"].points)
    r_max = max(abs(pts[:, 0]).max(), abs(pts[:, 1]).max()) + pad
    xs = np.linspace(0.0, r_max, N)
    ys = np.linspace(0.0, r_max, N)
    grid = [(x, y) for x in xs for y in ys]

    bz = sol.field_at_position(grid, zs=z_spin, units="T").magnitude  # (N²,)
    mp = bz * A_s * 1e-12                              # Φ due to one spin (Wb)
    dA = (xs[1] - xs[0]) * (ys[1] - ys[0]) * 1e-12     # m²

    mu_B = 9.274e-24
    integral = np.sum((mp / (A_s * 1e-12)) ** 2) * dA
    alpha = 8 * n * mu_B**2 * integral / np.log(1e9 / 1e-4)
    phi0 = 2.067833848e-15
    return np.sqrt(alpha) / phi0 * 1e6                 # µΦ₀ / √Hz


def task(shape: str, size: float, linewidth: float, Npts: int) -> Tuple[str, float, float]:
    """Execute one (shape, size, linewidth) simulation and return label, x, y."""
    label = f"{shape}_{linewidth:.2f}"
    dev = build_device(shape, size, linewidth, Npts=Npts)
    y_val = flux_noise_rms(dev)
    return label, size, y_val


# ─── main ────────────────────────────────────────────────────────

def main() -> None:
    total_runs = len(cases) * len(outer_sizes)
    results: Dict[str, Tuple[List[float], List[float], str]] = {}

    # Build argument list for the executor
    arg_list = []
    for label, shape, linewidth, marker, Npts in cases:
        for size in outer_sizes:
            arg_list.append((label, shape, linewidth, marker, Npts, size))

    # Submit to process pool (uses all available CPUs by default)
    with ProcessPoolExecutor() as pool, tqdm(total=total_runs, desc="Simulating", unit="run") as pbar:
        future_to_meta = {
            pool.submit(task, shape, size, linewidth, Npts): (label, marker)
            for (label, shape, linewidth, marker, Npts, size) in arg_list
        }
        for fut in as_completed(future_to_meta):
            label, marker = future_to_meta[fut]
            sim_label, x_val, y_val = fut.result()
            if label not in results:
                results[label] = ([], [], marker)
            results[label][0].append(x_val)  # xs
            results[label][1].append(y_val)  # ys
            pbar.update(1)

    # Sort each series by x for nice plotting
    for label in results:
        xs, ys, marker = results[label]
        xs, ys = map(list, zip(*sorted(zip(xs, ys))))
        results[label] = (xs, ys, marker)

    # ── static plot ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4))
    for label, (xs, ys, marker) in results.items():
        ax.plot(xs, ys, marker=marker, label=label, lw=1)

    ax.set_xlabel("Outer dimension (µm)")
    ax.set_ylabel(r"$\sqrt{S_{\Phi}\,(1\,\mathrm{Hz})}\;\left(\mu\Phi_0/\sqrt{\mathrm{Hz}}\right)$")
    ax.set_yscale("log")
    ax.set_title("Simulated flux noise vs washer size (multiprocessing)")
    ax.grid(True, which="major", lw=0.5)
    ax.grid(True, which="minor", lw=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()

    os.makedirs("plots", exist_ok=True)
    fname = datetime.now().strftime("plots/%Y-%m-%d-%H-%M.svg")
    fig.savefig(fname)
    print(f"Saved figure to {fname}")


if __name__ == "__main__":
    main()
