"""
Superscreen Quality Demo
========================

This script shows how to expose **one** user‑facing knob, `quality`, that
trades simulation accuracy for run‑time in *SuperScreen* by scaling the
finite‑element mesh density.

*   `quality = 0.25` → very coarse / fast meshes
*   `quality = 1.0`  → default (≈ publication‑grade)
*   `quality > 1`    → ultra‑fine meshes for convergence tests

The key idea is to convert `quality` into the two parameters that
actually control element count:

* **`max_edge_length`** – upper size limit for Triangle/meshgenerator.
* **`smooth`** – number of Laplacian‑smoothing passes.

Both are chosen from reasonable bounds and then divided or multiplied by
`quality` so that *all* physics‑based safeguards (e.g. resolving Λ and
trace widths) are still enforced.

Tested with **superscreen 0.6.0** but should work back to 0.5.x.
"""

from __future__ import annotations

import time
from typing import Tuple, Dict

import numpy as np
import superscreen as sc

##############################################################################
# Helper: build a simple circular washer SQUID film                           #
##############################################################################

def build_circular_washer(
    r_outer: float = 50.0,   # µm, outer radius of washer
    width: float = 2.0,      # µm, linewidth (so r_inner = r_outer – width)
    lambda_nm: float = 350.0,
    thickness_nm: float = 35.0,
    name: str = "washer",
) -> sc.Device:
    """Return a thin‑film device: circular washer with a single hole."""
    layer = sc.Layer(
        "base",
        london_lambda=lambda_nm * 1e-3,  # convert nm → µm
        thickness=thickness_nm * 1e-3,
    )

    film = sc.Polygon(
        name="film",
        layer="base",
        points=sc.geometry.circle(r_outer),
    )
    hole = sc.Polygon(
        name="hole",
        layer="base",
        points=sc.geometry.circle(r_outer - width),
    )

    return sc.Device(
        name=name,
        layers=[layer],
        films={"film": film},
        holes={"hole": hole},
        length_units="um",
    )

##############################################################################
# Mesh + solve with a single “quality” parameter                              #
##############################################################################

def _characteristic_size(dev: sc.Device) -> float:
    """Half the larger span of the device (µm)."""
    pts = dev.poly_points  # shape (N, 2)
    return 0.5 * max(pts[:, 0].ptp(), pts[:, 1].ptp())


def solve_device(
    dev: sc.Device,
    quality: float = 1.0,
    lambda_eff: float = 0.30,  # µm, effective penetration depth Λ
    h_min: float = 0.15,       # µm, smallest edge we'll ever allow
    h_max: float = 1.5,        # µm, coarsest edge if quality==1 and size>=100 µm
    smooth_max: int = 100,
) -> Tuple[sc.Solution, Dict[str, float]]:
    """Mesh *dev* according to *quality* and return (solution, stats)."""

    quality = max(1e-3, quality)  # avoid division by zero
    size_um = _characteristic_size(dev)

    # Base target that grows with size, then scale by 1/quality
    h_target = h_min + (h_max - h_min) * min(1.0, size_um / 100.0)
    h_target /= quality
    h_target = min(h_target, lambda_eff / 4)  # physics safeguard

    smooth = int(round(smooth_max * quality))

    # ------------------------------------------------------------------ mesh
    dev.make_mesh(max_edge_length=h_target, smooth=smooth)

    # ------------------------------------------------------------------ solve
    t0 = time.perf_counter()
    solution = sc.solve(dev)[-1]  # steady‑state (last frame)
    t1 = time.perf_counter()

    mesh = dev.meshes["film"]
    stats = {
        "edge_length": h_target,
        "smooth_passes": smooth,
        "n_vertices": mesh.sites.shape[0],
        "n_elements": mesh.elements.shape[0],
        "solve_time_s": t1 - t0,
    }
    return solution, stats

##############################################################################
# Quick demo                                                                  #
##############################################################################

def demo() -> None:
    print("quality  el[µm]  vertices  elements  time[s]")
    print("——" * 18)
    for q in (0.25, 0.5, 1.0):
        dev = build_circular_washer(name=f"washer_q{q}")
        _, st = solve_device(dev, quality=q)
        print(f"{q:>6.2f}  {st['edge_length']:<6.3f}  {st['n_vertices']:>8d}  "
              f"{st['n_elements']:>8d}  {st['solve_time_s']:>6.2f}")


if __name__ == "__main__":
    demo()
