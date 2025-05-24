"""
Live, non-blocking recreation of Fig. 3 from
Koch–DiVincenzo–Clarke PRL 98 (2007) using SuperScreen.

* Circular & square washers, two linewidths each
* √SΦ(1 Hz) vs outer size
* GUI stays responsive via multiprocessing
"""

import os, gc, time
from datetime import datetime
from multiprocessing import Process, Queue

import numpy as np
import matplotlib.pyplot as plt
import superscreen as sc
from superscreen.geometry import circle, box
from tqdm import tqdm

# ─── sweep parameters ────────────────────────────────────────────
outer_sizes = list(range(10, 100, 3))     # µm
cases = [                                 # label, linewidth, marker
    ("Circ 0.3 µm",   0.30, "o"),
    ("Circ 1.0 µm",   1.00, "s"),
    ("Square 0.3 µm", 0.30, "D"),
    ("Square 1.0 µm", 1.00, "v"),
]
min_points = 30_000                       # lighter meshes → faster

# ─── helpers ─────────────────────────────────────────────────────
def edge_length(size_um,  h_min=0.15, h_max=1.0,
                s_min=5, s_max=100, scale="linear"):
    t = np.clip((size_um - s_min)/(s_max - s_min), 0, 1)
    return h_min * (h_max/h_min)**t if scale == "log" else h_min + t*(h_max-h_min)

def circular_device(R_outer, linewidth, lam=0.1, t=0.025, Npts=40):
    layer = sc.Layer("Nb", london_lambda=lam, thickness=t, z0=0)
    film  = sc.Polygon("film", layer="Nb",
                       points=circle(R_outer)).resample(Npts)
    hole  = sc.Polygon("hole", layer="Nb",
                       points=circle(R_outer-linewidth)).resample(Npts)
    dev   = sc.Device("circ", layers=[layer], films=[film], holes=[hole],
                      length_units="um", solve_dtype="float32")
    dev.make_mesh(min_points=min_points, smooth=6)
    return dev

def square_device(side, linewidth, lam=0.1, t=0.025, Npts=100):
    layer = sc.Layer("Nb", london_lambda=lam, thickness=t, z0=0)
    film  = sc.Polygon("film", layer="Nb",
                       points=box(side, side, points=Npts))
    hole  = sc.Polygon("hole", layer="Nb",
                       points=box(side-2*linewidth,
                                  side-2*linewidth, points=Npts))
    dev   = sc.Device("square", layers=[layer], films=[film], holes=[hole],
                      length_units="um", solve_dtype="float32")
    dev.make_mesh(min_points=min_points, smooth=6)
    return dev

def flux_noise_rms(device, z_spin=0.02, A_s=0.10, n=5e17,
                   pad=50.0, N=150):
    model = sc.factorize_model(device=device, current_units="A")
    model.set_circulating_currents({"hole": 1.0})
    sol   = sc.solve(model=model)[-1]

    pts   = np.asarray(device.films["film"].points)
    Rmax  = max(abs(pts[:,0]).max(), abs(pts[:,1]).max()) + pad
    xs    = np.linspace(0, Rmax, N)
    ys    = np.linspace(0, Rmax, N)
    XY    = [(x, y) for x in xs for y in ys]

    Bz    = sol.field_at_position(XY, zs=z_spin, units="T").magnitude
    Mp    = Bz * A_s * 1e-12
    dA    = (xs[1]-xs[0]) * (ys[1]-ys[0]) * 1e-12
    integral = np.sum((Mp/(A_s*1e-12))**2) * dA

    mu_B  = 9.274e-24
    alpha = 8 * n * mu_B**2 * integral / np.log(1e9/1e-4)
    Phi0  = 2.067833848e-15
    return np.sqrt(alpha)/Phi0 * 1e6      # µΦ0 / √Hz

# ─── worker process ──────────────────────────────────────────────
def worker(label, linewidth, q):
    builder = circular_device if "Circ" in label else square_device
    for size in outer_sizes:
        y = flux_noise_rms(builder(size, linewidth))
        q.put((label, size, y))
    q.put((label, None, None))            # sentinel → this case is done

# ─── main (GUI) process ──────────────────────────────────────────
def main():
    # ── figure & empty lines ────────────────────────────────────
    plt.ion()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_xlabel("Outer dimension (µm)")
    ax.set_ylabel(r"$\sqrt{S_\Phi\,(1\;\mathrm{Hz})}$   (µ$\Phi_0$/√Hz)")
    ax.set_yscale("log")
    ax.set_title("Simulated flux noise vs washer size")
    ax.grid(True, which="major", lw=0.5)
    ax.grid(True, which="minor", lw=0.3)

    lines, xs_so_far, ys_so_far, queues, procs = {}, {}, {}, {}, {}
    for label, _, marker in cases:
        (line,) = ax.plot([], [], marker=marker, label=label, lw=1)
        lines[label] = line
        xs_so_far[label], ys_so_far[label] = [], []
        q = Queue();  queues[label] = q
        procs[label] = Process(target=worker, args=(label, _ , q))
        procs[label].start()

    ax.legend(frameon=False)
    finished = set()

    # ── event / update loop ─────────────────────────────────────
    while len(finished) < len(cases):
        for label, q in queues.items():
            while not q.empty():
                lbl, x, y = q.get()
                if x is None:                # sentinel: this label is done
                    finished.add(lbl)
                    break
                xs_so_far[lbl].append(x)
                ys_so_far[lbl].append(y)
                lines[lbl].set_data(xs_so_far[lbl], ys_so_far[lbl])
        ax.relim();  ax.autoscale_view()
        fig.canvas.draw_idle()
        plt.pause(0.01)                      # keep GUI responsive

    # ── clean up ────────────────────────────────────────────────
    for p in procs.values():
        p.join()
    plt.ioff()
    fig.tight_layout()
    os.makedirs("plots", exist_ok=True)
    fname = datetime.now().strftime("plots/%Y-%m-%d-%H-%M.svg")
    fig.savefig(fname)
    print(f"\nSaved final plot to {fname}")
    plt.show()

if __name__ == "__main__":
    main()
