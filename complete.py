"""
Recreates Fig. 3 of Koch–DiVincenzo–Clarke PRL 98 (2007)
and shows the plot **while** the data are still being generated.
"""

import numpy as np
import matplotlib.pyplot as plt
import superscreen as sc
from superscreen.geometry import circle, box
from tqdm import tqdm
from datetime import datetime

# ─── PARAMETERS ───────────────────────────────────────────────────
min_points   = 100_000
outer_sizes  = list(range(10, 100, 3))            # µm
cases = [
    ("Circ 0.3 µm",   0.30,  "o"),
    ("Circ 1.0 µm",   1.00,  "s"),
    ("Square 0.3 µm", 0.30,  "D"),
    ("Square 1.0 µm", 1.00,  "v"),
]

# ─── MESH HELPER ──────────────────────────────────────────────────
def edge_length(size_um, h_min=0.15, h_max=1.0,
                s_min=5, s_max=100, scale="linear"):
    t = np.clip((size_um - s_min)/(s_max - s_min), 0, 1)
    if scale == "log":
        return h_min * (h_max/h_min)**t
    return h_min + t*(h_max - h_min)

# ─── DEVICE BUILDERS ──────────────────────────────────────────────
def circular_device(R_outer, linewidth, lam=0.1, t=0.025,
                    Npts=40):
    layer = sc.Layer("Nb", london_lambda=lam, thickness=t, z0=0)
    film  = sc.Polygon("film", layer="Nb",
                       points=circle(R_outer)).resample(Npts)
    hole  = sc.Polygon("hole", layer="Nb",
                       points=circle(R_outer - linewidth)).resample(Npts)
    dev = sc.Device("circ", layers=[layer], films=[film], holes=[hole],
                    length_units="um", solve_dtype="float32")
    dev.make_mesh(min_points=min_points, smooth=10)
    return dev

def square_device(side, linewidth, lam=0.1, t=0.025, Npts=100):
    layer = sc.Layer("Nb", london_lambda=lam, thickness=t, z0=0)
    film  = sc.Polygon("film",  layer="Nb",
                       points=box(side, side, points=Npts))
    hole  = sc.Polygon("hole", layer="Nb",
                       points=box(side-2*linewidth,
                                  side-2*linewidth, points=Npts))
    dev = sc.Device("square", layers=[layer], films=[film], holes=[hole],
                    length_units="um", solve_dtype="float32")
    dev.make_mesh(min_points=min_points, smooth=10)
    return dev

# ─── FLUX-NOISE CALCULATOR ────────────────────────────────────────
def flux_noise_rms(device, z_spin=0.02, A_s=0.10, n=5e17,
                   pad=50.0, N=250):
    model = sc.factorize_model(device=device, current_units="A")
    model.set_circulating_currents({"hole": 1.0})
    sol = sc.solve(model)[-1]

    pts  = np.asarray(device.films["film"].points)
    Rmax = max(abs(pts[:, 0]).max(), abs(pts[:, 1]).max()) + pad
    xs   = np.linspace(0, Rmax, N)
    ys   = np.linspace(0, Rmax, N)
    XY   = [(x, y) for x in xs for y in ys]

    Bz   = sol.field_at_position(XY, zs=z_spin, units="T").magnitude
    Mp   = Bz * A_s * 1e-12                       # mutual-inductance map
    dA   = (xs[1] - xs[0]) * (ys[1] - ys[0]) * 1e-12
    integral = np.sum((Mp / (A_s*1e-12))**2) * dA

    mu_B = 9.274e-24
    alpha = 8 * n * mu_B**2 * integral / np.log(1e9 / 1e-4)
    Phi0  = 2.067833848e-15
    return np.sqrt(alpha)/Phi0 * 1e6              # µΦ0 / √Hz @ 1 Hz

# ─── LIVE PLOT SET-UP ─────────────────────────────────────────────
plt.ion()
fig, ax = plt.subplots(figsize=(6, 4))
ax.set_yscale("log")
ax.set_xlabel("Outer dimension (µm)")
ax.set_ylabel(r"$\sqrt{S_\Phi(1\;\mathrm{Hz})}$  (µ$\Phi_0$/√Hz)")
ax.set_title("Simulated flux noise vs washer size")
ax.grid(True, which="major", lw=0.5)
ax.grid(True, which="minor", lw=0.3)

# Create empty Line2D objects – one per case
lines, xs_so_far, ys_so_far = {}, {}, {}
for label, _, marker in cases:
    (line,) = ax.plot([], [], marker=marker, label=label)
    lines[label] = line
    xs_so_far[label], ys_so_far[label] = [], []
ax.legend(frameon=False)

# ─── MAIN SWEEP WITH LIVE UPDATES ─────────────────────────────────
for label, linewidth, _ in cases:
    builder = circular_device if "Circ" in label else square_device
    for size in tqdm(outer_sizes, desc=f"{label:>12}"):
        dev  = builder(size, linewidth)
        yval = flux_noise_rms(dev)

        # append and update the line
        xs_so_far[label].append(size)
        ys_so_far[label].append(yval)
        line = lines[label]
        line.set_data(xs_so_far[label], ys_so_far[label])

        # rescale axes to fit new data, then redraw
        ax.relim()
        ax.autoscale_view()
        fig.canvas.draw_idle()
        plt.pause(0.01)        # tiny pause lets the GUI process events

# ─── FINISH UP ────────────────────────────────────────────────────
plt.ioff()
fig.tight_layout()
today = datetime.now().strftime("%Y-%m-%d-%H-%M")
fig.savefig(f"plots/{today}.svg")
plt.show()
