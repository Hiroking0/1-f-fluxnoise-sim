"""
Recreates Fig. 3 of Koch–DiVincenzo–Clarke PRL 98 (2007) using SuperScreen.

• Circular and square washers, two linewidths each
• Flux-noise √SΦ(1 Hz) vs outer size
"""

import numpy as np
import matplotlib.pyplot as plt
import superscreen as sc
from superscreen.geometry import circle, box          # <-- NEW import
from packaging import version
from tqdm import tqdm
import time

def edge_length(size_um,
                h_min=0.15,          # µm, good for sub-10 µm devices
                h_max=1.0,           # µm, acceptable for >100 µm devices
                s_min=5, s_max=100,  # µm, sizes that bracket your sweep
                scale='linear'):
    """
    Return a size-dependent max_edge_length for mesh grading.
    """
    # normalised position in [0, 1]
    t = min(1.0, max(0.0, (size_um - s_min)/(s_max - s_min)))
    if scale == 'log':                     # logarithmic growth
        h = h_min * (h_max/h_min)**t
    else:                                  # linear growth
        h = h_min + t*(h_max - h_min)
    return h


# ─── device builders ──────────────────────────────────────────────
def circular_device(R_outer, linewidth, lam=0.1, t=0.025,
                    edge_small=0.15, edge_big=0.5, Npts=100):
    layer = sc.Layer("Nb", london_lambda=lam, thickness=t, z0=0)
    outer, inner = circle(R_outer), circle(R_outer - linewidth)
    # fewer boundary points → fewer triangles
    film = sc.Polygon("film", layer="Nb", points=outer).resample(Npts)
    hole = sc.Polygon("hole", layer="Nb", points=inner).resample(Npts)

    dev = sc.Device("circ", layers=[layer], films=[film], holes=[hole],
                    length_units="um", solve_dtype="float32")  # (3)
    
    start = time.time()
    el = edge_length(R_outer)          # or edge_length(side) for the square
    print("el: ",el)
    dev.make_mesh(max_edge_length=el, smooth=100)
    print(f"Meshing took {time.time() - start:.3f} seconds")
    mesh = dev.meshes["film"]
    print(mesh.n_sites)        # vertices
    print(mesh.n_elements)     # triangles
    return dev


def square_device(side, linewidth, *,                       # side = outer edge length (µm)
                  lam=0.1, t=0.025,
                  edge_small=0.15, edge_big=0.5,            # tweak as needed
                  Npts=100):
    """
    Build a square washer SQUID device with adaptive mesh density.

    Parameters
    ----------
    side : float
        Outer edge length of the square (µm).
    linewidth : float
        Trace width = side - inner_side  (µm).
    lam, t : float
        London penetration depth (µm) and film thickness (µm).
    edge_small, edge_big : float
        `max_edge_length` used for small vs. large devices.
    Npts : int
        Number of vertices on each polygon boundary before meshing.
    """
    layer = sc.Layer("Nb", london_lambda=lam, thickness=t, z0=0)

    # --- boundary polygons ----------
    outer = sc.box(side,     side,     points=Npts)
    inner = sc.box(side - 2*linewidth,
                   side - 2*linewidth, points=Npts)

    film = sc.Polygon("film", layer="Nb", points=outer)
    hole = sc.Polygon("hole", layer="Nb", points=inner)

    dev = sc.Device("square",
                    layers=[layer],
                    films=[film],
                    holes=[hole],
                    length_units="um",
                    solve_dtype="float32")      # cuts memory in half

    # --- choose mesh spacing based on size ----------
    el = edge_length(side)          # or edge_length(side) for the square
    dev.make_mesh(max_edge_length=el, smooth=10)
    
    return dev



# ─── flux-noise calculator (isotropic-spin assumption) ────────────
def flux_noise_rms(device, z_spin=0.02, A_s=0.10, n=5e17,
                   pad=50.0, N=250):
    # 1 A circulating current for reciprocity
    if version.parse(sc.__version__) >= version.parse("0.7"):
        model = sc.factorize_model(device=device, current_units="A")
        model.set_circulating_currents({"hole": 1.0})
        start = time.time()
        sol = sc.solve(model=model)[-1]
        print(f"solving took {time.time() - start:.3f} seconds")
    else:
        model = sc.factorize_model(device,
                                   current_units="A",
                                   circulating_currents={"hole": 1.0})
        start = time.time()
        sol = sc.solve(model)[-1]
        print(f"solving took {time.time() - start:.3f} seconds")

    # grid on +x,+y quadrant
    pts  = np.asarray(device.films["film"].points)        # (N,2) array
    Rmax = max(abs(pts[:,0]).max(), abs(pts[:,1]).max()) + pad

    xs = np.linspace(0, Rmax, N)
    ys = np.linspace(0, Rmax, N)
    XY = [(x, y) for x in xs for y in ys]

    # Bz at grid
    if hasattr(sol, "field_at_position"):
        Bz = sol.field_at_position(XY, zs=z_spin, units="T").magnitude
    else:
        from superscreen.postprocessing import field_at_positions
        Bz = field_at_positions(solution=sol, positions=XY,
                                zs=z_spin, units="T")
    Bz = Bz.reshape(N, N)

    # Mutual-inductance map
    Mp = Bz * A_s * 1e-12              # H
    dA = (xs[1] - xs[0]) * (ys[1] - ys[0]) * 1e-12
    integral = np.sum((Mp / (A_s * 1e-12)) ** 2) * dA  # ∫(M/A)²

    mu_B = 9.274e-24
    ms_flux = 8 * n * mu_B ** 2 * integral             # mean-square flux
    alpha = ms_flux / np.log(1e9 / 1e-4)               # 1/f coefficient

    Phi0 = 2.067833848e-15
    return np.sqrt(alpha) / Phi0 * 1e6                 # µΦ0/√Hz @ 1 Hz

# ─── sweep over geometries ───────────────────────────────────────
outer_sizes = [i for i in range(10,100,2)]        # µm
cases = [
    ("Circ 0.3 µm",   circular_device, 0.30)
    # ("Circ 1.0 µm",   circular_device, 1.00),
    # ("Square 0.3 µm", square_device, 0.30),
    # ("Square 1.0 µm", square_device, 1.00),
]

results = {}
for label, builder, linewidth in cases:
    vals = []
    for size in tqdm(outer_sizes):
        dev = builder(size, linewidth)
        vals.append(flux_noise_rms(dev))
    results[label] = vals

# ─── plot (qualitative Fig. 3) ───────────────────────────────────
plt.figure(figsize=(6, 4))
for label, vals in results.items():
    plt.plot(outer_sizes, vals, marker='o', label=label)
plt.xlabel("Outer dimension (µm)")                      # R or half-side
plt.ylabel(r"$\sqrt{S_\Phi(1\ \mathrm{Hz})}$  (µ$\Phi_0$/√Hz)")
plt.yscale("log")
plt.legend(frameon=False)
plt.title("Simulated flux noise vs washer size")
plt.tight_layout()
plt.show()
