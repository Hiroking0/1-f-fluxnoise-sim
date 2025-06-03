import numpy as np
import superscreen as sc
from superscreen.geometry import rotate  # handy for orientation
from matplotlib.colors import LogNorm, PowerNorm
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

def isosceles_polygon(
        name: str,
        layer,
        apex: tuple[float, float],
        height: float,
        base_width: float,
        angle: float = 0.0,
) -> sc.Polygon:
    """
    Construct an isosceles triangle as an sc.Polygon.

    Parameters
    ----------
    name : str
        Polygon name.
    layer : sc.Layer | str
        Superscreen layer object or its name.
    apex : (x, y)
        Coordinates of the apex (tip) of the triangle.
    height : float
        Distance from apex to the base’s midpoint.
    base_width : float
        Full width of the base (i.e. distance between the two base corners).
    angle : float, optional
        Rotation of the triangle in radians, measured from +x toward +y.
        angle = 0 → apex points along +x, π/2 → apex points along +y.

    Returns
    -------
    sc.Polygon
        Polygon with three vertices ordered CCW.
    """
    apex = np.asarray(apex, dtype=float)
    # Local triangle in its own frame (apex at origin, pointing +x):
    pts = np.array([
        [0.0, 0.0],                                  # apex
        [-height, -base_width / 2],                  # left base corner
        [-height,  base_width / 2],                  # right base corner
    ])
    # Rotate into global frame, then translate to apex
    pts = rotate(pts, angle) + apex
    return sc.Polygon(name=name, layer=layer, points=pts)
import numpy as np
import superscreen as sc

def arc_slot_polygon(
        name: str,
        layer,
        r_inner: float,
        r_outer: float,
        theta_inner: float,
        theta_outer: float,
        n_inner: int = 40,
        n_outer: int = 80,
        orientation: float = 0.0,
        **kwargs,
):
    """
    Build an sc.Polygon representing a curved slot bounded by two
    circular arcs (r_inner, r_outer) and two short straight segments.

    The slot is symmetric about the local +y axis before `orientation`
    is applied.

    Parameters
    ----------
    r_inner, r_outer : float
        Inner / outer radii (µm).
    theta_inner, theta_outer : float
        Half-angles (radians) at r_inner and r_outer, respectively.
    n_inner, n_outer : int
        Number of straight segments used to approximate the inner / outer arc.
    orientation : float
        Extra CCW rotation (radians).  0 → slot centred on +y.

    Returns
    -------
    sc.Polygon
    """
    # sample outer arc  ( +θₒ  →  −θₒ )
    tout  = np.linspace( theta_outer, -theta_outer, n_outer)
    outer = np.column_stack([r_outer*np.sin(tout),
                             r_outer*np.cos(tout)])

    # sample inner arc  ( −θᵢ  →  +θᵢ )
    tin   = np.linspace(-theta_inner,  theta_inner, n_inner)
    inner = np.column_stack([r_inner*np.sin(tin),
                             r_inner*np.cos(tin)])

    pts = np.vstack([outer, inner])           # CCW order

    if orientation != 0.0:
        c, s = np.cos(orientation), np.sin(orientation)
        pts  = pts @ np.array([[c, -s],
                               [s,  c]])

    return sc.Polygon(name=name, layer=layer, points=pts, **kwargs)


def flux_noise_rms(device: sc.Device, n=N_SPIN, A_s=A_SPIN,
                   pad=PAD_L, grid_N=300) -> float:
    """Return √S_Φ(1 Hz) in µΦ₀/√Hz for isotropic spins."""
    # Reciprocity: 1 A circulating current around the hole
    model = sc.factorize_model(device=device, current_units="A")
    model.set_circulating_currents({"hole": 1.0})
    solution = sc.solve(model=model)[-1]

    # ------------------------------------------------------------------
    # 1) First, plot the currents exactly as before, but tell plot_currents
    #    not to draw any streamlines.  We’ll handle streamlines manually:
    fig, axes = solution.plot_currents(
        streamplot=True,   # turn off built‐in streamlines
        figsize=(8, 8),
        cmap="plasma",      # a perceptually uniform colormap
    )

    # 2) For each axis, grab the QuadMesh and switch to a LogNorm (or PowerNorm).
    #    You need to pick a sensible vmin (e.g. 1e-3 or 1e-4 depending on your J range)
    for ax in axes:
        # There should be exactly one QuadMesh in ax.collections
        quad = ax.collections[0]
        J_data = quad.get_array()          # this is a 1D array of |J| values
        J_max = np.nanmax(J_data)
        J_min = 1e-3   # ← choose a floor below which you still want some contrast

        # Apply a LogNorm so that small currents are visible:
        quad.set_norm(LogNorm(vmin=J_min, vmax=J_max))
        quad.set_clim(J_min, J_max)

        # 4) Finally, redraw the device polygons on top so they stay sharp:
        device.plot_polygons(ax=ax, color="w", ls="--", lw=1)


    
    # Quadrant grid (x,y ≥ 0) out to outer‑edge + PAD_L
    outer = max(abs(device.films["film"].points).flatten())
    Rmax  = outer

    xs = np.linspace(0.0, Rmax, grid_N)
    ys = np.linspace(0.0, Rmax, grid_N)
    XY = [(x, y,Z_SPIN) for x in xs for y in ys]

    # B_z field of the 1‑A SQUID at spin plane
    Bz = solution.field_at_position(XY, units="T")
    Bz = Bz.reshape((grid_N, grid_N))

    # Mutual‑inductance density M(x,y) = Bz * A_s
    rng = np.random.default_rng(42)
    phi   = rng.uniform(0, 2*np.pi, (grid_N, grid_N))
    theta = rng.uniform(0, np.pi/2, (grid_N, grid_N))
    r   = 1
    n = np.array([
        np.sin(theta) * np.cos(phi),
         np.sin(theta) * np.sin(phi), 
         np.cos(theta)])  # random unit vectors in spherical coordinates
    
    print(f"n: {n.shape}, Bz: {Bz.shape}")
    
                       
    Mp = np.dot(Bz,n) * A_s * 1e-12           # convert µm² → m² (H = Wb/A)
    dA = (xs[1] - xs[0]) * (ys[1] - ys[0]) * 1e-12  # m² per pixel

    integral = np.sum((Mp / (A_s * 1e-12))**2) * dA   # ∫(M/A)² over quadrant

    ms_flux = 8.0 * n * MU_B**2 * integral           # factor 8: 4 quadrants × 2 spins
    alpha   = ms_flux / LN_BW                        # 1/f prefactor

    return np.sqrt(alpha) / PHI_0 * 1e6              # µΦ₀ / √Hz
