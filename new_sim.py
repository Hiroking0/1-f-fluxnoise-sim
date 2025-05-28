import numpy as np
import superscreen as sc
from superscreen.geometry import circle   # handy helper that returns a Nx2 array
import matplotlib.pyplot as plt
from helper import isosceles_polygon, arc_slot_polygon  # custom helper for isosceles triangles
# ─── 1.  Basic dimensions (µm) ───────────────────────────────────────────
R_outer = 205       # outer radius of the red ring
R_inner = 105       # inner radius of the red ring
slit_angle = np.deg2rad(8)   # angular width of each triangular slit
slit_depth = R_outer - R_inner

# ─── 2.  Helper to make an isosceles‐triangle slit pointing in +ŷ ────────
def radial_slit(depth, half_angle, apex=(0, R_outer)):
    """Returns 3×2 array of vertices for a radial triangular slit."""
    ax, ay = apex
    # two base points sit on the inner rim, offset by ±half_angle
    theta_left  = np.pi/2 + half_angle
    theta_right = np.pi/2 - half_angle
    p_left  = np.array([R_inner*np.cos(theta_left),  R_inner*np.sin(theta_left)])
    p_right = np.array([R_inner*np.cos(theta_right), R_inner*np.sin(theta_right)])
    return np.vstack([p_left, p_right, [ax, ay]])

# top-center slit
slit1_pts = radial_slit(depth=slit_depth, half_angle=slit_angle/2)

# clone it 180° away for the second slit
slit2_pts = slit1_pts # mirror in x → rotate 180° about origin

# ─── 3.  Geometry objects ────────────────────────────────────────────────
layer = sc.Layer("Nb", london_lambda=0.085, thickness=0.100)       # adjust to taste

# 3a. ONE big film polygon (slightly larger than the ring so all holes lie within)
margin = 10
outer_box = np.array([
    [-R_outer-margin, -R_outer-margin],
    [ R_outer+margin, -R_outer-margin],
    [ R_outer+margin,  R_outer+margin],
    [-R_outer-margin,  R_outer+margin],
])
film_poly = sc.Polygon("groundplane", layer="Nb", points=outer_box)

# 3b.  Holes = annulus + two slits
outer_ring = sc.Polygon("ring_outer", layer="Nb", points=circle(R_outer, points=400))
inner_ring = sc.Polygon("ring_inner", layer="Nb", points=circle(R_inner, points=200))
slit1      = sc.Polygon("slit1",      layer="Nb", points=slit1_pts).translate(dx=-20, dy=0)
slit2      = sc.Polygon("slit2",      layer="Nb", points=slit2_pts).translate(dx=20, dy=0)

trangle1 = isosceles_polygon(
        name="slit1",
        layer="Nb",
        apex=(0, R_outer),
        height=110,
        base_width=33,
        angle=90,
    ).translate(dx=-20, dy=0)

trangle2 = isosceles_polygon(
        name="slit2",
        layer="Nb",
        apex=(0, R_outer),
        height=110,
        base_width=33,
        angle=90,
    ).translate(dx=20, dy=0)
rectangle = sc.Polygon(
        name="rectangle",
        layer="Nb",
        points=np.array([
            [-20, R_outer-110],
            [ 20, R_outer-110],
            [ 20, R_outer-0],
            [-20, R_outer-0],
        ]),
    )
inner_ring = inner_ring.union(trangle1).union(trangle2).union(rectangle)

hole1 = outer_ring.difference(inner_ring)  # outer ring minus inner ring

# replace your straight-edge slot with a curvy one
theta_outer = np.arcsin(20 / R_outer)         # 5.55°
slope       = (33/2) / 108
x_inner     = 20 - slope*(R_outer - R_inner)  # 4.72 µm
theta_inner = np.arcsin(x_inner / R_inner)    # 2.58°

slot_hole = arc_slot_polygon(
    name        = "slot",
    layer       = "Nb",
    r_inner     = R_inner,
    r_outer     = R_outer,
    theta_inner = theta_inner,
    theta_outer = theta_outer,
    n_inner     = 40,
    n_outer     = 80,
    orientation = 0,        # puts slot on +y side
)



device = sc.Device(
    "dc_squid_mask",
    layers=[layer],
    films=[film_poly],
    holes=[hole1, slot_hole],
)



device.make_mesh(min_points=10000)

circulating_currents = {"slot": "1 uA"}

solution = sc.solve(
    device,
    applied_field=sc.sources.ConstantField(0),
    circulating_currents=circulating_currents,
    current_units="uA",
    progress_bar=True,
)[-1]
fig, axes = solution.plot_currents(
    streamplot=True,
    # cross_section_coords=sections,
    figsize=(13,8),
)
_ = device.plot_polygons(ax=axes[0], lw=2)