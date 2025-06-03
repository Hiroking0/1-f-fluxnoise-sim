import numpy as np
import superscreen as sc
from superscreen.geometry import circle   # handy helper that returns a Nx2 array
import matplotlib.pyplot as plt
from helper import isosceles_polygon, arc_slot_polygon, flux_noise_rms  # custom helper for isosceles triangles
# ─── 1.  Basic dimensions (µm) ───────────────────────────────────────────
sampling = 1000
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
margin = 20
outer_box = np.array([
    [-R_outer-margin, -R_outer-margin],
    [ R_outer+margin, -R_outer-margin],
    [ R_outer+margin,  R_outer+margin],
    [-R_outer-margin,  R_outer+margin],
])
film_poly = sc.Polygon("film", layer="Nb", points=outer_box)

# 3b.  Holes = annulus + two slits
outer_ring = sc.Polygon("ring_outer", layer="Nb", points=circle(R_outer, points=sampling))
inner_ring = sc.Polygon("ring_inner", layer="Nb", points=circle(R_inner, points=sampling))
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
inner_ring = inner_ring.union(trangle1).union(trangle2).union(rectangle).resample(sampling)

hole1 = outer_ring.difference(inner_ring).resample(sampling)  # outer ring minus inner ring

# replace your straight-edge slot with a curvy one
theta_outer = np.arcsin(20 / R_outer)         # 5.55°
slope       = (33/2) / 108
x_inner     = 20 - slope*(R_outer - R_inner)  # 4.72 µm
theta_inner = np.arcsin(x_inner / R_inner)    # 2.58°

slot_hole = arc_slot_polygon(
    name        = "hole",
    layer       = "Nb",
    r_inner     = R_inner,
    r_outer     = R_outer,
    theta_inner = theta_inner,
    theta_outer = theta_outer,
    n_inner     = 80,
    n_outer     = 100,
    orientation = 0,        # puts slot on +y side
)



# ─── 4.  Increase points where its needed ──────────────────────────
hole_sample_points = 400
R_theta = np.deg2rad(90 - 5.55)
R_ring = sc.Polygon("ring_inner", layer="Nb", points=circle(6, points=400,center=(R_outer * np.cos(R_theta),R_outer * np.sin(R_theta))))
L_theta = np.deg2rad(90 + 5.55)
L_ring = sc.Polygon("ring_outer", layer="Nb", points=circle(6, points=400,center=(R_outer * np.cos(L_theta),R_outer * np.sin(L_theta))))

L_ring_slot_hole = L_ring.intersection(slot_hole).resample(hole_sample_points)
R_ring_slot_hole = R_ring.intersection(slot_hole).resample(hole_sample_points)
slot_hole = slot_hole.union(L_ring_slot_hole).union(R_ring_slot_hole)

L_ring_hole1 = L_ring.intersection(hole1).resample(hole_sample_points)
R_ring_hole1 = R_ring.intersection(hole1).resample(hole_sample_points)
hole1 = hole1.union(L_ring_hole1).union(R_ring_hole1)

device = sc.Device(
    "dc_squid_mask",
    layers=[layer],
    films=[film_poly],
    holes=[ hole1, slot_hole],
)

fig, ax = device.draw(legend=True)

device.make_mesh(min_points=5000,
                 buffer = 0,
                 smooth=5)

fig,ax = device.plot_mesh(edge_color="k",
                          show_sites=False,
                          linewidth=0.8)
_ = device.plot_polygons(ax = ax, legend=True)


# noise = flux_noise_rms(device)
# print(f"Flux noise: {noise:.3f} µΦ₀/√Hz at 1 Hz")
plt.show()