import numpy as np
import superscreen as sc
from superscreen.geometry import rotate  # handy for orientation

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

import numpy as np
import superscreen as sc

def arc_slot_polygon(
        name: str,
        layer,
        *,
        r_inner: float,
        r_outer: float,
        theta_inner: float,
        theta_outer: float,
        n_inner: int = 40,
        n_outer: int = 80,
        n_side: int  = 0,          # ← NEW: extra points on each radial edge
        orientation: float = 0.0,
        **kwargs,
) -> sc.Polygon:
    """
    Return an sc.Polygon that looks like an annular sector (curved slot).

    Parameters
    ----------
    r_inner, r_outer          : radii (µm)
    theta_inner, theta_outer  : half-angles (rad) at r_inner / r_outer
    n_inner, n_outer          : # of linear segments on inner / outer arc
    n_side                    : # of *intermediate* vertices on each radial edge
    orientation               : extra rotation after building (rad)
    **kwargs                  : forwarded to sc.Polygon (e.g. color="r")

    Notes
    -----
    * vertices are ordered CCW so Superscreen treats this as a hole
    * set ``n_side`` ≥ 1 if you need finer control on the straight edges
    """
    # ----- outer arc   +θ_out  →  −θ_out   ------------------------------
    tout  = np.linspace(theta_outer, -theta_outer, n_outer)
    outer = np.column_stack([r_outer * np.sin(tout),
                             r_outer * np.cos(tout)])

    # ----- right radial edge  (outer → inner)  --------------------------
    p_out_R = outer[0]                                      # first outer point
    p_in_R  = np.array([r_inner*np.sin(theta_inner),
                        r_inner*np.cos(theta_inner)])
    if n_side > 0:
        t = np.linspace(0, 1, n_side + 2)[1:-1]             # skip endpoints
        right_edge = p_out_R + (p_in_R - p_out_R)[None, :] * t[:, None]
    else:
        right_edge = np.empty((0, 2))

    # ----- inner arc  −θ_in  →  +θ_in  (reverse order) ------------------
    tin   = np.linspace(-theta_inner, theta_inner, n_inner)
    inner = np.column_stack([r_inner * np.sin(tin),
                             r_inner * np.cos(tin)])

    # ----- left radial edge  (inner → outer)  ---------------------------
    p_out_L = outer[-1]
    p_in_L  = np.array([-r_inner*np.sin(theta_inner),
                         r_inner*np.cos(theta_inner)])
    if n_side > 0:
        left_edge  = p_in_L + (p_out_L - p_in_L)[None, :] * t[::-1, None]
    else:
        left_edge  = np.empty((0, 2))

    # ----- assemble vertices, CCW order ---------------------------------
    pts = np.vstack([outer,
                     right_edge,
                     inner,
                     left_edge])

    # ----- optional global rotation ------------------------------------
    if orientation:
        c, s = np.cos(orientation), np.sin(orientation)
        pts  = pts @ np.array([[c, -s], [s,  c]])

    return sc.Polygon(name=name, layer=layer, points=pts, **kwargs)


