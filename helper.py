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
