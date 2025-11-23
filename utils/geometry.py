# utils/geometry.py
import numpy as np

def transform_points(points, world_to_image: np.ndarray):
    """
    Convert a set of points in world coordinates to pixel coordinates using a 3x3 world_to_image matrix.
    points: (N,2) or (N,3) or (2,) or (3,)
    world_to_image: (3,3)
    Returns: (N,2) numpy array of pixel coordinates
    """
    pts = np.asarray(points)
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)
    ones = np.ones((pts.shape[0], 1), dtype=pts.dtype)
    # use only x,y
    pts_h = np.hstack([pts[:, :2], ones])
    pix = pts_h @ world_to_image.T
    return pix[:, :2]
