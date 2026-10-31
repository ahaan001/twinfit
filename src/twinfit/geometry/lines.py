"""3D line utilities. A line is (point p, direction a); a need not be unit length."""

from __future__ import annotations

import numpy as np

_PARALLEL_TOL = 1e-9


def line_distance(p1: np.ndarray, a1: np.ndarray, p2: np.ndarray, a2: np.ndarray) -> float:
    """Shortest distance between two lines, including the parallel case."""
    p1, a1, p2, a2 = (np.asarray(v, dtype=float).reshape(3) for v in (p1, a1, p2, a2))
    a1 = a1 / np.linalg.norm(a1)
    a2 = a2 / np.linalg.norm(a2)
    n = np.cross(a1, a2)
    dp = p2 - p1
    if np.linalg.norm(n) < _PARALLEL_TOL:
        return float(np.linalg.norm(np.cross(dp, a1)))
    return float(abs(np.dot(dp, n)) / np.linalg.norm(n))


def closest_point_to_origin(p: np.ndarray, a: np.ndarray) -> np.ndarray:
    """The point on the line through p with direction a that is closest to the origin."""
    p = np.asarray(p, dtype=float).reshape(3)
    a = np.asarray(a, dtype=float).reshape(3)
    return p - np.dot(p, a) / np.dot(a, a) * a
