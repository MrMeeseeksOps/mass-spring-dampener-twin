import numpy as np


def derivative(t, y, p):
    """State derivative in SI units; force is constant in this milestone."""
    x, v = y
    return np.array([v, (p.force - p.damping * v - p.stiffness * x) / p.mass])
