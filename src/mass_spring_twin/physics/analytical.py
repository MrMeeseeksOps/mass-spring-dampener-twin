import numpy as np


def analytical(p, times):
    """Exact unforced solution, initialized at t=0, stable across damping regimes."""
    if p.force != 0:
        raise ValueError("Analytical validation requires zero external force")
    t = np.asarray(times, dtype=float)
    alpha = p.damping / (2 * p.mass)
    if p.regime == "underdamped":
        w = p.omega_d
        b = (p.v0 + alpha * p.x0) / w
        q = p.x0 * np.cos(w * t) + b * np.sin(w * t)
        x = np.exp(-alpha * t) * q
        v = np.exp(-alpha * t) * (-alpha * q + w * (-p.x0 * np.sin(w * t) + b * np.cos(w * t)))
    elif p.regime == "critical":
        b = p.v0 + alpha * p.x0
        x = (p.x0 + b * t) * np.exp(-alpha * t)
        v = (b - alpha * (p.x0 + b * t)) * np.exp(-alpha * t)
    else:
        root = np.sqrt(alpha**2 - p.omega_n**2)
        r1 = -(p.omega_n**2) / (alpha + root)
        r2 = -alpha - root
        a = (p.v0 - r2 * p.x0) / (r1 - r2)
        b = p.x0 - a
        x = a * np.exp(r1 * t) + b * np.exp(r2 * t)
        v = r1 * a * np.exp(r1 * t) + r2 * b * np.exp(r2 * t)
    return np.column_stack((x, v))
