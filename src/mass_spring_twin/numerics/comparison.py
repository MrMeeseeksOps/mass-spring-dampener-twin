from time import perf_counter

import numpy as np

from mass_spring_twin.numerics.integrators import integrate


def compare(p, times, methods):
    reference = integrate(p, times, "reference")
    trajectories, rows = {}, []
    for method in methods:
        start = perf_counter()
        y = integrate(p, times, method)
        elapsed = perf_counter() - start
        error = y - reference
        trajectories[method] = y
        with np.errstate(over="ignore", invalid="ignore"):
            rows.append(
                {
                    "method": method,
                    "max_displacement_error": float(np.max(abs(error[:, 0]))),
                    "rmse": float(np.sqrt(np.mean(error[:, 0] ** 2))),
                    "final_state_error": float(np.linalg.norm(error[-1])),
                    "seconds": elapsed,
                    "steps": len(times) - 1,
                    "finite": bool(np.isfinite(y).all()),
                }
            )
    return trajectories, rows
