import numpy as np
from scipy.integrate import solve_ivp

from mass_spring_twin.physics.dynamics import derivative


def time_grid(duration, dt):
    if not np.isfinite([duration, dt]).all() or duration <= 0 or dt <= 0:
        raise ValueError("duration and timestep must be finite and positive")
    n = int(np.ceil(duration / dt))
    if n > 200000:
        raise ValueError("At most 200000 steps; increase timestep or shorten duration")
    return np.concatenate((np.arange(n) * dt, [duration]))


def integrate(p, times, method="rk4", initial=None):
    times = np.asarray(times, dtype=float)
    if (
        times.ndim != 1
        or len(times) == 0
        or not np.isfinite(times).all()
        or np.any(np.diff(times) <= 0)
    ):
        raise ValueError("times must be finite and strictly increasing")
    y0 = np.array([p.x0, p.v0] if initial is None else initial, dtype=float)
    if method not in ("euler", "semi-implicit", "rk4", "reference"):
        raise ValueError("Unknown integration method")
    if len(times) == 1:
        return y0[None, :]
    if method == "reference":
        result = solve_ivp(
            lambda t, y: derivative(t, y, p),
            (times[0], times[-1]),
            y0,
            t_eval=times,
            method="DOP853",
            rtol=1e-11,
            atol=1e-13,
        )
        if not result.success:
            raise RuntimeError(result.message)
        return result.y.T
    states = np.empty((len(times), 2))
    states[0] = y0
    with np.errstate(over="ignore", invalid="ignore"):
        for i, h in enumerate(np.diff(times)):
            t, y = times[i], states[i]
            a = derivative(t, y, p)
            if method == "euler":
                states[i + 1] = y + h * a
            elif method == "semi-implicit":
                v = y[1] + h * a[1]
                states[i + 1] = [y[0] + h * v, v]
            else:
                b = derivative(t + h / 2, y + h * a / 2, p)
                c = derivative(t + h / 2, y + h * b / 2, p)
                d = derivative(t + h, y + h * c, p)
                states[i + 1] = y + h * (a + 2 * b + 2 * c + d) / 6
    return states
