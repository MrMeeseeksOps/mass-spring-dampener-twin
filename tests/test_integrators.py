import numpy as np
import pytest

from mass_spring_twin.numerics.integrators import integrate, time_grid
from mass_spring_twin.physics.analytical import analytical
from mass_spring_twin.physics.parameters import Parameters


@pytest.mark.parametrize(
    "damping,regime", [(0, "underdamped"), (2 * np.sqrt(20), "critical"), (15, "overdamped")]
)
def test_analytical(damping, regime):
    p = Parameters(damping=damping, v0=0.2)
    t = time_grid(5, 0.01)
    assert p.regime == regime
    np.testing.assert_allclose(analytical(p, t), integrate(p, t, "reference"), atol=2e-10)


@pytest.mark.parametrize("method", ["euler", "semi-implicit", "rk4"])
def test_convergence(method):
    p = Parameters(damping=0)
    errors = []
    for dt in [0.04, 0.02, 0.01]:
        t = time_grid(1, dt)
        errors.append(np.linalg.norm(integrate(p, t, method)[-1] - analytical(p, t)[-1]))
    assert errors[0] > errors[1] > errors[2]
    if method == "rk4":
        assert 14 < errors[1] / errors[2] < 18


def test_grid_and_validation():
    np.testing.assert_allclose(time_grid(1, 0.3), [0, 0.3, 0.6, 0.9, 1])
    for dt in [0, -1, float("nan")]:
        with pytest.raises(ValueError):
            time_grid(1, dt)
    with pytest.raises(ValueError):
        integrate(Parameters(), [0, 0])
