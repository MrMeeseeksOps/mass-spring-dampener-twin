import numpy as np

from mass_spring_twin.numerics.integrators import integrate, time_grid
from mass_spring_twin.physics.analytical import analytical
from mass_spring_twin.physics.energy import energies
from mass_spring_twin.physics.parameters import Parameters


def test_conservation():
    p = Parameters(damping=0)
    t = time_grid(20, 0.01)
    e = energies(analytical(p, t), p)[2]
    np.testing.assert_allclose(e, e[0], atol=1e-14)


def test_dissipation():
    for c in [0.5, 2 * np.sqrt(20), 15]:
        p = Parameters(damping=c)
        e = energies(analytical(p, time_grid(10, 0.01)), p)[2]
        assert np.all(np.diff(e) <= 1e-14)


def test_euler_artificial_growth():
    p = Parameters(damping=0)
    e = energies(integrate(p, time_grid(10, 0.05), "euler"), p)[2]
    assert e[-1] > 100 * e[0]
