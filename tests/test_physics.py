import numpy as np
import pytest
from pydantic import ValidationError

from mass_spring_twin.numerics.integrators import integrate, time_grid
from mass_spring_twin.physics.dynamics import derivative
from mass_spring_twin.physics.parameters import Parameters


def test_newton():
    assert derivative(0, [0.1, 0], Parameters())[1] == pytest.approx(-2)


def test_validation():
    for field, value in [("mass", 0), ("stiffness", -1), ("damping", -1), ("x0", float("nan"))]:
        with pytest.raises(ValidationError):
            Parameters(**{field: value})


@pytest.mark.parametrize("method", ["euler", "semi-implicit", "rk4", "reference"])
def test_solvers(method):
    t = time_grid(1, 0.001)
    y = integrate(Parameters(), t, method)
    ref = integrate(Parameters(), t, "reference")
    assert np.max(abs(y - ref)) < 0.005
