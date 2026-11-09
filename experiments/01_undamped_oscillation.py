"""Reproducible laboratory experiment. Run from the repository root."""

from pathlib import Path

import numpy as np

from mass_spring_twin.numerics.integrators import time_grid
from mass_spring_twin.physics.analytical import analytical
from mass_spring_twin.physics.energy import energies
from mass_spring_twin.physics.parameters import Parameters

out = Path("outputs") / Path(__file__).stem
out.mkdir(parents=True, exist_ok=True)
p = Parameters(damping=0)
t = time_grid(20, 0.01)
y = analytical(p, t)
e = energies(y, p)[2]
np.savetxt(out / "trajectory.csv", np.column_stack((t, y, e)), delimiter=",", header="t,x,v,energy")
print("Maximum analytical energy drift (J):", max(abs(e - e[0])))
