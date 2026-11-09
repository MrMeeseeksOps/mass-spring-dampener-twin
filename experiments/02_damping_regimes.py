"""Reproducible laboratory experiment. Run from the repository root."""

from pathlib import Path

import numpy as np

from mass_spring_twin.numerics.integrators import time_grid
from mass_spring_twin.physics.analytical import analytical
from mass_spring_twin.physics.energy import energies
from mass_spring_twin.physics.parameters import Parameters

out = Path("outputs") / Path(__file__).stem
out.mkdir(parents=True, exist_ok=True)
t = time_grid(10, 0.01)
for c in [0.5, 2 * np.sqrt(20), 15]:
    p = Parameters(damping=c)
    y = analytical(p, t)
    np.savetxt(out / f"{p.regime}.csv", np.column_stack((t, y)), delimiter=",", header="t,x,v")
    print(p.regime, "final energy (J):", energies(y, p)[2][-1])
