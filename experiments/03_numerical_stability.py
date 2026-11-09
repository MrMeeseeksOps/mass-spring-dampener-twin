"""Reproducible laboratory experiment. Run from the repository root."""

import json
from pathlib import Path

from mass_spring_twin.numerics.comparison import compare
from mass_spring_twin.numerics.integrators import time_grid
from mass_spring_twin.physics.energy import energies
from mass_spring_twin.physics.parameters import Parameters

out = Path("outputs") / Path(__file__).stem
out.mkdir(parents=True, exist_ok=True)
p = Parameters(damping=0)
rows = []
for dt in [0.1, 0.05, 0.025, 0.0125]:
    t = time_grid(10, dt)
    ys, metrics = compare(p, t, ["euler", "semi-implicit", "rk4", "reference"])
    for row in metrics:
        row["dt"] = dt
        row["energy_ratio"] = float(
            energies(ys[row["method"]], p)[2][-1] / energies(ys[row["method"]], p)[2][0]
        )
        rows.append(row)
(out / "comparison.json").write_text(json.dumps(rows, indent=2))
print("Forward Euler energy ratios:", [r["energy_ratio"] for r in rows if r["method"] == "euler"])
