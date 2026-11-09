"""Reproducible laboratory experiment. Run from the repository root."""

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from mass_spring_twin.physics.parameters import Parameters
from mass_spring_twin.telemetry.sensors import SensorConfig
from mass_spring_twin.twin.emulator import Emulator
from mass_spring_twin.twin.evaluation import run_cycle

out = Path("outputs") / Path(__file__).stem
out.mkdir(parents=True, exist_ok=True)
observations = Emulator(Parameters()).observations(
    10, SensorConfig(seed=42, missing_probability=0.1)
)
rows = []
for k in [20, 15]:
    estimates, predictions, scores = run_cycle(Parameters(stiffness=k), observations, 1.0)
    errors = np.array([r["error"] for r in scores])
    rows.append(
        {"twin_stiffness": k, "rmse": float(np.sqrt(np.mean(errors**2))), "matured": len(scores)}
    )
    (out / f"predictions_k{k}.json").write_text(
        json.dumps([asdict(p) for p in predictions], indent=2)
    )
    (out / f"evaluation_k{k}.json").write_text(json.dumps(scores, indent=2))
(out / "summary.json").write_text(json.dumps(rows, indent=2))
print(rows)
