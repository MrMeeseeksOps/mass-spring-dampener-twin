import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from mass_spring_twin.telemetry.observations import Observation


class SensorConfig(BaseModel):
    model_config = ConfigDict(frozen=True, allow_inf_nan=False)
    interval: float = Field(default=0.1, gt=0)
    noise_std: float = Field(default=0.001, ge=0)
    missing_probability: float = Field(default=0.0, ge=0, le=1)
    seed: int = 42


def sample(times, displacements, config):
    rng = np.random.default_rng(config.seed)
    records = []
    for t, x in zip(times, displacements, strict=True):
        value = (
            None
            if rng.random() < config.missing_probability
            else float(x + rng.normal(0, config.noise_std))
        )
        records.append(Observation(float(t), value))
    return records
