from dataclasses import dataclass

import numpy as np

from mass_spring_twin.numerics.integrators import integrate
from mass_spring_twin.telemetry.sensors import sample


@dataclass
class Emulator:
    parameters: object

    def observations(self, duration, config):
        n = int(np.floor(duration / config.interval))
        if n > 20000:
            raise ValueError("At most 20000 sensor samples")
        times = np.arange(n + 1) * config.interval
        truth = integrate(self.parameters, times, "reference")
        return sample(times, truth[:, 0], config)
