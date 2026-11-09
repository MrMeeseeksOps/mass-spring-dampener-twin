import numpy as np

from mass_spring_twin.numerics.integrators import integrate


class Estimator:
    """Causal displacement reset plus backward secant velocity.

    First valid observation retains the configured velocity prior. Missing samples
    propagate the model. No emulator state or future observation is accepted.
    """

    def __init__(self, parameters):
        self.parameters = parameters
        self.state = np.array([parameters.x0, parameters.v0])
        self.time = 0.0
        self.previous = None
        self.last_record = None

    def update(self, observation):
        if observation.time < 0 or (
            self.last_record is not None and observation.time <= self.last_record
        ):
            raise ValueError("Observations must have increasing nonnegative timestamps")
        if observation.time > self.time:
            self.state = integrate(
                self.parameters, [self.time, observation.time], "reference", self.state
            )[-1]
        if observation.displacement is not None:
            velocity = self.state[1]
            if self.previous is not None:
                velocity = (observation.displacement - self.previous.displacement) / (
                    observation.time - self.previous.time
                )
            self.state = np.array([observation.displacement, velocity])
            self.previous = observation
        self.time = observation.time
        self.last_record = observation.time
        return self.state.copy()
