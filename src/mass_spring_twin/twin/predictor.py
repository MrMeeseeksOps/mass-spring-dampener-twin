from dataclasses import dataclass

from mass_spring_twin.numerics.integrators import integrate


@dataclass(frozen=True)
class Prediction:
    issue_time: float
    target_time: float
    displacement: float
    velocity: float


def predict(parameters, state, issue_time, target_times):
    if any(t <= issue_time for t in target_times):
        raise ValueError("Prediction targets must follow issue time")
    if len(target_times) == 0:
        return []
    states = integrate(parameters, [issue_time, *target_times], "reference", state)[1:]
    return [
        Prediction(issue_time, float(t), float(y[0]), float(y[1]))
        for t, y in zip(target_times, states, strict=True)
    ]
