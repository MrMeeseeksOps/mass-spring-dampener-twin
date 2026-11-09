from dataclasses import asdict

import numpy as np

from mass_spring_twin.twin.estimator import Estimator
from mass_spring_twin.twin.predictor import predict


def evaluate(predictions, observation, now):
    if observation.time > now:
        raise ValueError("Cannot evaluate future observations")
    return [
        dict(
            **asdict(p),
            observed=observation.displacement,
            error=p.displacement - observation.displacement,
        )
        for p in predictions
        if p.target_time <= now + 1e-10
        and abs(p.target_time - observation.time) < 1e-10
        and observation.displacement is not None
    ]


def run_cycle(parameters, observations, horizon=1.0):
    """Consume a chronological stream; targets use only known sampling schedule."""
    if horizon <= 0 or not np.isfinite(horizon):
        raise ValueError("Horizon must be finite and positive")
    estimator = Estimator(parameters)
    pending, predictions, evaluations, estimates = [], [], [], []
    interval = None
    previous_time = None
    for obs in observations:
        state = estimator.update(obs)
        estimates.append([obs.time, *state])
        evaluations.extend(evaluate(pending, obs, obs.time))
        pending = [p for p in pending if p.target_time > obs.time + 1e-10]
        if previous_time is not None:
            interval = obs.time - previous_time
        previous_time = obs.time
        if interval is not None:
            targets = (
                obs.time + np.arange(1, int(np.floor(horizon / interval + 1e-9)) + 1) * interval
            )
            issued = predict(parameters, state, obs.time, targets)
            predictions.extend(issued)
            pending.extend(issued)
    return np.asarray(estimates), predictions, evaluations
