import numpy as np
import pytest

from mass_spring_twin.physics.parameters import Parameters
from mass_spring_twin.telemetry.observations import Observation
from mass_spring_twin.twin.estimator import Estimator
from mass_spring_twin.twin.evaluation import evaluate, run_cycle
from mass_spring_twin.twin.predictor import Prediction


def test_causal_velocity_and_missing():
    e = Estimator(Parameters())
    np.testing.assert_allclose(e.update(Observation(0, 0.1)), [0.1, 0])
    np.testing.assert_allclose(e.update(Observation(0.1, 0.12)), [0.12, 0.2])
    assert np.isfinite(e.update(Observation(0.2, None))).all()
    with pytest.raises(ValueError):
        e.update(Observation(0.1, 0.1))


def test_future_separation():
    prefix = [Observation(i * 0.1, 0.1 * np.cos(i * 0.1)) for i in range(5)]
    _, a, _ = run_cycle(Parameters(), prefix + [Observation(0.5, 1)], 0.3)
    _, b, _ = run_cycle(Parameters(), prefix + [Observation(0.5, -100)], 0.3)
    assert [p for p in a if p.issue_time < 0.5] == [p for p in b if p.issue_time < 0.5]
    # The estimator accepts only observations and its own parameters, never an emulator.
    e = Estimator(Parameters(stiffness=10))
    assert not hasattr(e, "emulator")


def test_alignment_and_maturity():
    observations = [Observation(i * 0.1, 0.1) for i in range(10)]
    _, predictions, scores = run_cycle(Parameters(), observations, 0.3)
    assert predictions and scores
    assert all(p.target_time > p.issue_time for p in predictions)
    assert all(s["target_time"] <= observations[-1].time + 1e-12 for s in scores)
    p = Prediction(0, 1, 0.2, 0)
    assert evaluate([p], Observation(0.5, 0.1), 0.5) == []
    with pytest.raises(ValueError):
        evaluate([p], Observation(1, 0.1), 0.5)


def test_every_scheduled_target_scored_despite_roundoff():
    obs = [Observation(i * 0.1, 0.1) for i in range(10)]
    _, predictions, scores = run_cycle(Parameters(), obs, 0.3)
    expected = [p for p in predictions if p.target_time <= 0.9 + 1e-10]
    assert len(scores) == len(expected)
