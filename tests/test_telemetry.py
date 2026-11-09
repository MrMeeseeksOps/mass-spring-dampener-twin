from mass_spring_twin.physics.parameters import Parameters
from mass_spring_twin.telemetry.sensors import SensorConfig
from mass_spring_twin.twin.emulator import Emulator


def test_deterministic():
    e = Emulator(Parameters())
    c = SensorConfig(missing_probability=0.3)
    a = e.observations(2, c)
    assert a == e.observations(2, c)
    assert any(o.displacement is None for o in a)
    assert any(o.displacement is not None for o in a)
    assert all(abs(o.time - i * c.interval) < 1e-12 for i, o in enumerate(a))
