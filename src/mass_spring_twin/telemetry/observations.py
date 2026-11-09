from dataclasses import dataclass


@dataclass(frozen=True)
class Observation:
    time: float
    displacement: float | None
