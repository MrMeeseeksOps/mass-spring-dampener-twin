import math

from pydantic import BaseModel, ConfigDict, Field


class Parameters(BaseModel):
    model_config = ConfigDict(frozen=True, allow_inf_nan=False)
    mass: float = Field(default=1.0, gt=0)
    stiffness: float = Field(default=20.0, gt=0)
    damping: float = Field(default=0.5, ge=0)
    x0: float = 0.1
    v0: float = 0.0
    force: float = 0.0

    @property
    def omega_n(self):
        return math.sqrt(self.stiffness / self.mass)

    @property
    def zeta(self):
        return self.damping / (2 * math.sqrt(self.mass * self.stiffness))

    @property
    def regime(self):
        if math.isclose(self.zeta, 1, rel_tol=1e-12):
            return "critical"
        return "underdamped" if self.zeta < 1 else "overdamped"

    @property
    def omega_d(self):
        return self.omega_n * math.sqrt(1 - self.zeta**2) if self.zeta < 1 else None

    @property
    def period(self):
        return 2 * math.pi / self.omega_d if self.omega_d else None
