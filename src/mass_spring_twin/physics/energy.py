import numpy as np


def energies(states, p):
    states = np.asarray(states)
    kinetic = 0.5 * p.mass * states[..., 1] ** 2
    potential = 0.5 * p.stiffness * states[..., 0] ** 2
    return kinetic, potential, kinetic + potential
