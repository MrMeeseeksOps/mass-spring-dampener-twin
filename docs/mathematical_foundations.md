# Mathematical foundations

Run examples with `uv run python` from the repository root. All states use meters and m/s, forces use newtons, and time uses seconds. The model assumes a point mass on a frictionless horizontal guide, a linear spring, viscous damper and fixed coefficients.

## 1. Newton's second law

Momentum is p=mv. For constant mass, F_net=dp/dt=m dv/dt=m a. Thus acceleration is total force divided by mass, not merely the externally applied force divided by mass. Doubling mass halves instantaneous acceleration for the same net force.

```python
from mass_spring_twin.physics.parameters import Parameters
from mass_spring_twin.physics.dynamics import derivative

p = Parameters(mass=2, stiffness=20, damping=0.5, force=1)
print(derivative(0, [0.1, 0], p))  # [0, -0.5] in m/s and m/s²
```

Question: why is acceleration negative despite positive applied force? Experiment: double mass again while holding the state fixed and predict the result before running.

## 2. Hooke's law

Linearizing a smooth restoring force about stable equilibrium gives F_s(x)≈F_s(0)+F_s′(0)x=−kx, where k>0. The minus sign expresses restoration toward equilibrium. Integrating −F_s over displacement gives stored potential U=∫₀ˣ kq dq=kx²/2.

```python
from mass_spring_twin.physics.parameters import Parameters
from mass_spring_twin.physics.energy import energies

print(energies([[0.1, 0], [0.2, 0]], Parameters()))
```

Question: why does doubling displacement quadruple potential energy? Experiment: compare trajectories with x₀=0.1 and 0.2; test whether position scales linearly while energy scales quadratically.

## 3. Viscous damping

A linear drag law is F_d=−cv, c≥0. Its instantaneous mechanical power is F_d v=−cv²≤0, independent of the sign of velocity. Units require c in N s/m.

```python
c = 0.5
for v in [-2, 0, 2]:
    print(v, -c * v, -c * v * v)  # velocity, damping force, dissipated power
```

Question: does zero damping power at a turning point imply constant energy for the whole orbit? Experiment: run experiment 02 and inspect energy differences near turning points.

## 4. Governing equation

Sum forces along the guide: m ẍ=F−cv−kx. Rearranging gives m ẍ+c ẋ+kx=F. For constant nonzero force the equilibrium shifts to x_eq=F/k. Substituting q=x−F/k reduces the equation to the unforced homogeneous equation in q.

```python
from mass_spring_twin.physics.parameters import Parameters
from mass_spring_twin.physics.dynamics import derivative

p = Parameters(force=2)
print(derivative(0, [p.force / p.stiffness, 0], p))  # equilibrium
```

Question: why is mechanical energy relative to x=0 not monotone when F≠0? Experiment: add constant force in the dashboard and compare initial and final equilibrium positions.

## 5. State-space representation

Define y=(x,v)ᵀ with v=ẋ. Then ẏ=(v,(F−cv−kx)/m)ᵀ=Ay+b, where A=[[0,1],[−k/m,−c/m]], b=(0,F/m)ᵀ. A second-order equation needs two initial conditions; one displacement measurement alone leaves velocity unidentified.

```python
import numpy as np
from mass_spring_twin.physics.parameters import Parameters
from mass_spring_twin.physics.dynamics import derivative

p = Parameters()
A = np.array([[0, 1], [-p.stiffness / p.mass, -p.damping / p.mass]])
print(A @ [0.1, 0.2], derivative(0, [0.1, 0.2], p))
```

Question: can two states with identical position have different futures? Experiment: run with v₀=±0.2, keeping x₀ fixed.

## 6. Natural frequency and damping ratio

Divide the ODE by m and compare ẍ+2ζωₙẋ+ωₙ²x=0. Hence ωₙ=√(k/m), ζ=c/(2√(mk)). Eigenvalues solve r²+2ζωₙr+ωₙ²=0, yielding r=−ζωₙ±ωₙ√(ζ²−1). Complex conjugate roots cause oscillation; a repeated negative root marks critical damping; distinct negative real roots produce overdamping. For ζ<1, ω_d=ωₙ√(1−ζ²), T=2π/ω_d.

```python
from mass_spring_twin.physics.parameters import Parameters

p = Parameters()
print(p.omega_n, p.zeta, p.regime, p.omega_d, p.period)
```

Question: does overdamped motion have an oscillation period? Experiment: use c=2√(mk), then twice that value, and compare settling. Critical classification uses a relative tolerance of 1e−12 to absorb roundoff at the boundary.

## 7. Analytical solutions

Let α=c/(2m). For ζ<1, substitution of the roots into the exponential general solution yields

x=e^(−αt)[x₀ cos(ω_dt)+(v₀+αx₀)sin(ω_dt)/ω_d].

At ζ=1 the repeated root needs an independent t e^(−αt) solution:

x=[x₀+(v₀+αx₀)t]e^(−αt).

For ζ>1, r₁,₂=−α±√(α²−ωₙ²), x=Ae^(r₁t)+Be^(r₂t), A=(v₀−r₂x₀)/(r₁−r₂), B=x₀−A. Differentiation gives velocity; acceleration follows Newton's law. The implementation computes r₁=−ωₙ²/(α+√(α²−ωₙ²)) to avoid cancellation at high damping.

```python
from mass_spring_twin.physics.analytical import analytical
from mass_spring_twin.physics.parameters import Parameters

print(analytical(Parameters(), [0, 0.1, 1]))
```

Question: verify both initial conditions algebraically in each regime. Experiment: compare analytical states with DOP853 for c=0.5, 2√20, and 15; see `test_analytical` and experiment 02.

## 8. Mechanical energy

K=mv²/2, U=kx²/2, E=K+U. Differentiate and substitute the ODE:

Ė=mv v̇+kx ẋ=v(m v̇+kx)=Fv−cv².

For F=c=0 energy is constant. For F=0,c>0 it is nonincreasing, with instantaneous zero derivative when v=0. Forced motion exchanges work ∫Fv dt with the environment. Numerical energy drift is distinct from this physical work balance.

```python
import numpy as np
from mass_spring_twin.physics.parameters import Parameters
from mass_spring_twin.physics.analytical import analytical
from mass_spring_twin.physics.energy import energies

p = Parameters(damping=0)
t = np.linspace(0, 10, 1001)
e = energies(analytical(p, t), p)[2]
print(np.max(abs(e - e[0])))
```

Question: why does damping force vanish at rest while spring force may remain? Experiment: run 01, then set damping to 0.5 and confirm nonpositive energy differences within floating-point tolerance.
