# Numerical methods laboratory

Use the following shared setup for the executable examples:

```python
import numpy as np
from mass_spring_twin.physics.parameters import Parameters
from mass_spring_twin.physics.analytical import analytical
from mass_spring_twin.physics.energy import energies
from mass_spring_twin.numerics.integrators import integrate, time_grid

p = Parameters(damping=0)
```

## Integration from first principles

The fundamental theorem gives y(t+h)=y(t)+∫ₜᵗ⁺ʰ f(s,y(s)) ds. Replacing the unknown integrand with f(t,y(t)) gives Forward Euler: yₙ₊₁=yₙ+h f(tₙ,yₙ). This is a tangent-line extrapolation. Semi-implicit Euler first updates vₙ₊₁=vₙ+h a(xₙ,vₙ), then xₙ₊₁=xₙ+h vₙ₊₁. Damping remains explicit.

```python
t = time_grid(1, 0.02)
for method in ["euler", "semi-implicit"]:
    print(method, integrate(p, t, method)[-1])
```

Question: which state components use new versus old values? Experiment: manually compute the first step and compare with the two solvers.

## Euler versus classical RK4

RK4 samples four slopes: k₁=f(t,y), k₂=f(t+h/2,y+hk₁/2), k₃=f(t+h/2,y+hk₂/2), k₄=f(t+h,y+hk₃). Then yₙ₊₁=yₙ+h(k₁+2k₂+2k₃+k₄)/6. Taylor matching cancels errors through h⁴. Intermediate slopes approximate changes in acceleration during the step; four force evaluations buy greater accuracy on smooth solutions.

```python
for method in ["euler", "rk4", "reference"]:
    y = integrate(p, t, method)
    print(method, np.max(abs(y[:, 0] - analytical(p, t)[:, 0])))
```

Question: why is k₂ insufficient on its own for fourth order? Experiment: compare error and execution time in the dashboard at equal output timestamps. DOP853 uses adaptive internal steps with rtol=1e−11 and atol=1e−13; it is an independent numerical reference, not physical truth.

## Local and global truncation error

Taylor expansion y(t+h)=y+h ẏ+h²ÿ/2+… shows Euler's one-step state defect is O(h²). Over O(1/h) steps on a fixed smooth interval, stable propagation accumulates O(h) global error. RK4 has O(h⁵) local defect and O(h⁴) global error. Semi-implicit Euler is first order. These claims assume bounded derivatives and stable integration; roundoff eventually dominates as h becomes extremely small.

```python
errors = []
for h in [0.04, 0.02, 0.01]:
    t = time_grid(1, h)
    errors.append(np.linalg.norm(integrate(p, t, "rk4")[-1] - analytical(p, t)[-1]))
print(errors, errors[1] / errors[2])  # approximately 16
```

Question: derive the ratio 16 from fourth-order global error. Experiment: replace RK4 with Euler and inspect the ratio near 2. Avoid a final time where an error component accidentally vanishes; a full-state norm reduces that risk.

## Stability versus accuracy

For ẏ=λy, Euler amplifies by R(z)=1+z, z=hλ; stability requires |R(z)|≤1. An undamped oscillator has λ=±iωₙ, so |1+ihωₙ|=√(1+h²ωₙ²)>1 for every positive h. Small steps reduce growth over finite time but cannot make Euler energy-conserving. RK4 has R(z)=1+z+z²/2+z³/6+z⁴/24; on the imaginary axis its stability interval is |hωₙ|≤2√2. The undamped semi-implicit method has bounded oscillatory modes for hωₙ<2, but its physical energy fluctuates. Damping changes all stability constraints.

```python
for method in ["euler", "semi-implicit", "rk4"]:
    t = time_grid(10, 0.05)
    e = energies(integrate(p, t, method), p)[2]
    print(method, e[-1] / e[0])
```

Question: can a stable solution still have significant phase error? Experiment: increase h in the dashboard and compare peaks even when amplitude remains bounded. Nonfinite values trigger an instability message; large finite errors remain visible.

## Timestep selection

Choose h small relative to the fastest relevant timescale. Oscillation suggests hωₙ≪1; heavy damping can introduce a much faster decay mode than the oscillation period suggests. Refine h until quantities of interest converge within a stated tolerance. The grid uses requested spacing and a shortened final step when duration is not divisible by h, so every solver shares identical endpoints.

```python
print(time_grid(1, 0.3))  # [0, .3, .6, .9, 1]
for h in [0.02, 0.01]:
    t = time_grid(2, h)
    print(h, integrate(p, t, "rk4")[-1])
```

Question: why can overdamping require a smaller explicit timestep? Experiment: increase damping to 50 and compare Euler with the adaptive reference. Measure work as well as error; fixed solvers perform N Euler/semi-implicit or 4N RK4 derivative evaluations. The UI reports output intervals, not adaptive DOP853 work.

## Numerical energy drift

For undamped Euler, x′=x+hv and v′=v−hωₙ²x. Substitution into E=mv²/2+kx²/2 cancels cross terms and gives Eₙ₊₁=(1+h²ωₙ²)Eₙ. Thus systematic artificial energy growth is predictable, even though the physical oscillator is conservative. RK4 also has energy error; high-order accuracy does not imply exact conservation. Symplectic semi-implicit Euler bounds an altered energy for a stable fixed step in the undamped problem.

```python
t = time_grid(10, 0.05)
e = energies(integrate(p, t, "euler"), p)[2]
print(e[-1] / e[0], (1 + 0.05**2 * p.omega_n**2) ** (len(t) - 1))
```

Question: distinguish physical dissipation from a solver losing energy. Experiment: run 03 and compare each energy ratio with displacement error. Repeat with nonzero damping and reason from Ė=−cv² before interpreting the curves.
