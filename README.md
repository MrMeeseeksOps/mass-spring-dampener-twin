# Mass–spring–damper digital twin laboratory

A local, observation-driven **simulated digital twin prototype**, with an independent physical-system emulator. It has not been validated against hardware.

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/getting-started/installation/) on macOS or Linux. From this repository root:

```sh
make install
make test
make lint
make run
```

Open http://localhost:8501. Use Play, Pause, Reset and speed controls; changing parameters resets the simulation clock. The default system is m=1 kg, k=20 N/m, c=0.5 N s/m, x₀=0.1 m, v₀=0, F=0. Parameter controls expose damping regimes and constant external force. Choose numerical methods for synchronized position, velocity, acceleration and energy plots, with errors and timings. Analytical overlays appear for zero force.

The twin panel controls separate stiffness and damping, sensor interval, noise, missing probability, seed and forecast horizon. It shows observed displacement, updated state, a future forecast, and errors for matured predictions. Predictions beyond the simulation end remain unscored. Sensor RMSE includes measurement noise; it is not a pure model error metric.

```sh
make experiments
make format
```

Experiments write CSV trajectories and JSON metrics into gitignored `outputs/`. Experiment 03 demonstrates Forward Euler energy growth; experiment 04 compares matched and mismatched twin parameters with deterministic telemetry. Runtime measurements vary by machine.

Source modules separate `physics`, `numerics`, `telemetry`, `twin` and `ui`. See [mathematical foundations](docs/mathematical_foundations.md), [numerical methods](docs/numerical_methods.md), [architecture](docs/digital_twin_architecture.md), and [experiment guide](docs/experiments.md).

Tests cover Newton's law, validation, all damping regimes, analytical/reference agreement, energy conservation/dissipation, convergence including fourth-order RK4, deterministic telemetry, causal state updates and forecast maturity. Tolerances are explicit in tests: analytical/reference absolute error ≤2e−10, analytical energy drift ≤1e−14 J, RK4 refinement ratios 14–18 in a smooth test case.

Assumptions: linear spring, viscous damping, positive constant mass and stiffness, nonnegative damping, horizontal motion, constant applied force, point mass and ideal displacement sensor. The estimator resets position and uses backward differences for velocity; it amplifies noise and does not identify parameters. Missing observations propagate the model. No contact, nonlinear friction, parameter jumps, real sensors, Kalman filtering or machine learning are included. Large timesteps can be unstable and are deliberately exposed. Semi-implicit Euler with explicit damping also has a finite stability region.

The dashboard recomputes deterministic trajectories on input changes; playback is a clock over the computed trajectory. DOP853 drives the emulator animation and supplies reference truth; selected custom solvers are compared separately. The observation loop consumes timestamped reference samples chronologically. Solver accuracy and agreement with simulated truth do not establish physical fidelity.
