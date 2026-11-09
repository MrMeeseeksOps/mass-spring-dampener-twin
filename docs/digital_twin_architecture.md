# Observation and prediction architecture

`Emulator(parameters)` uses SciPy DOP853 to supply reference displacement at scheduled sensor timestamps. `SensorConfig` controls seeded independent Gaussian noise and independent missing samples; missing observations carry `None`. Each record includes time in seconds and displacement in meters. Sampling begins at zero and stops at the last scheduled time ≤ duration; no off-schedule endpoint is fabricated.

The twin owns separate parameters and an `Estimator`. Its only update argument is an `Observation`. At each timestamp it propagates its previous estimate with its own model, then replaces displacement with the valid measurement. Velocity is the secant slope between the two most recent valid observations. The first valid observation keeps the propagated velocity prior. Missing readings do not reset the state. Strict chronology is enforced.

The backward difference approximates velocity at a previous midpoint rather than precisely at the current time. For equal spacing Δ, independent position noise with standard deviation σ contributes velocity noise standard deviation √2 σ/Δ. Shorter sampling intervals can therefore worsen this baseline velocity estimate. Missing readings enlarge the secant interval and introduce more temporal averaging. No covariance or uncertainty calibration is claimed.

`run_cycle` updates state and issues immutable predictions containing issue and target times. It infers the regular schedule from elapsed intervals and forecasts on that schedule over the horizon. It never reads future observation values. Evaluation only scores forecasts whose target matches the newly arrived valid observation; missed targets expire unscored. Predictions beyond the final observation remain pending. Overlapping horizons contribute multiple correlated errors; RMSE is pooled over these predictions, not independent trials.

Only visualization reads reference truth directly. The estimator and predictor do not accept an emulator or its trajectory. Tests change future observations and confirm earlier issued predictions remain identical. The twin starts from a configurable prior (the dashboard shares initial conditions for convenience), never a hidden emulator state lookup.

The current emulator evaluates a full batch before streaming records; this is equivalent to sequential advancement for a deterministic constant-parameter ODE. Disturbances and changing parameters are optional future extensions. Modules intentionally avoid storage or service infrastructure.
