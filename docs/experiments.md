# Reproducible experiments

Run `make experiments` after `make install`. Results are overwritten on rerun.

| Script | Output | Investigation |
|---|---|---|
| 01 | CSV: time, position, velocity, energy | Explain why analytic energy drift is near roundoff. |
| 02 | One CSV per damping regime | Which case settles fastest without crossing equilibrium? |
| 03 | JSON solver errors, timings and energy ratios for four timesteps | Derive Euler's artificial growth and compare RK4 refinement. |
| 04 | JSON issued predictions, evaluations and summary | Separate measurement error, velocity-estimation error and parameter mismatch. |

Units are seconds, meters, m/s, joules. JSON prediction targets beyond 10 seconds are intentionally retained but never scored. The seed is 42; wall-clock timings are not reproducible bit for bit. Noise can obscure the benefit of correct parameters, so a mismatched model need not score worse in every finite noisy run.

Suggested extension: rerun 04 with zero noise, then increase noise and decrease sensor interval. Plot RMSE against lead time rather than pooling all horizons. Change one variable at a time and record your expectations before execution.
