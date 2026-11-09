import time

import numpy as np
import streamlit as st

from mass_spring_twin.numerics.integrators import integrate, time_grid
from mass_spring_twin.physics.parameters import Parameters
from mass_spring_twin.ui.visualization import mechanism

st.set_page_config(page_title="Mass–spring digital twin", layout="wide")
st.title("Mass–spring–damper laboratory")
st.caption("Simulated digital twin prototype • SI units • no real hardware validation")
with st.sidebar:
    st.header("Physical-system emulator")
    mass = st.number_input("Mass (kg)", 0.01, 100.0, 1.0)
    stiffness = st.number_input("Stiffness (N/m)", 0.01, 1000.0, 20.0)
    damping = st.number_input("Damping (N s/m)", 0.0, 100.0, 0.5)
    x0 = st.number_input("Initial displacement (m)", -5.0, 5.0, 0.1)
    v0 = st.number_input("Initial velocity (m/s)", -10.0, 10.0, 0.0)
    force = st.number_input("Force (N)", -100.0, 100.0, 0.0)
    duration = st.slider("Duration (s)", 1.0, 30.0, 10.0)
    dt = st.number_input("Timestep (s)", 0.001, 0.5, 0.01, format="%.3f")
p = Parameters(mass=mass, stiffness=stiffness, damping=damping, x0=x0, v0=v0, force=force)
times = time_grid(duration, dt)
states = integrate(p, times, "reference")
st.write(f"Regime: {p.regime} · ωn={p.omega_n:.4f} rad/s · ζ={p.zeta:.4f}")
if p.omega_d:
    st.write(f"ωd={p.omega_d:.4f} rad/s · period={p.period:.4f} s")
key = (p.model_dump_json(), duration, dt)
if st.session_state.get("experiment") != key:
    st.session_state.update(experiment=key, clock=0.0, playing=False, last=time.monotonic())
a, b, c = st.columns(3)
if a.button("Play"):
    st.session_state.update(playing=True, last=time.monotonic())
if b.button("Pause"):
    st.session_state.playing = False
if c.button("Reset"):
    st.session_state.update(clock=0.0, playing=False)
speed = st.slider("Simulation speed", 0.1, 4.0, 1.0)


@st.fragment(run_every=0.05)
def animation():
    now = time.monotonic()
    if st.session_state.playing:
        st.session_state.clock = min(
            duration, st.session_state.clock + (now - st.session_state.last) * speed
        )
        if st.session_state.clock >= duration:
            st.session_state.playing = False
    st.session_state.last = now
    x = np.interp(st.session_state.clock, times, states[:, 0])
    v = np.interp(st.session_state.clock, times, states[:, 1])
    st.plotly_chart(
        mechanism(x, v, max(0.3, float(np.nanmax(np.abs(states[:, 0]))) * 1.3)), key="animation"
    )
    st.caption(f"Simulation time: {st.session_state.clock:.2f} s")


animation()

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from mass_spring_twin.numerics.comparison import compare
from mass_spring_twin.physics.analytical import analytical
from mass_spring_twin.physics.energy import energies
from mass_spring_twin.telemetry.sensors import SensorConfig
from mass_spring_twin.twin.emulator import Emulator
from mass_spring_twin.twin.evaluation import run_cycle

st.subheader("Numerical methods and energy")
methods = st.multiselect(
    "Methods", ["euler", "semi-implicit", "rk4", "reference"], default=["rk4", "reference"]
)
trajectories, rows = compare(p, times, methods)
if p.force == 0:
    trajectories["analytical"] = analytical(p, times)
fig = make_subplots(
    rows=4,
    cols=1,
    shared_xaxes=True,
    subplot_titles=[
        "Displacement (m)",
        "Velocity (m/s)",
        "Acceleration (m/s²)",
        "Mechanical energy (J)",
    ],
)
for name, y in trajectories.items():
    if not np.isfinite(y).all():
        st.error(f"{name} overflowed: numerical instability. Reduce timestep.")
    acceleration = (p.force - p.damping * y[:, 1] - p.stiffness * y[:, 0]) / p.mass
    for row, values in enumerate([y[:, 0], y[:, 1], acceleration, energies(y, p)[2]], 1):
        fig.add_trace(
            go.Scatter(x=times, y=values, name=name, legendgroup=name, showlegend=row == 1),
            row=row,
            col=1,
        )
fig.update_layout(height=850, hovermode="x unified")
fig.update_xaxes(title_text="Time (s)", row=4, col=1)
st.plotly_chart(fig)
st.dataframe(rows)
st.caption(
    "Errors use DOP853 at identical output times. Final-state norm combines position and velocity in SI coordinates. Steps count output intervals; reference uses adaptive internal steps. Execution times vary."
)
st.write(
    dict(
        zip(
            ["Kinetic energy (J)", "Spring energy (J)", "Total energy (J)"],
            [float(e[-1]) for e in energies(states, p)],
            strict=True,
        )
    )
)
st.subheader("Observation-driven digital twin")
c1, c2, c3 = st.columns(3)
twin_k = c1.number_input("Twin stiffness (N/m)", 0.01, 1000.0, 18.0)
twin_c = c2.number_input("Twin damping (N s/m)", 0.0, 100.0, 0.4)
interval = c3.slider("Sensor interval (s)", 0.05, 1.0, 0.1)
noise = c1.number_input("Sensor noise σ (m)", 0.0, 0.1, 0.001, format="%.4f")
missing = c2.slider("Missing probability", 0.0, 1.0, 0.05)
horizon = c3.slider("Prediction horizon (s)", 0.1, 3.0, 1.0)
seed = c1.number_input("Random seed", 0, 100000, 42)
sensor = SensorConfig(interval=interval, noise_std=noise, missing_probability=missing, seed=seed)
observations = Emulator(p).observations(duration, sensor)
twin_p = p.model_copy(update={"stiffness": twin_k, "damping": twin_c})
estimates, predictions, evaluations = run_cycle(twin_p, observations, horizon)
twin_fig = go.Figure()
twin_fig.add_trace(
    go.Scatter(
        x=times, y=integrate(p, times, "reference")[:, 0], name="Emulator truth (evaluation only)"
    )
)
twin_fig.add_trace(
    go.Scatter(
        x=[o.time for o in observations],
        y=[o.displacement for o in observations],
        mode="markers",
        name="Sensor",
    )
)
twin_fig.add_trace(go.Scatter(x=estimates[:, 0], y=estimates[:, 1], name="Updated twin estimate"))
if predictions:
    last_issue = max(q.issue_time for q in predictions if q.issue_time <= duration)
    last = [q for q in predictions if q.issue_time == last_issue]
    twin_fig.add_trace(
        go.Scatter(
            x=[q.target_time for q in last],
            y=[q.displacement for q in last],
            name="Latest forecast",
        )
    )
twin_fig.update_layout(xaxis_title="Time (s)", yaxis_title="Displacement (m)")
st.plotly_chart(twin_fig)
if evaluations:
    errors = np.array([e["error"] for e in evaluations])
    st.metric(
        "Matured forecast RMSE against noisy observations (m)", f"{np.sqrt(np.mean(errors**2)):.6f}"
    )
    st.dataframe(evaluations[-100:])
else:
    st.info("No forecasts have reached a valid observed target yet.")
st.caption(
    "Position resets to each valid sensor value. Velocity uses a backward difference between the two latest valid readings; the first reading retains the velocity prior. Noise amplifies velocity error; missing readings propagate the model. Forecasts are stored at issue time and scored only when their target arrives. No parameter identification is performed."
)
