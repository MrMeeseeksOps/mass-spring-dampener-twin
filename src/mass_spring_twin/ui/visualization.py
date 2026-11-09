import numpy as np
import plotly.graph_objects as go


def mechanism(x, v, extent):
    wall = -extent
    end = x - 0.07 * extent
    sx = np.linspace(wall, end, 41)
    sy = np.zeros(41)
    sy[1:-1] = 0.04 * extent * np.where(np.arange(39) % 2, 1, -1)
    fig = go.Figure(go.Scatter(x=sx, y=sy, mode="lines", name="Spring"))
    fig.add_shape(type="line", x0=wall, x1=wall, y0=-extent / 4, y1=extent / 4)
    fig.add_shape(
        type="rect",
        x0=end,
        x1=x + 0.07 * extent,
        y0=-extent / 10,
        y1=extent / 10,
        fillcolor="#38bdf8",
    )
    fig.add_vline(x=0, line_dash="dot", annotation_text="Equilibrium")
    fig.add_annotation(
        x=x,
        y=extent / 5,
        text=f"x={x:.4f} m; v={v:+.4f} m/s {'→' if v > 0 else '←' if v < 0 else ''}",
        showarrow=False,
    )
    fig.update_layout(
        height=250,
        xaxis_range=[-extent, extent],
        yaxis_range=[-extent / 3, extent / 3],
        showlegend=False,
        margin={"t": 10, "b": 10},
    )
    fig.update_yaxes(visible=False)
    fig.update_xaxes(title="Displacement (m)")
    return fig
