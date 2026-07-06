#!/usr/bin/env python3

import hvplot.pandas  # Registers the .hvplot accessor
import numpy as np
import pandas as pd
import panel as pn

pn.extension("tabulator")

# ---------------------------------------------------------------------------
# Generate a line containing 1,000,000 samples
# ---------------------------------------------------------------------------

N_POINTS = 1_000_000

rng = np.random.default_rng(seed=42)

x = np.linspace(0.0, 1_000.0, N_POINTS, dtype=np.float64)

# Smooth signal plus high-frequency components and noise
y = (
    np.sin(x)
    + 0.25 * np.sin(8.0 * x)
    + 0.08 * np.sin(50.0 * x)
    + 0.03 * rng.standard_normal(N_POINTS)
)

df = pd.DataFrame(
    {
        "x": x,
        "y": y,
    }
)

# ---------------------------------------------------------------------------
# Create the interactive rasterized plot
#
# rasterize=True:
#     Datashader aggregates the complete dataset into the displayed pixel grid.
#
# dynamic=True:
#     Re-rasterizes the data when the visible range changes.
#
# line_width:
#     Applied by Datashader; values above zero enable antialiased lines.
# ---------------------------------------------------------------------------

plot = df.hvplot.line(
    x="x",
    y="y",
    rasterize=True,
    dynamic=True,
    width=1100,
    height=600,
    line_width=1,
    xlabel="x",
    ylabel="Signal",
    title="Rasterized line with 1,000,000 samples",
    responsive=False,
    tools=["pan", "wheel_zoom", "box_zoom", "reset", "save"],
)

# ---------------------------------------------------------------------------
# Panel application
# ---------------------------------------------------------------------------

description = pn.pane.Markdown(f"""
# One-million-point line with hvPlot + Datashader

This plot contains **{N_POINTS:,} samples**.

Zooming or panning causes Datashader to recompute the rasterized representation
for the currently visible range on the CPU in real-time.
""")

app = pn.Column(
    description,
    plot,
    sizing_mode="stretch_width",
)

# Makes the object discoverable by `panel serve`.
app.servable()
