import time

import numpy as np

from glplot import GPULinePlot


def run_verification():
    plot = GPULinePlot(title="GLPlot V2: Layer Architecture Verification")

    # 1. Background Filled Area (PatchLayer)
    # Sinusoidal band
    x = np.linspace(0, 10, 200)
    y1 = np.sin(x) - 0.5
    y2 = np.sin(x) + 0.5

    # Pre-calculate triangle strip vertices for fill_between
    vertices = []
    for i in range(len(x)):
        vertices.append([x[i], y1[i]])
        vertices.append([x[i], y2[i]])
    vertices = np.array(vertices, dtype=np.float32)

    plot.add_patch(vertices, mode="strip", face_color=(0.2, 0.4, 0.8, 0.3), label="Sinusoidal Band")

    # 2. Scatter with Outlines
    n_scat = 100
    xs = np.random.uniform(0, 10, n_scat)
    ys = np.random.uniform(-2, 2, n_scat)
    colors = np.ones((n_scat, 4), dtype=np.float32)
    colors[:, 0] = np.random.rand(n_scat)  # Random red
    colors[:, 2] = np.random.rand(n_scat)  # Random blue

    plot.add_scatter(xs, ys, colors, size=12.0)
    # Note: We can fine-tune outlines via the layer object afterward
    scat_layer = plot.scene.layers[-1]
    scat_layer.label = "Research Points"
    scat_layer.style.point_outline_enabled = True
    scat_layer.style.point_outline_color = (0, 0, 0, 1)
    scat_layer.style.point_outline_width = 1.5

    # 3. Polyline (Thick curve)
    xp = np.linspace(0, 10, 100)
    yp = np.cos(xp) * 1.5
    plot.add_line_strip(xp, yp, color=(1, 0.2, 0.1, 1), width=4.0)
    plot.scene.layers[-1].label = "Main Signal"

    # 4. Text Labels
    plot.add_text(5, 1.8, "Upper Bound", fontsize=15, color=(1, 1, 1, 1))
    plot.add_text(2, -1.8, "Minima reached", fontsize=12, color=(0.8, 0.1, 0.1, 1))

    # 5. Global Overrides Testing
    # We can set default overrides here
    plot.options.visual.overrides.alpha_multiplier = 0.9

    plot.run()


if __name__ == "__main__":
    run_verification()
