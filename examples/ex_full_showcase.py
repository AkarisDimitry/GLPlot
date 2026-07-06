import numpy as np
import time
import math
import glplot.pyplot as gplt


def generate_flow_field(n=200_000):
    """Generate a vector field of lines."""
    ab = np.zeros((n, 2), dtype=np.float32)
    # y = ax + b
    a = np.random.uniform(-0.1, 0.1, n).astype(np.float32)
    b = np.random.normal(0, 1.5, n).astype(np.float32)
    xr = (-5.0, 5.0)

    # Colors based on slope/intensity
    cols = np.ones((n, 4), dtype=np.float32)
    cols[:, 0] = 0.2 + 0.8 * (a + 0.1) / 0.2  # R based on slope
    cols[:, 1] = 0.7 + 0.3 * np.random.rand(n)  # G
    cols[:, 2] = 0.9  # B
    cols[:, 3] = 0.15  # Low alpha for layering

    return a, b, xr, cols


def generate_particles(m=10_000):
    """Generate random stars/particles."""
    pts = np.random.normal(0, 2.5, (m, 2)).astype(np.float32)
    # Subtle color variations
    cols = np.ones((m, 4), dtype=np.float32)
    cols[:, 0] = 1.0  # White/Redish
    cols[:, 1] = 0.9
    cols[:, 2] = 1.0
    return pts, cols


def generate_spiral(k=2000):
    """Generate a high-res spiral trajectory."""
    t = np.linspace(0, 20 * math.pi, k)
    x = t * np.cos(t) / 10.0
    y = t * np.sin(t) / 10.0
    return x.astype(np.float32), y.astype(np.float32)


def showcase_demo():
    print("Preparing GLPlot Full Showcase Demo...")

    # 1. Background Flow Field (Millions of Lines)
    a, b, xr, cols = generate_flow_field(250_000)
    # Use a higher alpha since we have dark background
    cols[:, 3] = 0.4  # Higher base alpha
    gplt.plot_lines(a, b, xr, colors=cols)

    # 2. Particle Cloud (Scatter)
    pts, pcols = generate_particles(20_000)
    gplt.scatter(pts[:, 0], pts[:, 1], size=6.0, color=(1, 1, 1, 0.9))

    # 3. Primary Trajectories (Polylines)
    sx, sy = generate_spiral(2000)
    gplt.plot(sx, sy, color=(0.4, 1.0, 0.6, 1.0))  # Neon Green
    gplt.plot(sx * 1.2, sy * 1.2, color=(1.0, 0.4, 0.8, 0.8))  # Pink

    # 4. Global Settings & Aesthetics
    gplt.title("GLPlot Core V1 Showcase")

    # Set premium visuals
    plot = gplt._get_or_create_plot()
    plot.autoscale()  # Ensure we see the data

    # Disable aggressive alpha scaling for this showcase to keep it bold
    plot.options.default_global_alpha = 1.0

    v = plot.options.visual

    # Neon Bloom - Lower threshold to capture the transparent lines
    # v.glow.enabled = True
    # v.glow.intensity = 2.0
    # v.glow.threshold = 0.05 # Lower to see more glow from faint lines
    # v.glow.radius_px = 6.0

    # Deep Dark Nebula Gradient (Slightly brighter than before)
    v.gradient_background.enabled = True
    v.gradient_background.top_color = (0.08, 0.08, 0.25)
    v.gradient_background.bottom_color = (0.02, 0.02, 0.05)

    print("Showcase Ready (High Visibility Mode). Opening Window...")
    gplt.show()


if __name__ == "__main__":
    showcase_demo()
