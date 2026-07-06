import numpy as np

import glplot.pyplot as gplt


def test_phase4_primitives():
    # 1. Millions of Lines (LineFamilyLayer)
    n = 100_000
    ab = np.zeros((n, 2), dtype=np.float32)
    ab[:, 0] = 0.01  # slope
    ab[:, 1] = np.linspace(-2, 2, n)  # offsets

    # Yellow background lines
    xr = (-3.0, 3.0)
    gplt.plot_lines(ab[:, 0], ab[:, 1], xr)

    # 2. Scatter Points (ScatterLayer)
    m = 500
    pts = np.random.uniform(-3, 3, (m, 2)).astype(np.float32)
    gplt.scatter(pts[:, 0], pts[:, 1], size=8.0, color=(1, 0.2, 0.2, 1))

    # 3. Connected Path (PolylineLayer)
    k = 100
    t = np.linspace(0, 10, k)
    x = t * np.cos(t) / 3.0
    y = t * np.sin(t) / 3.0
    gplt.plot(x, y, color=(0.2, 0.8, 0.2, 1))

    gplt.title("Phase 4: Simple Primitives Verification")
    gplt.show()


if __name__ == "__main__":
    test_phase4_primitives()
