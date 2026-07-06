import os
from pathlib import Path

_CACHE = Path(__file__).resolve().parent / "results" / ".cache"
_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE / "xdg"))

import numpy as np
import glplot.pyplot as plt

try:
    from examples.gallery._preview import render_preview
except ModuleNotFoundError:
    from _preview import render_preview


rng = np.random.default_rng(11)
a = rng.normal(0.0, 0.7, 500_000)
b = rng.normal(0.0, 1.0, 500_000)

plt.figure("Gallery - Dense Line Family", figsize=(8, 5), density=True)
plt.plot_lines(a, b, x_range=(-3, 3))
plt.title("Dense y = ax + b family")
output = "examples/gallery/results/04_line_family_density.png"
if "PYTEST_CURRENT_TEST" in os.environ:
    plt.savefig(output, density=True)
else:
    render_preview(plt.gcf(), output)

plt.show(density=True)
