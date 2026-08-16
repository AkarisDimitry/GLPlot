# GLPlot: GPU-Accelerated Plotting for Python

[![Tests](https://github.com/AkarisDimitry/GLPlot/workflows/Tests/badge.svg)](https://github.com/AkarisDimitry/GLPlot/actions/workflows/tests.yml)
[![Lint](https://github.com/AkarisDimitry/GLPlot/workflows/Lint/badge.svg)](https://github.com/AkarisDimitry/GLPlot/actions/workflows/lint.yml)
[![Build](https://github.com/AkarisDimitry/GLPlot/workflows/Build/badge.svg)](https://github.com/AkarisDimitry/GLPlot/actions/workflows/build.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/Python-3.9+-blue.svg)](pyproject.toml)

GLPlot is a Python plotting library with a Matplotlib-like API that renders on the GPU
through OpenGL instead of the CPU. Plots that make Matplotlib chug — millions of points,
dense line families, large 3D scenes — stay interactive: smooth pan, zoom, and rotation,
even at that scale. If you already know `plt.plot()` / `plt.scatter()`, most of what you
know carries over directly.

## Install

```bash
pip install glplot
```

or from source:

```bash
git clone https://github.com/AkarisDimitry/GLPlot.git
cd GLPlot
pip install -e .
```

Requires Python 3.9+. Core dependencies: numpy, scipy, matplotlib, glfw, PyOpenGL,
`imgui[glfw]` (for the on-screen control panel).

## 30-second example

```python
import numpy as np
import glplot.pyplot as plt

x = np.linspace(0, 10, 100)

plt.figure("My Plot", figsize=(8, 5))
plt.plot(x, np.sin(x), "r-", lw=2, label="sin(x)")
plt.scatter(x[::10], np.sin(x[::10]), c="blue", s=20)
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()
```

That's it — a real window opens, fully interactive (pan/zoom/rotate) from the first frame.

## What it looks like

| | |
|---|---|
| ![Scatter Fill](examples/gallery/results/02_scatter_fill.png) 10M-point spiral scatter | ![3D Cloud](examples/gallery/results/07_projected_3d_cloud.png) 1M-point 3D point cloud |
| ![Massive Density](examples/gallery/results/10_massive_hist2d_density.png) 10M-sample 2D density histogram | ![Volumetric Nebula](examples/gallery/results/13_volumetric_nebula.png) 1.75M-point volumetric nebula |
| ![3D Vector Field](examples/gallery/results/19_turbulent_vector_field_3d.png) 3D turbulent vector field | ![Chladni animation](examples/gallery/results/28_chladni_wave_animation.gif) Animated standing-wave pattern, `glplot.animation.FuncAnimation` |

**All of the above render at 60+ FPS** with interactive panning, zooming, and rotation,
regardless of point count. More in the [example gallery](examples/gallery/README.md)
(28 scripts) and the [showcase](examples/showcase/README.md) (four ~10-line demos).

## Features

- **Matplotlib-compatible API** — `plot`, `scatter`, `bar`, `hist`, `hist2d`, `imshow`,
  `contour`/`contourf`, `quiver`, format strings (`"r-o"`, `"b--"`), and more
- **Millions of points, still interactive** — GPU instancing and density accumulation
  instead of CPU-side geometry construction
- **Full 2D/3D** — lines, scatter, filled regions, bars, histograms, matrices, surfaces,
  wireframes, 3D bars, vector fields, with SSAO depth shading in 3D
- **Real multi-panel subplots** — `plt.subplots()`, per-panel interaction
- **Animation** — `glplot.animation.FuncAnimation`/`ArtistAnimation`, exportable to GIF/video
- **A live control panel** — `python -m glplot` opens a workstation for editing a scene,
  its layers, and its styling interactively, with undo history
- **Numerically stable at extreme zoom** — double-precision, viewport-relative coordinate
  transforms avoid the jitter that single-precision GPU pipelines show at large offsets

## More examples

**Filled regions, bars, and histograms**
```python
import numpy as np
import glplot.pyplot as plt

x = np.linspace(-3, 3, 250)
y = np.exp(-0.5 * x**2)

plt.figure("Common charts")
plt.fill_between(x, y, 0, color="tab:blue", alpha=0.25)
plt.plot(x, y, "b-", lw=2)
plt.scatter(x[::10], y[::10], c="tab:orange", s=20)
plt.show()
```

**3D scatter**
```python
import numpy as np
import glplot.pyplot as plt

t = np.linspace(0, 16 * np.pi, 100000)
x, y, z = (0.05 * t) * np.cos(t), (0.05 * t) * np.sin(t), 0.05 * t

plt.figure("Projected 3D")
plt.scatter3d(x, y, z, c=z, cmap="turbo", s=1.5)
plt.show()
```

**Readable 3D bars**
```python
import numpy as np
import glplot.pyplot as plt

x, y = np.meshgrid(np.arange(30), np.arange(30))
height = 1 + 4 * np.sin(x * 0.2) ** 2 * np.cos(y * 0.15) ** 2

plt.figure("3D Bars", ssao=True)
plt.bar3d(
    x.ravel(), y.ravel(), np.zeros(x.size),
    1, 1, height.ravel(),
    c=height.ravel(), cmap="turbo",
    gap=0.15, edge_color=(0, 0, 0, 0.75), edge_width=0.7, ssao=True,
)
plt.show()
```

**Massive 2D density**
```python
import numpy as np
import glplot.pyplot as plt

rng = np.random.default_rng(0)
x = rng.normal(size=1_000_000)
y = 0.5 * x + rng.normal(size=1_000_000)

plt.figure("Massive Density")
plt.hist2d(x, y, bins=350, cmap="inferno")
plt.show()
```

**A million lines at once (`plot_lines`)**

Ordinary plotting draws each line as CPU-generated geometry — a million calls to `plot()`
would build a million separate meshes. `plot_lines` instead uploads each line as an
`(a, b)` coefficient pair and lets the GPU work out what's visible:

```python
import numpy as np
import glplot.pyplot as plt

n = 1_000_000
a = np.random.randn(n)
b = np.random.randn(n)

plt.figure("Density")
plt.plot_lines(a, b, x_range=(-2, 2))
plt.show(density=True)
```

## How it compares

| Feature | GLPlot | Matplotlib | Plotly | Datashader | VisPy |
|---|---|---|---|---|---|
| GPU acceleration | ✓ (OpenGL) | ✗ | ✗ | ✓ | ✓ |
| Matplotlib-style API | ✓ | ✓ | ✗ | ✗ | ✗ |
| Millions of points, interactive | ✓ | ✗ | Limited | ✓ | ✓ |
| Interactive 3D | ✓ | Limited | ✓ | Limited | ✓ |
| Density visualization | ✓ (HDR) | Basic | Limited | ✓ | ✗ |
| Precision at extreme zoom | ✓ (double precision) | ✓ | Basic | ✓ | ✗ |

GLPlot sits between "familiar API, CPU-bound" (Matplotlib) and "GPU-fast, low-level"
(VisPy): a Matplotlib-shaped surface backed by a GPU renderer.

## Rendering architecture

GLPlot runs two rendering pipelines — one for 2D primitives (lines, scatter, density) and
one for 3D geometry (bars, surfaces, wireframes, `scatter3d`) — both driven from the CPU
but doing their actual work on the GPU. `glReadPixels` is only ever called for export; the
interactive path never reads pixels back to the CPU.

![GLPlot rendering pipeline](examples/dataflow.png)

See [GLPlot_Architecture_and_Mathematical_Formulation.md](GLPlot_Architecture_and_Mathematical_Formulation.md)
for the full derivation of each stage, including the density-accumulation math and the
viewport-relative projection that keeps zoom numerically stable.

## Testing

```bash
pytest                                   # run everything
pytest --cov=glplot --cov-report=html    # with coverage
pytest tests/test_pyplot.py::test_plot_accepts_y_only_and_returns_artists  # one test
```

6,800+ tests, run fully headless (no window is ever displayed) so they work in CI. The
matrix covers Python 3.9–3.12 on Ubuntu, macOS, and Windows; `black`, `isort`, and `flake8`
are enforced on every push. See `examples/benchmark/` for reproducible performance
comparisons against Matplotlib, VisPy, fastplotlib, Datashader, and hvPlot, and `tools/`
for GPU/environment diagnostics.

## Documentation

- API reference: docstrings in `glplot.pyplot`, or the built docs — see [docs/README.md](docs/README.md)
- Architecture: [GLPlot_Architecture_and_Mathematical_Formulation.md](GLPlot_Architecture_and_Mathematical_Formulation.md)
- Dev tools: [tools/README.md](tools/README.md)
- Contributing: [CONTRIBUTING.md](CONTRIBUTING.md) · Code of conduct: [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)

## Citation

```bibtex
@software{lombardi2026glplot,
  title={GLPlot: High-Performance GPU-Accelerated Plotting Library for Python},
  author={Lombardi, Juan Manuel},
  year={2026},
  url={https://github.com/AkarisDimitry/GLPlot},
  doi={10.5281/zenodo.PLACEHOLDER}
}
```

See [CITATION.cff](CITATION.cff) for other formats.

## License

[MIT](LICENSE).

## Acknowledgments

Built on PyOpenGL, GLFW, NumPy, SciPy, Matplotlib, and Dear ImGui — thanks to those
communities for the foundations this sits on.
