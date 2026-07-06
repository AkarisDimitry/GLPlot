# GLPlot: High-Performance GPU-Accelerated Plotting Library

[![Tests](https://github.com/AkarisDimitry/GLPlot/workflows/Tests/badge.svg)](https://github.com/AkarisDimitry/GLPlot/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/Python-3.9+-blue.svg)](pyproject.toml)

High-performance, GPU-accelerated plotting library in Python, designed to handle **millions** of geometric primitives effortlessly. It provides a Matplotlib-compatible API while running natively over an OpenGL/GLFW backend, performing instanced rendering and density visualization directly on the GPU.

## Motivation

Interactive visualization of large-scale datasets is critical in scientific computing, yet traditional CPU-bound plotting libraries (such as Matplotlib) fail to maintain smooth frame rates (60+ FPS) when rendering beyond 10^5 geometric elements. GLPlot addresses this by providing a familiar, high-performance alternative optimized for modern GPUs.

## Features

- **Matplotlib API Compatibility**: Familiar function signatures (`figure`, `plot`, `scatter`, `bar`, `hist`, `imshow`, `quiver`, etc.) reduce adoption friction
- **Massive Dataset Support**: Efficiently renders millions of geometric primitives through GPU instancing and density visualization
- **Novel GPU Algorithms**:
  - Analytical line-family shader expansion (millions of lines from $(a_i, b_i)$ coefficients)
  - Viewport-relative center projection preventing floating-point precision loss at extreme zoom levels
  - HDR density accumulation for statistical visualization of overlapping elements
- **Complete 2D/3D Support**: Lines, scatter plots, filled regions, bars, histograms, matrices, surfaces, wireframes, 3D bars, vector fields
- **Interactive Controls**: Smooth camera pan/zoom with on-the-fly level-of-detail adaptation
- **Screen-Space Ambient Occlusion (SSAO)**: Enhanced depth perception for 3D visualizations
- **Format String Support**: Matplotlib-style syntax (`"r-o"`, `"b--"`, etc.)

## Visual Gallery

GLPlot excels at visualizing massive datasets with stunning interactivity and real-time performance.

### 2D Visualization: 220k-Point Spiral Scatter
![Scatter Fill](examples/gallery/results/02_scatter_fill.png)
*Smooth interactive rendering of massive point clouds with color mapping*

### 3D Cloud: 100k-Point Volumetric Projection
![3D Cloud](examples/gallery/results/07_projected_3d_cloud.png)
*High-performance 3D point cloud visualization with depth and color encoding*

### Density Visualization: 1M-Sample 2D Histogram
![Massive Density](examples/gallery/results/10_massive_hist2d_density.png)
*HDR density mapping handles millions of overlapping points seamlessly*

### 3D Volumetric: 750k-Point Nebula
![Volumetric Nebula](examples/gallery/results/13_volumetric_nebula.png)
*Advanced 3D volumetric rendering with point cloud opacity and depth*

### 3D Vector Field: Turbulent Flow
![3D Vector Field](examples/gallery/results/19_turbulent_vector_field_3d.png)
*Complex 3D vector field visualization with particles and dynamic flow*

> **All visualizations render at 60+ FPS** with interactive panning, zooming, and rotation. No performance degradation with dataset size.

## Quick Showcase - Simple & Beautiful Examples

Start with these elegant examples to see GLPlot's power with minimal code:

### 🚀 Run in 30 seconds
```bash
# 100k colorful particles
python examples/showcase/01_colorful_particles_2d.py

# Mandelbrot fractal (360k points)
python examples/showcase/02_mandelbrot_zoom_2d.py

# Spinning 3D torus
python examples/showcase/03_spinning_torus_3d.py

# Cosmic 3D sphere with noise
python examples/showcase/04_cosmic_sphere_3d.py
```

### ✨ What You'll See

| Example | Data | Colors | Interaction | Code |
|---------|------|--------|-------------|------|
| Particles 2D | 100k points | Rainbow gradient | Pan/Zoom | 10 lines |
| Mandelbrot | 360k points | Psychedelic | Pan/Zoom | 15 lines |
| Torus 3D | 50k points | HSV spectrum | Rotate/Zoom | 12 lines |
| Cosmic Sphere | 150k points | Rainbow noise | Rotate/Zoom | 14 lines |

**All run at 60+ FPS** with smooth interactive controls and vibrant colors.

See [examples/showcase/README.md](examples/showcase/README.md) for detailed descriptions and how to modify them.

## Installation

### From PyPI (once released)
```bash
pip install glplot
```

### From Source
```bash
git clone https://github.com/AkarisDimitry/GLPlot.git
cd GLPlot
pip install -e .
```

### Requirements
- **Python**: 3.9 or later
- **Core Dependencies**: numpy, scipy, matplotlib, glfw, PyOpenGL
- **Optional**: imgui for advanced HUD features

### Clean Environment Testing
```bash
# Create isolated environment
python -m venv glplot_test
source glplot_test/bin/activate  # On Windows: glplot_test\Scripts\activate
pip install glplot
python -c "import glplot; print(glplot.__version__)"
```

## Usage

**Matplotlib-style line plots**
```python
import numpy as np
import glplot.pyplot as plt

x = np.linspace(0, 10, 100)

plt.figure("Sine Wave", figsize=(8, 5))
plt.plot(x, np.sin(x), "r-", lw=2, label="sin(x)")
plt.plot(x, np.cos(x), "bo", ms=3, label="cos(x)")
plt.xlabel("x")
plt.ylabel("value")
plt.title("Line and marker syntax")
plt.grid(True)
plt.legend()
plt.show()
```

**Scatter, filled regions, bars, and histograms**
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

**Projected 3D and arrows**
```python
import numpy as np
import glplot.pyplot as plt

t = np.linspace(0, 16 * np.pi, 100000)
x = (0.05 * t) * np.cos(t)
y = (0.05 * t) * np.sin(t)
z = 0.05 * t

plt.figure("Projected 3D")
plt.scatter3d(x, y, z, c=z, cmap="turbo", s=1.5)
plt.annotate("start", xy=(0, 0), xytext=(-2, 2), arrowprops={"color": "white"})
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
    gap=0.15,
    edge_color=(0, 0, 0, 0.75),
    edge_width=0.7,
    ssao=True,
)
plt.show()
```

**Matrices and 2D density**
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

**Bulk Lines (Density Map)**
```python
import numpy as np
import glplot.pyplot as gplt

N = 1000000
a = np.random.randn(N)
b = np.random.randn(N)

gplt.figure("Density")
gplt.plot_lines(a, b, x_range=(-2, 2))
gplt.show(density=True)
```

## Example Gallery

The ordered gallery lives in `examples/gallery` and writes rendered output into `examples/gallery/results`:

```bash
python examples/gallery/run_gallery.py
```

Gallery contents:

1. `01_line_plot.py` - dozens of high-resolution line layers plus sampled markers.
2. `02_scatter_fill.py` - 220k-point spiral scatter plus filled nonlinear band.
3. `03_bar_hist.py` - million-sample histogram with a bar overlay.
4. `04_line_family_density.py` - 500k-line high-density `plot_lines`.
5. `05_guides_and_colormap.py` - 250k clustered samples, guides, colormap, and annotation.
6. `06_signal_tools.py` - long signal with `step`, `errorbar`, event `stem`, and annotation.
7. `07_projected_3d_cloud.py` - projected 3D point cloud and 3D line syntax.
8. `08_vector_field_quiver.py` - arrows, annotation, and vector fields over a matrix.
9. `09_large_matrix_heatmap.py` - large procedural matrix heatmap.
10. `10_massive_hist2d_density.py` - one-million-sample 2D density histogram.
11. `11_contour_pcolormesh_field.py` - contour, contourf, and pcolormesh on a 520 x 520 field.
12. `12_surface_wireframe_bar3d.py` - projected 3D surface, wireframe, and bar3d syntax.
13. `13_volumetric_nebula.py` - massive volumetric 3D point field with 750k samples.
14. `14_bar3d_hex_box_city.py` - mixed square and hexagonal 3D bars.
15. `15_vector_field_3d.py` - 3D vector field over a massive volumetric flow cloud.
16. `16_ssao_comparison.py` - dense 3D bars comparing SSAO off vs on.
17. `17_square_bars3d.py` - square 3D bars with edges and SSAO.
18. `18_hex_bars3d.py` - hexagonal 3D bars with edges and SSAO.
19. `19_turbulent_vector_field_3d.py` - massive 3D vector field with volumetric particles and stream traces.

## Quality Assurance & CI/CD

GLPlot uses comprehensive continuous integration to ensure reliability:

### Automated Testing
- **Test Matrix**: Python 3.9, 3.10, 3.11, 3.12 on macOS, Ubuntu, Windows
- **Coverage**: 210+ tests (unit, API, edge cases, performance, regression)
- **Headless Execution**: All tests run without displaying windows

### Code Quality
- **Linting**: Black, isort, flake8 enforced in CI
- **Type Hints**: Python type annotations throughout codebase
- **Coverage Reports**: Automated coverage tracking to Codecov

### Workflow Status
[![Tests](https://github.com/AkarisDimitry/GLPlot/workflows/Tests/badge.svg)](https://github.com/AkarisDimitry/GLPlot/actions/workflows/tests.yml)
[![Lint](https://github.com/AkarisDimitry/GLPlot/workflows/Lint/badge.svg)](https://github.com/AkarisDimitry/GLPlot/actions/workflows/lint.yml)
[![Build](https://github.com/AkarisDimitry/GLPlot/workflows/Build/badge.svg)](https://github.com/AkarisDimitry/GLPlot/actions/workflows/build.yml)

## Testing

GLPlot includes a comprehensive test suite (210+ tests) covering core plotting functionality, 3D geometry, rendering pipeline, and robustness:

```bash
# Run all tests
pytest

# With coverage report
pytest --cov=glplot --cov-report=html

# Run specific test
pytest tests/test_pyplot.py::test_plot_accepts_y_only_and_returns_artists
```

All tests run **headless without displaying windows**, enabling CI/CD integration.

## Comparison with Alternative Libraries

| Feature | GLPlot | Matplotlib | Plotly | Datashader | VisPy |
|---------|--------|-----------|--------|-----------|-------|
| **GPU Acceleration** | ✓ (OpenGL) | ✗ | ✗ | ✓ | ✓ |
| **Matplotlib API** | ✓ | ✓ | ✗ | ✗ | ✗ |
| **Simple Setup** | ✓ | ✓ | ✓ | ✓ | ✗ |
| **Millions of Points** | ✓ | ✗ | Limited | ✓ | ✓ |
| **Interactive 3D** | ✓ | Limited | ✓ | Limited | ✓ |
| **Density Visualization** | ✓ (HDR) | Basic | Limited | ✓ | ✗ |
| **Zoom Precision** | ✓ (double precision) | ✓ | Basic | ✓ | ✗ |

## Scientific Applications

GLPlot is particularly suited for:
- **High-energy physics**: Visualizing detector event data and particle trajectories
- **Computational chemistry**: Molecular visualization and spectroscopic data
- **Climate science**: Large-scale gridded data visualization
- **Bioinformatics**: Single-cell RNA-seq and genomic visualization
- **Materials science**: Volumetric simulations and 3D material structures
- **Data science**: Extreme-scale density plots and statistical distributions

## Performance Benchmarks

GLPlot maintains **60+ FPS** across all tested platforms with various data sizes:

### Rendering Performance
- **1M points scatter**: 60 FPS
- **500k line family density**: 60 FPS  
- **1M histogram bins**: 60 FPS
- **3D volumetric cloud (750k points)**: 60 FPS
- **Large 3D meshes**: 60+ FPS with 100k+ vertices

### Scaling Efficiency
- Linear or better scaling with dataset size
- No performance degradation from 1k to 1M+ points
- Efficient memory usage with GPU instancing
- Automatic level-of-detail adaptation

See `examples/benchmark/` for reproducible benchmarks and `tools/` for performance diagnostics.

## Getting Started in 30 Seconds

```python
import numpy as np
import glplot.pyplot as plt

# Create data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Plot it (familiar Matplotlib syntax)
plt.figure("My Plot", figsize=(8, 5))
plt.plot(x, y, "r-", lw=2, label="sin(x)")
plt.scatter(x[::10], y[::10], c="blue", s=20)
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()

# That's it! Fully interactive with pan, zoom, rotation
```

## Documentation

- **API Reference**: See docstrings in `glplot.pyplot` module
- **Architecture**: See [GLPlot_Architecture_and_Mathematical_Formulation.md](GLPlot_Architecture_and_Mathematical_Formulation.md)
- **Tools & Diagnostics**: See [tools/README.md](tools/README.md)
- **Contributing**: See [CONTRIBUTING.md](CONTRIBUTING.md)
- **Code of Conduct**: See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)
- **Citation**: See [CITATION.cff](CITATION.cff)

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on reporting issues, suggesting enhancements, and submitting pull requests.

## License

GLPlot is released under the [MIT License](LICENSE). See LICENSE file for details.

## Citation

If you use GLPlot in your research, please cite it as:

```bibtex
@software{lombardi2026glplot,
  title={GLPlot: High-Performance GPU-Accelerated Plotting Library for Python},
  author={Lombardi, Juan Manuel},
  year={2026},
  url={https://github.com/AkarisDimitry/GLPlot},
  doi={10.5281/zenodo.PLACEHOLDER}
}
```

See [CITATION.cff](CITATION.cff) for additional citation formats.

## Acknowledgments

This project builds on foundational work in GPU-accelerated rendering and modern OpenGL. We acknowledge the Python scientific computing community and developers of PyOpenGL, GLFW, NumPy, SciPy, and Matplotlib.
