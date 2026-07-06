# GLPlot

High-performance, GPU-accelerated plotting library in Python, designed to handle **millions** of lines effortlessly. It provides an API similar to Matplotlib but runs natively over an OpenGL/GLFW backend, performing instanced rendering directly on the GPU.

## Features
- **Matplotlib-like API Compatibility**: Familiar `figure`, `subplots`, `plot`, `scatter`, `plot3d`, `scatter3d`, `plot_surface`, `plot_wireframe`, `bar3d`, `quiver3d`, `bar`, `hist`, `hist2d`, `imshow`, `matshow`, `pcolormesh`, `contour`, `contourf`, `fill_between`, `step`, `errorbar`, `stem`, `arrow`, `quiver`, `annotate`, `hlines`, `vlines`, `axhline`, `axvline`, `axline`, `ssao`, `xlim`, `ylim`, `axis`, `title`, `xlabel`, `ylabel`, `zlabel`, `grid`, `legend`, `show`, and `savefig`.
- **Matplotlib format strings**: Supports common forms such as `plot(y)`, `plot(x, y)`, `plot(x, y, "r-o")`, and repeated groups like `plot(x1, y1, "r-", x2, y2, "bo")`.
- **Phase Diagram Optimized (`plot_lines`)**: Explicitly supports passing millions of line parameters $(a, b)$ to calculate functions $y = ax + b$ securely bounded to bounds using shader math.
- **Logarithmic Density Heaps**: By displaying overlaps, `density=True` handles millions of parallel curves seamlessly for heatmaps.
- **Dynamic Camera**: Drag to pan, scroll to zoom with on-the-fly resolution subsampling.

## Installation

You can install this locally:

```bash
pip install .
```

Requirements: `numpy`, `glfw`, `PyOpenGL`, `scipy`, `matplotlib`.

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

## Testing

Uses `pytest` for unit and integration coverage without popping visible windows:
```bash
pytest
```
