---
title: 'GLPlot: High-Performance GPU-Accelerated Scientific Plotting for Python'
tags:
  - Python
  - plotting
  - visualization
  - GPU acceleration
  - OpenGL
  - scientific computing
authors:
  - name: Juan Manuel Lombardi
    affiliation: 1
affiliations:
  - name: Fritz Haber Institute of the Max Planck Society
    index: 1
date: 06 July 2026
bibliography: paper.bib
---

## Summary

GLPlot is a high-performance, GPU-accelerated plotting library for Python that enables interactive visualization of large-scale scientific datasets. Designed with a Matplotlib-compatible API, GLPlot leverages OpenGL/GLFW backend rendering to achieve superior performance when handling millions of geometric primitives. The library introduces novel algorithms for analytical line-family shader expansion, viewport-relative center projection for precision preservation during extreme zoom operations, and high-dynamic-range (HDR) density accumulation for visualizing massive overlapping datasets.

## Statement of Need

Interactive visualization is crucial in scientific computing, yet traditional CPU-bound plotting libraries (such as Matplotlib) fail to maintain smooth frame rates (60+ FPS) when rendering more than 10^5 geometric primitives. Existing GPU-accelerated alternatives exhibit steep learning curves (requiring manual GPU memory management) or precision limitations during extreme zoom operations. GLPlot addresses these challenges by providing a familiar, Matplotlib-like interface backed by efficient GPU computation.

## Key Features

- **Matplotlib-Compatible API**: Familiar function signatures (`plot`, `scatter`, `hist`, `bar`, `bar3d`, `quiver`, etc.) reduce adoption friction for existing scientists
- **Massive Dataset Handling**: Efficiently renders millions of geometric elements through instanced rendering and density visualization
- **Novel GPU Algorithms**: 
  - Analytical line-family shader expansion for phase diagrams
  - Viewport-relative center projection preventing floating-point catastrophic cancellation
  - HDR density accumulation for statistical visualization
- **Interactive Controls**: Smooth panning, zooming, and rotation with on-the-fly level-of-detail adaptation
- **3D Visualization**: Full 3D point cloud, surface, wireframe, and volumetric rendering support
- **Comprehensive Gallery**: 19+ example scripts demonstrating scientific use cases

## Technical Innovation

GLPlot's core innovation lies in its GPU-side analytical geometry expansion. Rather than pre-computing line segments in CPU memory, the library transfers only line coefficients $(a_i, b_i)$ to the GPU and computes visible segments analytically within the vertex shader. This dramatically reduces memory overhead when visualizing families of curves, such as phase diagrams containing millions of state trajectories.

Additionally, GLPlot implements viewport-relative center projection to prevent floating-point precision loss during extreme zooming (sub-micron scales on large offsets), a critical feature for scientific applications in physics and microscopy.

## Architecture

GLPlot adopts a modular, decoupled design:
- **Frontend**: High-level `glplot.pyplot` API
- **Control Plane**: Scene graph, rendering policy manager, and level-of-detail decisions
- **Rendering Pipeline**: Multi-pass post-processing with HDR tone-mapping
- **GPU Subsystems**: Primitive renderers, picking, and interaction handling

The engine operates reactively: if scene and camera remain unchanged, rendering halts and the thread sleeps, reducing idle resource consumption to near zero.

## Usage Example

```python
import numpy as np
import glplot.pyplot as plt

# Generate high-resolution data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Create and display plot
plt.figure("Sine Wave", figsize=(8, 5))
plt.plot(x, y, "r-", lw=2, label="sin(x)")
plt.xlabel("x")
plt.ylabel("value")
plt.title("High-Performance Plotting")
plt.legend()
plt.show()
```

For massive datasets:

```python
# Visualize 1 million samples
N = 1_000_000
x = np.random.randn(N)
y = 0.5 * x + np.random.randn(N)

plt.figure("Massive Density")
plt.hist2d(x, y, bins=350, cmap="inferno")
plt.show(density=True)
```

## Testing and Validation

The project includes a comprehensive test suite covering:
- Core plotting functionality (65+ tests)
- Camera and coordinate transformation math
- 3D geometry rendering
- Rendering pipeline robustness
- Gallery integration tests ensuring all examples build successfully

All tests run headless without displaying windows, enabling CI/CD integration.

## Availability

- **Repository**: https://github.com/AkarisDimitry/GLPlot
- **License**: MIT
- **Installation**: `pip install glplot`
- **Documentation**: See README.md and architecture documentation
- **Examples**: 19 gallery scripts covering scientific use cases

## Acknowledgments

This project builds on foundational work in GPU-accelerated rendering and modern OpenGL practices. We acknowledge the contributions of the open-source community, particularly the developers of PyOpenGL, GLFW, NumPy, and ImGui.

## References
