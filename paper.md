---
title: "GLPlot: GPU-accelerated interactive scientific visualization with a Matplotlib-like interface"
tags:
  - Python
  - scientific visualization
  - GPU rendering
  - OpenGL
  - large-scale plotting
  - interactive visualization
authors:
  - name: Juan Manuel Lombardi
    corresponding: true
    affiliation: 1
    # orcid: 0000-0000-0000-0000  # TODO: Add the author's ORCID.
affiliations:
  - name: "TODO: Add the author's institute, department/research group, city, and country"
    index: 1
    # ror: "TODO"  # TODO: Add the institutional ROR identifier, if available.
date: 6 July 2026
bibliography: paper.bib
---

# Summary

Interactive visualization is part of the scientific reasoning process: researchers use plots to inspect convergence, identify outliers, compare simulations, assess uncertainty, and decide which analyses or experiments to perform next. The complexity of these tasks rapidly escalates when a figure contains a large number of points, vertices, curves, or other geometric primitives. Conventional plotting workflows may incur substantial costs from CPU-side geometry construction, per-object draw calls, repeated rasterization, and transfer of expanded geometry to the graphics device.

Here, we propose `GLPlot`, an open-source Python plotting package that exposes a familiar, `pyplot`-style interface while using an OpenGL rendering engine in the backend. It is intentionally developed to support common two- and three-dimensional scientific visualizations, including polylines, scatter plots, histograms, matrix images, contour and pseudocolor plots, vector fields, surfaces, meshes. A more dedicated `plot_lines` interface represents large families of straight lines by their slope and intercept coefficients and expands their visible geometry analytically on the GPU. This latter feature was developed [girorun]

The rendering engine separates scene representation from GPU execution. It combines event-driven updates, shader-based geometry generation, stable viewport transformations, hardware clipping, density accumulation, adaptive level-of-detail control, and GPU-based picking. These operations are exposed to the user through a high-level Python interface, allowing large scientific datasets to be explored without direct interaction with low-level OpenGL components.

# Statement of need

GPU-oriented visualization libraries can provide high rendering throughput, but they often require users to adopt a different programming model or work closer to graphics concepts than is desirable in routine scientific analysis [@vispy; @fastplotlib]. Conversely, pixel-based aggregation systems efficiently summarize very large datasets, but do not necessarily preserve the semantics or interactive selection of individual geometric objects [@datashader; @stevens2015holoviews].
`GLPlot` aims at targeting the space between these approaches. The intended users are computational scientists, engineers, and data analysts who require interactive rendering of large two- and three-dimensional datasets and yet retaining a concise interface close to standard plotting libraries, such as Matplotlib idiom. The software is particularly suited to cases where the plotted entities remain meaningful geometric objects, such as line families in chemical-potential, feasibility diagrams, trajectory ensembles, vector fields, meshes, and selectable point clouds.
The design is guided by five requirements:

1. avoid CPU-side geometry expansion when equivalent geometry can be generated analytically on the GPU;
2. maintain stable coordinate transforms during deep zoom operations at large coordinate offsets;
3. preserve density information when many primitives overlap;
4. avoid unnecessary CPU and GPU work when a scene is unchanged; and
5. support interactive inspection and selection without CPU searches that scale linearly with the number of rendered objects.

# State of the field
Scientific visualization tools span a broad range of abstraction levels and rendering strategies. General-purpose plotting libraries prioritize flexibility and publication-quality output, GPU-oriented frameworks expose programmable rendering pipelines, and large-data visualization systems aggregate observations into display-resolution representations. These approaches address different requirements and therefore provide complementary rather than interchangeable solutions.

Matplotlib is the reference general-purpose visualization library for scientific Python, supporting static, animated, and interactive figures across a broad range of plotting and publication workflows [@hunter2007matplotlib]. Its artist-based architecture and extensive layout, annotation, and backend systems make it appropriate for most conventional scientific figures. However, performance can deteriorate when a figure contains very large numbers of individual primitives, because geometry construction, artist management, coordinate transformation, and redraw operations are largely coordinated on the CPU. This limitation becomes particularly relevant for dense line families, large point clouds, trajectory ensembles, and complex three-dimensional scenes requiring repeated interactive updates.

VisPy provides high-performance interactive two- and three-dimensional visualization through an OpenGL-based rendering framework [@vispy]. Its interfaces range from relatively low-level graphics abstractions, including buffers, shaders, and rendering programs, to higher-level scene graphs and reusable visual components. This flexibility makes VisPy well suited to the development of custom visualization applications and specialized rendering techniques. At the same time, achieving highly specialized behaviour may require users to work directly with graphics concepts or implement custom visual components, which can impose a substantial development burden for routine scientific plotting.

Datashader addresses large-data visualization through rasterization and aggregation into the finite pixel grid of the display [@datashader]. This strategy is highly effective when the primary objective is to preserve the statistical or spatial distribution of a dataset despite severe overplotting. Because the resulting representation is defined at the level of aggregate cells or image pixels, however, the original geometric objects are not necessarily retained as persistent elements of an interactive scene. This limits workflows in which individual lines, trajectories, points, or surfaces must remain separately identifiable, selectable, or subject to object-specific styling.

PyQtGraph is designed for scientific and engineering applications requiring responsive desktop graphics, data-acquisition displays, and rapidly updated signals [@pyqtgraph]. Its integration with Qt makes it particularly suitable for instrumentation, monitoring interfaces, and custom graphical applications. Its principal strengths lie in fast two-dimensional visualization and application development, whereas highly specialized three-dimensional rendering, analytical shader-side primitive generation, density accumulation, and large-scale GPU-based object selection are not its central design focus.

GLPlot targets the gap between these approaches. It retains a concise procedural interface close to Matplotlib’s pyplot idiom while adopting a GPU-resident rendering architecture for large two- and three-dimensional scientific scenes. Its design combines analytical shader-side generation of specialized primitives, numerically stable coordinate transformations during deep zoom, density-aware rendering of overlapping geometry, suppression of unnecessary redraws, interaction caching, and GPU-based object picking. GLPlot is therefore intended for workloads in which large numbers of geometric objects must remain individually represented, interactive, and semantically identifiable without requiring users to construct a custom graphics pipeline.
# Software design
GLPlot implements a rendering pipeline in which numerical plotting data are converted into OpenGL-ready buffers on the CPU, while the expensive per-frame operations are delegated to the GPU. For 2D data, line coordinates or analytical functions are staged as contiguous float32 vertex-buffer objects and rendered through shader-based wide-line and density-accumulation paths, including alpha blending, logarithmic density normalization, and colormap resolution. For 3D data, GLPlot constructs geometry buffers and model–view–projection transformations on the CPU, then performs perspective scaling, depth testing, fragment shading, ambient-occlusion/rim enhancement, transparency composition, and axis-overlay rendering on the GPU. Importantly, CPU readback is restricted to explicit export operations through glReadPixels, avoiding unnecessary host–device transfers during interactive rendering.

![](https://pad.gwdg.de/uploads/a6bbc8a8-1a80-49d6-a0d4-c548d2c09479.png)
*Figure 1. GLPlot OpenGL rendering pipeline for 2D and 3D visualization. The schematic separates CPU-side data preparation from GPU-side rendering operations. In the 2D pipeline, GLPlot supports both direct line rendering and a density-rendering branch based on additive accumulation, density normalization, and colormap resolution. In the 3D pipeline, geometry construction and MVP-matrix preparation are followed by GPU-side vertex transformation, perspective-dependent point sizing, depth testing, fragment shading, SSAO/rim enhancement, alpha blending, and guide-overlay composition. Pixel transfer back to the CPU is performed only for image export, while the rendering workload remains on the GPU..*


### User-facing interface

The primary entry point is `glplot.pyplot`. Calls such as `plot`, `scatter`, `hist2d`, `pcolormesh`, `contour`, `quiver`, `plot_surface`, `mesh3d`, and `bar3d` create layers in a scene. The API accepts NumPy arrays and follows common Matplotlib calling conventions, including format strings for basic line and marker styling. NumPy supplies the array model used at the interface [@harris2020array], while SciPy is used by selected numerical operations [@virtanen2020scipy].
A specialized interface,

```python
import numpy as np
import glplot.pyplot as plt

n_lines = 1_000_000
slopes = np.random.default_rng(0).normal(size=n_lines)
intercepts = np.random.default_rng(1).normal(size=n_lines)

plt.figure()
plt.plot_lines(
    slopes,
    intercepts,
    x_range=(-5.0, 5.0),
    density=True,
    gain=0.8,
    cmap="magma",
)
plt.xlabel("x")
plt.ylabel("a x + b")
plt.show()
```

represents each line as a coefficient pair rather than as CPU-generated endpoints. This data model reduces host-side geometry construction and allows the active viewport to determine the visible segment in the vertex shader.
### Rendering architecture

The front end stores data arrays, styles, coordinate transforms, visibility, and renderer-specific metadata as scene layers. A rendering manager maps each layer to a specialized primitive renderer and selects the required framebuffer and post-processing passes. This separation permits the rendering policy to change without changing the user-facing plotting calls.

The event loop is reactive. Layers, camera transforms, styles, and framebuffer-dependent effects carry dirty state. When no state has changed, the window loop waits for events rather than continuously redrawing. User interaction, viewport changes, data updates, or scheduled animations request new frames. This behavior is intended to reduce unnecessary resource use when a scientific figure remains static for inspection.

For a line family \(y_i(x)=a_i x+b_i\), `GLPlot` uploads \((a_i,b_i)\) as instanced attributes. The vertex shader evaluates and clips the endpoints against the current viewport, avoiding explicit endpoint generation for every line on the CPU. Renderers that admit planar clip constraints use OpenGL clip distances so that out-of-domain fragments can be rejected before unnecessary fragment processing [@opengl].

To improve numerical behavior during deep zoom, the viewport center is computed in double precision on the CPU. World coordinates are expressed relative to this center before conversion to normalized device coordinates. This keeps active coordinates close to the display range even when absolute coordinate offsets are large.
### Density, interaction, and level of detail

For dense overlapping geometry, `GLPlot` can accumulate contributions additively in a floating-point framebuffer. A logarithmic normalization maps accumulated density to the display range, and optional post-processing uses Reinhard tone mapping to preserve contrast in bright regions [@reinhard2002photographic]. During rapid pan or zoom operations, the engine can reproject a cached offscreen image over
an expanded viewport and schedule an exact redraw when interaction stops. This impostor strategy reduces the amount of exact geometry work required for every pointer-motion event.
The level-of-detail policy estimates projected fill-rate cost rather than relying only on primitive count. It accounts for viewport dimensions, line width, and approximate projected polyline length. When the estimated cost exceeds a viewport-scaled budget, deterministic shader-side thinning is used. Since the decision is derived from the instance identifier, a static view does not exhibit random frame-to-frame flicker.

Selection is implemented through a deferred integer picking pass. Selectable primitives write an encoded identifier to an integer framebuffer. A click requires a one-pixel readback, after which the identifier maps directly to a layer and element index. The CPU lookup cost is therefore independent of scene size, apart from the fixed framebuffer readback overhead.

<!-- TODO: Add a compact architecture figure if it materially improves the paper. The
repository's existing architecture document contains a suitable pipeline description, but the
final figure should be provided as a publication-quality vector or high-resolution raster
asset and referenced here. -->

# Research impact

`GLPlot` is designed for research workflows in which visualization is a bottleneck between numerical computation and interpretation. Representative applications include the inspection of large trajectory projections, chemical-potential line families, phase-stability or feasibility constructions, energy distributions, structural descriptors, spatial fields, large vector fields, and uncertainty ensembles. Related visualization patterns occur in atomistic simulation, computational chemistry and physics, bioinformatics, engineering simulation, and exploratory data analysis.

The principal scholarly contribution is the integration of multiple GPU rendering techniques into a coherent scientific plotting interface: analytical primitive expansion, viewport- relative transforms, hardware clipping, floating-point density accumulation, interaction caching, fill-rate-aware level of detail, lightweight depth cues for three-dimensional primitives, and integer-ID picking. Although these techniques are established individually in computer graphics, their combination is intended to make them usable by researchers who should not need to manage shaders and framebuffer objects directly.

<!-- TODO (required before JOSS submission): Replace or extend this paragraph with documented evidence of actual research use. Current JOSS screening requires demonstrated research impact, not only plausible future applications. Cite at least one publication, preprint, public research workflow, benchmark study, or externally documented project that uses GLPlot. Name the research question and explain what GLPlot enabled. -->
<!-- TODO (strongly recommended): Add reproducible comparative benchmarks on a documented hardware/software stack. At minimum compare representative large-line, line-family, and scatter workloads with relevant alternatives. Report data-transfer/setup time separately from steady-state interaction or frame time, and avoid unsupported universal performance claims. --> 

<!-- TODO: Add links or citations for any external adopters, integrations, teaching use, or research groups evaluating the software. -->

# Limitations

`GLPlot` focuses on high-performance interactive visualization rather than exhaustive Matplotlib compatibility. Complex publication layouts, specialized axis artists, advanced tick formatting, and some annotation or backend features may remain better served by Matplotlib. GPU rendering also introduces hardware, driver, and context-creation dependencies. Reproducible performance reports should therefore record the GPU model, driver and OpenGL versions, operating system, Python version, package version, viewport size, and rendering configuration.

The package currently requires a functional OpenGL context and compatible driver. Headless rendering and continuous-integration procedures should be documented explicitly for automated report generation and reviewer testing.

<!-- TODO: Verify the minimum supported OpenGL version and operating-system support against automated tests or a documented compatibility matrix. -->
# AI usage disclosure

<!-- TODO (required by the current JOSS review criteria): Provide an accurate disclosure of AI assistance used in the design, implementation, testing, documentation, or preparation of this manuscript. Identify the tools and the nature of their use. Also state which human contributors were responsible for problem formulation, architectural decisions, verification, and final approval. Do not leave this section blank. -->
# Acknowledgements

<!-- TODO: Add funding sources, grant identifiers, institutional support, collaborators, beta testers, and computational resources. -->
# References