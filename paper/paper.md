---
title: "GLPlot: GPU-accelerated visualization with an MCP server for AI agents and a Matplotlib-like interface"
tags:
  - Python
  - scientific visualization
  - GPU rendering
  - OpenGL
  - large-scale plotting
  - interactive visualization
  - Model Context Protocol
  - AI agents
  - large language models
authors:
  - name: Juan Manuel Lombardi
    corresponding: true
    email: lombardi@fhi.mpg.de
    affiliation: 1
    orcid: 0009-0002-3961-1426
  - name: Felix Riccius
    equal-contrib: true
    affiliation: 1
    orcid: 0000-0002-1331-4469
  - name: Julian Holland
    equal-contrib: true
    affiliation: 1
    orcid: 0000-0003-0134-1790
  - name: Gianmarco Ducci
    equal-contrib: true
    affiliation: 1
affiliations:
  - name: "Fritz Haber Institute of the Max Planck Society, Berlin, Germany"
    index: 1
    ror: "03k9qs827"
date: 2 November 2026
bibliography: paper.bib
thanks: "Software repository: \\url{https://github.com/AkarisDimitry/GLPlot}"
---

# Summary

Interactive visualization is part of the scientific reasoning process: researchers use plots to inspect convergence, identify outliers, compare simulations, assess uncertainty, and decide which analyses or experiments to perform next. These tasks become slow when a figure contains very many points, vertices, curves, or other geometric primitives, because conventional plotting workflows pay for CPU-side geometry construction, per-object draw calls, repeated rasterization, and the transfer of expanded geometry to the graphics device. Waiting on every view change is more than an irritation: fewer lines of reasoning are pursued when each one is slow to check.

Here, we propose `GLPlot`, an open-source Python plotting package that exposes a familiar `pyplot`-style interface while using an OpenGL rendering engine in the backend. It supports the common two- and three-dimensional scientific plot types. A more dedicated `plot_lines` interface represents large families of straight lines by their slope and intercept coefficients and expands their visible geometry analytically on the GPU.

The rendering engine combines event-driven updates, shader-based geometry generation, stable viewport transforms, density accumulation, adaptive level of detail, and GPU picking behind that interface, so large datasets can be explored without touching OpenGL. This paper describes version 0.3.0, which adds a Model Context Protocol (MCP) server [@mcp] through which AI agents build, inspect, and animate figures, an interactive workstation, and headless export of figures and animations.

# Statement of need

GPU-oriented visualization libraries can provide high rendering throughput, but they often require users to adopt a different programming model or work closer to graphics concepts than is desirable in routine scientific analysis [@vispy; @fastplotlib]. Conversely, pixel-based aggregation systems efficiently summarize very large datasets, but do not necessarily preserve the semantics or interactive selection of individual geometric objects [@datashader; @stevens2015holoviews].
`GLPlot` targets the space between these approaches. The intended users are computational scientists, engineers, and data analysts who require interactive rendering of large two- and three-dimensional datasets while retaining a concise interface close to the idiom of standard plotting libraries such as Matplotlib. The software is particularly suited to cases where the plotted entities remain meaningful geometric objects, such as line families in chemical-potential diagrams, feasibility diagrams, trajectory ensembles, vector fields, meshes, and selectable point clouds.
The design is guided by five requirements:

1. avoid CPU-side geometry expansion when equivalent geometry can be generated analytically on the GPU;
2. maintain stable coordinate transforms during deep zoom operations at large coordinate offsets;
3. preserve density information when many primitives overlap;
4. avoid unnecessary CPU and GPU work when a scene is unchanged; and
5. support interactive inspection and selection without CPU searches that scale linearly with the number of rendered objects.

Analyses are increasingly carried out by AI agents built on large language models (LLMs), which need to produce a figure and check it without running arbitrary code; `GLPlot` exposes its plotting to them as typed tools over MCP.

# State of the field

Scientific visualization tools span a range of abstraction levels and rendering strategies; general-purpose plotting libraries, browser-based libraries [@bokeh; @plotly], VTK-based applications [@schroeder2006vtk; @ramachandran2011mayavi; @ahrens2005paraview], GPU frameworks, and aggregation systems answer different needs and complement rather than replace one another.

Matplotlib is the reference plotting library of scientific Python [@hunter2007matplotlib]. Its artist-based architecture coordinates geometry, transforms, and redraws largely on the CPU, so performance deteriorates for very many primitives: dense line families, large point clouds, and interactive three-dimensional scenes.

VisPy offers high-performance OpenGL visualization from buffers and shaders up to scene graphs [@vispy], and fastplotlib a similar model on a modern GPU API [@fastplotlib]; both suit custom applications but ask users to work with graphics concepts.

Datashader aggregates data into the display's pixel grid [@datashader], preserving distributions but not individual, selectable objects.

PyQtGraph targets responsive desktop graphics for instrumentation [@pyqtgraph].

`GLPlot` sits between these approaches. Contributing these capabilities to an existing package would have meant changing its core rendering model, so `GLPlot` keeps a procedural interface close to Matplotlib's `pyplot` and builds a GPU-resident architecture underneath it, for workloads in which many geometric objects must remain individually represented, interactive, and identifiable without a custom graphics pipeline.

# Software design

`GLPlot` stages data as float32 buffers on the CPU and leaves per-frame work to the GPU: 2D data goes through shader-based wide-line and density-accumulation paths, 3D data through model-view-projection transforms with depth testing, ambient-occlusion shading, and transparency. The interactive path never reads pixels back; `glReadPixels` is used only for explicit export. Rendering uses OpenGL [@opengl] through PyOpenGL [@pyopengl] and GLFW [@glfw].

![GLPlot's 2D and 3D rendering pipelines, from CPU-side data staging through GPU shading, blending, and post-processing to optional export.](dataflow.png)

## User-facing interface

`GLPlot` is available on PyPI and installs with `pip install glplot`. The primary entry point is `glplot.pyplot`. Calls such as `plot`, `scatter`, `hist2d`, `pcolormesh`, `contour`, `quiver`, `plot_surface`, `mesh3d`, and `bar3d` create layers in a scene, accept NumPy arrays [@harris2020array], and follow common Matplotlib calling conventions, including format strings for line and marker styling; select numerical routines use SciPy [@virtanen2020scipy]. A more specialized `plot_lines` interface instead takes a line family as coefficient pairs:

```python
import numpy as np
import glplot.pyplot as plt

slopes = np.random.default_rng(0).normal(size=1_000_000)
intercepts = np.random.default_rng(1).normal(size=1_000_000)

plt.figure()
plt.plot_lines(slopes, intercepts, x_range=(-5.0, 5.0))
plt.set_cmap("magma")
plt.density_gain(0.8)
plt.xlabel("x")
plt.ylabel("a x + b")
plt.show(density=True)
```

## Rendering architecture and precision

Data, styles, and transforms are stored as scene layers, each mapped to a primitive renderer and the framebuffer passes it needs. The render loop redraws only when a layer, the camera, or an effect is marked dirty, so an unchanged figure costs nothing between events.

For a line family $y_i(x) = a_i x + b_i$, `GLPlot` uploads $(a_i, b_i)$ as instanced attributes; the vertex shader analytically recomputes each line's visible endpoints against the current viewport bounds and discards out-of-domain instances before rasterization, rather than the CPU generating explicit per-line geometry. For stability during deep zoom, the viewport center is computed in double precision on the CPU and coordinates are expressed relative to it, so the values the GPU sees stay near the display range at any offset.

## Density, interaction, and level of detail

Dense, overlapping geometry is accumulated additively into a floating-point framebuffer, log-normalized to the display range, and optionally tone-mapped in the style of @reinhard2002photographic to preserve contrast. During rapid pan or zoom a cached offscreen image is reprojected and an exact redraw follows when interaction stops. Level of detail is chosen from estimated fill-rate cost (viewport size, line width, projected length) rather than primitive count, deterministically per instance, so a static view does not flicker. Selection uses an integer picking pass: primitives write an identifier to an integer framebuffer, and a click resolves from a single-pixel readback, independent of scene size.

## The MCP server for AI agents

`glplot-mcp` serves GLPlot to AI agents over the Model Context Protocol (\autoref{fig:mcp}). An agent works through typed tools rather than generated code: it creates figures and datasets, plots, applies style presets by name, animates in stages, cuts scenes into a video, and exports the equivalent Python script. Every render returns the image with a deterministic review of provable problems (a missing label, overlapping ticks, a colour lost against the page), each paired with the call that fixes it, so the agent sees and corrects its own output. Knowledge tools (`get_guide`, `search_api`, `get_api_doc`, `get_example`) serve the documentation installed with the package, so the agent learns the API instead of guessing it from Matplotlib. No arbitrary code runs, inputs are checked against schemas and resource limits, and file access is confined to directories the user chooses, so the server can be connected to an agent safely. It works after a plain `pip install`, and `glplot-mcp config` writes the settings for clients such as Claude Code, Cursor, VS Code, and Codex.

![The MCP loop an AI agent runs for a figure (left) and what the same session can deliver (right). The review in step 7 is the server's real output.\label{fig:mcp}](mcp.png)

## Workstation and export

`python -m glplot` opens a workstation, built with Dear ImGui [@imgui], for editing layers, data, styles, and layouts live, with undo history and native controls; a headless path rebuilds any figure through Matplotlib for publication output and for animations written as GIF or video.

# Research impact statement

`GLPlot` is designed for research workflows in which visualization is a bottleneck between numerical computation and interpretation. It has been used in this capacity in `EZGA`, a published evolutionary structure-exploration framework for computational materials science, to visualize energy landscapes, convex-hull reconstructions, and surface phase diagrams produced across its benchmark systems [@lombardi2026ezga]. The same patterns (trajectory projections, chemical-potential line families, phase-stability constructions, uncertainty ensembles) recur across atomistic simulation, computational chemistry and physics, and engineering.

The principal contribution is to make established GPU rendering techniques usable by researchers who should not need to manage shaders and framebuffers.

# Limitations

`GLPlot` aims at fast interactive visualization, not exhaustive Matplotlib compatibility; complex publication layouts and specialized axis artists may remain better served by Matplotlib. GPU rendering also introduces hardware and driver dependencies, so performance reports should record the GPU, driver and OpenGL versions, operating system, and viewport size.

The package requires an OpenGL 3.3 core-profile context, the minimum its shaders request. Continuous integration runs the test suite on Linux, macOS, and Windows with Python 3.11 to 3.13, using Mesa's software OpenGL under a virtual display on Linux.

# AI usage disclosure

Generative AI was used in developing `GLPlot`. Claude Opus 5.5 (Anthropic), through Claude Code, and OpenAI Codex were used as coding assistants for parts of the implementation, the test suite, packaging and continuous-integration configuration, documentation and examples, and for editing this manuscript. The authors made the core design decisions, defined the problems to be solved, and reviewed, tested, and validated all AI-assisted output: changes were accepted only after passing the automated test suite and manual inspection of the rendered figures. The authors take full responsibility for the software and this paper.

# Author contributions

J.M.L. conceived and developed the software, wrote the documentation and the test suite, and drafted the manuscript. F.R., J.H., and G.D. contributed equally to the writing of the manuscript, to the design of features, and to testing the software in real research applications.

# Acknowledgements

This work was supported by the Fritz Haber Institute of the Max Planck Society, which had no role in the design of the software or in the writing of this paper.

# Conflicts of interest

The authors declare no conflicts of interest.

# References