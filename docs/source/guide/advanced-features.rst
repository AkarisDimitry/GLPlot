Advanced Features and Performance Optimization
================================================

This guide covers advanced GLPlot features for visualization optimization, special effects, and performance tuning. These techniques are essential when working with large datasets or requiring fine-grained control over rendering behavior.

Density Visualization and HDR Histograms
----------------------------------------

Density visualization transforms overlapping point clouds into smooth heatmaps, revealing data concentration patterns that individual points would obscure. This is particularly effective for scatter plots with 10k+ overlapping points.

Enabling Density Rendering
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import glplot as gplt
   import numpy as np

   # Generate overlapping points
   n = 500000
   x = np.concatenate([np.random.normal(-1, 0.5, n//2), np.random.normal(1, 0.5, n//2)])
   y = np.concatenate([np.random.normal(0, 0.5, n//2), np.random.normal(0, 0.5, n//2)])

   gplt.scatter(x, y, s=1, alpha=0.3)
   gplt.toggle_density()  # Press D key or call this
   gplt.show()

**Performance Impact**: Density rendering reduces per-vertex overhead. A 500k point scatter that would struggle at 30 FPS renders at 60+ FPS with density enabled. The GPU handles histogram accumulation; interactive zooming and panning remain smooth.

Density Gain and Color Schemes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Density gain controls how aggressively the histogram is stretched. Logarithmic scaling prevents bright cores from saturating:

.. code-block:: python

   fig = gplt.gcf()

   # Set density gain (higher = more visible detail in low-density areas)
   fig.set_density_gain(1.5)

   # Cycle through color schemes (0-11 standard schemes)
   # Press C key to cycle interactively, or:
   fig.next_density_scheme()  # Advance to next scheme
   fig.previous_density_scheme()  # Go to previous scheme

**Expected Performance**:

- 100k points: 60+ FPS, density optional
- 1M points: 40-50 FPS with density, 5-15 FPS without
- 10M points: Density recommended (30-45 FPS), raw scatter unusable (<2 FPS)

Logarithmic vs. Linear Scaling
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By default, density uses logarithmic scaling to handle the wide dynamic range of histogram values:

.. code-block:: python

   from glplot.options import EngineOptions

   # Create figure with linear density scaling
   opts = EngineOptions(density_is_log=False)
   fig = gplt.figure("Linear Density", engine_options=opts)
   gplt.scatter(x, y, s=1)
   gplt.toggle_density()
   gplt.show()

Linear scaling is useful when you need to map actual count magnitudes; logarithmic scaling excels for perception—humans perceive brightness logarithmically.

Weighted Density Accumulation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By default, each point contributes 1.0 to the histogram. Enable weighted accumulation to sum alpha values instead:

.. code-block:: python

   from glplot.options import EngineOptions

   opts = EngineOptions(density_weighted=True)
   fig = gplt.figure("Weighted Density", engine_options=opts)

   # Points with alpha=0.5 contribute 0.5 to histogram
   gplt.scatter(x, y, s=1, alpha=0.5)
   gplt.toggle_density()
   gplt.show()

This is useful when points carry uncertainty or confidence information you want to preserve in the density estimate.


Level-of-Detail (LOD) Rendering
-------------------------------

LOD rendering dynamically reduces geometry complexity during interaction. When the user drags to pan, GLPlot switches to a lower-resolution representation, then restores full quality after interaction stops. This keeps frame rates smooth during exploration.

Enabling and Configuring LOD
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

LOD is enabled by default. Configure it via engine options or the ``set_lod()`` method:

.. code-block:: python

   import glplot as gplt
   import numpy as np

   # Create dense line plot
   x = np.linspace(0, 100, 1000000)
   y = np.cumsum(np.random.randn(1000000) * 0.1)

   fig = gplt.figure("LOD Example")
   gplt.plot(x, y)

   # Enable LOD with max 8 lines per screen pixel
   fig.set_lod(enabled=True, max_lines_per_px=8)
   gplt.show()

**LOD Parameters**:

- ``enabled``: Turn LOD on/off
- ``max_lines_per_px``: Target density during interaction. Higher values preserve detail but may stutter. Typical range: 2-16 lines per pixel.

LOD Performance Behavior
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from glplot.options import EngineOptions

   # Tune via engine options
   opts = EngineOptions(
       lod_enabled=True,
       lod_target_coverage=0.35,  # 35% of screen pixels covered during LOD
       default_line_budget_per_px=8,  # Lines per pixel budget
   )
   fig = gplt.figure("Tuned LOD", engine_options=opts)
   gplt.plot(x, y)
   gplt.show()

**Expected Results**:

- **Without LOD**: 1M vertices, 15-20 FPS during pan/zoom
- **With LOD (8px budget)**: 1M vertices, 55-60 FPS during pan/zoom, 50-55 FPS idle

When LOD activates, the display switches to every *n*-th vertex. After interaction ends (~100ms), full resolution restores. Users experience smooth 60 FPS interaction with no visual lag.

Disabling LOD for Exact Rendering
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Some applications require exact pixel-perfect rendering without simplification:

.. code-block:: python

   opts = EngineOptions(lod_enabled=False, always_lod=False)
   fig = gplt.figure("Exact Mode", engine_options=opts)
   gplt.plot(x, y)
   gplt.show()

**Warning**: Without LOD, 1M+ point datasets may drop below 30 FPS during interaction. Combine with density visualization or downsampling for interactive performance.


Colormap Customization
----------------------

GLPlot supports both Matplotlib-style colormaps and custom color arrays. Colormaps map data values to RGBA tuples for scalable color-by-value visualization.

Matplotlib Colormap Integration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use any Matplotlib colormap with GLPlot:

.. code-block:: python

   import glplot as gplt
   import numpy as np
   import matplotlib.cm as cm

   n = 100000
   x = np.random.randn(n)
   y = np.random.randn(n)
   c = x**2 + y**2  # Radial distance

   # Use viridis (default Matplotlib colormap)
   gplt.scatter(x, y, c=c, cmap='viridis', s=5)
   gplt.colorbar()
   gplt.show()

**Available Colormaps**: Any Matplotlib colormap works: ``'viridis'``, ``'plasma'``, ``'inferno'``, ``'magma'``, ``'cividis'``, ``'twilight'``, ``'rainbow'``, ``'hot'``, ``'cool'``, ``'spring'``, ``'summer'``, ``'autumn'``, ``'winter'``, ``'Greys'``, ``'Purples'``, ``'Blues'``, ``'Greens'``, ``'Oranges'``, ``'Reds'``, etc.

Custom Colormap from Arrays
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Create custom colormaps by passing RGB(A) arrays directly:

.. code-block:: python

   import glplot as gplt
   import numpy as np

   n = 50000
   x = np.random.randn(n)
   y = np.random.randn(n)
   c = np.random.rand(n)

   # Create custom 256-color map: black -> blue -> cyan -> white
   colors = np.zeros((256, 4), dtype=np.float32)
   for i in range(256):
       t = i / 255.0
       if t < 0.33:
           # Black to blue
           colors[i] = [0, 0, t*3, 1]
       elif t < 0.66:
           # Blue to cyan
           colors[i] = [0, (t-0.33)*3, 1, 1]
       else:
           # Cyan to white
           colors[i] = [(t-0.66)*3, 1, 1, 1]

   gplt.scatter(x, y, c=c, cmap=colors, s=5)
   gplt.colorbar()
   gplt.show()

Diverging and Perceptually Uniform Maps
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For data with a meaningful center value (e.g., -100 to +100), use diverging colormaps:

.. code-block:: python

   n = 100000
   x = np.random.randn(n)
   y = np.random.randn(n)
   c = x - y  # Diverging around 0

   # RdBu: Red-Blue diverging
   gplt.scatter(x, y, c=c, cmap='RdBu', vmin=-3, vmax=3, s=5)
   gplt.colorbar()
   gplt.show()

**Perceptually Uniform Recommendation**: Use ``'viridis'``, ``'plasma'``, or ``'cividis'`` for quantitative data. They preserve perceived magnitude under color-blind vision and grayscale conversion.

Line Colormaps (1D Line Segments)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Color individual line vertices by a scalar field:

.. code-block:: python

   x = np.linspace(0, 10, 10000)
   y = np.sin(x) + np.random.randn(10000) * 0.1
   c = x  # Color by x position (age/time)

   gplt.plot(x, y, c=c, cmap='plasma')
   gplt.colorbar()
   gplt.show()


3D View Parameters and Camera Control
-------------------------------------

GLPlot's 3D mode uses an interactive camera system with both programmatic and manual control. Camera orientation is specified using elevation (pitch) and azimuth (yaw) angles, with optional zoom and pan.

Camera Position and Orientation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Set camera view explicitly:

.. code-block:: python

   import glplot as gplt
   import numpy as np

   # Generate 3D data
   t = np.linspace(0, 4*np.pi, 1000)
   x = np.cos(t)
   y = np.sin(t)
   z = t

   fig = gplt.figure("3D Camera Control")
   gplt.plot3d(x, y, z, 'b-', linewidth=2)

   # Set camera: elevation=45°, azimuth=45°, zoom_factor=1.5
   fig.set_3d_view(elev=45, azim=45, zoom_factor=1.5)
   gplt.show()

**Camera Parameters**:

- ``elev``: Elevation angle in degrees (-90 to +90). 0° is side view, 90° is top-down.
- ``azim``: Azimuth (rotation around vertical axis) in degrees (0 to 360).
- ``zoom_factor``: Zoom level (1.0 = default, 2.0 = 2x magnification).

Common Camera Positions
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Isometric view (45-45-45 perspective)
   fig.set_3d_view(elev=30, azim=45, zoom_factor=1.2)

   # Top-down orthographic
   fig.set_3d_view(elev=90, azim=0, zoom_factor=1.0)

   # Side view
   fig.set_3d_view(elev=0, azim=0, zoom_factor=1.0)

   # Front view with slight tilt
   fig.set_3d_view(elev=15, azim=0, zoom_factor=1.0)

   # Reset to default
   fig.reset_3d_view()

Programmatic Rotation Animation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Create smooth 3D rotations by updating camera parameters in a loop:

.. code-block:: python

   import glplot as gplt
   import numpy as np
   import time

   # Create 3D mesh
   u = np.linspace(0, 2*np.pi, 50)
   v = np.linspace(0, np.pi, 30)
   U, V = np.meshgrid(u, v)
   X = np.sin(V) * np.cos(U)
   Y = np.sin(V) * np.sin(U)
   Z = np.cos(V)

   fig = gplt.figure("Rotating Sphere")
   gplt.plot_surface(X, Y, Z, cmap='viridis')

   # Rotate for 10 seconds
   start = time.time()
   while time.time() - start < 10:
       azim = (time.time() - start) * 36  # 360° every 10 seconds
       fig.set_3d_view(elev=30, azim=azim, zoom_factor=1.2)
       # Window updates automatically

Auto-Centering and Bounds
^^^^^^^^^^^^^^^^^^^^^^^^^

GLPlot automatically centers and scales the view to fit 3D data. Override this with explicit bounds:

.. code-block:: python

   # Get current bounds (min_x, max_x, min_y, max_y, min_z, max_z)
   bounds = fig.get_3d_bounds()
   print(f"Data bounds: {bounds}")

   # Data automatically scales to fit view, but you can manually:
   fig.set_3d_view(zoom_factor=2.0)  # Manually zoom in

Interactive 3D Controls
^^^^^^^^^^^^^^^^^^^^^^^

While a figure is displayed:

- **Right-click + Drag**: Rotate view (change azim/elev)
- **Scroll Wheel**: Zoom in/out
- **Shift + Scroll**: Pan camera
- **R / Home Key**: Reset view to default


Screen-Space Ambient Occlusion (SSAO)
-------------------------------------

SSAO adds depth-aware shadowing to 3D visualizations, enhancing depth perception by darkening crevices and concave regions. It's particularly effective for surface and mesh visualizations.

Enabling SSAO
^^^^^^^^^^^^^

.. code-block:: python

   from glplot.options import EngineOptions, SSAOOptions

   ssao_opts = SSAOOptions(
       enabled=True,
       strength=0.5,   # Occlusion darkness (0.0-1.0)
       radius=1.0,     # Sampling radius in screen space
   )
   opts = EngineOptions(visual__ssao=ssao_opts)
   fig = gplt.figure("SSAO Demo", engine_options=opts)

   # Create 3D surface
   u = np.linspace(0, 2*np.pi, 100)
   v = np.linspace(0, np.pi, 50)
   U, V = np.meshgrid(u, v)
   X = np.sin(V) * np.cos(U)
   Y = np.sin(V) * np.sin(U)
   Z = np.cos(V)

   gplt.plot_surface(X, Y, Z, cmap='twilight')
   gplt.show()

SSAO Parameters
^^^^^^^^^^^^^^^

- **strength** (0.0-1.0): Controls occlusion darkness. 0.0 = no effect, 1.0 = heavy shadows. Typical: 0.4-0.7.
- **radius** (0.5-3.0): Screen-space sampling radius. Larger radius captures coarser occlusion; smaller radius is faster. Typical: 0.8-1.5.

**Performance Cost**: SSAO adds ~10-20% GPU overhead on large meshes (50k+ triangles). On smaller datasets, overhead is <5%.

SSAO with Glow Effects
^^^^^^^^^^^^^^^^^^^^^^

Combine SSAO with glow for enhanced visual drama:

.. code-block:: python

   from glplot.options import (
       EngineOptions,
       SSAOOptions,
       GlowOptions,
       VisualOptions,
   )

   ssao = SSAOOptions(enabled=True, strength=0.6, radius=1.0)
   glow = GlowOptions(
       enabled=True,
       threshold=0.6,
       intensity=1.0,
       radius_px=8.0,
   )
   visual = VisualOptions(ssao=ssao, glow=glow)
   opts = EngineOptions(visual=visual)

   fig = gplt.figure("SSAO + Glow", engine_options=opts)
   # ... add 3D geometry ...
   gplt.show()

**Expected FPS**: High-quality SSAO + glow on 100k+ triangles: 30-40 FPS. Consider reducing mesh resolution or disabling on lower-end GPUs.


Matplotlib Integration and Figure Embedding
-------------------------------------------

GLPlot figures can be embedded into Matplotlib layouts, enabling hybrid visualizations combining GPU acceleration with Matplotlib's ecosystem.

Embedding GLPlot in Matplotlib Subplots
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Create a GLPlot figure and embed it alongside Matplotlib plots:

.. code-block:: python

   import glplot as gplt
   import matplotlib.pyplot as plt
   import numpy as np

   # Create Matplotlib figure with subplots
   fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

   # Left: Matplotlib scatter
   x = np.random.randn(10000)
   y = np.random.randn(10000)
   ax1.scatter(x, y, s=1, alpha=0.3)
   ax1.set_title("Matplotlib Scatter (10k points)")

   # Right: GLPlot render saved to file, then displayed
   fig_gl = gplt.figure("GLPlot 3D", width=600, height=500)
   t = np.linspace(0, 4*np.pi, 1000)
   gplt.plot3d(np.cos(t), np.sin(t), t)
   gplt.savefig('/tmp/glplot_render.png', scale=1.0)

   from PIL import Image
   img = Image.open('/tmp/glplot_render.png')
   ax2.imshow(img)
   ax2.set_title("GLPlot 3D (GPU-rendered)")
   ax2.axis('off')

   plt.tight_layout()
   plt.show()

Combining Matplotlib Colormaps
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use Matplotlib colormaps seamlessly in GLPlot:

.. code-block:: python

   import glplot as gplt
   import matplotlib.pyplot as plt
   import numpy as np

   n = 100000
   x = np.random.randn(n)
   y = np.random.randn(n)
   c = np.sqrt(x**2 + y**2)

   # Use Matplotlib's cmap directly
   gplt.scatter(x, y, c=c, cmap=plt.cm.get_cmap('plasma'), s=3)
   gplt.colorbar()
   gplt.show()

Exporting for Publication
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   fig = gplt.figure("Publication Ready")
   gplt.plot([1, 2, 3], [1, 2, 3])

   # Export at 300 DPI equivalent (scale=4.0 for ~1200px width)
   gplt.savefig('figure.png', scale=4.0)
   # Result: High-quality PNG suitable for print


Exporting High-Resolution Images
---------------------------------

GLPlot uses off-screen rendering (FBO) to export images at arbitrary resolutions. The ``scale`` parameter multiplies internal render resolution.

Basic Export
^^^^^^^^^^^^

.. code-block:: python

   import glplot as gplt

   fig = gplt.figure("Export Demo")
   gplt.plot([1, 2, 3, 4], [1, 4, 2, 3])

   # Export at 2x resolution (typical for screen display)
   gplt.savefig('plot_2x.png', scale=2.0)

   # Export at 4x resolution (for print/publication)
   gplt.savefig('plot_print.png', scale=4.0)

   # Export at 1x (same as window, fastest)
   gplt.savefig('plot_screen.png', scale=1.0)

Export Parameters and Density Visualization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When exporting with density visualization enabled:

.. code-block:: python

   n = 500000
   x = np.random.randn(n)
   y = np.random.randn(n)

   fig = gplt.figure("Dense Plot")
   gplt.scatter(x, y, s=1, alpha=0.2)
   gplt.toggle_density()

   # Export preserves density rendering at high resolution
   gplt.savefig('density_plot_hires.png', scale=4.0, density=True)

   # Or force density off for raw points:
   gplt.savefig('raw_plot.png', scale=2.0, density=False)

**Export Performance**:

- **scale=1.0 (1400x900)**: ~50ms export
- **scale=2.0 (2800x1800)**: ~150ms export
- **scale=4.0 (5600x3600)**: ~400ms export

For 4K exports (8000x6000), use ``scale=5.7`` and expect ~1 second render time.

Multi-Figure Export
^^^^^^^^^^^^^^^^^^^

Export multiple figures in sequence for montage creation:

.. code-block:: python

   figures = []
   for i in range(3):
       fig = gplt.figure(f"Plot {i}", width=800, height=600)
       x = np.linspace(0, 10, 100)
       gplt.plot(x, np.sin(x + i), label=f'sin(x+{i})')
       gplt.legend()
       gplt.savefig(f'plot_{i}.png', scale=2.0)

Preserving Colors in 3D Exports
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

3D visualizations with colormaps export as-rendered. Ensure sufficient resolution to preserve colormap gradients:

.. code-block:: python

   # Generate 3D surface
   u = np.linspace(0, 2*np.pi, 100)
   v = np.linspace(0, np.pi, 50)
   U, V = np.meshgrid(u, v)
   X = np.sin(V) * np.cos(U)
   Y = np.sin(V) * np.sin(U)
   Z = np.cos(V)

   fig = gplt.figure("3D Surface", width=1000, height=800)
   gplt.plot_surface(X, Y, Z, cmap='twilight')

   # Export at high resolution to preserve colormap detail
   gplt.savefig('surface.png', scale=3.0)


Performance Profiling and Optimization Tips
-------------------------------------------

GLPlot includes a built-in profiler for monitoring GPU/CPU performance in real-time. Press **F3** to toggle the profiler HUD overlay.

Built-in Profiler
^^^^^^^^^^^^^^^^^

Enable the profiler programmatically:

.. code-block:: python

   import glplot as gplt

   fig = gplt.figure("Profiled")
   gplt.plot(np.random.randn(100000).cumsum())
   gplt.set_hud_enabled(True)  # Show HUD including profiler
   gplt.show()

   # Press F3 while window is open for profiler overlay

The profiler displays:

- **FPS**: Current frame rate (target: 60 FPS)
- **Frame Time**: Total frame render time (ms)
- **Render Stage Breakdown**: Time for geometry, density, compositing
- **Memory Usage**: VRAM allocation (GB)
- **Vertex Count**: Active geometry

Memory Profiling
^^^^^^^^^^^^^^^^

Monitor VRAM usage for large datasets:

.. code-block:: python

   import glplot as gplt
   import numpy as np

   sizes = [100e3, 1e6, 10e6]
   for size in sizes:
       fig = gplt.figure(f"{int(size/1e6)}M Points")
       x = np.random.randn(int(size))
       y = np.random.randn(int(size))
       gplt.scatter(x, y, s=1)
       gplt.set_hud_enabled(True)
       # Check HUD for memory usage
       gplt.show()

**VRAM Usage Estimates**:

- 100k points: ~3 MB
- 1M points: ~30 MB
- 10M points: ~300 MB
- 100M points: ~3 GB

Optimization Checklist
^^^^^^^^^^^^^^^^^^^^^^

When performance degrades, apply these optimizations in order:

1. **Enable LOD**: ``fig.set_lod(enabled=True, max_lines_per_px=8)`` — Maintains 60 FPS during panning.
2. **Use Density**: ``fig.toggle_density()`` — Reduces per-vertex overhead by 50-80% for overlapping data.
3. **Reduce Alpha**: Set ``alpha=0.1`` instead of 1.0; GPU skips fully transparent pixels.
4. **Downsample Data**: Use every *n*-th point: ``data[::10]`` for 10M point datasets.
5. **Disable SSAO/Glow**: Remove screen-space effects if FPS < 30.
6. **Reduce Export Scale**: Use ``scale=2.0`` instead of 4.0 for quick exports.
7. **Upgrade GPU**: Older integrated GPUs (Intel HD 4000) are 5-10x slower than modern discrete GPUs.

CPU vs. GPU Bottlenecks
^^^^^^^^^^^^^^^^^^^^^^^

Identify bottlenecks:

- **High frame time despite low FPS**: GPU-bound. Reduce complexity (LOD, density, SSAO).
- **Laggy panning/scrolling despite high idle FPS**: CPU-bound. Reduce update frequency or data transfer rate.

.. code-block:: python

   import time

   fig = gplt.figure("Benchmark")
   x = np.linspace(0, 100, 1000000)
   y = np.cumsum(np.random.randn(1000000) * 0.1)
   gplt.plot(x, y)

   # Measure interaction latency
   start = time.time()
   for _ in range(60):  # Simulate 60 frames of rendering
       pass  # Window renders; check profiler F3
   elapsed = time.time() - start
   avg_frame_ms = (elapsed / 60) * 1000
   print(f"Avg frame time: {avg_frame_ms:.2f} ms")


Large Dataset Handling (10M+ Points)
------------------------------------

Visualizing 10M+ points requires careful strategy. Raw point rendering is infeasible; density visualization, LOD, and subsampling are essential.

Strategy: 10M-Point Scatter
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import glplot as gplt
   import numpy as np

   # Generate 10M points
   print("Generating 10M points...")
   n = 10_000_000
   x = np.random.laplace(0, 1, n)
   y = np.random.laplace(0, 1, n)

   print("Creating figure...")
   fig = gplt.figure("10M Points", width=1400, height=900)
   gplt.scatter(x, y, s=1, alpha=0.3)

   print("Enabling density visualization...")
   fig.set_density_enabled(True)
   fig.set_density_gain(1.5)

   print("Setting LOD...")
   fig.set_lod(enabled=True, max_lines_per_px=8)

   print("Rendering (should achieve 40-50 FPS)...")
   gplt.show()

**Performance**: 40-50 FPS interactive exploration. Memory: ~300 MB VRAM.

Strategy: Subsampling + Density
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For truly massive datasets (100M+ points), subsample first:

.. code-block:: python

   import glplot as gplt
   import numpy as np

   n_total = 100_000_000
   n_sample = 10_000_000  # Keep 10% of data

   # Load/generate data
   print(f"Subsampling {n_total} to {n_sample} points...")
   idx = np.random.choice(n_total, n_sample, replace=False)
   x = x[idx]
   y = y[idx]

   # Visualize subsampled data
   fig = gplt.figure("Subsampled 100M")
   gplt.scatter(x, y, s=1, alpha=0.2)
   fig.toggle_density()
   gplt.show()

**Result**: Subsampling 100M → 10M is mathematically sound; density histograms show the same distribution.

Multi-Layer Large Dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^

Combine multiple data distributions on one plot:

.. code-block:: python

   n = 5_000_000

   fig = gplt.figure("Multi-layer 10M")

   # Layer 1: Gaussian cluster
   x1 = np.random.normal(-2, 1, n)
   y1 = np.random.normal(-2, 1, n)
   gplt.scatter(x1, y1, s=1, alpha=0.2, color='red')

   # Layer 2: Uniform cloud
   x2 = np.random.uniform(-5, 5, n)
   y2 = np.random.uniform(-5, 5, n)
   gplt.scatter(x2, y2, s=1, alpha=0.2, color='blue')

   fig.toggle_density()
   gplt.show()

**Performance**: Still ~40 FPS with density enabled. Each layer renders independently in GPU.

Streaming / Progressive Loading
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For disk-resident datasets, load progressively:

.. code-block:: python

   import glplot as gplt
   import numpy as np
   import h5py

   fig = gplt.figure("Progressive Load")

   # Load HDF5 dataset in chunks
   with h5py.File('large_dataset.h5', 'r') as f:
       chunk_size = 1_000_000
       for i in range(0, len(f['x']), chunk_size):
           x = f['x'][i:i+chunk_size]
           y = f['y'][i:i+chunk_size]
           gplt.scatter(x, y, s=1, alpha=0.2)
           print(f"Loaded {i+chunk_size} points...")

   fig.toggle_density()
   gplt.show()

This avoids loading the entire dataset into memory upfront.


Real-Time Plotting and Animation Patterns
------------------------------------------

GLPlot supports dynamic updates for real-time data streaming and animation. The window remains interactive during updates.

Basic Animation Loop
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import glplot as gplt
   import numpy as np
   import time

   fig = gplt.figure("Sine Wave Animation")

   # Animation loop
   for frame in range(200):
       t = np.linspace(0, 2*np.pi, 100)
       y = np.sin(t + frame * 0.1)

       fig.clear()  # Clear previous frame
       gplt.plot(t, y, 'b-', linewidth=2)
       gplt.ylim(-1.5, 1.5)
       gplt.xlabel('x')
       gplt.ylabel(f'sin(x + {frame*0.1:.2f})')

       # Window updates automatically
       # Press Esc to exit early

**Frame Rate**: Animation runs at GPU refresh rate (~60 FPS). The ``fig.clear()`` call is fast (<1ms).

Real-Time Data Streaming
^^^^^^^^^^^^^^^^^^^^^^^^^

Append new data points to an existing plot:

.. code-block:: python

   import glplot as gplt
   import numpy as np
   import time

   fig = gplt.figure("Data Stream")
   t_start = time.time()

   # Simulate streaming sensor data
   while time.time() - t_start < 30:  # Run for 30 seconds
       elapsed = time.time() - t_start
       x = np.array([elapsed])
       y = np.array([np.sin(elapsed) + np.random.randn() * 0.1])

       gplt.plot(x, y, 'r.', markersize=5)
       gplt.xlim(elapsed - 10, elapsed + 1)  # Sliding window
       gplt.xlabel('Time (s)')
       gplt.ylabel('Value')

       time.sleep(0.05)  # Update 20x per second

   gplt.show()

**Performance**: Updates at 100+ Hz with <10 points per update. Latency: ~1-2 ms.

Multi-Series Real-Time Update
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Update multiple series simultaneously:

.. code-block:: python

   import glplot as gplt
   import numpy as np
   import time

   fig = gplt.figure("Multi-Series Stream")
   t_start = time.time()

   series_x = []
   series_y1 = []
   series_y2 = []
   series_y3 = []

   while time.time() - t_start < 60:
       elapsed = time.time() - t_start
       series_x.append(elapsed)
       series_y1.append(np.sin(elapsed))
       series_y2.append(np.cos(elapsed))
       series_y3.append(np.sin(elapsed * 2))

       fig.clear()
       gplt.plot(series_x, series_y1, 'r-', label='sin(t)')
       gplt.plot(series_x, series_y2, 'g-', label='cos(t)')
       gplt.plot(series_x, series_y3, 'b-', label='sin(2t)')
       gplt.legend()
       gplt.xlim(max(0, elapsed - 10), elapsed + 1)
       gplt.ylim(-1.5, 1.5)

       time.sleep(0.05)

   gplt.show()

Efficient Update Pattern (Retain Structure)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To avoid redundant GPU transfers, update plot data in-place where possible:

.. code-block:: python

   import glplot as gplt
   import numpy as np

   fig = gplt.figure("Efficient Update")

   # Pre-allocate data
   n_history = 1000
   x = np.arange(n_history)
   y = np.zeros(n_history)

   layer = gplt.plot(x, y, 'b-')[0]  # Get the layer handle

   # Update by rolling and appending
   for i in range(10000):
       new_y = np.sin(i * 0.01) + np.random.randn() * 0.1
       y = np.roll(y, -1)
       y[-1] = new_y

       # Re-plot (GPU handles incremental updates efficiently)
       fig.clear()
       gplt.plot(x, y, 'b-')
       gplt.ylim(-2, 2)

       if i % 100 == 0:
           print(f"Frame {i}")

   gplt.show()

Frame Rate and Update Frequency
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Desktop rendering target: 60 FPS = 16.67 ms per frame.

.. code-block:: python

   # Safe update frequency: 10-30 Hz
   update_hz = 20
   sleep_ms = 1.0 / update_hz

   for _ in range(300):  # 15 seconds at 20 Hz
       # Update and render (GPU-limited, ~10-15 ms)
       time.sleep(sleep_ms)

Higher update frequencies (100+ Hz) may exceed GPU render capacity, causing frame drops.

Particle Animation
^^^^^^^^^^^^^^^^^^

Animate point clouds with position updates:

.. code-block:: python

   import glplot as gplt
   import numpy as np

   n_particles = 50000
   positions = np.random.randn(n_particles, 2)
   velocities = np.random.randn(n_particles, 2) * 0.1

   fig = gplt.figure("Particle Animation")

   for frame in range(500):
       positions += velocities

       # Wrap around boundaries
       positions[positions > 5] = -5
       positions[positions < -5] = 5

       fig.clear()
       gplt.scatter(positions[:, 0], positions[:, 1], s=2, alpha=0.6)
       gplt.xlim(-5, 5)
       gplt.ylim(-5, 5)

       if frame % 50 == 0:
           print(f"Frame {frame}: {np.mean(np.linalg.norm(velocities, axis=1)):.2f} px/frame")

   gplt.show()

**Performance**: 50k particles at 60 FPS on mid-range GPU.


Summary and Key Takeaways
-------------------------

**Density Visualization**: Essential for 10k+ overlapping points. Reduces per-vertex overhead by 50-80% while improving visual clarity.

**LOD Rendering**: Automatically maintains 60 FPS during interaction on 1M+ vertex datasets.

**Colormaps**: Use perceptually uniform maps (viridis, plasma) for scientific accuracy.

**3D Camera**: Programmatically control view with ``set_3d_view(elev, azim, zoom_factor)``.

**SSAO/Glow**: Enhance 3D depth perception at 10-20% GPU cost.

**High-Resolution Export**: Use ``scale=4.0`` for publication-quality images.

**Performance Profiling**: Press F3 to inspect frame times, memory, and render stages.

**10M+ Points**: Combine density + LOD + optional subsampling for 40-60 FPS.

**Real-Time Updates**: Achieve 100+ Hz update rates with efficient in-place data modifications.

**Matplotlib Integration**: Export GLPlot renders and embed in Matplotlib layouts for hybrid visualizations.


See Also
--------

- :doc:`basic-plotting` for fundamental operations
- :doc:`../../api/plotting` for complete API reference
- :doc:`../../api/core` for engine and layer classes
