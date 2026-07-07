Advanced Features & Optimization
==================================

Deep dive into GLPlot's advanced capabilities for specialized use cases.

Density Visualization
---------------------

**When to Use:**

- Overlapping point clouds (100k+ points)
- Heatmap-style visualization
- Exploring data concentration

**Enable Density Mode:**

.. code-block:: python

   import glplot as gplt
   import numpy as np

   # 10M point dataset
   x = np.random.normal(0, 1, 10000000)
   y = np.random.normal(0, 1, 10000000)

   gplt.scatter(x, y, s=1)
   gplt.toggle_density()  # Press D key or call this
   gplt.show()

Density mode creates an HDR histogram where darker = more points, brighter = fewer.

Level-of-Detail (LOD) Rendering
--------------------------------

**Adaptive Quality System:**

LOD automatically reduces geometry detail during interaction for smooth 60+ FPS.

.. code-block:: python

   fig = gplt.gcf()
   
   # Enable LOD with target of 8 lines per pixel
   fig.set_lod(enabled=True, max_lines_per_px=8)
   
   # More aggressive LOD (faster, less detail)
   fig.set_lod(enabled=True, max_lines_per_px=12)
   
   # Less aggressive LOD (slower, more detail)
   fig.set_lod(enabled=True, max_lines_per_px=4)

Screen-Space Ambient Occlusion (SSAO)
--------------------------------------

**Enhanced Depth Perception for 3D:**

SSAO adds realistic shadowing to 3D surfaces and point clouds.

.. code-block:: python

   fig = gplt.gcf()
   gplt.plot_surface(X, Y, Z)
   fig.set_ssao_enabled(True)  # Enable for key frames
   gplt.savefig('output.png', scale=2.0)  # High-res export

Note: SSAO has performance cost; use primarily for exports, not real-time.

Colormap Customization
----------------------

**Built-in Colormaps:**

.. code-block:: python

   cmaps = ['viridis', 'plasma', 'inferno', 'hot', 'cool', 
            'spring', 'summer', 'autumn', 'winter', 'gray']
   
   for cmap in cmaps:
       gplt.scatter(x, y, c=values, cmap=cmap)
       gplt.savefig(f'output_{cmap}.png')

**Colormap Normalization:**

.. code-block:: python

   # Auto normalization (default)
   gplt.scatter(x, y, c=values, cmap='viridis')
   
   # Manual bounds
   gplt.scatter(x, y, c=values, cmap='plasma', vmin=0, vmax=100)

3D View Parameters
------------------

**Camera Control:**

.. code-block:: python

   fig = gplt.gcf()
   gplt.plot3d(x, y, z)
   
   # Set viewing angle
   fig.set_3d_view(elev=45, azim=30, scale_z=1.0)
   
   # Reset to default
   fig.reset_3d_view()

Parameters:
- **elev**: Elevation angle (degrees, 0-90)
- **azim**: Azimuth angle (degrees, 0-360)
- **scale_z**: Z-axis scale relative to X-Y

Matplotlib Integration
----------------------

**Embed in Matplotlib Figure:**

.. code-block:: python

   import matplotlib.pyplot as plt
   import glplot as gplt

   # Create GLPlot
   gplt.plot([1, 2, 3, 4], [1, 4, 2, 3])
   
   # Export snapshot
   fig, ax, artist = gplt.to_matplotlib()
   
   # Customize in Matplotlib
   ax.set_title('GLPlot in Matplotlib')
   plt.tight_layout()
   plt.savefig('integrated.png', dpi=150)

**Real-Time Transfer:**

.. code-block:: python

   # Set transfer target
   fig_ax = plt.subplots()[1]
   gplt_fig = gplt.gcf()
   gplt_fig.set_matplotlib_transfer_target(fig_ax)
   
   # Press 'M' key in window to transfer current frame
   # Or call: gplt_fig.transfer_to_matplotlib_default()

Exporting High-Resolution Images
---------------------------------

**Basic Export:**

.. code-block:: python

   gplt.plot(x, y)
   gplt.savefig('output.png')  # Default resolution

**High-Resolution Export:**

.. code-block:: python

   # 2x resolution
   gplt.savefig('output_2x.png', scale=2.0)
   
   # 4x resolution (highest quality)
   gplt.savefig('output_4x.png', scale=4.0)

Timing grows with scale² (4x scale → 16x slower).

Real-Time Plotting & Animation
-------------------------------

**Loop-Based Animation:**

.. code-block:: python

   fig = gplt.figure()
   
   for frame in range(100):
       fig.clear()  # Clear previous frame
       
       # Generate frame data
       t = frame / 10.0
       x = np.cos(t)
       y = np.sin(t)
       z = t
       
       fig.plot3d(x, y, z)
       # Window auto-updates; press Esc to stop

**Smooth Interaction During Loop:**

The window remains responsive even during plotting loops due to async rendering.

Large Dataset Handling (10M+ Points)
------------------------------------

**Strategy for 10M+ Points:**

1. **Use Density Visualization**
2. **Reduce Point Size** (s=0.5-1.0)
3. **Enable Aggressive LOD**
4. **Consider Subsampling**

.. code-block:: python

   # 100M point dataset
   n = 100000000
   x = np.random.randn(n)
   y = np.random.randn(n)
   
   # Strategy: Subsample and use density
   sample_idx = np.random.choice(n, size=10000000, replace=False)
   gplt.scatter(x[sample_idx], y[sample_idx], s=0.5)
   gplt.toggle_density()
   gplt.show()

Global Alpha & Blending
-----------------------

**Adjust Overall Transparency:**

.. code-block:: python

   fig = gplt.gcf()
   gplt.plot(x, y)
   
   # Reduce global alpha for speed
   fig.set_global_alpha(0.7)  # 70% opacity
   gplt.show()

Lower alpha can actually speed up rendering (reduced fill rate pressure).

Blending Mode Control
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   fig = gplt.gcf()
   fig.set_blending_mode('auto')  # Automatic (default)
   # Or: 'on' (always blend), 'off' (no transparency)

Performance Profiling
---------------------

**Enable HUD Profiler:**

.. code-block:: python

   fig = gplt.figure()
   fig.set_hud_enabled(True)
   gplt.plot(x, y)
   gplt.show()
   
   # Press F3 key to toggle profiler

Profiler shows:
- FPS (frames per second)
- GPU memory usage
- Primitive counts
- CPU frame time

Batch Plotting Optimization
----------------------------

**Efficient Multi-Series Plotting:**

.. code-block:: python

   fig = gplt.figure()
   
   # Batch-add many series at once
   for i in range(1000):
       x = np.linspace(0, 10, 100)
       y = np.sin(x + i * 0.01)
       fig.plot(x, y, alpha=0.1)  # Low alpha for speed
   
   gplt.show()

Tips:
- Add all layers before show()
- Use low alpha for dense overlays
- Avoid intermediate savefig() calls

See Also
--------

- :doc:`performance-tips` - General optimization
- :doc:`3d-visualization` - 3D-specific features
- :doc:`2d-plotting` - 2D plotting reference
