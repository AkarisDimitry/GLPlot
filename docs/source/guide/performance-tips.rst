Performance Tips & Optimization
================================

Guide to optimizing GLPlot for large datasets and smooth interactivity.

Dataset Scaling
---------------

**Expected Performance:**

- **10k points**: 60+ FPS, full quality
- **100k points**: 60+ FPS interactive
- **1M points**: 30-60 FPS, interactive with zoom latency
- **10M points**: Density visualization recommended
- **100M+ points**: Specialized handling needed

Level-of-Detail (LOD)
---------------------

**Enable Adaptive LOD:**

.. code-block:: python

   fig = gplt.gcf()
   fig.set_lod(enabled=True, max_lines_per_px=8)
   gplt.show()

The LOD system automatically reduces geometry detail during pan/zoom for smooth interaction.

Density Visualization
---------------------

**For Overlapping Data:**

.. code-block:: python

   x = np.random.randn(10000000)  # 10M points
   y = np.random.randn(10000000)
   gplt.scatter(x, y, s=1)
   
   gplt.toggle_density()  # Press D key or call this
   gplt.show()

Density mode creates a heat map of point concentrations, essential for extremely large datasets.

Colormaps & SSAO
----------------

**Optimize Colormap Performance:**

.. code-block:: python

   # Use built-in colormaps (viridis, plasma, hot)
   # Avoid creating custom colormaps per-render
   gplt.scatter(x, y, c=values, cmap='plasma')

**Screen-Space Ambient Occlusion (SSAO):**

.. code-block:: python

   fig = gplt.gcf()
   gplt.plot3d(x, y, z)
   
   # Use only for key frames or stills, not real-time
   # SSAO adds depth perception but has performance cost
   fig.set_ssao_enabled(True)
   gplt.savefig('output.png', scale=2.0)

Global Alpha & Transparency
----------------------------

.. code-block:: python

   fig = gplt.gcf()
   fig.set_global_alpha(0.7)  # Reduces fill rate by 30%
   gplt.show()

Lower alpha can actually improve performance for large datasets.

Point Size Optimization
-----------------------

- **s=1-2**: Best for 1M+ points
- **s=5-10**: Good for 100k-1M points
- **s=20+**: Only for small datasets (<10k)

.. code-block:: python

   n = 1000000
   gplt.scatter(x, y, s=1, alpha=0.3)  # Optimized
   gplt.show()

Real-Time Plotting
-------------------

**Loop Update Pattern:**

.. code-block:: python

   fig = gplt.gcf()
   for frame in range(100):
       fig.clear()
       # Generate frame data
       gplt.plot(...)
       # Window auto-updates; press Esc to stop

Memory Management
-----------------

- **NumPy Arrays**: GLPlot takes reference copies; avoid duplicating
- **Figure Clearing**: ``fig.clear()`` frees GPU memory
- **Large Exports**: Use ``scale`` parameter for high-res saves
  
.. code-block:: python

   gplt.savefig('high_res.png', scale=3.0)  # 3x resolution

Profiling & Monitoring
----------------------

**Show Performance Profiler:**

.. code-block:: python

   fig = gplt.gcf()
   fig.set_hud_enabled(True)
   gplt.show()

In the HUD, press F3 key to toggle profiler showing:
- FPS (frames per second)
- GPU memory usage
- Geometry counts
- CPU frame times

Benchmarking
------------

**Test Your Hardware:**

.. code-block:: python

   import time
   
   for n_points in [100e3, 1e6, 10e6]:
       x = np.random.randn(int(n_points))
       y = np.random.randn(int(n_points))
       
       fig = gplt.figure()
       t0 = time.time()
       gplt.scatter(x, y, s=1)
       render_time = time.time() - t0
       
       print(f"{int(n_points):,} points: {render_time:.3f}s render")

Common Bottlenecks
------------------

1. **Too many colors** - Use single color or simple colormap
2. **High alpha blending** - Reduce alpha or disable for overlaps
3. **Huge markers** - Keep point sizes small for datasets >100k
4. **Excessive annotations** - Limit text labels for real-time plots
5. **Rapid updates** - Batch updates; avoid per-point refresh

Advanced Settings
-----------------

.. code-block:: python

   fig = gplt.gcf()
   
   # Disable features you don't need
   fig.set_hud_enabled(False)  # Hide HUD for slight speed boost
   
   # Optimize for specific use cases
   fig.set_lod(True, max_lines_per_px=12)  # Aggressive LOD
   
   # Export settings
   gplt.savefig('output.png', scale=1.0)  # Keep scale reasonable

See Also
--------

- :doc:`advanced-features` - More optimization techniques
- :doc:`3d-visualization` - 3D-specific tips
