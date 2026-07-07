Quick Start Guide
=================

Creating Your First Plot
-------------------------

The most basic plot in GLPlot follows Matplotlib conventions:

.. code-block:: python

   import glplot as gplt
   import numpy as np

   # Create some data
   x = np.linspace(0, 2*np.pi, 100)
   y = np.sin(x)

   # Plot it
   gplt.plot(x, y, 'b-', label='sin(x)')
   gplt.legend()
   gplt.xlabel('x')
   gplt.ylabel('y')
   gplt.title('Sine Wave')
   gplt.show()

Key Concepts
------------

**Plot Object**: Each call to a plotting function (``plot``, ``scatter``, ``bar``, etc.) adds layers to the current plot. Access the underlying ``GPULinePlot`` object via:

.. code-block:: python

   fig = gplt.gcf()  # Get current figure
   print(fig.width, fig.height)  # Window dimensions

**Interactive Controls**:
- **Left Mouse + Drag**: Pan the view
- **Scroll Wheel**: Zoom in/out
- **Right Mouse + Drag**: Rotate (in 3D mode)
- **Home / R Key**: Reset view to fit data
- **D Key**: Toggle density visualization
- **C Key**: Cycle line color schemes
- **F3 Key**: Show performance profiler
- **Esc Key**: Close window

**Multiple Figures**:

.. code-block:: python

   # Create first figure
   gplt.figure("Window 1")
   gplt.plot([1, 2, 3], [1, 2, 3])

   # Create second figure
   gplt.figure("Window 2")
   gplt.scatter([1, 2, 3], [3, 2, 1], color='red')

   # Show both
   gplt.show()

2D Plotting Examples
--------------------

**Line Plots with Markers**:

.. code-block:: python

   x = np.linspace(0, 10, 50)
   y1 = np.sin(x)
   y2 = np.cos(x)

   gplt.plot(x, y1, 'r-o', label='sin(x)')  # Red line with circles
   gplt.plot(x, y2, 'b--s', label='cos(x)')  # Blue dashed line with squares
   gplt.legend()
   gplt.show()

**Scatter Plot with Color Mapping**:

.. code-block:: python

   n = 10000
   x = np.random.randn(n)
   y = np.random.randn(n)
   c = x**2 + y**2  # Color by distance from origin

   gplt.scatter(x, y, c=c, cmap='viridis', s=20, alpha=0.6)
   gplt.colorbar()
   gplt.show()

**Histogram**:

.. code-block:: python

   data = np.random.normal(0, 1, 100000)
   gplt.hist(data, bins=50)
   gplt.show()

3D Plotting Examples
--------------------

**3D Line Plot**:

.. code-block:: python

   t = np.linspace(0, 4*np.pi, 1000)
   x = np.cos(t)
   y = np.sin(t)
   z = t

   gplt.plot3d(x, y, z, 'b-')
   gplt.show()

**3D Scatter Cloud**:

.. code-block:: python

   n = 50000
   x = np.random.randn(n)
   y = np.random.randn(n)
   z = np.random.randn(n)

   gplt.scatter3d(x, y, z, s=5, alpha=0.5)
   gplt.show()

**3D Vector Field**:

.. code-block:: python

   x, y, z = np.meshgrid(
       np.linspace(-2, 2, 8),
       np.linspace(-2, 2, 8),
       np.linspace(-2, 2, 8)
   )
   u = -y
   v = x
   w = np.zeros_like(z)

   gplt.quiver3d(x, y, z, u, v, w)
   gplt.show()

Performance Hints
-----------------

GLPlot excels with large datasets. Here are typical performance expectations:

- **100k points**: Smooth 60+ FPS interaction
- **1M points**: Still interactive, slight zoom/pan latency
- **10M points**: Density visualization recommended
- **100M+ points**: Requires density or subsampling

Use density visualization for overlapping point clouds:

.. code-block:: python

   x = np.random.randn(1000000)
   y = np.random.randn(1000000)
   gplt.scatter(x, y, s=1)  # 1M points
   gplt.toggle_density()  # Press D key or call this
   gplt.show()

Common Patterns
----------------

**Comparing Multiple Datasets**:

.. code-block:: python

   for i in range(5):
       x = np.linspace(0, 10, 100)
       y = np.sin(x + i)
       gplt.plot(x, y, label=f'sin(x+{i})')
   gplt.legend()
   gplt.show()

**Real-time Updating** (via loop):

.. code-block:: python

   fig = gplt.gcf()
   for frame in range(100):
       fig.clear()  # Clear previous frame
       t = np.linspace(0, 2*np.pi, 100)
       gplt.plot(np.cos(t + frame*0.1), np.sin(t + frame*0.1))
       # Window auto-updates; press Esc to exit loop

**Exporting Figures**:

.. code-block:: python

   gplt.plot([1, 2, 3, 4], [1, 4, 2, 3])
   gplt.show()
   gplt.savefig('my_plot.png', scale=2.0)  # High-res PNG

Next Steps
----------

- Explore :doc:`basic-plotting` for fundamental operations
- Learn :doc:`2d-plotting` for comprehensive 2D examples
- Discover :doc:`3d-visualization` for 3D capabilities
- Browse :doc:`../gallery/gallery` for advanced examples
