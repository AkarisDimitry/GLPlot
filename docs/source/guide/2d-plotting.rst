2D Plotting Guide
=================

This guide covers essential 2D plotting techniques in GLPlot, with practical examples for every plotting type.
GLPlot provides a Matplotlib-compatible API optimized for GPU-accelerated rendering of large datasets.


Line Plots
----------

Basic Line Plot
^^^^^^^^^^^^^^^

The simplest plot shows a line connecting (x, y) points:

.. code-block:: python

   import numpy as np
   import glplot.pyplot as gplt

   x = np.linspace(0, 2*np.pi, 100)
   y = np.sin(x)

   gplt.plot(x, y)
   gplt.xlabel('x')
   gplt.ylabel('y')
   gplt.title('Sine Wave')
   gplt.show()


Line Styles and Colors
^^^^^^^^^^^^^^^^^^^^^^

Control line appearance with format strings (Matplotlib-style):

.. code-block:: python

   x = np.linspace(0, 10, 100)

   # Solid lines with different colors
   gplt.plot(x, np.sin(x), 'b-', label='solid blue')      # Blue solid line
   gplt.plot(x, np.cos(x), 'r--', label='dashed red')     # Red dashed line
   gplt.plot(x, np.sin(x)*0.5, 'g-.', label='dash-dot')   # Green dash-dot
   gplt.plot(x, np.cos(x)*0.5, 'k:', label='dotted')      # Black dotted

   gplt.legend()
   gplt.show()

**Format String Syntax**: ``'[color][linestyle]'``

- **Colors**: ``r`` (red), ``g`` (green), ``b`` (blue), ``c`` (cyan), ``m`` (magenta), ``y`` (yellow), ``k`` (black), ``w`` (white), or hex (``#FF0000``)
- **Line Styles**: ``-`` (solid), ``--`` (dashed), ``-.`` (dash-dot), ``:`` (dotted)


Line Plots with Markers
^^^^^^^^^^^^^^^^^^^^^^^

Add markers to line plots to emphasize individual points:

.. code-block:: python

   x = np.linspace(0, 10, 25)
   y1 = np.sin(x)
   y2 = np.cos(x)

   # Line with markers
   gplt.plot(x, y1, 'ro-', label='line + circles')      # Red line with circle markers
   gplt.plot(x, y2, 'b^--', label='line + triangles')   # Blue dashed with triangles

   gplt.legend()
   gplt.show()

**Marker Styles**: ``o`` (circle), ``s`` (square), ``^`` (triangle up), ``v`` (triangle down),
``x`` (cross), ``+`` (plus), ``*`` (star), ``d`` (diamond), and more.


Styling with Keyword Arguments
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For fine-grained control, use keyword arguments:

.. code-block:: python

   x = np.linspace(0, 10, 100)
   y = np.sin(x)

   # Named color and line width
   gplt.plot(x, y, color='navy', linewidth=2.5, label='Thick line')

   # Custom RGBA color
   gplt.plot(x, y + 0.3, color=(0.8, 0.2, 0.1, 1.0), alpha=0.7, label='Custom color')

   # Marker size control
   gplt.plot(x[::5], y[::5], 'go', markersize=8, label='Large markers')

   gplt.legend()
   gplt.show()

**Styling Options**:

- ``color``: Color name, hex string, or (r, g, b, a) tuple
- ``linewidth`` / ``lw``: Line thickness in pixels
- ``alpha``: Transparency (0.0-1.0)
- ``markersize`` / ``ms``: Marker size in pixels
- ``label``: Legend label


Multiple Series
^^^^^^^^^^^^^^^

Plot multiple lines on the same figure:

.. code-block:: python

   x = np.linspace(0, 2*np.pi, 100)

   for i in range(4):
       phase = i * np.pi / 4
       gplt.plot(x, np.sin(x + phase), label=f'sin(x + {i}π/4)')

   gplt.grid(True)
   gplt.legend()
   gplt.show()


Scatter Plots
-------------

Basic Scatter Plot
^^^^^^^^^^^^^^^^^^

Plot individual points without connecting lines:

.. code-block:: python

   import numpy as np
   import glplot.pyplot as gplt

   # Generate random data
   n = 1000
   x = np.random.randn(n)
   y = np.random.randn(n)

   gplt.scatter(x, y, size=5, color='blue')
   gplt.xlabel('X')
   gplt.ylabel('Y')
   gplt.show()

**Parameters**:

- ``x, y``: Data coordinates
- ``size`` / ``s``: Point size in pixels
- ``color``: Single color for all points


Scatter with Color Mapping
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Map point colors to data values for scientific visualization:

.. code-block:: python

   n = 5000
   x = np.random.randn(n)
   y = np.random.randn(n)
   values = x**2 + y**2  # Distance from origin

   gplt.scatter(x, y, c=values, cmap='viridis', s=15, alpha=0.6)
   gplt.colorbar()
   gplt.show()

**Color Mapping Parameters**:

- ``c``: Numeric array of values to map to colors
- ``cmap``: Colormap name ('viridis', 'plasma', 'cool', 'hot', 'RdBu', etc.)
- ``vmin, vmax``: Normalization range (auto-detected if not provided)


Scatter Plots with Per-Point Colors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Assign custom RGBA colors to individual points:

.. code-block:: python

   n = 1000
   x = np.random.randn(n)
   y = np.random.randn(n)

   # Generate per-point RGBA colors
   colors = np.random.rand(n, 4).astype(np.float32)
   colors[:, 3] = 0.7  # Set alpha

   gplt.scatter(x, y, color=colors, size=6)
   gplt.show()

This approach works well for very large point clouds (millions of points) with transparency blending.


Histograms
----------

1D Histogram
^^^^^^^^^^^^

Bin data and visualize the distribution:

.. code-block:: python

   import numpy as np
   import glplot.pyplot as gplt

   # Generate normal distribution
   data = np.random.normal(loc=100, scale=15, size=100000)

   counts, edges, patches = gplt.hist(data, bins=50, color='steelblue')

   gplt.xlabel('Value')
   gplt.ylabel('Frequency')
   gplt.title('Distribution of Data')
   gplt.show()

**Parameters**:

- ``x``: Data array
- ``bins``: Number of bins (or array of bin edges)
- ``density``: If True, normalize to probability density
- ``color``: Bar color


Custom Bin Edges
^^^^^^^^^^^^^^^^

Specify custom bin boundaries:

.. code-block:: python

   data = np.random.exponential(scale=2.0, size=50000)

   # Custom bins: smaller bins at low values, larger bins at high values
   custom_bins = [0, 1, 2, 3, 5, 8, 13, 20]
   counts, edges, patches = gplt.hist(data, bins=custom_bins, color='coral')

   gplt.xlabel('Value')
   gplt.ylabel('Count')
   gplt.show()


Density Normalization
^^^^^^^^^^^^^^^^^^^^^

Normalize histogram to probability density for comparison:

.. code-block:: python

   # Compare two distributions
   data1 = np.random.normal(0, 1, 50000)
   data2 = np.random.normal(2, 1.5, 50000)

   gplt.hist(data1, bins=40, density=True, color='blue', alpha=0.5, label='Distribution A')
   gplt.hist(data2, bins=40, density=True, color='red', alpha=0.5, label='Distribution B')

   gplt.legend()
   gplt.show()


2D Histogram (Heatmap)
^^^^^^^^^^^^^^^^^^^^^

Visualize bivariate density with a 2D histogram:

.. code-block:: python

   # Generate correlated 2D data
   n = 50000
   x = np.random.normal(0, 1, n)
   y = x + np.random.normal(0, 0.5, n)  # Correlated with noise

   counts, xedges, yedges, layer = gplt.hist2d(
       x, y,
       bins=50,
       cmap='magma'
   )

   gplt.xlabel('X')
   gplt.ylabel('Y')
   gplt.title('2D Histogram')
   gplt.colorbar()
   gplt.show()

**Parameters**:

- ``x, y``: Data coordinates
- ``bins``: Number of bins (or separate (xbins, ybins))
- ``cmap``: Colormap for heatmap
- ``density``: Normalize by total count


Bar Charts and Error Bars
--------------------------

Basic Bar Chart
^^^^^^^^^^^^^^^

Display categorical data as rectangular bars:

.. code-block:: python

   import numpy as np
   import glplot.pyplot as gplt

   categories = ['A', 'B', 'C', 'D', 'E']
   values = [23, 45, 56, 78, 32]
   x_positions = np.arange(len(categories))

   gplt.bar(x_positions, values, width=0.6, color='steelblue')

   gplt.xlabel('Category')
   gplt.ylabel('Value')
   gplt.xticks(x_positions, categories)
   gplt.show()

**Parameters**:

- ``x``: Bar x-positions
- ``height``: Bar heights
- ``width``: Bar width in data units
- ``color``: Bar color


Stacked Bars
^^^^^^^^^^^^

Create stacked bar charts with multiple ``bar()`` calls:

.. code-block:: python

   categories = ['Q1', 'Q2', 'Q3', 'Q4']
   x_pos = np.arange(len(categories))

   # First data series (base)
   series1 = [10, 15, 12, 18]
   gplt.bar(x_pos, series1, width=0.5, color='steelblue', label='Series 1')

   # Second data series (stacked on top)
   series2 = [5, 8, 7, 6]
   gplt.bar(x_pos, series2, bottom=series1, width=0.5, color='coral', label='Series 2')

   gplt.legend()
   gplt.show()


Error Bars
^^^^^^^^^^

Show uncertainty with error bars:

.. code-block:: python

   x = np.array([1, 2, 3, 4, 5])
   y = np.array([2.1, 3.9, 6.2, 7.8, 9.1])
   yerr = np.array([0.3, 0.2, 0.4, 0.3, 0.5])  # Per-point error

   gplt.errorbar(x, y, yerr=yerr, fmt='o-', color='steelblue',
                 ecolor='red', elinewidth=2, capsize=3)

   gplt.xlabel('X')
   gplt.ylabel('Y')
   gplt.show()

**Parameters**:

- ``x, y``: Data coordinates
- ``yerr``: Y-direction error (single value or per-point)
- ``xerr``: X-direction error
- ``ecolor``: Error bar color
- ``elinewidth``: Error bar line width
- ``capsize``: Cap size at error bar ends
- ``fmt``: Format string for data points


Error Bars on Scatter Plot
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Show asymmetric errors:

.. code-block:: python

   x = np.array([1, 2, 3, 4])
   y = np.array([5, 6, 7, 8])

   # Asymmetric errors
   yerr = [[0.2, 0.3, 0.2, 0.4], [0.3, 0.2, 0.3, 0.2]]  # [lower, upper]

   gplt.errorbar(x, y, yerr=yerr, fmt='o', color='blue', alpha=0.7)
   gplt.show()


Filled Regions and Contour Plots
---------------------------------

Fill Between Two Curves
^^^^^^^^^^^^^^^^^^^^^^^

Shade the region between two lines (e.g., confidence intervals):

.. code-block:: python

   import numpy as np
   import glplot.pyplot as gplt

   x = np.linspace(0, 10, 200)
   y_mean = np.sin(x)
   y_lower = y_mean - 0.1
   y_upper = y_mean + 0.1

   # Plot filled region
   gplt.fill_between(x, y_upper, y_lower, color='lightblue', alpha=0.5, label='Confidence band')

   # Plot mean line
   gplt.plot(x, y_mean, 'b-', linewidth=2, label='Mean')

   gplt.legend()
   gplt.show()

**Parameters**:

- ``x``: X-coordinates
- ``y1``: Upper boundary y-values
- ``y2``: Lower boundary y-values (or scalar for baseline)
- ``color``: Fill color
- ``alpha``: Transparency


Stacked Area Plot
^^^^^^^^^^^^^^^^^

Combine multiple filled regions to show composition:

.. code-block:: python

   x = np.linspace(0, 10, 100)
   y1 = np.sin(x)
   y2 = np.cos(x) * 0.5

   # Stack areas
   gplt.fill_between(x, y1, 0, color='steelblue', alpha=0.5, label='Component A')
   gplt.fill_between(x, y1 + y2, y1, color='coral', alpha=0.5, label='Component B')

   gplt.legend()
   gplt.show()


Contour Lines
^^^^^^^^^^^^^

Draw level curves of a 2D field:

.. code-block:: python

   # Create a 2D field
   x = np.linspace(-3, 3, 100)
   y = np.linspace(-3, 3, 100)
   X, Y = np.meshgrid(x, y)
   Z = np.exp(-(X**2 + Y**2))

   gplt.contour(X, Y, Z, levels=10, cmap='viridis')

   gplt.xlabel('X')
   gplt.ylabel('Y')
   gplt.title('Contour Lines')
   gplt.colorbar()
   gplt.show()

**Parameters**:

- ``X, Y, Z``: 2D coordinate arrays and values
- ``levels``: Number of levels or array of level values
- ``cmap``: Colormap for level colors
- ``linewidths``: Width of contour lines


Filled Contours
^^^^^^^^^^^^^^^

Color-fill regions between contour levels:

.. code-block:: python

   # Gaussian function
   x = np.linspace(-4, 4, 80)
   y = np.linspace(-4, 4, 80)
   X, Y = np.meshgrid(x, y)
   Z = np.exp(-(X**2 + Y**2)/2)

   gplt.contourf(X, Y, Z, levels=15, cmap='cool')

   gplt.xlabel('X')
   gplt.ylabel('Y')
   gplt.colorbar()
   gplt.show()


Image Visualization
-------------------

Image Display (imshow)
^^^^^^^^^^^^^^^^^^^^^^

Display a 2D array as a colored image:

.. code-block:: python

   import numpy as np
   import glplot.pyplot as gplt

   # Random image data
   image = np.random.rand(64, 64)

   gplt.imshow(image, cmap='viridis')
   gplt.title('Random Image')
   gplt.colorbar()
   gplt.show()

**Parameters**:

- ``X``: 2D array to display
- ``cmap``: Colormap ('viridis', 'hot', 'gray', etc.)
- ``origin``: 'upper' (image coords) or 'lower' (math coords)
- ``extent``: (left, right, bottom, top) in data coordinates
- ``vmin, vmax``: Colormap normalization range


Image with Data Coordinates
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Map 2D array to custom data range:

.. code-block:: python

   # Create mathematical function
   x = np.linspace(-2, 2, 100)
   y = np.linspace(-2, 2, 100)
   X, Y = np.meshgrid(x, y)
   Z = np.sin(np.sqrt(X**2 + Y**2))

   gplt.imshow(Z, extent=[-2, 2, -2, 2], cmap='RdBu', origin='lower')

   gplt.xlabel('X')
   gplt.ylabel('Y')
   gplt.colorbar()
   gplt.show()


Colored Mesh (pcolormesh)
^^^^^^^^^^^^^^^^^^^^^^^^^

Display values on a regular or irregular mesh:

.. code-block:: python

   # Regular mesh
   x = np.linspace(0, 10, 30)
   y = np.linspace(0, 5, 20)
   X, Y = np.meshgrid(x, y)
   Z = np.sin(X) * np.cos(Y)

   gplt.pcolormesh(X, Y, Z, cmap='plasma')

   gplt.xlabel('X')
   gplt.ylabel('Y')
   gplt.colorbar()
   gplt.show()

**Parameters**:

- ``X, Y``: 2D coordinate arrays
- ``C``: 2D value array
- ``cmap``: Colormap
- ``shading``: 'auto' or 'flat'


Vector Fields (Quiver)
---------------------

2D Vector Field
^^^^^^^^^^^^^^^

Plot arrows representing 2D vectors:

.. code-block:: python

   import numpy as np
   import glplot.pyplot as gplt

   # Create a grid of positions
   x = np.linspace(-2, 2, 12)
   y = np.linspace(-2, 2, 12)
   X, Y = np.meshgrid(x, y)

   # Vector field: rotational
   U = -Y
   V = X

   gplt.quiver(X.ravel(), Y.ravel(), U.ravel(), V.ravel(),
               color='steelblue', scale=0.5)

   gplt.xlabel('X')
   gplt.ylabel('Y')
   gplt.title('Rotational Vector Field')
   gplt.show()

**Parameters**:

- ``x, y``: Arrow positions
- ``u, v``: X and Y components of vectors
- ``scale``: Scaling factor for arrow lengths
- ``width``: Shaft line width
- ``head_width, head_length``: Arrowhead dimensions
- ``color``: Arrow color


Gradient Field
^^^^^^^^^^^^^^

Visualize the gradient of a scalar function:

.. code-block:: python

   # Scalar field
   x = np.linspace(-3, 3, 15)
   y = np.linspace(-3, 3, 15)
   X, Y = np.meshgrid(x, y)
   F = X**2 + Y**2

   # Gradient (pointing up the slope)
   U = 2 * X
   V = 2 * Y
   magnitude = np.sqrt(U**2 + V**2)
   U_norm = U / (magnitude + 1e-6)
   V_norm = V / (magnitude + 1e-6)

   gplt.quiver(X.ravel(), Y.ravel(), U_norm.ravel(), V_norm.ravel(),
               color='red', scale=0.8, head_length=0.15)

   gplt.xlabel('X')
   gplt.ylabel('Y')
   gplt.show()


Annotations and Text Labels
----------------------------

Add Text to Plot
^^^^^^^^^^^^^^^^

Place text at specified coordinates:

.. code-block:: python

   import numpy as np
   import glplot.pyplot as gplt

   x = np.linspace(0, 10, 50)
   y = np.sin(x)

   gplt.plot(x, y)

   # Add text annotation
   gplt.text(5.0, 0.5, 'Peak', fontsize=14, color='red')
   gplt.text(2.5, -0.5, 'Trough', fontsize=14, color='blue')

   gplt.show()

**Parameters**:

- ``x, y``: Text position in data coordinates
- ``s``: Text string
- ``fontsize``: Font size
- ``color``: Text color


Annotate with Arrows
^^^^^^^^^^^^^^^^^^^^

Add annotations with pointing arrows:

.. code-block:: python

   x = np.linspace(0, 10, 100)
   y = np.sin(x)

   gplt.plot(x, y, 'b-')

   # Annotation with arrow
   gplt.annotate(
       'Maximum',
       xy=(np.pi/2, 1.0),                    # Point on data
       xytext=(np.pi/2 + 2, 0.5),            # Text location
       arrowprops={'color': 'red', 'width': 1}
   )

   gplt.show()

**Parameters**:

- ``text_value``: Annotation text
- ``xy``: Point being annotated (data coordinates)
- ``xytext``: Text location
- ``arrowprops``: Dictionary of arrow properties (color, width, etc.)


Horizontal and Vertical Lines
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Add guide lines at specific positions:

.. code-block:: python

   x = np.linspace(0, 10, 100)
   y = np.sin(x)

   gplt.plot(x, y)

   # Add horizontal reference line
   gplt.axhline(y=0, color='red', linestyle='--', linewidth=1, label='y=0')

   # Add vertical reference line
   gplt.axvline(x=np.pi/2, color='green', linestyle='--', linewidth=1, label='x=π/2')

   gplt.legend()
   gplt.show()

Alternatively, use ``hlines`` and ``vlines`` for multiple lines:

.. code-block:: python

   # Multiple horizontal lines
   gplt.hlines([0, 0.5, -0.5], xmin=0, xmax=10, colors='gray', linestyles=':')

   # Multiple vertical lines
   gplt.vlines([1, 3, 5, 7], ymin=-1, ymax=1, colors='gray', linestyles=':')

   gplt.show()


Combining Multiple Plot Types
------------------------------

Complex Multi-Type Plot
^^^^^^^^^^^^^^^^^^^^^^^

Combine different plot types for comprehensive visualization:

.. code-block:: python

   import numpy as np
   import glplot.pyplot as gplt

   # Generate data
   x = np.linspace(0, 10, 100)
   y_true = np.sin(x)
   y_measured = y_true + np.random.normal(0, 0.05, len(x))
   y_std = 0.1

   # Plot components
   gplt.fill_between(x, y_true + y_std, y_true - y_std,
                     color='lightblue', alpha=0.3, label='±1σ')
   gplt.plot(x, y_true, 'b-', linewidth=2, label='Theory')
   gplt.scatter(x[::5], y_measured[::5], color='red', size=8, label='Measurements')
   gplt.errorbar(x[::5], y_measured[::5], yerr=0.1,
                 fmt='none', ecolor='red', elinewidth=1)

   gplt.xlabel('x')
   gplt.ylabel('y')
   gplt.title('Comparison: Theory vs. Measurements')
   gplt.legend()
   gplt.grid(True)
   gplt.show()


Performance Tips
----------------

Large Dataset Handling
^^^^^^^^^^^^^^^^^^^^^^

GLPlot efficiently handles millions of points. For very large datasets:

.. code-block:: python

   # Scatter plot with 1 million points
   n = 1_000_000
   x = np.random.randn(n)
   y = np.random.randn(n)

   gplt.figure("Large Scatter", width=1280, height=800, lod=True, budget=100)
   gplt.scatter(x, y, size=2, alpha=0.3)
   gplt.show()

**Optimization Strategies**:

- Use **LOD** (Level of Detail): ``gplt.figure(..., lod=True)``
- Enable **density visualization**: ``gplt.toggle_density()`` or press ``D`` key
- Reduce point size and use transparency
- Use per-point RGBA colors instead of colormaps for very large sets
- Consider subsampling for exploration, then plot full dataset for final output


Reusing Figures
^^^^^^^^^^^^^^^

Efficiently update plots without creating new windows:

.. code-block:: python

   fig = gplt.figure("Interactive Plot")

   for i in range(10):
       fig.clear()  # Clear previous frame
       x = np.linspace(0, 10, 100)
       y = np.sin(x + i * 0.5)
       gplt.plot(x, y)
       # Window auto-updates; press Esc to exit


See Also
--------

- :doc:`basic-plotting` — Figure management and fundamental operations
- :doc:`quickstart` — Quick introduction with code examples
- :doc:`../api/plotting` — Complete API reference for plotting functions
