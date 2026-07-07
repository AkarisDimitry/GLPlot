Basic Plotting Operations
==========================

Figure Management
-----------------

Creating and Managing Figures
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Create a new figure with optional title:

.. code-block:: python

   import glplot as gplt

   fig = gplt.figure("My Plot", width=1280, height=800)
   fig.plot([1, 2, 3], [1, 2, 3])

Get the current active figure:

.. code-block:: python

   fig = gplt.gcf()  # Get current figure

Clear a figure:

.. code-block:: python

   fig.clear()  # Remove all layers
   fig.plot([1, 2, 3], [1, 2, 3])  # Add new plot

Close a figure:

.. code-block:: python

   fig.close()

Axes & Labels
-------------

Setting Axis Labels and Title
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import glplot as gplt
   import numpy as np

   x = np.linspace(0, 10, 100)
   y = np.sin(x)

   gplt.plot(x, y)
   gplt.xlabel('Time (s)')
   gplt.ylabel('Amplitude')
   gplt.title('Sinusoidal Signal')
   gplt.show()

Setting Axis Limits
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   gplt.plot(x, y)
   gplt.xlim(0, 10)  # Set x-axis limits
   gplt.ylim(-1.5, 1.5)  # Set y-axis limits
   gplt.show()

Grid & Legends
^^^^^^^^^^^^^^

.. code-block:: python

   gplt.plot([1, 2, 3, 4], [1, 4, 2, 3], label='Data 1')
   gplt.plot([1, 2, 3, 4], [2, 3, 1, 4], label='Data 2')
   gplt.grid(True)  # Show grid
   gplt.legend()  # Show legend
   gplt.show()

Color & Styling
---------------

Format Strings (Matplotlib-style)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   gplt.plot([1, 2, 3], [1, 2, 3], 'r-')   # Red solid line
   gplt.plot([1, 2, 3], [3, 2, 1], 'b--')  # Blue dashed line
   gplt.plot([1, 2, 3], [2, 1, 3], 'go')   # Green circles
   gplt.plot([1, 2, 3], [1, 3, 2], 'k^')   # Black triangles
   gplt.show()

**Format String Syntax**: ``'[color][linestyle][marker]'``

Colors:
- ``r``, ``g``, ``b``, ``c``, ``m``, ``y``, ``k``, ``w`` (or hex: ``#FF0000``)

Line Styles:
- ``-`` (solid), ``--`` (dashed), ``-.`` (dash-dot), ``:`` (dotted)

Markers:
- ``o`` (circle), ``s`` (square), ``^`` (triangle), ``+`` (plus), ``x`` (cross), etc.

Named Color Parameters
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Pass color as keyword argument
   gplt.plot([1, 2, 3], [1, 2, 3], color='red', linewidth=2, label='Line 1')
   gplt.scatter([1, 2, 3], [3, 2, 1], color='blue', size=15, alpha=0.7, label='Points')
   gplt.legend()
   gplt.show()

Transparency (Alpha)
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   x = np.linspace(0, 10, 100)
   for i in range(5):
       y = np.sin(x + i)
       gplt.plot(x, y, alpha=0.2 * (i + 1))  # Gradually more opaque
   gplt.show()

Data Types & Arrays
-------------------

Supported Input Types
^^^^^^^^^^^^^^^^^^^^^

GLPlot accepts various array-like inputs:

.. code-block:: python

   import glplot as gplt
   import numpy as np

   # Python lists
   gplt.plot([1, 2, 3], [1, 4, 2])

   # NumPy arrays
   gplt.plot(np.array([1, 2, 3]), np.array([1, 4, 2]))

   # Even 2D arrays (treated as separate series for some functions)
   X = np.random.rand(100, 50)
   gplt.imshow(X)

   gplt.show()

Multi-Series Plots
-------------------

Adding Multiple Series
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   x = np.linspace(0, 2*np.pi, 100)

   gplt.plot(x, np.sin(x), 'r-', label='sin(x)')
   gplt.plot(x, np.cos(x), 'b-', label='cos(x)')
   gplt.plot(x, np.tan(x), 'g-', label='tan(x)')

   gplt.legend()
   gplt.show()

Using the Figure Object Directly
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   fig = gplt.figure("Multi-Plot")

   x = np.linspace(0, 10, 100)
   for i in range(5):
       y = np.sin(x - i * np.pi/5)
       fig.plot(x, y, label=f'Wave {i+1}')

   fig.legend()
   fig.show()

Clearing & Resetting
---------------------

Clear All Layers
^^^^^^^^^^^^^^^^

.. code-block:: python

   fig = gplt.gcf()
   fig.clear()  # Remove all plots, layers, etc.

Reset View to Fit All Data
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   gplt.autoscale()  # Auto-scale axes to fit data
   # Or press 'Home' / 'R' key interactively

Reset View to Default
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   gplt.reset_view()  # Reset pan/zoom to initial state

Interactive Features
---------------------

Show/Hide HUD (Head-Up Display)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   fig = gplt.gcf()
   fig.set_hud_enabled(True)  # Show HUD
   # Or press H key interactively

Toggle Density Visualization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   fig = gplt.gcf()
   fig.toggle_density()  # Switch density on/off
   # Or press D key interactively

Accessing Plot Data
--------------------

Get Figure/Plot Object
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   fig = gplt.gcf()  # Current figure
   print(fig.width, fig.height)  # Window size
   print(fig.N)  # Number of lines plotted
   print(fig.scene.layers)  # All layers

Accessing Layer Data
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   layer = gplt.plot([1, 2, 3], [1, 2, 3])
   print(layer[0].label)  # Layer label
   print(layer[0].style)  # Style information (alpha, color, etc.)
   print(layer[0].metadata)  # Additional metadata

See Also
--------

- :doc:`2d-plotting` for advanced 2D operations
- :doc:`3d-visualization` for 3D plotting
- :doc:`advanced-features` for performance tuning and special effects
