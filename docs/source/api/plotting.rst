Plotting Functions (pyplot)
=============================

High-level pyplot-style plotting interface, compatible with Matplotlib conventions.

.. autosummary::
   :toctree: generated
   :template: autosummary/module.rst

   glplot.pyplot

2D Plotting Functions
---------------------

.. autofunction:: glplot.pyplot.plot
.. autofunction:: glplot.pyplot.scatter
.. autofunction:: glplot.pyplot.bar
.. autofunction:: glplot.pyplot.hist
.. autofunction:: glplot.pyplot.errorbar
.. autofunction:: glplot.pyplot.hlines
.. autofunction:: glplot.pyplot.vlines
.. autofunction:: glplot.pyplot.step

View-Driven Plots
-----------------

Plots that are **recomputed from the current view** rather than stored as fixed samples,
so zooming reveals detail that was never computed before. These have no matplotlib
equivalent — matplotlib's zoom magnifies what it was given.

.. autofunction:: glplot.pyplot.function
.. autofunction:: glplot.pyplot.mandelbrot
.. autofunction:: glplot.pyplot.julia

Matrix & Grid Visualization
-----------------------------

.. autofunction:: glplot.pyplot.imshow
.. autofunction:: glplot.pyplot.pcolormesh
.. autofunction:: glplot.pyplot.contour
.. autofunction:: glplot.pyplot.contourf

3D Visualization Functions
----------------------------

.. autofunction:: glplot.pyplot.plot3d
.. autofunction:: glplot.pyplot.scatter3d
.. autofunction:: glplot.pyplot.bar3d
.. autofunction:: glplot.pyplot.plot_surface
.. autofunction:: glplot.pyplot.plot_wireframe
.. autofunction:: glplot.pyplot.quiver3d

Vector Field Visualization
----------------------------

.. autofunction:: glplot.pyplot.quiver
.. autofunction:: glplot.pyplot.arrow

Annotations & Text
-------------------

.. autofunction:: glplot.pyplot.text
.. autofunction:: glplot.pyplot.annotate
