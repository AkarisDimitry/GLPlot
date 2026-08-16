Layer System
============

Internal layer abstraction for representing geometric primitives on the GPU.

.. autosummary::
   :toctree: generated

   glplot.core.layers.BaseLayer
   glplot.core.layers.LineFamilyLayer
   glplot.core.layers.ScatterLayer
   glplot.core.layers.PolylineLayer
   glplot.core.layers.PatchLayer
   glplot.core.layers.FunctionLayer
   glplot.core.layers.FractalLayer
   glplot.core.layers.TextLayer
   glplot.core.layers.Layer3D

Base Layer
----------

.. autoclass:: glplot.core.layers.BaseLayer
   :members:
   :undoc-members:
   :show-inheritance:

2D Layers
---------

.. autoclass:: glplot.core.layers.LineFamilyLayer
   :members:
   :show-inheritance:

.. autoclass:: glplot.core.layers.ScatterLayer
   :members:
   :show-inheritance:

.. autoclass:: glplot.core.layers.PolylineLayer
   :members:
   :show-inheritance:

.. autoclass:: glplot.core.layers.PatchLayer
   :members:
   :show-inheritance:

View-Driven Layers
------------------

Layers whose contents are a function of the **current view** and are recomputed when it
changes, rather than sampled once at creation. See :func:`glplot.pyplot.function` and
:func:`glplot.pyplot.mandelbrot`.

.. autoclass:: glplot.core.layers.FunctionLayer
   :members:
   :show-inheritance:

.. autoclass:: glplot.core.layers.FractalLayer
   :members:
   :show-inheritance:

3D & Text Layers
-----------------

.. autoclass:: glplot.core.layers.TextLayer
   :members:
   :show-inheritance:

.. autoclass:: glplot.core.layers.Layer3D
   :members:
   :show-inheritance:

Layer Style
-----------

.. autoclass:: glplot.core.layers.LayerStyle
   :members:
   :show-inheritance:
