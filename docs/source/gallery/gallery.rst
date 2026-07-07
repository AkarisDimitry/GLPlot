Gallery & Examples
==================

This gallery showcases GLPlot's capabilities with runnable examples.

Quick Start Examples
--------------------

The fastest way to see GLPlot in action:

.. code-block:: bash

   python examples/showcase/01_colorful_particles_2d.py  # 100k particles
   python examples/showcase/02_mandelbrot_zoom_2d.py    # Mandelbrot fractal
   python examples/showcase/03_spinning_torus_3d.py     # 3D torus
   python examples/showcase/04_cosmic_sphere_3d.py      # 3D sphere with noise

See `examples/showcase/ <https://github.com/AkarisDimitry/GLPlot/tree/main/examples/showcase>`_
for the complete showcase gallery with detailed descriptions.

Key Features Demonstrated
--------------------------

**2D Visualization**
   - Massive scatter plots with 220k+ points and smooth color mapping
   - Line plots with millions of data points
   - Histogram and density visualization
   - Image display (imshow, pcolormesh)

**3D Visualization**
   - Point clouds with 100k+ points
   - Surface and wireframe plots
   - 3D vector fields
   - Volumetric data rendering

**Performance**
   - Real-time interaction with 1M+ points
   - Level-of-detail (LOD) rendering
   - Density-based visualization for extreme datasets
   - Screen-space ambient occlusion (SSAO)

**Integration**
   - Matplotlib compatibility
   - High-resolution export
   - Real-time plotting loops
   - Embedding in applications

Dataset Sizes Tested
--------------------

GLPlot has been tested with:

- **100k points**: Smooth 60+ FPS interaction
- **1M points**: Interactive with minor latency
- **10M points**: Requires density visualization
- **100M+ points**: Special handling recommended

See :doc:`../guide/performance-tips` for optimization strategies.

Running Examples Locally
-------------------------

Clone and explore the repository:

.. code-block:: bash

   git clone https://github.com/AkarisDimitry/GLPlot.git
   cd GLPlot
   pip install -e .

   # Run showcase examples
   cd examples/showcase
   python 01_colorful_particles_2d.py

   # Run gallery examples (advanced)
   cd ../gallery
   python <example_name>.py

Code Repository
---------------

For more examples and detailed source code, visit the
`GitHub repository <https://github.com/AkarisDimitry/GLPlot>`_.

Building Your Own Examples
---------------------------

To create your own interactive plots, start with the :doc:`../guide/quickstart`
and then refer to:

- :doc:`../guide/2d-plotting` for 2D specific functions
- :doc:`../guide/3d-visualization` for 3D plotting
- :doc:`../guide/advanced-features` for performance tuning
- :doc:`../api/plotting` for the complete API reference

External Resources
-------------------

- `NumPy Documentation <https://numpy.org/doc/>`_ - Data manipulation
- `Matplotlib Documentation <https://matplotlib.org/stable/>`_ - Compatibility reference
- `OpenGL Tutorial <https://learnopengl.com/>`_ - Understanding GPU rendering concepts
- `GLFW Documentation <https://www.glfw.org/>`_ - Window and input handling
