Installation
=============

System Requirements
-------------------

Before installing GLPlot, ensure you have:

- **Python**: 3.9 or later
- **OpenGL**: 3.3+ capable GPU driver
- **Git**: For installing from source

Required Dependencies:
- `numpy <https://numpy.org/>`_ >= 1.23
- `scipy <https://scipy.org/>`_ >= 1.9
- `matplotlib <https://matplotlib.org/>`_ >= 3.6
- `glfw <https://github.com/FlorianBrucker/pyglfw>`_ >= 2.5
- `PyOpenGL <https://pyopengl.sourceforge.net/>`_ >= 3.1.6

Installing from PyPI
---------------------

Once GLPlot is published to PyPI, install via pip:

.. code-block:: bash

   pip install glplot

Installing from Source
----------------------

For development or the latest unstable features:

.. code-block:: bash

   git clone https://github.com/AkarisDimitry/GLPlot.git
   cd GLPlot
   pip install -e .

This installs GLPlot in editable mode, allowing you to modify the code and see changes immediately.

Verifying Installation
----------------------

Test that GLPlot imports successfully:

.. code-block:: python

   import glplot as gplt
   print(gplt.__version__)

Or run a simple plotting example:

.. code-block:: python

   import glplot as gplt
   import numpy as np

   x = np.linspace(0, 10, 1000)
   y = np.sin(x)
   gplt.plot(x, y)
   gplt.show()

GPU Driver Notes
----------------

**Linux (Ubuntu/Debian):**

.. code-block:: bash

   # Install OpenGL development libraries
   sudo apt-get install libgl1-mesa-glx libglu1-mesa

**macOS:**

OpenGL support is built-in. For M1/M2 Macs, PyOpenGL may require special configuration:

.. code-block:: bash

   ARCHFLAGS=-Wno-error=unused-command-line-argument-hard-error-in-future pip install PyOpenGL

**Windows:**

Update your graphics driver via Device Manager or manufacturer's website (NVIDIA/AMD/Intel).

Troubleshooting
---------------

**ImportError: No module named 'glfw'**
   Ensure GLFW is installed: ``pip install glfw``

**ImportError: No OpenGL support**
   Verify OpenGL drivers are installed and updated. Check via:

   .. code-block:: bash

      glxinfo | grep "OpenGL version"  # Linux
      system_profiler SPDisplaysDataType | grep OpenGL  # macOS

**Segmentation Fault on macOS**
   This may indicate a version mismatch. Try:

   .. code-block:: bash

      pip install --force-reinstall --upgrade PyOpenGL

**Performance Issues**
   Ensure your GPU driver is up-to-date. Check your GPU's DirectX/OpenGL capabilities match the minimum requirements (GL 3.3+).

Next Steps
----------

- Read :doc:`quickstart` to plot your first graph
- Explore :doc:`basic-plotting` for common operations
- Check :doc:`../gallery/gallery` for example scripts
