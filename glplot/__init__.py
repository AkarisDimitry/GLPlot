"""GLPlot: High-Performance GPU-Accelerated Plotting Library for Python.

A GPU-accelerated plotting library providing a Matplotlib-compatible API
with OpenGL/GLFW backend support. Efficiently handles millions of geometric
primitives through instanced rendering and advanced density visualization.

Key modules:
    pyplot: Matplotlib-style high-level plotting interface
    engine: Core GPU rendering engine
    options: Configuration and rendering mode options

Example:
    >>> import numpy as np
    >>> import glplot.pyplot as plt
    >>> x = np.linspace(0, 10, 100)
    >>> plt.figure("Example")
    >>> plt.plot(x, np.sin(x), "r-", label="sin(x)")
    >>> plt.legend()
    >>> plt.show()

License: MIT
Author: Juan Manuel Lombardi
Repository: https://github.com/AkarisDimitry/GLPlot
"""

from .engine import GPULinePlot
from .options import BlendMode, EngineOptions, RenderMode

__version__ = "0.1.2"
__author__ = "Juan Manuel Lombardi"
__license__ = "MIT"

__all__ = ["GPULinePlot", "EngineOptions", "RenderMode", "BlendMode"]
