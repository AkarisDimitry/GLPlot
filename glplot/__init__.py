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

try:
    # Must run before anything imports the `glfw` package (`.engine` does, right below).
    # imgui-bundle points pip's `glfw` at its own bundled libglfw via the PYGLFW_LIBRARY
    # env var, but only if it gets a chance to set that *before* `glfw` loads its own
    # copy -- otherwise both libglfw.3.dylib's end up loaded in the same process, and
    # macOS logs "Class GLFWWindow is implemented in both ..." for every GLFW ObjC class
    # (verified: importing imgui_bundle first eliminates the warning entirely; importing
    # it after `glfw`, as a naive import order would, does not).
    import imgui_bundle as _imgui_bundle  # noqa: F401
except Exception:  # pragma: no cover - imgui_bundle is optional at import time (GL-less use)
    pass

from .engine import GPULinePlot
from .options import BlendMode, EngineOptions, RenderMode

__version__ = "0.1.6"
__author__ = "Juan Manuel Lombardi"
__license__ = "MIT"

__all__ = ["GPULinePlot", "EngineOptions", "RenderMode", "BlendMode"]
