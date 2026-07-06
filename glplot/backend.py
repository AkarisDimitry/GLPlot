"""
Backward compatibility shim for glplot.backend.
New code should import from glplot or specific modules.
"""

import warnings

from .engine import GPULinePlot
from .options import BlendMode, EngineOptions, RenderMode

warnings.warn(
    "Importing from glplot.backend is deprecated. " "Please import from 'glplot' directly.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["GPULinePlot", "EngineOptions", "RenderMode", "BlendMode"]
