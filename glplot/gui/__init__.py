"""GLPlot mathematical workstation GUI.

Import-safe by construction: nothing here imports imgui (or any other GL-touching
module) at module level, so ``import glplot.gui`` works headless. The handful of names
re-exported below are resolved lazily on first attribute access (PEP 562), which keeps
``from glplot.gui import Workspace`` working without paying for imgui at import time.

Submodules are still imported by explicit path, matching the convention of the other
subpackages::

    from glplot.gui.commands import CommandQueue
    from glplot.gui.history import Command, UndoStack
"""

from __future__ import annotations

from typing import Any

__all__ = ["Command", "CommandQueue", "UndoStack", "Workspace"]

# Attribute name -> (submodule, symbol). Resolved on first access, never at import.
_LAZY = {
    "CommandQueue": ("commands", "CommandQueue"),
    "Command": ("history", "Command"),
    "UndoStack": ("history", "UndoStack"),
    "Workspace": ("workspace", "Workspace"),  # imgui-dependent: imported only on demand
}


def __getattr__(name: str) -> Any:
    """Lazily resolve the re-exported names (PEP 562)."""
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    module = import_module(f"{__name__}.{target[0]}")
    value = getattr(module, target[1])
    globals()[name] = value
    return value


def __dir__() -> list:
    """Expose the lazy names to ``dir()`` and tab completion."""
    return sorted(set(list(globals().keys()) + __all__))
