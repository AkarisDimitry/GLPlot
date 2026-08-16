"""Animation support for GLPlot: processes that evolve, and the frames they produce.

Import-safe by construction, exactly like :mod:`glplot.gui`: nothing here imports imgui,
the engine or OpenGL at module level, so ``import glplot.anim`` works headless and the
simulation code underneath is testable without a window. The names re-exported below are
resolved lazily on first attribute access (PEP 562).

Submodules are still imported by explicit path, matching the convention of the other
subpackages::

    from glplot.anim.processes import PROCESSES, Process, State
"""

from __future__ import annotations

from typing import Any

__all__ = ["PROCESSES", "Process", "ProcessError", "ProcessParam", "State", "process"]

# Attribute name -> (submodule, symbol). Resolved on first access, never at import.
_LAZY = {
    "PROCESSES": ("processes", "PROCESSES"),
    "Process": ("processes", "Process"),
    "ProcessError": ("processes", "ProcessError"),
    "ProcessParam": ("processes", "ProcessParam"),
    "State": ("processes", "State"),
    "process": ("processes", "process"),
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
    """Include the lazily-resolved names in ``dir()`` so tab-completion finds them."""
    return sorted(set(globals()) | set(_LAZY))
