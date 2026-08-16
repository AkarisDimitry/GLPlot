"""Workspace panels.

:class:`~glplot.gui.panels.base.Panel` is re-exported eagerly: it is the contract every
panel implements, and it imports nothing but ``typing``, so it is safe to pull in
headless.

The concrete panels are **not** re-exported. They import imgui, and eagerly importing
them here would make ``import glplot.gui.panels`` fail on a GL-less host. Import them by
explicit module path, matching the convention of the other subpackages::

    from glplot.gui.panels.base import Panel
    from glplot.gui.panels.scene import ScenePanel

:class:`~glplot.gui.panels.scene.ScenePanel`, :class:`~glplot.gui.panels.help.HelpPanel`
and :class:`~glplot.gui.panels.palette.CommandPalette` are additionally available as lazy
attributes (PEP 562, as in ``glplot/gui/__init__.py``) — ``panels.HelpPanel`` resolves on
first access, so the imgui import is paid only when something actually asks for it.

Note the palette is **not** a rail panel: the workspace keeps it in ``ws.palette``, draws
it itself, and must never put it in ``ws.panels`` (it owns its own modal window). It is
exported here because it is a panel module, not because it is one of the panel windows.
"""

from __future__ import annotations

from typing import Any

from .base import Panel

__all__ = [
    "AnnotatePanel",
    "CommandPalette",
    "HelpPanel",
    "Panel",
    "ScenePanel",
    "SelectionPanel",
]

# Attribute name -> (submodule, symbol). Resolved on first access, never at import.
_LAZY = {
    "AnnotatePanel": ("annotate", "AnnotatePanel"),
    "CommandPalette": ("palette", "CommandPalette"),
    "HelpPanel": ("help", "HelpPanel"),
    "ScenePanel": ("scene", "ScenePanel"),
    "SelectionPanel": ("selection", "SelectionPanel"),
}


def __getattr__(name: str) -> Any:
    """Lazily resolve the imgui-dependent panel names (PEP 562)."""
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
