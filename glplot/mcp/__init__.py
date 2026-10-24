"""Model Context Protocol (MCP) server: let a language model make plots with GLPlot.

Run it with ``python -m glplot.mcp`` (or the ``glplot-mcp`` script) and point an MCP
client at it. The model gets tools to build figures and datasets, add plots, restyle, and
*look at the result* -- every render comes back as an image, so it can iterate on its own
output.

The package is layered so that only one file needs the optional ``mcp`` SDK:

* :mod:`glplot.mcp.state` -- limits, the file sandbox, datasets, bounded expressions.
* :mod:`glplot.mcp.toolbox` -- every tool as a plain method (importable and testable with
  no SDK installed).
* :mod:`glplot.mcp.server` -- describes those methods to the protocol. The only importer of
  ``mcp``.

Importing this package is cheap and never imports the SDK.
"""

from __future__ import annotations

from typing import Any

__all__ = ["Limits", "McpError", "Toolbox", "build_server", "main"]


def __getattr__(name: str) -> Any:
    """Resolve the public names lazily so ``import glplot.mcp`` stays light."""
    if name in ("Limits", "McpError"):
        from . import state

        return getattr(state, name)
    if name == "Toolbox":
        from .toolbox import Toolbox

        return Toolbox
    if name in ("build_server", "main"):
        from . import server

        return getattr(server, name)
    raise AttributeError(f"module 'glplot.mcp' has no attribute {name!r}")
