"""``python -m glplot.mcp`` -- run the MCP server (or ``config`` / ``doctor`` / ``tools``).

A thin shim over :mod:`glplot.mcp.cli`, which needs nothing beyond GLPlot itself: with the optional
``mcp`` package installed the official-SDK server runs, without it the built-in stdio server does,
so ``pip install glplot`` is enough for an MCP client. It is also the target of the ``glplot-mcp``
console script.
"""

import sys


def main() -> int:
    """Delegate to :func:`glplot.mcp.cli.main`."""
    from .cli import main as cli_main

    return cli_main()


if __name__ == "__main__":
    sys.exit(main())
