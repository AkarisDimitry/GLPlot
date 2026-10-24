"""``python -m glplot.mcp`` -- run the MCP server.

A thin shim, like :mod:`glplot.__main__`: the SDK import lives in :mod:`glplot.mcp.server`,
so importing this module (or the package) never needs the optional ``mcp`` extra. It is also
the target of the ``glplot-mcp`` console script, which is why a missing extra is reported
here as one readable line instead of a traceback.
"""

import sys


def main() -> int:
    """Delegate to :func:`glplot.mcp.server.main`."""
    try:
        from .server import main as server_main
    except ImportError as exc:
        print(
            f"glplot-mcp: the MCP server needs the optional 'mcp' package ({exc}). "
            'Install it with: pip install "glplot[mcp]"',
            file=sys.stderr,
        )
        return 2
    return server_main()


if __name__ == "__main__":
    sys.exit(main())
