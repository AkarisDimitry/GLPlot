"""``python -m glplot`` — open the mathematical workstation.

Kept to a thin shim so the CLI lives with the rest of the GUI. The import is inside
``main`` rather than at module level: ``glplot/gui/app.py`` pulls in imgui and every
panel, and importing this module must not do that as a side effect.
"""

from __future__ import annotations

import sys


def main() -> int:
    """Delegate to :func:`glplot.gui.app.main`."""
    from .gui.app import main as app_main

    return app_main()


if __name__ == "__main__":
    sys.exit(main())
