from __future__ import annotations

import os
import runpy
from pathlib import Path

from glplot.engine import GPULinePlot

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    cache_dir = RESULTS / ".cache"
    cache_dir.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir / "xdg"))
    # Defensive only: every script here drives its own animation.FuncAnimation.save()
    # call, which is already headless-safe on its own (same no-window fallback every
    # static gallery script relies on) -- this just guards against a script that
    # accidentally left a live plt.show() call active, matching run_gallery.py's own
    # safety net one directory up.
    original_run = GPULinePlot.run
    GPULinePlot.run = lambda self: None
    try:
        for script in sorted(ROOT.glob("[0-9][0-9]_*.py")):
            print(f"running {script.name}")
            runpy.run_path(str(script), run_name="__main__")
    finally:
        GPULinePlot.run = original_run


if __name__ == "__main__":
    main()
