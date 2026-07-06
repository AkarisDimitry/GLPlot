from __future__ import annotations

import os
import runpy
from pathlib import Path

from glplot.engine import GPULinePlot

try:
    from examples.gallery._preview import render_preview
except ModuleNotFoundError:
    from _preview import render_preview


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    cache_dir = RESULTS / ".cache"
    cache_dir.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir / "xdg"))
    original_savefig = GPULinePlot.savefig
    original_run = GPULinePlot.run
    GPULinePlot.savefig = render_preview
    GPULinePlot.run = lambda self: None
    try:
        for script in sorted(ROOT.glob("[0-9][0-9]_*.py")):
            print(f"running {script.name}")
            runpy.run_path(str(script), run_name="__main__")
    finally:
        GPULinePlot.savefig = original_savefig
        GPULinePlot.run = original_run


if __name__ == "__main__":
    main()
