"""The standalone mathematical workstation — ``python -m glplot``.

Opens an empty GLPlot window with the HUD and workspace up and the Data panel focused,
so the first thing a user sees is somewhere to paste a table or write a function. With a
path argument it loads that file as a dataset first::

    python -m glplot                # empty workstation
    python -m glplot data.csv       # opens with data.csv loaded in the Data panel

Parsing goes through :mod:`glplot.gui.clipboard`, which is the same code path as Ctrl+V:
the delimiter (tab/comma/semicolon/whitespace) and the presence of a header row are both
detected, so ``.csv``, ``.tsv`` and whitespace-aligned ``.txt`` all work with no flags.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import List, Optional, Sequence

from ..engine import GPULinePlot
from .clipboard import parse_table
from .datasets import Column, DataSet

logger = logging.getLogger(__name__)


def load_table_file(path: str) -> DataSet:
    """Parse a delimited text file into a :class:`~glplot.gui.datasets.DataSet`.

    Raises ``ValueError`` with a human-readable message on an unreadable or unparseable
    file — the caller turns that into a CLI error, not a traceback.
    """
    try:
        with open(path, "r", encoding="utf-8-sig", errors="replace") as handle:
            text = handle.read()
    except OSError as exc:
        raise ValueError(f"Could not read {path}: {exc}") from exc

    # parse_table raises ValueError with its own readable message; prefix the file so the
    # user knows which one of several failed.
    try:
        headers, data = parse_table(text)
    except ValueError as exc:
        raise ValueError(f"Could not parse {path}: {exc}") from exc

    columns: List[Column] = [
        Column(headers[i], data[:, i].astype(float)) for i in range(data.shape[1])
    ]
    dataset = DataSet(os.path.splitext(os.path.basename(path))[0] or "data", columns)
    dataset.source = "clipboard"
    return dataset


def build_parser() -> argparse.ArgumentParser:
    """The CLI. Every argument is optional: bare ``python -m glplot`` must work."""
    parser = argparse.ArgumentParser(
        prog="glplot",
        description="Open the GLPlot mathematical workstation.",
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=None,
        help="Optional CSV/TSV/whitespace-delimited file to load into the Data panel.",
    )
    parser.add_argument("--width", type=int, default=1400, help="Window width (default: 1400).")
    parser.add_argument("--height", type=int, default=900, help="Window height (default: 900).")
    parser.add_argument("--light", action="store_true", help="Use the light theme/background.")
    return parser


def create_app(
    path: Optional[str] = None,
    *,
    width: int = 1400,
    height: int = 900,
    light: bool = False,
) -> GPULinePlot:
    """Build the workstation plot without running its loop.

    Split out from :func:`main` so it is testable: constructing ``GPULinePlot`` creates no
    window and touches no GL (``plot.window`` stays None until ``run()``), so this whole
    function is safe headless.
    """
    plot = GPULinePlot(width=width, height=height, title="GLPlot — Workstation")
    plot.options.enable_hud = True
    if light:
        plot.options.light_bg_mode = True
        plot.options.visual.background_color = (0.96, 0.97, 0.98)

    workspace = plot.hud.workspace
    if workspace is None:
        # imgui is a hard dependency, so this means a GL-less import failure. The window
        # would come up with no UI at all, which is worse than saying so.
        raise RuntimeError(
            "The GLPlot workstation needs a working imgui install (pip install 'imgui[glfw]')."
        )

    if path is not None:
        dataset = load_table_file(path)
        workspace.store.add(dataset)
        # The Data panel auto-selects the first dataset, and the empty-scene hint hides
        # itself once the store is non-empty, so there is nothing else to wire here.
    workspace.open["data"] = True
    return plot


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point. Returns a process exit code; never raises for user error."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = build_parser().parse_args(argv)

    try:
        plot = create_app(args.path, width=args.width, height=args.height, light=args.light)
    except (ValueError, RuntimeError) as exc:
        print(f"glplot: {exc}", file=sys.stderr)
        return 2

    plot.run()
    return 0
