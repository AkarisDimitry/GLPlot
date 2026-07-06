from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from ..core.context import RenderContext
    from ..engine import GPULinePlot


@dataclass
class AxisTicks:
    major: np.ndarray = field(default_factory=lambda: np.array([]))
    labels: List[str] = field(default_factory=list)


class AxisManager:
    """
    Logical manager for coordinate axes.
    Generates 'nice' ticks and labels based on the visible data range.
    """

    def __init__(self, plot: GPULinePlot):
        self.plot = plot
        self.options = plot.options
        self.ticks_x = AxisTicks()
        self.ticks_y = AxisTicks()

    def update(self, ctx: RenderContext) -> None:
        """Recalculate ticks for the current view."""
        # Target ~7 ticks for X, ~6 for Y as suggested by user
        target_x = max(4, int(ctx.width_px / 160))
        target_y = max(4, int(ctx.height_px / 120))

        win = ctx.window_world
        self.ticks_x = self._generate_ticks(win[0], win[1], target_x)
        self.ticks_y = self._generate_ticks(win[2], win[3], target_y)

    def _generate_ticks(self, vmin: float, vmax: float, target_count: int) -> AxisTicks:
        if vmin >= vmax:
            return AxisTicks()

        span = vmax - vmin
        raw_step = span / max(1, target_count)

        # Nice Step Algorithm (1, 2, 5 x 10^n)
        p = 10 ** np.floor(np.log10(raw_step))
        m = raw_step / p

        if m < 1.5:
            step = 1.0 * p
        elif m < 3.5:
            step = 2.0 * p
        elif m < 7.5:
            step = 5.0 * p
        else:
            step = 10.0 * p

        # Calculate start/end
        start = np.ceil(vmin / step) * step
        end = np.floor(vmax / step) * step

        if start > end:
            return AxisTicks()

        major = np.arange(start, end + step / 2, step)

        # Generate labels
        labels = []
        precision = max(0, int(-np.floor(np.log10(step)))) if step < 1 else 0
        fmt = f"{{:.{precision}f}}"
        for v in major:
            labels.append(fmt.format(v))

        return AxisTicks(major=major, labels=labels)
