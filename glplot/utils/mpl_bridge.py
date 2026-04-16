from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np

@dataclass
class GLPlotSnapshot:
    """
    Serializable container for a snapshot of a GLPlot viewport.
    This can be used to transfer high-fidelity renders to other 
    plotting libraries like Matplotlib.
    """
    rgba: np.ndarray                     # H x W x 4 uint8
    extent: Tuple[float, float, float, float]   # xmin, xmax, ymin, ymax
    xlim: Tuple[float, float]
    ylim: Tuple[float, float]
    width_px: int
    height_px: int
    transparent: bool

def snapshot_to_matplotlib(
    snapshot: GLPlotSnapshot, 
    ax=None, 
    interpolation: str = "nearest",
    preserve_aspect: bool = False,
    set_limits: bool = True,
    zorder: float = 0.0
):
    """
    Standalone utility to embed a GLPlotSnapshot into a Matplotlib axis.
    Does not require a live OpenGL context.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    xmin, xmax, ymin, ymax = snapshot.extent

    # aspect 'auto' as default
    artist = ax.imshow(
        snapshot.rgba,
        extent=(xmin, xmax, ymin, ymax),
        aspect='auto',
        interpolation=interpolation,
        zorder=zorder
    )

    if set_limits:
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

    if preserve_aspect:
        # Matches GLPlot's likely aspect if it was consistent
        ax.set_aspect("equal", adjustable="box")

    return fig, ax, artist
