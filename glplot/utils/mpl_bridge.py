from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np


@dataclass
class GLPlotSnapshot:
    """
    Serializable container for a snapshot of a GLPlot viewport.
    This can be used to transfer high-fidelity renders to other
    plotting libraries like Matplotlib.
    """

    rgba: np.ndarray  # H x W x 4 uint8, top row first
    extent: Tuple[float, float, float, float]  # xmin, xmax, ymin, ymax
    xlim: Tuple[float, float]
    ylim: Tuple[float, float]
    width_px: int
    height_px: int
    transparent: bool
    #: True when the scene is a 3D projection (``elev``/``azim``), which has no 2D data
    #: mapping: the raster is a view of a rotated volume, so ``extent`` is only the
    #: camera's window and pinning x/y ticks to it would label the picture with numbers
    #: that mean nothing. Receivers should show such a snapshot as a plain image.
    projected_3d: bool = False


def snapshot_to_matplotlib(
    snapshot: GLPlotSnapshot,
    ax: Optional[Any] = None,
    interpolation: str = "nearest",
    preserve_aspect: bool = False,
    set_limits: bool = True,
    zorder: float = 0.0,
) -> tuple[Any, Any, Any]:
    """
    Standalone utility to embed a GLPlotSnapshot into a Matplotlib axis.
    Does not require a live OpenGL context.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    if snapshot.projected_3d:
        # No data extent: the picture is a projection of a rotated volume, so it goes in
        # as a plain image. 'equal' keeps that projection from being stretched, and the
        # frame comes off rather than carry ticks that would be pure fiction.
        artist = ax.imshow(
            snapshot.rgba,
            aspect="equal",
            interpolation=interpolation,
            zorder=zorder,
        )
        ax.set_axis_off()
        return fig, ax, artist

    xmin, xmax, ymin, ymax = snapshot.extent

    # aspect 'auto' as default
    artist = ax.imshow(
        snapshot.rgba,
        extent=(xmin, xmax, ymin, ymax),
        aspect="auto",
        interpolation=interpolation,
        zorder=zorder,
    )

    if set_limits:
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

    if preserve_aspect:
        # Matches GLPlot's likely aspect if it was consistent
        ax.set_aspect("equal", adjustable="box")

    return fig, ax, artist
