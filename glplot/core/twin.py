"""Twin axes (``twinx`` / ``twiny``): a second y (or x) scale for one panel.

A panel is projected by ONE orthographic camera, so a second y scale cannot be a second
projection. It is an affine *relation* to the first instead. Each twin layer keeps its true
values in ``metadata["twin_raw"]`` and carries a *baked* copy in its normal ``pts`` /
``vertices``, mapped into the host axis' coordinates::

    host coordinate = a * twin(scaled value) + b          (``TwinAxis.affine``)

Everything downstream -- the renderers, level-of-detail, culling, bounds -- keeps reading
``pts`` in host coordinates and needs no idea twins exist. And because the relation between
the two axes is fixed, zooming or panning the host moves the twin with it, which is what
matplotlib's interactive navigation does to a ``twinx`` pair.

``a`` and ``b`` come from matching the twin's own range (its data, or an explicit
``set_ylim``) to the host window at fit time. See :func:`fit`.

Pure functions over a :class:`~glplot.core.panel.Panel`; nothing here touches OpenGL, and
nothing imports ``pyplot``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from ..utils import scale as _scale

#: Column of a layer's ``(N, 2)`` coordinates that belongs to each axis letter.
AXIS_INDEX = {"x": 0, "y": 1}

#: Layer types whose coordinates can be mapped. Anything else (an image, a line family, a 3D
#: layer) stays on the host axis, and the caller is told.
SUPPORTED_LAYER_TYPES = frozenset({"polyline", "scatter", "patch", "text"})

#: Artists built from a scatter/polyline/patch layer that are nevertheless drawn from their own
#: geometry (an image's ``extent``, a contour's grid), so mapping the points would not move them.
UNSUPPORTED_ARTISTS = frozenset(
    {"imshow", "contour", "contour_line", "contourf", "contourf_fill", "pcolormesh"}
)


@dataclass
class TwinAxis:
    """One duplicated axis of a panel: ``"y"`` for ``twinx()``, ``"x"`` for ``twiny()``.

    Its label, scale, limits and ticks are its own -- the host's are untouched -- and all
    are in the twin's *data* units. ``affine`` is ``None`` until :func:`fit` has run.
    """

    axis: str
    label: str = ""
    label_color: Any = None
    label_fontsize: Optional[float] = None
    tick_color: Any = None
    #: An explicit range in the twin's data units (``set_ylim``); ``None`` fits the data.
    lim: Optional[Tuple[float, float]] = None
    scale: str = "linear"
    scale_params: Optional[dict] = None
    ticks: Optional[Tuple[float, ...]] = None
    tick_labels: Optional[Tuple[str, ...]] = None
    #: A matplotlib ``Formatter`` (called as ``formatter(value, position)``) or ``None``.
    formatter: Any = None
    locator: Any = None
    inverted: bool = False
    #: ``(a, b)``: host scaled coordinate = ``a * twin scaled value + b``.
    affine: Optional[Tuple[float, float]] = None


@dataclass
class SecondaryAxis:
    """``secondary_xaxis`` / ``secondary_yaxis``: ticks in other units along one edge.

    No data of its own: a pair of functions ``(forward, inverse)`` converts the host's units to
    the secondary's, and its ticks are the round numbers *in those units*, placed where the
    inverse says they fall -- so unlike a twin the relation need not be affine (Hz <-> period).
    """

    axis: str
    location: str
    functions: Tuple[Any, Any]
    label: str = ""
    label_color: Any = None
    label_fontsize: Optional[float] = None
    tick_color: Any = None
    ticks: Optional[Tuple[float, ...]] = None
    tick_labels: Optional[Tuple[str, ...]] = None


#: Where each ``location=`` puts the strip, by axis letter.
SECONDARY_LOCATIONS = {
    "x": {"top": "top", "bottom": "bottom"},
    "y": {"left": "left", "right": "right"},
}


def owner_axis(layer: Any) -> Optional[str]:
    """``"x"``/``"y"`` if ``layer`` was drawn on a twin axis, else ``None`` (the host)."""
    return getattr(layer, "metadata", {}).get("twin_axis")


def layers_of(layers: Iterable[Any], axis: str) -> List[Any]:
    """The layers drawn on the twin of ``axis``."""
    return [layer for layer in layers if owner_axis(layer) == axis]


def host_layers(layers: Iterable[Any]) -> List[Any]:
    """The layers drawn on the host axes, in order."""
    return [layer for layer in layers if owner_axis(layer) is None]


# ---------------------------------------------------------------------------------------
# Coordinates: capture the true values, bake the mapped ones
# ---------------------------------------------------------------------------------------


def _read_coords(layer: Any) -> Optional[np.ndarray]:
    kind = layer.layer_type
    if kind in ("polyline", "scatter"):
        pts = getattr(layer, "pts", None)
        return None if pts is None else np.asarray(pts, dtype=np.float64)
    if kind == "patch":
        vertices = getattr(layer, "vertices", None)
        return None if vertices is None else np.asarray(vertices, dtype=np.float64)
    if kind == "text":
        return np.array([[layer.x, layer.y]], dtype=np.float64)
    return None


def capture(layer: Any, axis: str) -> bool:
    """Mark ``layer`` as drawn on the ``axis`` twin and remember its true coordinates.

    Returns False -- leaving the layer alone -- for a layer type that cannot be mapped.
    """
    if layer.layer_type not in SUPPORTED_LAYER_TYPES:
        return False
    if layer.metadata.get("artist") in UNSUPPORTED_ARTISTS:
        return False
    coords = _read_coords(layer)
    if coords is None:
        return False
    layer.metadata["twin_axis"] = axis
    layer.metadata["twin_raw"] = np.array(coords, dtype=np.float64, copy=True)
    return True


def raw_coords(layer: Any) -> Optional[np.ndarray]:
    """The layer's true ``(N, 2)`` coordinates: what the user plotted, not the baked copy."""
    raw = layer.metadata.get("twin_raw")
    return raw if raw is not None else _read_coords(layer)


def _write_coords(layer: Any, coords: np.ndarray) -> None:
    kind = layer.layer_type
    if kind in ("polyline", "scatter"):
        from .layers import _replace_points

        _replace_points(layer, np.ascontiguousarray(coords, dtype=np.float32))
    elif kind == "patch":
        layer.vertices = np.ascontiguousarray(coords, dtype=np.float32)
        layer.dirty.data_dirty = True
        layer.dirty.gpu_dirty = True
        layer.dirty.bounds_dirty = True
    elif kind == "text":
        layer.x, layer.y = float(coords[0, 0]), float(coords[0, 1])


def bake(
    layer: Any,
    axis: str,
    twin: TwinAxis,
    host_scale: str = "linear",
    host_params: Optional[dict] = None,
) -> None:
    """Write the host-space copy of ``layer``'s coordinates, from its true ones.

    The renderers apply the host's own axis scale to ``pts`` at upload, so the mapped value
    (which lives in the host's *scaled* space) is passed back through the scale's inverse.
    """
    if twin.affine is None:
        return
    raw = raw_coords(layer)
    if raw is None:
        return
    a, b = twin.affine
    column = AXIS_INDEX[axis]
    coords = np.array(raw, dtype=np.float64, copy=True)
    scaled = _scale.forward(coords[:, column], twin.scale, twin.scale_params)
    coords[:, column] = _scale.inverse(a * scaled + b, host_scale, host_params)
    _write_coords(layer, coords)


# ---------------------------------------------------------------------------------------
# Ranges and the fit
# ---------------------------------------------------------------------------------------


def scaled_limit(twin: TwinAxis) -> Optional[Tuple[float, float]]:
    """``twin.lim`` in the twin's scaled space, or ``None`` if unset or not representable."""
    if twin.lim is None:
        return None
    lo, hi = _scale.forward(np.array(twin.lim, dtype=np.float64), twin.scale, twin.scale_params)
    if not (np.isfinite(lo) and np.isfinite(hi)) or lo == hi:
        return None
    return float(lo), float(hi)


def data_range(
    layers: Iterable[Any], axis: str, twin: TwinAxis, margin: float
) -> Optional[Tuple[float, float]]:
    """The scaled range that fits the twin's data, padded the way the host autoscale pads."""
    column = AXIS_INDEX[axis]
    lo, hi = np.inf, -np.inf
    for layer in layers:
        if not layer.style.visible:
            continue
        coords = raw_coords(layer)
        if coords is None or len(coords) == 0:
            continue
        values = _scale.forward(coords[:, column], twin.scale, twin.scale_params)
        values = values[np.isfinite(values)]
        if values.size:
            lo, hi = min(lo, float(values.min())), max(hi, float(values.max()))
    if not np.isfinite(lo):
        return None
    pad = (hi - lo) * margin if hi - lo >= 1e-9 else 0.5
    return lo - pad, hi + pad


def derived_range(
    twin: TwinAxis, host_window: Tuple[float, float]
) -> Optional[Tuple[float, float]]:
    """The twin's scaled range currently visible, read back through its affine relation.

    ``host_window`` is the host's ``(low, high)`` along the twinned axis. Always ordered
    ``low < high``; an inverted twin is a sign on ``a``, not a reversed range.
    """
    if twin.affine is None or twin.affine[0] == 0.0:
        return None
    a, b = twin.affine
    lo, hi = (host_window[0] - b) / a, (host_window[1] - b) / a
    return (lo, hi) if lo <= hi else (hi, lo)


def fit(
    panel: Any,
    window: Tuple[float, float, float, float],
    scales: Dict[str, Tuple[str, Optional[dict]]],
    margin: float,
    keep: Optional[Dict[str, Tuple[float, float]]] = None,
    flips: Optional[Dict[str, bool]] = None,
) -> None:
    """(Re)compute every twin's affine against the host ``window`` and re-bake its layers.

    ``window`` is the host's ``(left, right, bottom, top)`` in scaled space. A twin's range is,
    in order: ``keep[axis]`` (the caller is preserving what was on screen), an explicit
    ``set_ylim``, the extent of its data, and failing all of those ``(0, 1)``.
    ``scales`` gives the host's ``(mode, params)`` per axis letter, for the inverse in
    :func:`bake`. ``flips`` says which host axes are inverted: inverting ``ax1`` does not
    invert ``ax2`` in matplotlib, so a twin on an inverted host is drawn the other way round
    in *host* coordinates to stay upright on screen.
    """
    for axis, twin in panel.twins.items():
        layers = layers_of(panel.scene.layers, axis)
        rng = (keep or {}).get(axis) or scaled_limit(twin)
        if rng is None:
            rng = data_range(layers, axis, twin, margin)
        lo, hi = rng if rng is not None else (0.0, 1.0)
        if hi - lo < 1e-12:
            lo, hi = lo - 0.5, hi + 0.5
        host_lo, host_hi = (window[0], window[1]) if axis == "x" else (window[2], window[3])
        if twin.inverted != bool((flips or {}).get(axis, False)):
            slope = (host_lo - host_hi) / (hi - lo)
            twin.affine = (slope, host_hi - slope * lo)
        else:
            slope = (host_hi - host_lo) / (hi - lo)
            twin.affine = (slope, host_lo - slope * lo)
        mode, params = scales.get(axis, ("linear", None))
        for layer in layers:
            bake(layer, axis, twin, mode, params)
