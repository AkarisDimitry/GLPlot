from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

import numpy as np

from .layers import BaseLayer


@dataclass
class LineDataset:
    ab: Optional[np.ndarray] = None  # shape (N, 2)
    colors: Optional[np.ndarray] = None  # shape (N, 4)
    x_range: Tuple[float, float] = (-3.0, 3.0)

    def validate(self) -> None:
        if self.ab is None:
            return
        if self.ab.ndim != 2 or self.ab.shape[1] != 2:
            raise ValueError("ab must have shape (N,2)")
        if self.ab.dtype != np.float32:
            raise ValueError("ab must be float32")
        if self.colors is not None:
            if self.colors.shape != (self.ab.shape[0], 4):
                raise ValueError("colors must have shape (N,4)")
            if self.colors.dtype != np.float32:
                raise ValueError("colors must be float32")

    @property
    def count(self) -> int:
        return 0 if self.ab is None else int(self.ab.shape[0])


@dataclass
class ScatterDataset:
    pts: Optional[np.ndarray] = None  # shape (M,2)
    colors: Optional[np.ndarray] = None  # shape (M,4)
    size: float = 5.0


@dataclass
class StripDataset:
    pts: Optional[np.ndarray] = None  # shape (K,2)
    color: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)


@dataclass
class SceneData:
    layers: List[BaseLayer] = field(default_factory=list)
    lines: LineDataset = field(default_factory=LineDataset)
    scatters: List[ScatterDataset] = field(default_factory=list)
    strips: List[StripDataset] = field(default_factory=list)
    texts: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class CameraState:
    cx: float = 0.0
    cy: float = 0.0
    zoom_x: float = 1.0
    zoom_y: float = 1.0
    zoom_min: float = 1e-12
    zoom_max: float = 1e12

    @property
    def zoom(self) -> float:
        """Legacy property: returns the vertical zoom."""
        return self.zoom_y

    @zoom.setter
    def zoom(self, value: float) -> None:
        """Legacy setter: scales both axes proportionally to preserve anisotropy."""
        if self.zoom_y == 0:
            self.zoom_x = self.zoom_y = float(value)
        else:
            factor = float(value) / self.zoom_y
            self.zoom_x *= factor
            self.zoom_y *= factor


#: What ``Selection`` accepts wherever it takes element indices.
IndexLike = Union[int, np.integer, Iterable[int], np.ndarray]


def _normalize_indices(indices: IndexLike) -> Set[int]:
    """Coerce a scalar / iterable / ndarray of element indices to a set of ints."""
    if isinstance(indices, (int, np.integer)):
        return {int(indices)}
    if isinstance(indices, np.ndarray):
        return {int(i) for i in indices.ravel()}
    return {int(i) for i in indices}


class Selection:
    """Selected data elements, per layer.

    Element ids are indices into the owning layer's arrays, using the same
    scheme the picking pass encodes (``managers/picking.py``): a point index
    for ``scatter`` layers and a line index for ``line_family`` layers.
    ``polyline`` and ``patch`` layers contribute a single element -- index 0 --
    standing for the whole layer.

    Keys are stable ``layer.layer_id`` values, never list positions, so a
    selection survives reordering. Layers deleted from the scene must be
    dropped with :meth:`discard_layer`.
    """

    __slots__ = ("by_layer",)

    def __init__(self) -> None:
        self.by_layer: Dict[int, Set[int]] = {}

    # -- mutation ------------------------------------------------------

    def add(self, layer_id: int, indices: IndexLike) -> None:
        """Add ``indices`` to the selection for ``layer_id``."""
        new = _normalize_indices(indices)
        if not new:
            return
        self.by_layer.setdefault(int(layer_id), set()).update(new)

    def remove(self, layer_id: int, indices: IndexLike) -> None:
        """Remove ``indices`` from ``layer_id``. Unselected indices are ignored."""
        key = int(layer_id)
        current = self.by_layer.get(key)
        if current is None:
            return
        current -= _normalize_indices(indices)
        if not current:
            del self.by_layer[key]

    def toggle(self, layer_id: int, indices: IndexLike) -> None:
        """Toggle ``indices`` for ``layer_id`` (selected ones go out, the rest come in)."""
        key = int(layer_id)
        current = self.by_layer.get(key, set())
        self.by_layer[key] = current ^ _normalize_indices(indices)
        if not self.by_layer[key]:
            del self.by_layer[key]

    def set(self, layer_id: int, indices: IndexLike) -> None:
        """Replace the WHOLE selection with ``indices`` on ``layer_id`` alone."""
        self.by_layer = {}
        self.add(layer_id, indices)

    def set_all(self, mapping: Dict[int, IndexLike]) -> None:
        """Replace the whole selection with ``{layer_id: indices}``."""
        self.by_layer = {}
        self.update(mapping)

    def update(self, mapping: Dict[int, IndexLike]) -> None:
        """Merge ``{layer_id: indices}`` into the selection additively."""
        for layer_id, indices in mapping.items():
            self.add(layer_id, indices)

    def invert(self, layer_id: int, n: int) -> None:
        """Invert the selection for ``layer_id`` over an element count of ``n``."""
        key = int(layer_id)
        current = self.by_layer.get(key, set())
        new = {i for i in range(int(n)) if i not in current}
        if new:
            self.by_layer[key] = new
        else:
            self.by_layer.pop(key, None)

    def discard_layer(self, layer_id: int) -> None:
        """Drop every element of ``layer_id``. Call this when a layer is removed."""
        self.by_layer.pop(int(layer_id), None)

    def clear(self) -> None:
        """Deselect everything."""
        self.by_layer = {}

    # -- queries -------------------------------------------------------

    def count(self) -> int:
        """Total number of selected elements across all layers."""
        return sum(len(v) for v in self.by_layer.values())

    def count_in(self, layer_id: int) -> int:
        """Number of selected elements in ``layer_id``."""
        return len(self.by_layer.get(int(layer_id), ()))

    def layers(self) -> List[int]:
        """Layer ids holding at least one selected element, in insertion order."""
        return [k for k, v in self.by_layer.items() if v]

    def contains(self, layer_id: int, idx: int) -> bool:
        """True when element ``idx`` of ``layer_id`` is selected."""
        return int(idx) in self.by_layer.get(int(layer_id), ())

    def indices(self, layer_id: int) -> np.ndarray:
        """Sorted int64 array of the selected indices in ``layer_id``."""
        return np.array(sorted(self.by_layer.get(int(layer_id), ())), dtype=np.int64)

    def as_mask(self, layer_id: int, n: int) -> np.ndarray:
        """Boolean mask of length ``n`` marking the selected elements of ``layer_id``.

        Indices outside ``[0, n)`` are ignored, so a stale selection cannot
        raise against a shrunken layer.
        """
        n = int(n)
        mask = np.zeros(max(n, 0), dtype=bool)
        idx = self.indices(layer_id)
        if idx.size and n > 0:
            mask[idx[(idx >= 0) & (idx < n)]] = True
        return mask

    def is_empty(self) -> bool:
        """True when nothing at all is selected."""
        return not any(self.by_layer.values())

    def copy(self) -> "Selection":
        """Deep-enough copy: independent per-layer sets."""
        other = Selection()
        other.by_layer = {k: set(v) for k, v in self.by_layer.items()}
        return other

    def __len__(self) -> int:
        return self.count()

    def __repr__(self) -> str:
        return f"Selection({self.count()} elements across {len(self.layers())} layers)"


@dataclass
class InteractionState:
    drag_active: bool = False
    drag_confirmed: bool = False
    right_drag_active: bool = False
    #: Middle-button drag. Its own flag rather than a mode of ``drag_active``: the middle
    #: button can be pressed and released while the left one is held, and folding it into
    #: the left button's state machine would let one release cancel the other's gesture.
    middle_drag_active: bool = False
    last_mouse: Tuple[float, float] = (0.0, 0.0)
    press_mouse: Tuple[float, float] = (0.0, 0.0)
    right_press_mouse: Tuple[float, float] = (0.0, 0.0)
    shift_down: bool = False
    ctrl_down: bool = False
    alt_down: bool = False

    #: "pan", "move", "ratio", "marquee" (2D) or "rotate3d", "pan3d", "roll3d" (3D).
    drag_mode: str = "pan"
    selected_layer_id: Optional[int] = None
    drag_start_world: Tuple[float, float] = (0.0, 0.0)
    drag_start_translation: Optional[Tuple[float, float]] = None
    drag_start_zoom_x: float = 1.0
    drag_start_zoom_y: float = 1.0
    drag_start_azim: float = 0.0
    drag_start_elev: float = 0.0
    drag_start_roll: float = 0.0
    explicit_pick_requested: bool = False  # Set on Shift+Click; bypasses the shift_down gate
    # Shift press seen; the pick is deferred to RELEASE because the gesture may
    # still turn out to be a marquee drag rather than a click.
    pick_press_requested: bool = False
    # Set only by a completed Shift+Click: lets the deferred pick edit `selection`.
    selection_pick_requested: bool = False

    hover_idx: int = -1
    hover_type: Optional[str] = None
    selected_idx: Any = -1
    selected_type: Optional[str] = None

    # Multi-element selection (Shift+Click = one, Shift+Drag = marquee).
    # selected_idx/selected_type above stay as the single-pick legacy mirror.
    selection: Selection = field(default_factory=Selection)
    # Set at press time from MOD_ALT: additive/toggle instead of replace.
    selection_additive: bool = False

    last_hover_pick_time: float = 0.0
    hover_resume_time: float = 0.0


@dataclass
class CacheState:
    active: bool = False
    capture_window: Optional[Tuple[float, float, float, float]] = None
    refresh_requested: bool = False
    last_capture_time: float = 0.0
    release_deadline: float = 0.0


@dataclass
class FrameState:
    dirty_scene: bool = True
    dirty_ui: bool = True
    dirty_pick: bool = True
    last_frame_time: float = 0.0
    fps_estimate: float = 0.0
