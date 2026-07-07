from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, List, Optional, Protocol, Tuple

import numpy as np

if TYPE_CHECKING:
    from .context import RenderContext


@dataclass
class LayerStyle:
    """Encapsulates all non-geometric visual properties of a layer."""

    visible: bool = True
    alpha: float = 1.0
    zorder: int = 0
    pickable: bool = False

    # Colors
    color: Optional[Tuple[float, float, float, float]] = None  # Primary (Lines, edges)
    edge_color: Optional[Tuple[float, float, float, float]] = None  # Edges for patches
    face_color: Optional[Tuple[float, float, float, float]] = None  # Fill for patches

    # Geometry
    line_width: float = 1.0
    point_size: float = 6.0

    # Scatter Polish
    point_outline_enabled: bool = False
    point_outline_color: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    point_outline_width: float = 1.0

    # Colormapping
    use_colormap: bool = False
    cmap: Optional[str] = None
    vmin: Optional[float] = None
    vmax: Optional[float] = None

    # Text
    text_size_px: float = 12.0


@dataclass
class LayerDirtyState:
    """Fine-grained invalidation flags to optimize GPU updates."""

    data_dirty: bool = True
    style_dirty: bool = True
    gpu_dirty: bool = True
    bounds_dirty: bool = True

    def clear(self) -> None:
        self.data_dirty = False
        self.style_dirty = False
        self.gpu_dirty = False
        self.bounds_dirty = False


class CompiledLayer:
    """GPU-ready geometry and cached bounds."""

    def __init__(self, layer_id: int) -> None:
        self.layer_id = layer_id
        self.bounds_world: Optional[Tuple[float, float, float, float]] = None
        self.gpu_initialized: bool = False


class BaseLayer:
    """Abstract base for all visual primitives."""

    def __init__(self, layer_type: str, label: str = "") -> None:
        self.layer_id = uuid.uuid4().int & (1 << 31) - 1
        self.layer_type = layer_type
        self.label = label
        self.style = LayerStyle()
        self.dirty = LayerDirtyState()
        self.bounds_world: Optional[Tuple[float, float, float, float]] = None
        self.translation: Tuple[float, float] = (0.0, 0.0)
        self.metadata: dict[str, Any] = {}

    def get_intrinsic_bounds(self) -> Optional[Tuple[float, float, float, float]]:
        return None


@dataclass
class LineFamilyLayer(BaseLayer):
    """High-performance layer for millions of lines y = ax + b."""

    ab: Optional[np.ndarray] = None
    colors: Optional[np.ndarray] = None
    x_range: Tuple[float, float] = (-1.0, 1.0)

    def __init__(
        self,
        ab: Optional[np.ndarray] = None,
        colors: Optional[np.ndarray] = None,
        x_range: Tuple[float, float] = (-1.0, 1.0),
        label: str = "",
    ) -> None:
        super().__init__(layer_type="line_family", label=label)
        self.ab = ab
        self.colors = colors
        self.x_range = x_range

    def get_intrinsic_bounds(self) -> Optional[Tuple[float, float, float, float]]:
        if self.ab is None or len(self.ab) == 0:
            return None
        x0, x1 = self.x_range
        y_at_x0 = self.ab[:, 0] * x0 + self.ab[:, 1]
        y_at_x1 = self.ab[:, 0] * x1 + self.ab[:, 1]
        return (
            x0,
            x1,
            float(min(np.min(y_at_x0), np.min(y_at_x1))),
            float(max(np.max(y_at_x0), np.max(y_at_x1))),
        )


@dataclass
class ScatterLayer(BaseLayer):
    """Layer for point clouds."""

    pts: Optional[np.ndarray] = None
    colors: Optional[np.ndarray] = None

    def __init__(
        self,
        pts: Optional[np.ndarray] = None,
        colors: Optional[np.ndarray] = None,
        size: float = 6.0,
        label: str = "",
    ) -> None:
        super().__init__(layer_type="scatter", label=label)
        self.pts = pts
        self.colors = colors
        self.style.point_size = size

    def get_intrinsic_bounds(self) -> Optional[Tuple[float, float, float, float]]:
        if self.pts is None or len(self.pts) == 0:
            return None
        return (
            float(np.min(self.pts[:, 0])),
            float(np.max(self.pts[:, 0])),
            float(np.min(self.pts[:, 1])),
            float(np.max(self.pts[:, 1])),
        )


@dataclass
class PolylineLayer(BaseLayer):
    """Layer for connected line segments (Polyline)."""

    pts: Optional[np.ndarray] = None

    def __init__(
        self,
        pts: Optional[np.ndarray] = None,
        color: Optional[Tuple[float, float, float, float]] = None,
        width: float = 1.0,
        label: str = "",
    ) -> None:
        super().__init__(layer_type="polyline", label=label)
        self.pts = pts
        if color:
            self.style.color = color
        self.style.line_width = width

    def get_intrinsic_bounds(self) -> Optional[Tuple[float, float, float, float]]:
        if self.pts is None or len(self.pts) == 0:
            return None
        return (
            float(np.min(self.pts[:, 0])),
            float(np.max(self.pts[:, 0])),
            float(np.min(self.pts[:, 1])),
            float(np.max(self.pts[:, 1])),
        )


@dataclass
class PatchLayer(BaseLayer):
    """Layer for filled areas (tri-strips, bars, rects)."""

    vertices: Optional[np.ndarray] = None  # (N, 2)
    indices: Optional[np.ndarray] = None  # (M,)
    mode: str = "strip"  # "strip", "triangles", "rects"

    def __init__(
        self,
        vertices: Optional[np.ndarray] = None,
        indices: Optional[np.ndarray] = None,
        mode: str = "strip",
        label: str = "",
    ) -> None:
        super().__init__(layer_type="patch", label=label)
        self.vertices = vertices
        self.indices = indices
        self.mode = mode

    def get_intrinsic_bounds(self) -> Optional[Tuple[float, float, float, float]]:
        if self.vertices is None or len(self.vertices) == 0:
            return None
        return (
            float(np.min(self.vertices[:, 0])),
            float(np.max(self.vertices[:, 0])),
            float(np.min(self.vertices[:, 1])),
            float(np.max(self.vertices[:, 1])),
        )


@dataclass
class TextLayer(BaseLayer):
    """Layer for text labels proyected from world coordinates."""

    x: float = 0.0
    y: float = 0.0
    text: str = ""

    def __init__(self, x: float = 0.0, y: float = 0.0, text: str = "", label: str = "") -> None:
        super().__init__(layer_type="text", label=label)
        self.x = x
        self.y = y
        self.text = text

    def get_intrinsic_bounds(self) -> Optional[Tuple[float, float, float, float]]:
        # Text does not participate in autoscale by default
        return None


@dataclass
class Layer3D(BaseLayer):
    """Generic GPU 3D geometry layer."""

    vertices: Optional[np.ndarray] = None  # (N, 3)
    colors: Optional[np.ndarray] = None  # (N, 4)
    indices: Optional[np.ndarray] = None
    primitive: str = "points"  # points, lines, triangles

    def __init__(
        self,
        vertices: Optional[np.ndarray] = None,
        colors: Optional[np.ndarray] = None,
        indices: Optional[np.ndarray] = None,
        primitive: str = "points",
        label: str = "",
        layer_type: str = "geometry3d",
    ) -> None:
        super().__init__(layer_type=layer_type, label=label)
        self.vertices = vertices
        self.colors = colors
        self.indices = indices
        self.primitive = primitive
        self.style.point_size = 3.0

    def get_intrinsic_bounds(self) -> Optional[Tuple[float, float, float, float]]:
        if self.vertices is None or len(self.vertices) == 0:
            return None
        return (
            float(np.min(self.vertices[:, 0])),
            float(np.max(self.vertices[:, 0])),
            float(np.min(self.vertices[:, 1])),
            float(np.max(self.vertices[:, 1])),
        )

    def get_bounds_3d(self) -> Optional[Tuple[float, float, float, float, float, float]]:
        if self.vertices is None or len(self.vertices) == 0:
            return None
        return (
            float(np.min(self.vertices[:, 0])),
            float(np.max(self.vertices[:, 0])),
            float(np.min(self.vertices[:, 1])),
            float(np.max(self.vertices[:, 1])),
            float(np.min(self.vertices[:, 2])),
            float(np.max(self.vertices[:, 2])),
        )
