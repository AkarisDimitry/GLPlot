from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any
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


@dataclass
class InteractionState:
    drag_active: bool = False
    drag_confirmed: bool = False
    right_drag_active: bool = False
    last_mouse: Tuple[float, float] = (0.0, 0.0)
    press_mouse: Tuple[float, float] = (0.0, 0.0)
    right_press_mouse: Tuple[float, float] = (0.0, 0.0)
    shift_down: bool = False
    ctrl_down: bool = False
    alt_down: bool = False

    drag_mode: str = "pan"  # "pan", "move", "ratio", "rotate3d"
    selected_layer_id: Optional[int] = None
    drag_start_world: Tuple[float, float] = (0.0, 0.0)
    drag_start_translation: Optional[Tuple[float, float]] = None
    drag_start_zoom_x: float = 1.0
    drag_start_zoom_y: float = 1.0
    drag_start_azim: float = 0.0
    drag_start_elev: float = 0.0
    explicit_pick_requested: bool = False  # Set on Shift+Click; bypasses the shift_down gate

    hover_idx: int = -1
    hover_type: Optional[str] = None
    selected_idx: Any = -1
    selected_type: Optional[str] = None

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
