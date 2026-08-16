from __future__ import annotations

from enum import Flag, auto
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple, Type

import numpy as np

if TYPE_CHECKING:
    from ..engine import GPULinePlot
    from ..core.layers import BaseLayer
    from ..options import EngineOptions

from ..renderers.axis import AxisRenderer
from ..renderers.fractal import FractalRenderer
from ..renderers.geometry3d import Geometry3DRenderer
from ..renderers.line_family import LineFamilyRenderer
from ..renderers.patch import PatchRenderer
from ..renderers.polyline import PolylineRenderer
from ..renderers.scatter import ScatterRenderer
from ..renderers.text import TextRenderer


class LayerCapability(Flag):
    """Flags defining what a layer can participate in."""

    NONE = 0
    EXACT = auto()  # Can be rendered in the main pass
    DENSITY = auto()  # Can be rendered in the density pass
    PICKING = auto()  # Can be rendered in the picking pass
    EXPORT = auto()  # Can be exported to image
    OVERLAY = auto()  # Is an overlay (HUD-like)


class RendererManager:
    """
    Orchestrates the rendering of layers, sorting by z-order,
    and dispatching to specialized primitive renderers.
    """

    def __init__(self, plot: GPULinePlot) -> None:
        self.plot = plot
        self.options = plot.options

        # Capability-based renderer registration
        # Map: layer_type -> renderer instance
        self.renderers: Dict[str, Any] = {}

        # Performance cache
        self._sorted_layers: List[BaseLayer] = []
        self._last_scene_hash: int = 0

    def initialize(self) -> None:
        """Initialize all registered primitive renderers."""
        self.renderers["line_family"] = LineFamilyRenderer(self.options)
        self.renderers["polyline"] = PolylineRenderer(self.options)
        self.renderers["scatter"] = ScatterRenderer(self.options)
        self.renderers["patch"] = PatchRenderer(self.options)
        self.renderers["text"] = TextRenderer(self.options)
        self.renderers["axis"] = AxisRenderer(self.options)
        geometry3d = Geometry3DRenderer(self.options)
        # ``geometry3d`` is ``Layer3D``'s own default ``layer_type``, so a layer built by
        # constructing one directly had no renderer registered and silently drew nothing.
        self.renderers["geometry3d"] = geometry3d
        self.renderers["scatter3d"] = geometry3d
        # ``surface3d`` was already accepted by ``datasets._LAYER_BINDINGS`` but never
        # routed here, which made it the one 3D layer type the editor could bind and the
        # renderer could not draw.
        self.renderers["surface3d"] = geometry3d
        self.renderers["mesh3d"] = geometry3d
        self.renderers["wireframe3d"] = geometry3d
        self.renderers["bars3d"] = geometry3d
        self.renderers["volume3d"] = geometry3d
        # The GPU escape-time field (Mandelbrot / Julia). A view-driven layer: its pixels
        # are computed per screen fragment every frame, so it re-detailes as the camera
        # zooms. See ``renderers/fractal.py``.
        self.renderers["fractal"] = FractalRenderer(self.options)
        initialized = set()
        for renderer in self.renderers.values():
            if id(renderer) in initialized:
                continue
            initialized.add(id(renderer))
            renderer.initialize()

    def filter_layers(
        self, layers: Iterable[BaseLayer], capability: LayerCapability
    ) -> List[BaseLayer]:
        """
        Filter and sort layers based on visibility and capability.
        This fulfils the Phase 2 Capability Filtering requirement.
        """
        # 1. Filter by visibility and capability metadata
        # (In V1, we assume all layers support EXACT/EXPORT unless specified)
        eligible = []
        for l in layers:
            if not l.style.visible:
                continue

            # Capability check
            # Default to EXACT | EXPORT | DENSITY if not specified
            caps = l.metadata.get(
                "capabilities",
                LayerCapability.EXACT | LayerCapability.EXPORT | LayerCapability.DENSITY,
            )
            if capability in caps:
                eligible.append(l)

        # 2. Sort by zorder, maintaining stable insertion order for ties
        return sorted(eligible, key=lambda l: l.style.zorder)

    def draw_density(
        self,
        layers: List[BaseLayer],
        context: Any,
        target_fbo: int = 0,
        target_size: Optional[Tuple[int, int]] = None,
        target_viewport: Optional[Tuple[int, int, int, int]] = None,
    ) -> None:
        """Modular DENSITY pass render loop."""
        sorted_layers = self.filter_layers(layers, LayerCapability.DENSITY)
        tint = self.density_tint_active(sorted_layers)

        # 1. Prepare the density manager for accumulation
        target = self.plot.density_renderer.accum_target
        self.plot.density_renderer.begin_accum()

        # 2. Accumulate each layer (with context override for the density surface size)
        old_w, old_h = context.fb_width, context.fb_height
        context.fb_width, context.fb_height = target.width, target.height

        try:
            for layer in sorted_layers:
                renderer = self.renderers.get(layer.layer_type)
                if renderer and hasattr(renderer, "draw_density"):
                    renderer.draw_density(layer, context)
        finally:
            context.fb_width, context.fb_height = old_w, old_h

        # 3. Resolve to the target FBO
        self.plot.density_renderer.resolve(
            target_fbo=target_fbo,
            target_size=target_size,
            target_viewport=target_viewport,
            tint=tint,
        )

    @staticmethod
    def density_tint_active(layers: Iterable[BaseLayer]) -> bool:
        """Whether this density pass should be painted in the layers' own colours.

        On when any participating layer was given a colour *by the caller* --
        ``scatter(color=...)``, ``scatter(c=...)``, ``plot(color=...)``. That is the whole
        rule, and it is deliberately about what was asked for rather than about what colour
        came out: every layer ends up with a colour (a cycle colour, or scatter's default
        black), so "has colours" is true of everything and would tint every density plot
        ever made -- including the ones whose layers are black, which would paint the image
        in black on black.

        A layer that inherited its colour from the cycle therefore keeps the colormap, and
        so do the bulk verbs (``plot_lines`` and friends) where the heatmap *is* the point
        and the colours are usually a flat fill chosen to weigh every line the same.
        """
        return any(l.metadata.get("explicit_color") for l in layers)

    def draw_exact(self, layers: List[BaseLayer], context: Any) -> None:
        """Main EXACT pass render loop."""
        sorted_layers = self.filter_layers(layers, LayerCapability.EXACT)

        # Separate system overlay layers so they render in the correct order:
        #   1. floor3d  — behind all data (semi-transparent floor plane)
        #   2. data     — user geometry
        #   3. axis3d   — bounding-box wireframe always on top
        floor_layers = [l for l in sorted_layers if l.metadata.get("artist") == "floor3d"]
        axis_layers = [l for l in sorted_layers if l.metadata.get("artist") == "axis3d"]
        data_layers = self._order_by_opacity(
            [l for l in sorted_layers if l.metadata.get("artist") not in ("floor3d", "axis3d")]
        )

        # Pre-count types for colourmap normalisation (data layers only)
        type_totals = {}
        for l in data_layers:
            type_totals[l.layer_type] = type_totals.get(l.layer_type, 0) + 1

        def _draw_batch(batch: list) -> None:
            type_counters = {}
            for layer in batch:
                total = type_totals.get(layer.layer_type, 1)
                current = type_counters.get(layer.layer_type, 0)
                id_norm = float(current) / float(max(1, total - 1)) if total > 1 else 0.5
                self._dispatch_draw(layer, context, id_norm=id_norm)
                type_counters[layer.layer_type] = current + 1

        _draw_batch(floor_layers)
        _draw_batch(data_layers)
        _draw_batch(axis_layers)

    @staticmethod
    def _order_by_opacity(layers: List[BaseLayer]) -> List[BaseLayer]:
        """Draw opaque **3D** layers before translucent ones, order-stable within each group.

        A translucent layer can only blend against what is *already in the framebuffer*, so
        declaration order decided whether transparency worked at all. The nebula gallery
        example is exactly that: ``volume3d`` is declared before its ``scatter3d`` probes, so
        the cloud went down first and the opaque probes then painted over it -- and where the
        cloud's own depth writes had claimed a pixel first (see ``renderers/geometry3d.py``),
        the probes behind it vanished instead. Opaque first means the depth buffer is already
        filled when the cloud draws: fragments genuinely behind the probes are rejected, the
        rest blend over them, and both layers read as being in the same space.

        2D layers are untouched -- for them ``zorder`` *is* the contract and there is no depth
        test to arbitrate anything -- so a batch with no 3D layer is returned unchanged and
        every 2D plot renders bit-for-bit as before.
        """
        from ..core.layers import Layer3D
        from ..renderers.geometry3d import is_translucent

        if not any(isinstance(l, Layer3D) for l in layers):
            return layers
        opaque = [l for l in layers if not (isinstance(l, Layer3D) and is_translucent(l))]
        translucent = [l for l in layers if isinstance(l, Layer3D) and is_translucent(l)]
        return opaque + translucent

    def draw_axes(self, axis_manager: Any, context: Any) -> None:
        """Draw the coordinate framework."""
        self.renderers["axis"].draw(axis_manager, context)

    def _dispatch_draw(self, layer: BaseLayer, context: Any, id_norm: float = 0.0) -> None:
        """Internal dispatcher for primitive types."""
        renderer = self.renderers.get(layer.layer_type)
        if renderer:
            # Check if renderer expects id_norm
            import inspect

            sig = inspect.signature(renderer.draw)
            if "id_norm" in sig.parameters:
                renderer.draw(layer, context, id_norm=id_norm)
            else:
                renderer.draw(layer, context)
        else:
            # Fallback for Phase 0-2: legacy bridge handles drawing
            pass

    def get_bounds(self, layers: List[BaseLayer]) -> Optional[Tuple[float, float, float, float]]:
        """Calculate collective bounds of all layers contributing to autoscale."""
        xmin, xmax, ymin, ymax = float("inf"), float("-inf"), float("inf"), float("-inf")
        found = False

        for layer in layers:
            # Autoscale policy: visible layers + no semantic guide lines
            if not layer.style.visible:
                continue

            # Policy check: does this layer type participate in autoscale?
            # V1 NON-GOAL: Text and guide lines (axvline) don't count by default
            is_guide = layer.layer_type in ["semantic_line", "semantic_span", "text"]
            if is_guide:
                continue

            b = layer.get_intrinsic_bounds()
            if b:
                if not all(np.isfinite(b)):
                    continue
                tx, ty = layer.translation
                xmin = min(xmin, b[0] + tx)
                xmax = max(xmax, b[1] + tx)
                ymin = min(ymin, b[2] + ty)
                ymax = max(ymax, b[3] + ty)
                found = True

        if not found or xmin == float("inf"):
            return None

        return (xmin, xmax, ymin, ymax)
