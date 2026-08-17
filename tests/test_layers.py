"""Test core layer logic in glplot.core.layers.

Focus on layer creation, properties, bounds calculation, and metadata
without requiring OpenGL or GPU.
"""

import numpy as np

from glplot.core.layers import (
    BaseLayer,
    CompiledLayer,
    Layer3D,
    LayerDirtyState,
    LayerStyle,
    LineFamilyLayer,
    PatchLayer,
    PolylineLayer,
    ScatterLayer,
    TextLayer,
)


class TestLayerStyle:
    """Test LayerStyle dataclass."""

    def test_default_initialization(self):
        """Test default LayerStyle."""
        style = LayerStyle()
        assert style.visible is True
        assert style.alpha == 1.0
        assert style.zorder == 0
        assert style.pickable is False
        assert style.line_width == 1.0
        assert style.point_size == 6.0

    def test_visibility_toggle(self):
        """Test visibility flag."""
        style_visible = LayerStyle(visible=True)
        style_hidden = LayerStyle(visible=False)
        assert style_visible.visible is True
        assert style_hidden.visible is False

    def test_alpha_modification(self):
        """Test alpha value modification."""
        style = LayerStyle(alpha=0.5)
        assert style.alpha == 0.5

    def test_alpha_bounds(self):
        """Test various alpha values."""
        for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
            style = LayerStyle(alpha=alpha)
            assert style.alpha == alpha

    def test_zorder_positive(self):
        """Test positive zorder."""
        style = LayerStyle(zorder=10)
        assert style.zorder == 10

    def test_zorder_negative(self):
        """Test negative zorder."""
        style = LayerStyle(zorder=-5)
        assert style.zorder == -5

    def test_color_tuple(self):
        """Test color tuple."""
        color = (1.0, 0.0, 0.0, 1.0)
        style = LayerStyle(color=color)
        assert style.color == color

    def test_color_none(self):
        """Test color None."""
        style = LayerStyle(color=None)
        assert style.color is None

    def test_edge_color(self):
        """Test edge color."""
        color = (0.0, 0.0, 0.0, 1.0)
        style = LayerStyle(edge_color=color)
        assert style.edge_color == color

    def test_face_color(self):
        """Test face color."""
        color = (1.0, 1.0, 1.0, 1.0)
        style = LayerStyle(face_color=color)
        assert style.face_color == color

    def test_line_width(self):
        """Test line width."""
        style_thin = LayerStyle(line_width=0.5)
        style_thick = LayerStyle(line_width=5.0)
        assert style_thin.line_width == 0.5
        assert style_thick.line_width == 5.0

    def test_point_size(self):
        """Test point size."""
        style_small = LayerStyle(point_size=2.0)
        style_large = LayerStyle(point_size=20.0)
        assert style_small.point_size == 2.0
        assert style_large.point_size == 20.0

    def test_point_outline_enabled(self):
        """Test point outline flag."""
        style_outline = LayerStyle(point_outline_enabled=True)
        style_no_outline = LayerStyle(point_outline_enabled=False)
        assert style_outline.point_outline_enabled is True
        assert style_no_outline.point_outline_enabled is False

    def test_point_outline_color(self):
        """Test point outline color."""
        color = (1.0, 1.0, 1.0, 1.0)
        style = LayerStyle(point_outline_color=color)
        assert style.point_outline_color == color

    def test_point_outline_width(self):
        """Test point outline width."""
        style = LayerStyle(point_outline_width=2.0)
        assert style.point_outline_width == 2.0

    def test_colormap_usage(self):
        """Test colormap usage flag."""
        style_cmap = LayerStyle(use_colormap=True, cmap="viridis")
        style_no_cmap = LayerStyle(use_colormap=False)
        assert style_cmap.use_colormap is True
        assert style_cmap.cmap == "viridis"
        assert style_no_cmap.use_colormap is False

    def test_colormap_vmin_vmax(self):
        """Test colormap vmin/vmax."""
        style = LayerStyle(vmin=0.0, vmax=1.0)
        assert style.vmin == 0.0
        assert style.vmax == 1.0

    def test_text_size(self):
        """Test text size in pixels."""
        style_small = LayerStyle(text_size_px=10.0)
        style_large = LayerStyle(text_size_px=24.0)
        assert style_small.text_size_px == 10.0
        assert style_large.text_size_px == 24.0


class TestLayerDirtyState:
    """Test LayerDirtyState dataclass."""

    def test_default_initialization(self):
        """Test default dirty state (all dirty)."""
        dirty = LayerDirtyState()
        assert dirty.data_dirty is True
        assert dirty.style_dirty is True
        assert dirty.gpu_dirty is True
        assert dirty.bounds_dirty is True

    def test_partial_dirty(self):
        """Test partial dirty state."""
        dirty = LayerDirtyState(data_dirty=True, style_dirty=False)
        assert dirty.data_dirty is True
        assert dirty.style_dirty is False
        assert dirty.gpu_dirty is True
        assert dirty.bounds_dirty is True

    def test_all_clean(self):
        """Test all clean state."""
        dirty = LayerDirtyState(
            data_dirty=False, style_dirty=False, gpu_dirty=False, bounds_dirty=False
        )
        assert dirty.data_dirty is False
        assert dirty.style_dirty is False
        assert dirty.gpu_dirty is False
        assert dirty.bounds_dirty is False

    def test_clear_method(self):
        """Test clear method resets all flags."""
        dirty = LayerDirtyState()
        dirty.clear()
        assert dirty.data_dirty is False
        assert dirty.style_dirty is False
        assert dirty.gpu_dirty is False
        assert dirty.bounds_dirty is False

    def test_clear_preserves_flags(self):
        """Test that clear sets all flags to False."""
        dirty = LayerDirtyState(data_dirty=True, gpu_dirty=False)
        dirty.clear()
        # All should be False after clear
        assert all(
            [
                not dirty.data_dirty,
                not dirty.style_dirty,
                not dirty.gpu_dirty,
                not dirty.bounds_dirty,
            ]
        )


class TestCompiledLayer:
    """Test CompiledLayer."""

    def test_initialization(self):
        """Test CompiledLayer initialization."""
        layer = CompiledLayer(layer_id=123)
        assert layer.layer_id == 123
        assert layer.bounds_world is None
        assert layer.gpu_initialized is False

    def test_bounds_assignment(self):
        """Test setting bounds."""
        layer = CompiledLayer(layer_id=1)
        bounds = (0.0, 1.0, 0.0, 1.0)
        layer.bounds_world = bounds
        assert layer.bounds_world == bounds

    def test_gpu_initialized_flag(self):
        """Test GPU initialization flag."""
        layer = CompiledLayer(layer_id=1)
        assert layer.gpu_initialized is False
        layer.gpu_initialized = True
        assert layer.gpu_initialized is True


class TestBaseLayer:
    """Test BaseLayer base class."""

    def test_initialization(self):
        """Test BaseLayer initialization."""
        layer = BaseLayer(layer_type="test", label="TestLayer")
        assert layer.layer_type == "test"
        assert layer.label == "TestLayer"
        assert layer.layer_id > 0
        assert isinstance(layer.style, LayerStyle)
        assert isinstance(layer.dirty, LayerDirtyState)
        assert layer.bounds_world is None
        assert layer.translation == (0.0, 0.0)
        assert layer.metadata == {}

    def test_default_label(self):
        """Test default empty label."""
        layer = BaseLayer(layer_type="test")
        assert layer.label == ""

    def test_unique_layer_ids(self):
        """Test that each layer gets unique ID."""
        layer1 = BaseLayer(layer_type="test")
        layer2 = BaseLayer(layer_type="test")
        assert layer1.layer_id != layer2.layer_id

    def test_translation_setting(self):
        """Test translation tuple."""
        layer = BaseLayer(layer_type="test")
        layer.translation = (10.0, 20.0)
        assert layer.translation == (10.0, 20.0)

    def test_metadata_access(self):
        """Test metadata dictionary."""
        layer = BaseLayer(layer_type="test")
        layer.metadata["key"] = "value"
        assert layer.metadata["key"] == "value"

    def test_get_intrinsic_bounds_default(self):
        """Test get_intrinsic_bounds returns None by default."""
        layer = BaseLayer(layer_type="test")
        assert layer.get_intrinsic_bounds() is None

    def test_style_modification(self):
        """Test style property access."""
        layer = BaseLayer(layer_type="test")
        layer.style.alpha = 0.5
        assert layer.style.alpha == 0.5


class TestScatterLayer:
    """Test ScatterLayer."""

    def test_initialization_empty(self):
        """Test empty ScatterLayer initialization."""
        layer = ScatterLayer()
        assert layer.layer_type == "scatter"
        assert layer.pts is None
        assert layer.colors is None
        assert layer.style.point_size == 6.0

    def test_initialization_with_data(self):
        """Test ScatterLayer with data."""
        pts = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
        colors = np.array([[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]], dtype=np.float32)
        layer = ScatterLayer(pts=pts, colors=colors, size=10.0)
        assert np.array_equal(layer.pts, pts)
        assert np.array_equal(layer.colors, colors)
        assert layer.style.point_size == 10.0

    def test_label(self):
        """Test layer label."""
        layer = ScatterLayer(label="My Points")
        assert layer.label == "My Points"

    def test_get_intrinsic_bounds_empty(self):
        """Test bounds calculation with no data."""
        layer = ScatterLayer()
        assert layer.get_intrinsic_bounds() is None

    def test_get_intrinsic_bounds_single_point(self):
        """Test bounds with single point."""
        pts = np.array([[0.5, 0.5]], dtype=np.float32)
        layer = ScatterLayer(pts=pts)
        bounds = layer.get_intrinsic_bounds()
        assert bounds == (0.5, 0.5, 0.5, 0.5)

    def test_get_intrinsic_bounds_multiple_points(self):
        """Test bounds with multiple points."""
        pts = np.array([[0.0, 0.0], [1.0, 2.0], [3.0, 1.0]], dtype=np.float32)
        layer = ScatterLayer(pts=pts)
        bounds = layer.get_intrinsic_bounds()
        assert bounds == (0.0, 3.0, 0.0, 2.0)

    def test_get_intrinsic_bounds_negative_values(self):
        """Test bounds with negative coordinates."""
        pts = np.array([[-1.0, -2.0], [1.0, 2.0]], dtype=np.float32)
        layer = ScatterLayer(pts=pts)
        bounds = layer.get_intrinsic_bounds()
        assert bounds == (-1.0, 1.0, -2.0, 2.0)


class TestPolylineLayer:
    """Test PolylineLayer."""

    def test_initialization_empty(self):
        """Test empty PolylineLayer."""
        layer = PolylineLayer()
        assert layer.layer_type == "polyline"
        assert layer.pts is None

    def test_initialization_with_data(self):
        """Test PolylineLayer with data."""
        pts = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]], dtype=np.float32)
        color = (1.0, 0.0, 0.0, 1.0)
        layer = PolylineLayer(pts=pts, color=color, width=2.0)
        assert np.array_equal(layer.pts, pts)
        assert layer.style.color == color
        assert layer.style.line_width == 2.0

    def test_get_intrinsic_bounds_empty(self):
        """Test bounds with no data."""
        layer = PolylineLayer()
        assert layer.get_intrinsic_bounds() is None

    def test_get_intrinsic_bounds(self):
        """Test bounds calculation."""
        pts = np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 2.0]], dtype=np.float32)
        layer = PolylineLayer(pts=pts)
        bounds = layer.get_intrinsic_bounds()
        assert bounds == (0.0, 4.0, 1.0, 3.0)


class TestPatchLayer:
    """Test PatchLayer."""

    def test_initialization_empty(self):
        """Test empty PatchLayer."""
        layer = PatchLayer()
        assert layer.layer_type == "patch"
        assert layer.vertices is None
        assert layer.indices is None
        assert layer.mode == "strip"

    def test_initialization_with_data(self):
        """Test PatchLayer with data."""
        vertices = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]], dtype=np.float32)
        indices = np.array([0, 1, 2], dtype=np.uint32)
        layer = PatchLayer(vertices=vertices, indices=indices, mode="triangles")
        assert np.array_equal(layer.vertices, vertices)
        assert np.array_equal(layer.indices, indices)
        assert layer.mode == "triangles"

    def test_mode_variants(self):
        """Test different patch modes."""
        for mode in ["strip", "triangles", "rects"]:
            layer = PatchLayer(mode=mode)
            assert layer.mode == mode

    def test_get_intrinsic_bounds_empty(self):
        """Test bounds with no vertices."""
        layer = PatchLayer()
        assert layer.get_intrinsic_bounds() is None

    def test_get_intrinsic_bounds(self):
        """Test bounds calculation."""
        vertices = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float32)
        layer = PatchLayer(vertices=vertices)
        bounds = layer.get_intrinsic_bounds()
        assert bounds == (0.0, 1.0, 0.0, 1.0)

    def test_get_intrinsic_bounds_negative(self):
        """Test bounds with negative coordinates."""
        vertices = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]], dtype=np.float32)
        layer = PatchLayer(vertices=vertices)
        bounds = layer.get_intrinsic_bounds()
        assert bounds == (-1.0, 1.0, -1.0, 1.0)


class TestTextLayer:
    """Test TextLayer."""

    def test_initialization_empty(self):
        """Test empty TextLayer."""
        layer = TextLayer()
        assert layer.layer_type == "text"
        assert layer.x == 0.0
        assert layer.y == 0.0
        assert layer.text == ""

    def test_initialization_with_text(self):
        """Test TextLayer with text."""
        layer = TextLayer(x=1.0, y=2.0, text="Hello", label="label")
        assert layer.x == 1.0
        assert layer.y == 2.0
        assert layer.text == "Hello"
        assert layer.label == "label"

    def test_get_intrinsic_bounds_none(self):
        """Test that text does not participate in autoscale."""
        layer = TextLayer(x=100.0, y=200.0, text="Large values")
        assert layer.get_intrinsic_bounds() is None


class TestLayer3D:
    """Test Layer3D."""

    def test_initialization_empty(self):
        """Test empty Layer3D."""
        layer = Layer3D()
        assert layer.layer_type == "geometry3d"
        assert layer.vertices is None
        assert layer.colors is None
        assert layer.indices is None
        assert layer.primitive == "points"
        assert layer.style.point_size == 3.0

    def test_initialization_with_data(self):
        """Test Layer3D with data."""
        vertices = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32)
        colors = np.array([[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]], dtype=np.float32)
        indices = np.array([0, 1], dtype=np.uint32)
        layer = Layer3D(
            vertices=vertices, colors=colors, indices=indices, primitive="lines", label="3D"
        )
        assert np.array_equal(layer.vertices, vertices)
        assert np.array_equal(layer.colors, colors)
        assert np.array_equal(layer.indices, indices)
        assert layer.primitive == "lines"
        assert layer.label == "3D"

    def test_primitive_types(self):
        """Test different primitive types."""
        for prim in ["points", "lines", "triangles"]:
            layer = Layer3D(primitive=prim)
            assert layer.primitive == prim

    def test_custom_layer_type(self):
        """Test custom layer type."""
        layer = Layer3D(layer_type="custom_3d")
        assert layer.layer_type == "custom_3d"

    def test_get_intrinsic_bounds_empty(self):
        """Test 2D bounds with no vertices."""
        layer = Layer3D()
        assert layer.get_intrinsic_bounds() is None

    def test_get_intrinsic_bounds_2d(self):
        """Test 2D bounds calculation."""
        vertices = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 5.0]], dtype=np.float32)
        layer = Layer3D(vertices=vertices)
        bounds = layer.get_intrinsic_bounds()
        # 2D bounds: (xmin, xmax, ymin, ymax)
        assert bounds == (0.0, 1.0, 0.0, 2.0)

    def test_get_bounds_3d_empty(self):
        """Test 3D bounds with no vertices."""
        layer = Layer3D()
        assert layer.get_bounds_3d() is None

    def test_get_bounds_3d(self):
        """Test 3D bounds calculation."""
        vertices = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [0.5, 1.5, 2.5]], dtype=np.float32)
        layer = Layer3D(vertices=vertices)
        bounds = layer.get_bounds_3d()
        # 3D bounds: (xmin, xmax, ymin, ymax, zmin, zmax)
        assert bounds == (0.0, 1.0, 0.0, 2.0, 0.0, 3.0)

    def test_get_bounds_3d_negative(self):
        """Test 3D bounds with negative coordinates."""
        vertices = np.array([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]], dtype=np.float32)
        layer = Layer3D(vertices=vertices)
        bounds = layer.get_bounds_3d()
        assert bounds == (-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)


class TestLayer3DBoundsCache:
    """``get_bounds_3d`` is memoised: a camera nudge asks for it ~20 times a frame.

    Uncached it was six full scans of the vertex array per call, which on a large cloud was
    the whole of a rotate drag's frame budget and, whenever the array fell out of cache, the
    hitch that made the view jump.
    """

    def _layer(self, values=((0.0, 0.0, 0.0), (1.0, 2.0, 3.0))):
        layer = Layer3D(vertices=np.array(values, dtype=np.float32))
        layer.dirty.clear()
        return layer

    def test_a_repeat_call_does_not_rescan(self, monkeypatch):
        layer = self._layer()
        layer.get_bounds_3d()

        def explode(*args, **kwargs):  # pragma: no cover - only runs on failure
            raise AssertionError("bounds were recomputed on a cache hit")

        monkeypatch.setattr(np, "min", explode)
        monkeypatch.setattr(np, "max", explode)
        assert layer.get_bounds_3d() == (0.0, 1.0, 0.0, 2.0, 0.0, 3.0)

    def test_replacing_the_array_invalidates_it(self):
        layer = self._layer()
        assert layer.get_bounds_3d() == (0.0, 1.0, 0.0, 2.0, 0.0, 3.0)
        layer.vertices = np.array([[-5.0, -5.0, -5.0], [5.0, 5.0, 5.0]], dtype=np.float32)
        assert layer.get_bounds_3d() == (-5.0, 5.0, -5.0, 5.0, -5.0, 5.0)

    def test_an_in_place_edit_invalidates_it_through_the_dirty_flag(self):
        """The contract for mutating an array in place (``gui/datasets.py`` CONTRACT 1.4)."""
        layer = self._layer()
        assert layer.get_bounds_3d()[5] == 3.0
        layer.vertices[1, 2] = 42.0
        layer.dirty.gpu_dirty = True
        assert layer.get_bounds_3d()[5] == 42.0

    def test_bounds_dirty_invalidates_it_too(self):
        layer = self._layer()
        layer.get_bounds_3d()
        layer.vertices[1, 0] = -9.0
        layer.dirty.bounds_dirty = True
        assert layer.get_bounds_3d()[0] == -9.0

    def test_emptying_the_layer_is_not_a_stale_hit(self):
        layer = self._layer()
        layer.get_bounds_3d()
        layer.vertices = None
        assert layer.get_bounds_3d() is None

    def test_intrinsic_bounds_agree_with_the_cached_3d_ones(self):
        layer = self._layer()
        x0, x1, y0, y1, _, _ = layer.get_bounds_3d()
        assert layer.get_intrinsic_bounds() == (x0, x1, y0, y1)


class TestLineFamilyLayer:
    """Test LineFamilyLayer (high-performance line rendering)."""

    def test_initialization_empty(self):
        """Test empty LineFamilyLayer."""
        layer = LineFamilyLayer()
        assert layer.layer_type == "line_family"
        assert layer.ab is None
        assert layer.colors is None
        assert layer.x_range == (-1.0, 1.0)

    def test_initialization_with_data(self):
        """Test LineFamilyLayer with data."""
        ab = np.array([[1.0, 0.0], [2.0, 1.0]], dtype=np.float32)
        colors = np.array([[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]], dtype=np.float32)
        layer = LineFamilyLayer(ab=ab, colors=colors, x_range=(0.0, 10.0), label="lines")
        assert np.array_equal(layer.ab, ab)
        assert np.array_equal(layer.colors, colors)
        assert layer.x_range == (0.0, 10.0)
        assert layer.label == "lines"

    def test_get_intrinsic_bounds_empty(self):
        """Test bounds with no data."""
        layer = LineFamilyLayer()
        assert layer.get_intrinsic_bounds() is None

    def test_get_intrinsic_bounds(self):
        """Test bounds calculation (y = ax + b)."""
        # Lines: y = x (a=1, b=0) and y = -x + 2 (a=-1, b=2)
        ab = np.array([[1.0, 0.0], [-1.0, 2.0]], dtype=np.float32)
        layer = LineFamilyLayer(ab=ab, x_range=(0.0, 2.0))
        bounds = layer.get_intrinsic_bounds()
        # At x=0: y = 0 and y = 2
        # At x=2: y = 2 and y = 0
        # So ymin=0, ymax=2
        assert bounds == (0.0, 2.0, 0.0, 2.0)

    def test_get_intrinsic_bounds_negative_slope(self):
        """Test bounds with negative slopes."""
        ab = np.array([[-1.0, 1.0], [-2.0, 3.0]], dtype=np.float32)
        layer = LineFamilyLayer(ab=ab, x_range=(-1.0, 1.0))
        bounds = layer.get_intrinsic_bounds()
        # At x=-1: y = -(-1) + 1 = 2 and y = -2(-1) + 3 = 5
        # At x=1: y = -(1) + 1 = 0 and y = -2(1) + 3 = 1
        assert bounds[0] == -1.0  # xmin
        assert bounds[1] == 1.0  # xmax


class TestLayersIntegration:
    """Test layer interactions and consistency."""

    def test_multiple_layer_unique_ids(self):
        """Test that multiple layers have unique IDs."""
        layers = [
            ScatterLayer(),
            PolylineLayer(),
            PatchLayer(),
            TextLayer(),
            Layer3D(),
            LineFamilyLayer(),
        ]
        ids = [layer.layer_id for layer in layers]
        assert len(ids) == len(set(ids))

    def test_layer_style_independence(self):
        """Test that layer styles are independent."""
        layer1 = ScatterLayer()
        layer2 = ScatterLayer()
        layer1.style.alpha = 0.5
        assert layer2.style.alpha == 1.0

    def test_layer_metadata_independence(self):
        """Test that layer metadata is independent."""
        layer1 = ScatterLayer()
        layer2 = ScatterLayer()
        layer1.metadata["key"] = "value1"
        assert "key" not in layer2.metadata

    def test_bounds_consistency(self):
        """Test that bounds are consistent across layer types."""
        pts = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
        scatter = ScatterLayer(pts=pts)
        poly = PolylineLayer(pts=pts)
        patch = PatchLayer(vertices=pts)

        scatter_bounds = scatter.get_intrinsic_bounds()
        poly_bounds = poly.get_intrinsic_bounds()
        patch_bounds = patch.get_intrinsic_bounds()

        assert scatter_bounds == poly_bounds == patch_bounds
