"""Test the 3D plot-kind registry: grid detection, geometry, tagging and the builders.

The geometry half is pure numpy. The builder half needs a ``GPULinePlot``, which
constructs without a window or a GL context, so these still run headless.
"""

from __future__ import annotations

import numpy as np
import pytest

from glplot.engine import GPULinePlot
from glplot.gui import layerops3d as l3


@pytest.fixture
def plot():
    """A 3D figure. ``GPULinePlot()`` opens no window and touches no GL."""
    fig = GPULinePlot()
    fig.set_ndim(3)
    return fig


def _grid(nx: int = 7, ny: int = 5):
    """A gridded table: x/y a full lattice, z a function of it."""
    gx, gy = np.meshgrid(np.linspace(-2.0, 2.0, nx), np.linspace(-1.0, 1.0, ny))
    return gx.ravel(), gy.ravel(), np.sin(gx * gy).ravel()


def _path(n: int = 50):
    t = np.linspace(0.0, 4.0 * np.pi, n)
    return np.cos(t), np.sin(t), t / 10.0


class TestGridDetection:
    """A surface is defined by connectivity; a bag of points has none."""

    def test_a_full_lattice_is_detected(self):
        xs, ys, _ = _grid(7, 5)
        assert l3.grid_shape(xs, ys) == (7, 5)

    def test_scattered_points_are_not_a_lattice(self):
        rng = np.random.default_rng(0)
        assert l3.grid_shape(rng.random(35), rng.random(35)) is None

    def test_a_partial_lattice_is_rejected(self):
        """Unique counts multiplying out is necessary but not sufficient."""
        xs = np.array([0.0, 0.0, 1.0, 1.0])
        ys = np.array([0.0, 0.0, 1.0, 1.0])  # (0,1) and (1,0) missing
        assert l3.grid_shape(xs, ys) is None

    def test_too_few_distinct_values_is_not_a_grid(self):
        assert l3.grid_shape(np.zeros(4), np.arange(4.0)) is None

    def test_mismatched_lengths_are_rejected(self):
        assert l3.grid_shape(np.arange(4.0), np.arange(5.0)) is None

    def test_reshape_recovers_row_major_meshgrid_order(self):
        xs, ys, zs = _grid(7, 5)
        gx, gy, gz = l3.reshape_to_grid(xs, ys, zs)
        assert gx.shape == gy.shape == gz.shape == (5, 7)
        assert np.allclose(gz.ravel(), zs)

    def test_reshape_of_scattered_points_raises_with_the_counts(self):
        rng = np.random.default_rng(1)
        with pytest.raises(l3.Grid3DError, match="regular grid"):
            l3.reshape_to_grid(rng.random(20), rng.random(20), rng.random(20))

    def test_resolve_grid_prefers_an_explicit_shape(self):
        """A parametric surface's (x, y) is not a lattice; only nu/nv can recover it."""
        u, v = np.meshgrid(np.linspace(0, 2 * np.pi, 9), np.linspace(0, np.pi, 6))
        xs = (np.sin(v) * np.cos(u)).ravel()
        ys = (np.sin(v) * np.sin(u)).ravel()
        zs = np.cos(v).ravel()
        assert l3.grid_shape(xs, ys) is None  # a sphere is not an (x, y) lattice
        gx, _, _ = l3.resolve_grid(xs, ys, zs, {"nu": 9, "nv": 6})
        assert gx.shape == (6, 9)

    def test_a_wrong_explicit_shape_raises_rather_than_silently_reshaping(self):
        xs, ys, zs = _grid(7, 5)
        with pytest.raises(l3.Grid3DError, match="does not match"):
            l3.resolve_grid(xs, ys, zs, {"nu": 4, "nv": 4})

    def test_zero_shape_falls_back_to_detection(self):
        xs, ys, zs = _grid(7, 5)
        gx, _, _ = l3.resolve_grid(xs, ys, zs, {"nu": 0, "nv": 0})
        assert gx.shape == (5, 7)


class TestKindRegistry:
    """Every kind is a complete, self-describing record."""

    def test_every_key_resolves_to_a_matching_spec(self):
        for key in l3.KIND3D_KEYS:
            assert l3.kind3d_spec(key).key == key

    def test_unknown_kind_raises(self):
        with pytest.raises(ValueError, match="kind must be one of"):
            l3.kind3d_spec("hypercube")

    def test_every_kind_has_defaults_and_an_icon(self):
        from glplot.gui import icons

        for key in l3.KIND3D_KEYS:
            spec = l3.kind3d_spec(key)
            assert isinstance(l3.default_kind3d_options(key), dict)
            assert spec.icon in icons.ICON_SHAPES
            assert spec.description
            assert spec.primitive in ("points", "lines", "triangles")
            assert spec.layer_type.endswith("3d")

    def test_direct_kinds_are_exactly_the_non_derived_ones(self):
        assert set(l3.DIRECT_KIND3D_KEYS) == {
            k for k in l3.KIND3D_KEYS if not l3.kind3d_spec(k).derived
        }
        assert set(l3.DIRECT_KIND3D_KEYS) == {"scatter3d", "volume3d"}

    def test_unknown_kinds_are_treated_as_derived(self):
        """Refusing to write back is the safe side of the bet."""
        assert l3.kind3d_is_derived(None) is True
        assert l3.kind3d_is_derived("nope") is True

    def test_is_kind3d_discriminates(self):
        assert l3.is_kind3d("surface3d") is True
        assert l3.is_kind3d("line") is False
        assert l3.is_kind3d(None) is False

    def test_options_reject_a_key_the_kind_has_no_use_for(self):
        with pytest.raises(ValueError, match="has no option"):
            l3.resolve_kind3d_options("scatter3d", {"bins": 10})

    def test_options_merge_over_the_defaults(self):
        resolved = l3.resolve_kind3d_options("bar3d", {"gap": 0.5})
        assert resolved["gap"] == 0.5
        assert "baseline" in resolved


class TestGeometry:
    """Each kind's ``geom``, called directly."""

    @pytest.mark.parametrize("key", l3.KIND3D_KEYS)
    def test_every_kind_builds_finite_geometry(self, key):
        spec = l3.kind3d_spec(key)
        xs, ys, zs = _grid(7, 5) if spec.needs_grid else _path()
        opts = l3.default_kind3d_options(key)
        if key == "quiver3d":
            opts.update({"u": np.ones_like(xs), "v": np.zeros_like(xs), "w": np.ones_like(xs)})
        if spec.needs_grid:
            opts.update({"nu": 7, "nv": 5})
        verts, indices, values = spec.geom(xs, ys, zs, opts)
        assert verts.ndim == 2 and verts.shape[1] == 3
        assert len(verts) > 0
        assert np.all(np.isfinite(verts))
        if indices is not None:
            assert indices.max() < len(verts)
        if values is not None:
            assert len(values) in (len(verts), len(np.ravel(zs)))

    def test_points_geometry_is_row_for_row(self):
        xs, ys, zs = _path(17)
        verts, indices, _ = l3.points_geometry(xs, ys, zs, {})
        assert len(verts) == 17 and indices is None
        assert np.allclose(verts[:, 2], zs, atol=1e-6)

    def test_path_geometry_expands_to_segment_pairs(self):
        xs, ys, zs = _path(10)
        verts, _, _ = l3.path_geometry(xs, ys, zs, {})
        assert len(verts) == 2 * (10 - 1)
        # Segment i joins row i to row i+1.
        assert np.allclose(verts[0], [xs[0], ys[0], zs[0]], atol=1e-6)
        assert np.allclose(verts[1], [xs[1], ys[1], zs[1]], atol=1e-6)

    def test_a_single_point_path_does_not_raise(self):
        verts, _, _ = l3.path_geometry(np.array([1.0]), np.array([2.0]), np.array([3.0]), {})
        assert len(verts) == 1

    def test_stems_run_from_the_baseline(self):
        xs, ys, zs = _path(6)
        verts, _, _ = l3.stem_geometry(xs, ys, zs, {"baseline": -2.0})
        assert len(verts) == 12
        assert np.allclose(verts[0::2, 2], -2.0)
        assert np.allclose(verts[1::2, 2], zs, atol=1e-6)

    def test_surface_indices_form_two_triangles_per_cell(self):
        xs, ys, zs = _grid(7, 5)
        verts, indices, _ = l3.surface_geometry(xs, ys, zs, {"nu": 0, "nv": 0})
        assert len(verts) == 35
        assert len(indices) == 6 * (7 - 1) * (5 - 1)

    def test_wireframe_stride_thins_the_mesh(self):
        xs, ys, zs = _grid(9, 9)
        dense, _, _ = l3.wireframe_geometry(xs, ys, zs, {"stride": 1, "nu": 0, "nv": 0})
        sparse, _, _ = l3.wireframe_geometry(xs, ys, zs, {"stride": 4, "nu": 0, "nv": 0})
        assert len(sparse) < len(dense)

    def test_bars_are_closed_boxes(self):
        xs, ys, zs = _grid(3, 3)
        verts, _, values = l3.bar_geometry(xs, ys, zs, l3.default_kind3d_options("bar3d"))
        assert len(verts) == 9 * 36
        assert len(values) == len(verts)

    def test_bar_gap_shrinks_the_footprint(self):
        xs, ys, zs = _grid(3, 3)
        opts = l3.default_kind3d_options("bar3d")
        wide, _, _ = l3.bar_geometry(xs, ys, zs, {**opts, "gap": 0.0})
        narrow, _, _ = l3.bar_geometry(xs, ys, zs, {**opts, "gap": 0.6})
        assert float(np.ptp(narrow[:, 0])) < float(np.ptp(wide[:, 0]))

    def test_auto_spacing_is_eighty_percent_of_the_median_gap(self):
        assert l3.auto_spacing(np.array([0.0, 1.0, 2.0, 3.0])) == pytest.approx(0.8)
        assert l3.auto_spacing(np.array([5.0])) == pytest.approx(1.0)

    def test_quiver_arrows_have_a_shaft_and_two_barbs(self):
        xs, ys, zs = _path(4)
        verts, _, _ = l3.quiver_geometry(
            xs,
            ys,
            zs,
            {"u": np.ones(4), "v": np.zeros(4), "w": np.zeros(4), "scale": 1.0, "head": 0.25},
        )
        assert len(verts) == 4 * 6
        # The shaft runs from the base to the tip, which is base + (u, v, w).
        assert np.allclose(verts[1] - verts[0], [1.0, 0.0, 0.0], atol=1e-5)

    def test_quiver_rejects_a_wrong_length_component(self):
        xs, ys, zs = _path(4)
        with pytest.raises(ValueError, match="has 3 values"):
            l3.quiver_geometry(
                xs, ys, zs, {"u": np.ones(3), "v": None, "w": None, "scale": 1.0, "head": 0.25}
            )

    def test_quiver_tolerates_a_zero_length_vector(self):
        """A stationary sample must not produce NaNs from a divide by its own length."""
        xs, ys, zs = _path(3)
        verts, _, _ = l3.quiver_geometry(
            xs,
            ys,
            zs,
            {"u": np.zeros(3), "v": np.zeros(3), "w": np.zeros(3), "scale": 1.0, "head": 0.25},
        )
        assert np.all(np.isfinite(verts))

    def test_ribbon_hangs_from_the_path_to_the_baseline(self):
        xs, ys, zs = _path(5)
        verts, _, _ = l3.ribbon_geometry(xs, ys, zs, {"baseline": -1.0})
        assert len(verts) == 6 * (5 - 1)
        assert float(verts[:, 2].min()) == pytest.approx(-1.0)


class TestColourStretching:
    """A per-row colour column spread over a kind's per-vertex array."""

    def test_equal_lengths_pass_through(self):
        raw = np.arange(5.0)
        assert np.allclose(l3.stretch_to_vertices(raw, 5), raw)

    def test_an_exact_multiple_repeats(self):
        assert np.allclose(l3.stretch_to_vertices(np.array([0.0, 1.0]), 6), [0, 0, 0, 1, 1, 1])

    def test_a_path_length_resamples_rather_than_dropping_the_column(self):
        """2(N-1) is not a multiple of N; the old code silently fell back to flat colour."""
        out = l3.stretch_to_vertices(np.arange(4.0), 6)
        assert out is not None and len(out) == 6
        assert out[0] == pytest.approx(0.0) and out[-1] == pytest.approx(3.0)
        assert np.all(np.diff(out) > 0)

    def test_nothing_to_map_returns_none(self):
        assert l3.stretch_to_vertices(np.array([]), 10) is None
        assert l3.stretch_to_vertices(np.arange(3.0), 0) is None


class TestBuilders:
    """``add_xyz_layer`` / ``update_layer_xyz`` against a real (windowless) engine."""

    @pytest.mark.parametrize("key", l3.KIND3D_KEYS)
    def test_every_kind_lands_in_the_scene(self, plot, key):
        spec = l3.kind3d_spec(key)
        xs, ys, zs = _grid(7, 5) if spec.needs_grid else _path()
        options = {"nu": 7, "nv": 5} if spec.needs_grid else {}
        if key == "quiver3d":
            options = {"u": np.ones_like(xs), "v": np.zeros_like(xs), "w": np.ones_like(xs)}
        layer = l3.add_xyz_layer(
            plot, xs, ys, zs, kind=key, label=f"L {key}", options=options or None
        )
        assert layer in plot.scene.layers
        assert layer.layer_type == spec.layer_type
        assert layer.primitive == spec.primitive
        assert l3.layer_kind3d(layer) == key

    def test_a_label_is_required(self, plot):
        xs, ys, zs = _path()
        with pytest.raises(ValueError, match="label is required"):
            l3.add_xyz_layer(plot, xs, ys, zs, kind="scatter3d", label="")

    def test_mismatched_column_lengths_are_refused(self, plot):
        with pytest.raises(ValueError, match="same length"):
            l3.add_xyz_layer(
                plot,
                np.arange(5.0),
                np.arange(4.0),
                np.arange(5.0),
                kind="scatter3d",
                label="bad",
            )

    def test_z_drives_the_colormap_by_default(self, plot):
        xs, ys, zs = _path()
        layer = l3.add_xyz_layer(plot, xs, ys, zs, kind="scatter3d", label="c")
        assert layer.colors is not None
        assert len(np.unique(layer.colors, axis=0)) > 1

    def test_an_explicit_colour_column_wins(self, plot):
        xs, ys, zs = _path(20)
        values = np.linspace(0.0, 1.0, 20)
        layer = l3.add_xyz_layer(
            plot, xs, ys, zs, kind="scatter3d", label="c", c=values, cmap="magma"
        )
        assert layer.metadata["cmap"] == "magma"
        assert np.allclose(layer.metadata["cvalues"], values)

    def test_a_colour_column_survives_a_derived_kind(self, plot):
        """The regression: line3d dropped the column and drew one flat colour."""
        xs, ys, zs = _path(30)
        values = np.linspace(0.0, 1.0, 30)
        layer = l3.add_xyz_layer(plot, xs, ys, zs, kind="line3d", label="c", c=values)
        assert layer.colors is not None
        assert len(np.unique(layer.colors, axis=0)) > 1
        assert len(layer.metadata["cvalues"]) == len(layer.vertices)

    def test_flat_colour_when_asked(self, plot):
        xs, ys, zs = _path()
        layer = l3.add_xyz_layer(
            plot,
            xs,
            ys,
            zs,
            kind="scatter3d",
            label="flat",
            color=(1.0, 0.0, 0.0, 1.0),
            colormap_by_z=False,
        )
        assert np.allclose(layer.colors, [1.0, 0.0, 0.0, 1.0])

    def test_alpha_scales_the_colours(self, plot):
        xs, ys, zs = _path()
        layer = l3.add_xyz_layer(
            plot, xs, ys, zs, kind="scatter3d", label="a", colormap_by_z=False, alpha=0.25
        )
        assert float(layer.colors[:, 3].max()) == pytest.approx(0.25)

    def test_per_point_sizes_reach_the_layer(self, plot):
        xs, ys, zs = _path(12)
        sizes = np.linspace(2.0, 20.0, 12)
        layer = l3.add_xyz_layer(plot, xs, ys, zs, kind="scatter3d", label="s", s=sizes)
        assert layer.sizes is not None and len(layer.sizes) == 12

    def test_sizes_are_ignored_by_a_line_kind(self, plot):
        """A size array on a line has no meaning; it must not reach a mismatched buffer."""
        xs, ys, zs = _path(12)
        layer = l3.add_xyz_layer(
            plot, xs, ys, zs, kind="line3d", label="s", s=np.linspace(2.0, 20.0, 12)
        )
        assert layer.sizes is None

    def test_a_new_layer_is_framed_by_the_panel_camera(self, plot):
        plot.set_3d_view(elev=11.0, azim=22.0)
        xs, ys, zs = _path()
        layer = l3.add_xyz_layer(plot, xs, ys, zs, kind="scatter3d", label="f")
        assert layer.metadata["camera"]["elev"] == pytest.approx(11.0)
        assert layer.metadata["scene_bounds"] is not None

    def test_update_re_derives_in_place(self, plot):
        xs, ys, zs = _grid(7, 5)
        layer = l3.add_xyz_layer(
            plot, xs, ys, zs, kind="surface3d", label="s", options={"nu": 7, "nv": 5}
        )
        before = layer.layer_id
        nx2, ny2 = 9, 6
        xs2, ys2, zs2 = _grid(nx2, ny2)
        l3.update_layer_xyz(plot, layer, xs2, ys2, zs2, options={"nu": nx2, "nv": ny2})
        assert layer.layer_id == before
        assert len(layer.vertices) == nx2 * ny2
        assert layer.dirty.gpu_dirty is True

    def test_update_refuses_an_untagged_layer(self, plot):
        layer = plot.add_geometry3d(np.zeros((4, 3), dtype=np.float32), label="raw")
        with pytest.raises(ValueError, match="no 3D kind tag"):
            l3.update_layer_xyz(plot, layer, *_path(4))

    def test_update_keeps_a_flat_colour_flat(self, plot):
        """Re-deriving geometry is not a request to restyle."""
        xs, ys, zs = _path(10)
        layer = l3.add_xyz_layer(
            plot,
            xs,
            ys,
            zs,
            kind="scatter3d",
            label="k",
            color=(0.0, 1.0, 0.0, 1.0),
            colormap_by_z=False,
        )
        l3.update_layer_xyz(plot, layer, *_path(10))
        assert np.allclose(layer.colors, [0.0, 1.0, 0.0, 1.0])


class TestTagging:
    """The kind tag is what makes a 3D layer re-derivable and re-typable."""

    def test_tag_records_the_kind_and_options(self, plot):
        xs, ys, zs = _path()
        layer = l3.add_xyz_layer(
            plot, xs, ys, zs, kind="stem3d", label="t", options={"baseline": 1.5}
        )
        assert l3.layer_kind3d(layer) == "stem3d"
        assert l3.layer_kind3d_options(layer)["baseline"] == pytest.approx(1.5)

    def test_source_columns_are_retained_for_a_derived_kind(self, plot):
        xs, ys, zs = _path(20)
        layer = l3.add_xyz_layer(plot, xs, ys, zs, kind="line3d", label="t")
        source = l3.layer_source_xyz(layer)
        assert source is not None
        assert np.allclose(source[0], xs)

    def test_a_direct_kind_recovers_its_source_from_its_vertices(self, plot):
        xs, ys, zs = _path(20)
        layer = l3.add_xyz_layer(plot, xs, ys, zs, kind="scatter3d", label="t")
        layer.metadata.pop("gui_source_xyz", None)
        source = l3.layer_source_xyz(layer)
        assert source is not None and np.allclose(source[2], zs, atol=1e-6)

    def test_an_untagged_layer_has_no_kind_or_source(self, plot):
        layer = plot.add_geometry3d(np.zeros((3, 3), dtype=np.float32), label="raw")
        assert l3.layer_kind3d(layer) is None
        assert l3.layer_source_xyz(layer) is None
