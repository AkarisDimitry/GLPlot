"""Test the 3D object catalogue: every generator's shape, finiteness and reproducibility.

The catalogue is data, so these tests are largely a walk over it. That is the point of
storing it as data: a new object cannot be added without these assertions applying to it.
"""

from __future__ import annotations

import numpy as np
import pytest

from glplot.gui import generators3d as g3
from glplot.gui import icons
from glplot.gui import layerops3d as l3

ALL_KEYS = list(g3.GENERATOR_KEYS)
SURFACES = [k for k in ALL_KEYS if g3.generator(k).category == "surface"]
FIELDS = [k for k in ALL_KEYS if g3.generator(k).category == "field"]


class TestCatalogueIntegrity:
    """Every entry is complete and self-consistent, checked without running it."""

    def test_there_are_generators_in_every_category(self):
        for key, _label in g3.CATEGORIES:
            assert g3.by_category(key), f"category {key!r} is empty"

    @pytest.mark.parametrize("key", ALL_KEYS)
    def test_every_generator_is_well_formed(self, key):
        spec = g3.generator(key)
        assert spec.key == key
        assert spec.label and spec.description
        assert spec.category in dict(g3.CATEGORIES)
        assert spec.icon in icons.ICON_SHAPES
        assert spec.kind in l3.KIND3D_KEYS
        assert len(spec.plot_columns) == 3
        assert set(spec.plot_columns) <= set(spec.columns)
        if spec.color_column is not None:
            assert spec.color_column in spec.columns

    @pytest.mark.parametrize("key", ALL_KEYS)
    def test_parameter_defaults_are_inside_their_bounds(self, key):
        for param in g3.generator(key).params:
            assert param.vmin <= param.default <= param.vmax, param.name

    def test_unknown_generator_raises(self):
        with pytest.raises(g3.GeneratorError, match="unknown generator"):
            g3.generator("dodecahedron")

    def test_category_label_falls_back_to_the_key(self):
        assert g3.category_label("curve") == "Space curves"
        assert g3.category_label("nope") == "nope"


class TestGeneration:
    """Running each one."""

    @pytest.mark.parametrize("key", ALL_KEYS)
    def test_output_is_finite_and_rectangular(self, key):
        table = g3.generator(key).generate(samples=1024)
        assert set(table) == set(g3.generator(key).columns)
        lengths = {len(values) for values in table.values()}
        assert len(lengths) == 1
        for name, values in table.items():
            assert np.all(np.isfinite(values)), f"{key}.{name} is not finite"

    @pytest.mark.parametrize("key", ALL_KEYS)
    def test_output_is_not_degenerate(self, key):
        """Every generator must produce a shape with extent, not a single point."""
        spec = g3.generator(key)
        table = spec.generate(samples=1024)
        spans = [float(np.ptp(table[name])) for name in spec.plot_columns]
        assert max(spans) > 1e-9, f"{key} collapsed to a point"

    @pytest.mark.parametrize("key", ALL_KEYS)
    def test_generation_is_reproducible(self, key):
        """Seeded, so a slider drag re-tunes the same cloud instead of reshuffling it."""
        first = g3.generator(key).generate(samples=512)
        second = g3.generator(key).generate(samples=512)
        for name in first:
            assert np.allclose(first[name], second[name])

    def test_sample_count_is_clamped_at_both_ends(self):
        spec = g3.generator("helix")
        assert len(spec.generate(samples=-5)["t"]) >= g3.MIN_SAMPLES
        assert len(spec.generate(samples=10)["t"]) == 10

    def test_parameters_are_clamped_to_their_bounds(self):
        spec = g3.generator("helix")
        resolved = spec.resolve({"radius": 1e9})
        assert resolved["radius"] == pytest.approx(
            next(p.vmax for p in spec.params if p.name == "radius")
        )

    def test_integer_parameters_are_rounded(self):
        spec = g3.generator("torus_knot")
        assert spec.resolve({"p": 2.7})["p"] == pytest.approx(3.0)

    def test_unknown_parameters_are_ignored_not_rejected(self):
        """A panel keeping one dict per generator would otherwise raise on every switch."""
        assert "nonsense" not in g3.generator("helix").resolve({"nonsense": 1.0})

    def test_parameters_change_the_output(self):
        spec = g3.generator("torus")
        small = spec.generate({"R": 1.0, "r": 0.1}, 512)
        large = spec.generate({"R": 4.0, "r": 0.1}, 512)
        assert float(np.ptp(large["x"])) > float(np.ptp(small["x"]))


class TestSurfaceGrids:
    """A surface's connectivity is its sampling shape, not its (x, y)."""

    @pytest.mark.parametrize("key", SURFACES)
    def test_the_uv_lattice_is_full(self, key):
        table = g3.generator(key).generate(samples=900)
        assert l3.grid_shape(table["u"], table["v"]) is not None

    @pytest.mark.parametrize("key", SURFACES)
    def test_the_reported_shape_matches_the_array(self, key):
        spec = g3.generator(key)
        table = spec.generate(samples=900)
        shape = spec.grid_shape(900)
        assert shape is not None
        assert shape[0] * shape[1] == len(table["x"])

    @pytest.mark.parametrize("key", SURFACES)
    def test_the_reported_shape_triangulates_the_output(self, key):
        """The end-to-end contract: generator shape -> resolve_grid -> a surface."""
        spec = g3.generator(key)
        table = spec.generate(samples=900)
        verts, indices, _ = l3.surface_geometry(
            table["x"], table["y"], table["z"], spec.kind_options(900)
        )
        assert len(verts) == len(table["x"])
        assert indices is not None and len(indices) > 0

    def test_a_parametric_surface_is_not_an_xy_lattice(self):
        """The reason ``nu``/``nv`` exist: a sphere's x and y both cycle."""
        table = g3.generator("sphere").generate(samples=900)
        assert l3.grid_shape(table["x"], table["y"]) is None

    def test_non_surfaces_report_no_grid_shape(self):
        for key in ALL_KEYS:
            spec = g3.generator(key)
            if spec.category != "surface":
                assert spec.grid_shape(900) is None
                assert spec.kind_options(900) == {}

    def test_surface_resolution_is_the_square_root(self):
        assert g3.surface_resolution(10_000) == 100
        assert g3.surface_resolution(1) >= 3


class TestVectorFields:
    """Fields carry a vector as well as a position."""

    @pytest.mark.parametrize("key", FIELDS)
    def test_a_field_has_uvw_columns(self, key):
        spec = g3.generator(key)
        assert {"u", "v", "w"} <= set(spec.columns)
        assert spec.kind == "quiver3d"

    @pytest.mark.parametrize("key", FIELDS)
    def test_the_vectors_are_not_all_zero(self, key):
        table = g3.generator(key).generate(samples=512)
        magnitude = np.hypot(np.hypot(table["u"], table["v"]), table["w"])
        assert float(magnitude.max()) > 1e-9

    def test_the_dipole_singularity_stays_finite(self):
        """A 1/r^5 field sampled through the origin must not produce inf."""
        table = g3.generator("dipole").generate(samples=343)
        for name in ("u", "v", "w"):
            assert np.all(np.isfinite(table[name]))


class TestEndToEnd:
    """Every generator, through its own default kind, into a real scene."""

    @pytest.mark.parametrize("key", ALL_KEYS)
    def test_it_plots(self, key):
        from glplot.engine import GPULinePlot

        plot = GPULinePlot()
        plot.set_ndim(3)
        spec = g3.generator(key)
        table = spec.generate(samples=900)
        options = dict(spec.kind_options(900))
        if spec.kind == "quiver3d":
            options.update({name: table[name] for name in ("u", "v", "w")})
        cx, cy, cz = spec.plot_columns
        layer = l3.add_xyz_layer(
            plot,
            table[cx],
            table[cy],
            table[cz],
            kind=spec.kind,
            label=spec.label,
            options=options or None,
            c=table[spec.color_column] if spec.color_column else None,
        )
        assert len(layer.vertices) > 0
        assert np.all(np.isfinite(layer.vertices))
        assert plot.get_3d_bounds() is not None
