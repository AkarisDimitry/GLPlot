"""Test the properties of the generators added on top of the original catalogue.

``tests/test_generators3d.py`` walks the whole catalogue and enforces the structural
invariants (finite, non-degenerate, reproducible, griddable, plottable) on every entry,
new ones included. This file tests the thing a walk cannot: whether each new object is
actually the object it claims to be -- that the Roman surface satisfies Steiner's quartic,
that the helicoid is ruled, that the honeycomb has three-fold coordination rather than the
six-fold of the triangular lattice it is built from, that the quasilattice really is
aperiodic.

A generator whose shape is merely finite and non-degenerate is not worth having; these are
the assertions that would fail if a formula were subtly wrong.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from glplot.gui import generators3d as g3
from glplot.gui import layerops3d as l3

#: The keys added in this pass, by category. Kept explicit rather than derived so that
#: dropping one of them from the catalogue fails here loudly.
NEW_CURVES = ["figure_eight_knot", "granny_knot", "coiled_coil", "rose3d", "twisted_cubic"]
NEW_SURFACES = [
    "enneper",
    "catenoid",
    "helicoid",
    "dini",
    "roman",
    "boy",
    "supertoroid",
    "superellipsoid",
    "kuen",
    "pseudosphere",
    "monkey_saddle",
    "knot_tube",
]
NEW_CLOUDS = ["multivariate_normal", "swiss_roll", "s_curve", "two_moons", "uniform_cube"]
NEW_LATTICES = [
    "bcc_lattice",
    "diamond_lattice",
    "hcp_lattice",
    "honeycomb",
    "quasilattice",
    "fibonacci_sphere",
]
NEW_FIELDS = ["taylor_green", "hill_vortex", "linear_field", "roessler_field", "lorenz_field"]
NEW_KEYS = NEW_CURVES + NEW_SURFACES + NEW_CLOUDS + NEW_LATTICES + NEW_FIELDS


def _grid(table, key, samples):
    """Reshape a surface's columns to ``(nv, nu)``: v is the slow axis, u the fast one."""
    nu, nv = g3.generator(key).grid_shape(samples)
    return {name: values.reshape(nv, nu) for name, values in table.items()}


class TestCatalogueGrew:
    """The additions are present and the catalogue is still self-consistent."""

    @pytest.mark.parametrize("key", NEW_KEYS)
    def test_the_key_exists(self, key):
        assert g3.generator(key).key == key

    @pytest.mark.parametrize(
        "keys,category",
        [
            (NEW_CURVES, "curve"),
            (NEW_SURFACES, "surface"),
            (NEW_CLOUDS, "cloud"),
            (NEW_LATTICES, "lattice"),
            (NEW_FIELDS, "field"),
        ],
    )
    def test_each_lands_in_its_category(self, keys, category):
        for key in keys:
            assert g3.generator(key).category == category

    def test_labels_are_unique(self):
        """The picker shows labels; two objects with one name is a bug the user sees."""
        labels = [spec.label for spec in g3.GENERATORS.values()]
        assert len(labels) == len(set(labels))

    def test_parameter_names_are_unique_within_a_generator(self):
        for key in g3.GENERATOR_KEYS:
            names = [p.name for p in g3.generator(key).params]
            assert len(names) == len(set(names)), key

    def test_every_category_has_several_entries(self):
        for category, _label in g3.CATEGORIES:
            assert len(g3.by_category(category)) >= 5, category


class TestNewCurves:
    """Each new curve is the curve it says it is."""

    @pytest.mark.parametrize("key", ["figure_eight_knot", "granny_knot", "rose3d"])
    def test_the_closed_ones_close(self, key):
        table = g3.generator(key).generate(samples=2001)
        first = np.array([table[c][0] for c in "xyz"])
        last = np.array([table[c][-1] for c in "xyz"])
        assert np.allclose(first, last, atol=1e-9)

    def test_the_figure_eight_knot_is_not_planar(self):
        table = g3.generator("figure_eight_knot").generate(samples=2000)
        assert float(np.ptp(table["z"])) > 1.0

    def test_the_knots_scale_linearly(self):
        small = g3.generator("granny_knot").generate({"scale": 1.0}, 512)
        large = g3.generator("granny_knot").generate({"scale": 2.0}, 512)
        assert np.allclose(large["x"], 2.0 * small["x"])

    def test_the_coiled_coil_winds_the_stated_number_of_times(self):
        """One radial oscillation per 'coils' per turn: count the radius maxima.

        Eleven interior peaks, not twelve: the twelfth is the shared start/end point of the
        closed loop, which the interior comparison cannot see.
        """
        table = g3.generator("coiled_coil").generate(
            {"turns": 1.0, "coils": 12.0, "minor": 0.3, "pitch": 0.0}, 4001
        )
        radius = np.hypot(table["x"], table["y"])
        interior = radius[1:-1]
        peaks = int(np.count_nonzero((interior > radius[:-2]) & (interior > radius[2:])))
        assert peaks == 11
        assert radius[0] == pytest.approx(radius[-1])

    def test_the_rose_radius_is_bounded_by_its_parameter(self):
        table = g3.generator("rose3d").generate({"radius": 2.0, "k": 5.0}, 4001)
        assert float(np.hypot(table["x"], table["y"]).max()) == pytest.approx(2.0, rel=1e-3)

    def test_the_twisted_cubic_is_exactly_t_t2_t3(self):
        spec = g3.generator("twisted_cubic")
        table = spec.generate({"extent": 2.0, "scale": 1.0}, 501)
        assert np.allclose(table["y"], table["x"] ** 2)
        assert np.allclose(table["z"], table["x"] ** 3)


class TestNewSurfaces:
    """The surfaces, checked against the identities that define them."""

    def test_enneper_satisfies_its_height_relation(self):
        table = g3.generator("enneper").generate({"scale": 1.0}, 900)
        assert np.allclose(table["z"], table["u"] ** 2 - table["v"] ** 2)

    def test_the_catenoid_is_narrowest_at_its_waist(self):
        table = g3.generator("catenoid").generate({"waist": 0.5, "height": 1.0}, 2500)
        radius = np.hypot(table["x"], table["y"])
        assert float(radius.min()) == pytest.approx(0.5, rel=0.01)
        # cosh grows, so the widest ring is at |v| = height.
        assert float(radius.max()) == pytest.approx(0.5 * math.cosh(2.0), rel=0.01)

    def test_the_helicoid_is_ruled(self):
        """Every line of constant u is straight — that is what 'ruled' means."""
        samples = 900
        table = _grid(g3.generator("helicoid").generate(samples=samples), "helicoid", samples)
        pts = np.stack([table["x"], table["y"], table["z"]], axis=-1)
        for column in range(pts.shape[1]):
            line = pts[:, column, :]
            direction = line[-1] - line[0]
            offsets = line[1:-1] - line[0]
            assert np.allclose(np.cross(offsets, direction), 0.0, atol=1e-9)

    def test_dini_twists_linearly_along_u(self):
        samples = 900
        table = _grid(g3.generator("dini").generate({"twist": 0.5}, samples), "dini", samples)
        row = table["z"][0]
        assert np.allclose(np.diff(row, 2), 0.0, atol=1e-9)
        assert float(row[-1] - row[0]) > 0.0

    def test_the_roman_surface_satisfies_steiners_quartic(self):
        """x²y² + y²z² + z²x² = r² xyz — the defining equation of the Steiner surface."""
        radius = 1.3
        table = g3.generator("roman").generate({"radius": radius}, 900)
        x, y, z = table["x"], table["y"], table["z"]
        lhs = x * x * y * y + y * y * z * z + z * z * x * x
        rhs = radius * radius * x * y * z
        assert np.allclose(lhs, rhs, atol=1e-9)

    def test_boys_surface_never_divides_by_zero(self):
        """Its denominator is bounded below by 2 - sqrt(2), so z keeps one sign."""
        table = g3.generator("boy").generate({"scale": 1.0}, 2500)
        assert float(table["z"].min()) >= 0.0
        assert float(table["z"].max()) > 1.0

    def test_the_supertoroid_at_exponent_one_is_an_ordinary_torus(self):
        # 2601 -> a 51 x 51 lattice, which is odd and so actually samples u = v = 0 where
        # the extremes of the ring are; an even lattice straddles them and misses by 0.1%.
        table = g3.generator("supertoroid").generate(
            {"R": 2.0, "r": 0.5, "e1": 1.0, "e2": 1.0}, 2601
        )
        ring = np.hypot(table["x"], table["y"])
        assert float(ring.max()) == pytest.approx(2.5, rel=1e-6)
        assert float(ring.min()) == pytest.approx(1.5, rel=1e-6)
        # Distance from the ring circle is the tube radius everywhere.
        tube = np.hypot(ring - 2.0, table["z"])
        assert np.allclose(tube, 0.5, atol=1e-9)

    def test_the_superellipsoid_at_exponent_one_is_a_sphere(self):
        table = g3.generator("superellipsoid").generate({"radius": 2.0, "e1": 1.0, "e2": 1.0}, 900)
        r = np.sqrt(table["x"] ** 2 + table["y"] ** 2 + table["z"] ** 2)
        assert np.allclose(r, 2.0, atol=1e-9)

    def test_a_low_superellipsoid_exponent_squares_it_off(self):
        """e -> 0 pushes the surface out to the corners of the box."""
        boxy = g3.generator("superellipsoid").generate({"radius": 1.0, "e1": 0.1, "e2": 0.1}, 900)
        corner = np.sqrt(boxy["x"] ** 2 + boxy["y"] ** 2 + boxy["z"] ** 2)
        assert float(corner.max()) > 1.4

    def test_the_pseudosphere_is_widest_at_its_cusp(self):
        # An odd lattice, so u = 0 (the cusp, where sech u = 1) is actually sampled.
        table = g3.generator("pseudosphere").generate({"radius": 1.0, "extent": 3.0}, 2601)
        ring = np.hypot(table["x"], table["y"])
        assert float(ring.max()) == pytest.approx(1.0, rel=1e-6)
        assert float(ring.min()) < 0.2

    def test_the_monkey_saddle_has_three_descents(self):
        """z(theta) = cos(3 theta) on a circle: three ups and three downs."""
        table = g3.generator("monkey_saddle").generate({"extent": 1.0, "scale": 1.0}, 900)
        theta = np.arctan2(table["v"], table["u"])
        radius = np.hypot(table["u"], table["v"])
        edge = radius > 0.9 * radius.max()
        assert np.allclose(
            table["z"][edge], radius[edge] ** 3 * np.cos(3.0 * theta[edge]), atol=1e-9
        )

    def test_the_knot_tube_stays_at_its_radius_from_the_centre_curve(self):
        tube = 0.3
        spec = g3.generator("knot_tube")
        table = spec.generate({"p": 2.0, "q": 3.0, "tube": tube}, 2500)
        u = table["u"]
        ring = 2.0 + np.cos(3.0 * u)
        centre = np.stack([ring * np.cos(2.0 * u), ring * np.sin(2.0 * u), -np.sin(3.0 * u)], -1)
        pts = np.stack([table["x"], table["y"], table["z"]], axis=-1)
        assert np.allclose(np.linalg.norm(pts - centre, axis=-1), tube, atol=1e-6)

    @pytest.mark.parametrize("key", NEW_SURFACES)
    def test_the_lattice_is_griddable_and_triangulates(self, key):
        spec = g3.generator(key)
        table = spec.generate(samples=900)
        assert l3.grid_shape(table["u"], table["v"]) is not None
        verts, indices, _ = l3.surface_geometry(
            table["x"], table["y"], table["z"], spec.kind_options(900)
        )
        assert indices is not None and len(indices) > 0
        assert len(verts) == len(table["x"])


class TestStatisticalClouds:
    """The clouds whose whole value is a statistical property."""

    def test_the_covariance_is_the_one_that_was_asked_for(self):
        table = g3.generator("multivariate_normal").generate(
            {"sx": 1.0, "sy": 2.0, "sz": 0.5, "rxy": 0.7, "rxz": -0.4, "ryz": 0.1}, 200_000
        )
        pts = np.column_stack([table["x"], table["y"], table["z"]])
        corr = np.corrcoef(pts, rowvar=False)
        assert corr[0, 1] == pytest.approx(0.7, abs=0.02)
        assert corr[0, 2] == pytest.approx(-0.4, abs=0.02)
        assert corr[1, 2] == pytest.approx(0.1, abs=0.02)
        assert float(pts[:, 1].std()) == pytest.approx(2.0, rel=0.05)

    def test_an_impossible_correlation_triple_still_produces_a_cloud(self):
        """(0.9, 0.9, -0.9) is not a covariance matrix; the repair must not produce nan."""
        table = g3.generator("multivariate_normal").generate(
            {"rxy": 0.9, "rxz": 0.9, "ryz": -0.9}, 20_000
        )
        pts = np.column_stack([table["x"], table["y"], table["z"]])
        assert np.all(np.isfinite(pts))
        eigenvalues = np.linalg.eigvalsh(np.cov(pts, rowvar=False))
        assert float(eigenvalues.min()) > -1e-9

    def test_the_mahalanobis_column_is_a_distance_not_a_radius(self):
        """It is the norm of the *uncorrelated* draw, so it is chi(3)-distributed."""
        table = g3.generator("multivariate_normal").generate({"sx": 10.0}, 50_000)
        assert float(table["d"].min()) >= 0.0
        assert float(table["d"].mean()) == pytest.approx(1.5957, rel=0.05)

    def test_the_swiss_roll_lies_on_its_spiral(self):
        table = g3.generator("swiss_roll").generate({"noise": 0.0}, 20_000)
        assert np.allclose(np.hypot(table["x"], table["y"]), np.abs(table["t"]))
        assert float(table["t"].min()) == pytest.approx(1.5 * np.pi, rel=0.01)

    def test_the_swiss_roll_noise_thickens_the_sheet(self):
        clean = g3.generator("swiss_roll").generate({"noise": 0.0}, 5_000)
        noisy = g3.generator("swiss_roll").generate({"noise": 0.5}, 5_000)
        clean_error = np.abs(np.hypot(clean["x"], clean["y"]) - clean["t"])
        noisy_error = np.abs(np.hypot(noisy["x"], noisy["y"]) - noisy["t"])
        assert float(noisy_error.mean()) > float(clean_error.mean()) + 0.1

    def test_the_s_curve_is_the_sklearn_one(self):
        table = g3.generator("s_curve").generate({"noise": 0.0}, 10_000)
        assert np.allclose(table["x"], np.sin(table["t"]))
        assert np.allclose(table["z"], np.sign(table["t"]) * (np.cos(table["t"]) - 1.0))

    def test_two_moons_returns_exactly_the_rows_asked_for(self):
        table = g3.generator("two_moons").generate(samples=1001)
        assert len(table["x"]) == 1001
        counts = np.bincount(table["moon"].astype(int))
        assert abs(int(counts[0]) - int(counts[1])) <= 1

    def test_two_moons_separates_its_classes_along_z(self):
        table = g3.generator("two_moons").generate({"lift": 2.0, "noise": 0.0}, 2_000)
        upper = table["z"][table["moon"] > 0.5]
        lower = table["z"][table["moon"] < 0.5]
        assert float(lower.max()) < float(upper.min())

    def test_the_uniform_box_respects_its_half_widths(self):
        table = g3.generator("uniform_cube").generate({"sx": 1.0, "sy": 5.0, "sz": 0.25}, 20_000)
        assert float(np.abs(table["x"]).max()) <= 1.0
        assert float(np.abs(table["y"]).max()) <= 5.0
        assert float(np.abs(table["z"]).max()) <= 0.25
        assert float(np.abs(table["y"]).max()) > 4.5


class TestCrystalLattices:
    """The lattices, checked by coordination and by site count."""

    @pytest.mark.parametrize("key", NEW_LATTICES)
    def test_no_site_is_duplicated(self, key):
        table = g3.generator(key).generate(samples=1000)
        pts = np.column_stack([table["x"], table["y"], table["z"]])
        assert len(np.unique(np.round(pts, 9), axis=0)) == len(pts)

    @pytest.mark.parametrize(
        "key,per_cell", [("bcc_lattice", 2), ("diamond_lattice", 8), ("hcp_lattice", 2)]
    )
    def test_the_row_count_is_whole_unit_cells(self, key, per_cell):
        table = g3.generator(key).generate(samples=1000)
        assert len(table["x"]) % per_cell == 0
        cells = len(table["x"]) // per_cell
        assert round(cells ** (1.0 / 3.0)) ** 3 == cells

    def test_bcc_has_a_site_at_the_centre_of_every_cell(self):
        table = g3.generator("bcc_lattice").generate({"spacing": 2.0, "jitter": 0.0}, 1000)
        pts = np.column_stack([table["x"], table["y"], table["z"]])
        assert np.any(np.all(np.isclose(pts, [1.0, 1.0, 1.0]), axis=1))

    def test_diamond_nearest_neighbours_are_a_quarter_body_diagonal(self):
        table = g3.generator("diamond_lattice").generate({"spacing": 1.0, "jitter": 0.0}, 512)
        pts = np.column_stack([table["x"], table["y"], table["z"]])
        distances = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
        np.fill_diagonal(distances, np.inf)
        assert float(distances.min()) == pytest.approx(math.sqrt(3.0) / 4.0, rel=1e-6)

    def test_hcp_has_twelve_nearest_neighbours(self):
        """The signature of a close packing: coordination 12 at the ideal c/a."""
        table = g3.generator("hcp_lattice").generate({"spacing": 1.0, "jitter": 0.0}, 2000)
        pts = np.column_stack([table["x"], table["y"], table["z"]])
        distances = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
        np.fill_diagonal(distances, np.inf)
        nearest = float(distances.min())
        assert nearest == pytest.approx(1.0, rel=1e-6)
        counts = np.count_nonzero(np.isclose(distances, nearest, rtol=1e-6), axis=1)
        assert int(counts.max()) == 12

    def test_the_honeycomb_is_three_fold_not_six_fold(self):
        """A triangular lattice would give 6 neighbours; the two-atom basis gives 3."""
        table = g3.generator("honeycomb").generate({"spacing": 1.0, "jitter": 0.0}, 200)
        pts = np.column_stack([table["x"], table["y"], table["z"]])
        distances = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
        np.fill_diagonal(distances, np.inf)
        assert float(distances.min()) == pytest.approx(1.0, rel=1e-9)
        counts = np.count_nonzero(np.isclose(distances, 1.0, rtol=1e-9), axis=1)
        assert int(counts.max()) == 3

    def test_stacking_the_honeycomb_makes_graphite(self):
        one = g3.generator("honeycomb").generate({"layers": 1.0}, 400)
        four = g3.generator("honeycomb").generate({"layers": 4.0}, 400)
        assert float(np.ptp(one["z"])) == 0.0
        assert len(np.unique(np.round(four["z"], 6))) == 4

    def test_the_quasilattice_has_two_spacings_in_the_golden_ratio(self):
        table = g3.generator("quasilattice").generate({"spacing": 1.0, "jitter": 0.0}, 1000)
        axis = np.unique(np.round(table["x"], 9))
        gaps = np.unique(np.round(np.diff(axis), 6))
        assert len(gaps) == 2
        assert float(gaps[1] / gaps[0]) == pytest.approx((1.0 + math.sqrt(5.0)) / 2.0, rel=1e-5)

    def test_the_quasilattice_is_aperiodic(self):
        """A periodic chain would repeat its gap word; a Sturmian one never does."""
        table = g3.generator("quasilattice").generate({"spacing": 1.0, "jitter": 0.0}, 8000)
        axis = np.unique(np.round(table["x"], 9))
        word = (np.diff(axis) > np.mean(np.diff(axis))).astype(int)
        assert len(word) >= 10
        for period in range(1, len(word) // 2):
            head, tail = word[:-period], word[period:]
            assert not np.array_equal(head, tail), f"repeats with period {period}"

    def test_the_fibonacci_sphere_puts_every_point_on_the_sphere(self):
        table = g3.generator("fibonacci_sphere").generate({"radius": 2.0, "jitter": 0.0}, 5000)
        r = np.sqrt(table["x"] ** 2 + table["y"] ** 2 + table["z"] ** 2)
        assert np.allclose(r, 2.0, atol=1e-9)
        assert len(r) == 5000

    def test_the_fibonacci_sphere_is_more_even_than_a_random_one(self):
        """The point of the golden angle: no clumps. Compare worst-case z gaps."""
        table = g3.generator("fibonacci_sphere").generate({"jitter": 0.0}, 4000)
        spiral_gap = float(np.diff(np.sort(table["z"])).max())
        random_z = np.sort(np.random.default_rng(0).uniform(-1.0, 1.0, 4000))
        assert spiral_gap < float(np.diff(random_z).max())


class TestNewFields:
    """Vector fields, checked by the symmetries that identify them."""

    def test_taylor_green_has_no_vertical_velocity(self):
        table = g3.generator("taylor_green").generate(samples=512)
        assert np.all(table["w"] == 0.0)
        assert float(np.abs(table["u"]).max()) > 0.1

    def test_taylor_green_is_antisymmetric_under_swapping_x_and_y(self):
        table = g3.generator("taylor_green").generate(samples=512)
        index = {
            (round(x, 9), round(y, 9), round(z, 9)): i
            for i, (x, y, z) in enumerate(zip(table["x"], table["y"], table["z"]))
        }
        for i, (x, y, z) in enumerate(zip(table["x"], table["y"], table["z"])):
            mirrored = index[(round(y, 9), round(x, 9), round(z, 9))]
            assert table["u"][i] == pytest.approx(-table["v"][mirrored], abs=1e-12)

    def test_hills_vortex_tends_to_a_uniform_stream_far_away(self):
        table = g3.generator("hill_vortex").generate(
            {"radius": 1.0, "speed": 1.0, "extent": 8.0}, 512
        )
        r = np.sqrt(table["x"] ** 2 + table["y"] ** 2 + table["z"] ** 2)
        far = r > 10.0
        assert np.any(far)
        assert np.allclose(table["w"][far], 1.0, atol=0.05)
        assert np.allclose(np.hypot(table["u"][far], table["v"][far]), 0.0, atol=0.05)

    def test_hills_vortex_moves_faster_inside_the_sphere(self):
        """The core carries the fluid along faster than the stream it sits in."""
        table = g3.generator("hill_vortex").generate(
            {"radius": 4.0, "speed": 1.0, "extent": 6.0}, 512
        )
        r = np.sqrt(table["x"] ** 2 + table["y"] ** 2 + table["z"] ** 2)
        core = int(np.argmin(r))
        assert float(r[core]) < 4.0
        assert float(table["w"][core]) > 1.1
        # 1.5 U bounds both branches: the axial speed at the centre of Hill's vortex, and
        # the equatorial speed of potential flow past a sphere. Nothing may exceed it.
        assert float(table["w"].max()) <= 1.5 + 1e-9

    def test_the_linear_field_is_exactly_its_three_gains(self):
        table = g3.generator("linear_field").generate({"a": 2.0, "b": -1.0, "c": 0.5}, 512)
        assert np.allclose(table["u"], 2.0 * table["x"])
        assert np.allclose(table["v"], -1.0 * table["y"])
        assert np.allclose(table["w"], 0.5 * table["z"])

    def test_the_linear_field_defaults_to_an_incompressible_saddle(self):
        spec = g3.generator("linear_field")
        gains = spec.defaults()
        assert gains["a"] + gains["b"] + gains["c"] == pytest.approx(0.0)
        assert min(gains["a"], gains["b"], gains["c"]) < 0.0 < max(gains["a"], gains["b"])

    def test_the_lorenz_field_has_the_lorenz_symmetry(self):
        """(x, y, z) -> (-x, -y, z) must map the field to (-u, -v, w)."""
        table = g3.generator("lorenz_field").generate(samples=512)
        index = {
            (round(x, 9), round(y, 9), round(z, 9)): i
            for i, (x, y, z) in enumerate(zip(table["x"], table["y"], table["z"]))
        }
        for i, (x, y, z) in enumerate(zip(table["x"], table["y"], table["z"])):
            mirrored = index[(round(-x, 9), round(-y, 9), round(z, 9))]
            assert table["u"][i] == pytest.approx(-table["u"][mirrored], abs=1e-9)
            assert table["w"][i] == pytest.approx(table["w"][mirrored], abs=1e-9)

    def test_the_roessler_field_folds_through_z(self):
        """Its only nonlinearity is z(x - c), so w must depend on x wherever z is not 0."""
        table = g3.generator("roessler_field").generate({"b": 0.0, "c": 5.7}, 512)
        assert np.allclose(table["w"], table["z"] * (table["x"] - 5.7))
        assert np.allclose(table["u"], -table["y"] - table["z"])

    @pytest.mark.parametrize("key", NEW_FIELDS)
    def test_every_new_field_carries_a_non_zero_vector(self, key):
        table = g3.generator(key).generate(samples=512)
        magnitude = np.sqrt(table["u"] ** 2 + table["v"] ** 2 + table["w"] ** 2)
        assert float(magnitude.max()) > 1e-9
        assert np.all(np.isfinite(magnitude))


class TestNewGeneratorsPlot:
    """Every new object, through its own default kind, into a real scene.

    The same contract ``TestEndToEnd`` applies to the whole catalogue, asserted here for
    the additions specifically so that a failure names the new object rather than a
    parametrised sweep.
    """

    @pytest.mark.parametrize("key", NEW_KEYS)
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

    @pytest.mark.parametrize("key", NEW_KEYS)
    def test_it_survives_both_ends_of_every_slider(self, key):
        """A slider the user can drag into a nan is a slider that ships broken."""
        spec = g3.generator(key)
        for param in spec.params:
            for value in (param.vmin, param.vmax):
                table = spec.generate({param.name: value}, 512)
                for name, values in table.items():
                    assert np.all(np.isfinite(values)), f"{key}.{param.name}={value} -> {name}"
