"""Test the 2-D multivariate operations in glplot.gui.mathops2d.

Every function under test takes numpy arrays and returns new numpy arrays, so no OpenGL
context, no window, and no live engine scene is required: these tests run fully headless,
mirroring tests/test_gui_mathops.py's conventions (closed-form/synthetic references, an
explicit fallback test per optional scipy accelerator).
"""

from __future__ import annotations

import math
import sys

import numpy as np
import pytest

from glplot.gui import mathops2d


def _blobs(centers, spread=0.3, n_per=100, seed=0):
    """Points drawn from well-separated Gaussian blobs at ``centers``."""
    rng = np.random.default_rng(seed)
    parts = [rng.normal(c, spread, (n_per, 2)) for c in centers]
    pts = np.vstack(parts)
    truth_labels = np.repeat(np.arange(len(centers)), n_per)
    return pts[:, 0], pts[:, 1], truth_labels


def _matched_centroid_error(centers, cx, cy):
    """Best-matching (greedy) distance from each true center to a recovered centroid."""
    centers = np.asarray(centers, dtype=np.float64)
    recovered = np.column_stack([cx, cy])
    total = 0.0
    used = set()
    for center in centers:
        dists = np.hypot(recovered[:, 0] - center[0], recovered[:, 1] - center[1])
        order = np.argsort(dists)
        for idx in order:
            if idx not in used:
                used.add(int(idx))
                total += float(dists[idx])
                break
    return total / len(centers)


class TestKmeansClusterRecovery:
    """kmeans_cluster must recover well-separated cluster centers."""

    def test_two_well_separated_blobs(self):
        x, y, _truth = _blobs([(0.0, 0.0), (10.0, 10.0)], seed=1)
        labels, cx, cy = mathops2d.kmeans_cluster(x, y, 2, seed=1)
        assert cx.size == 2 and cy.size == 2
        assert _matched_centroid_error([(0.0, 0.0), (10.0, 10.0)], cx, cy) < 0.3

    def test_three_well_separated_blobs(self):
        centers = [(0.0, 0.0), (10.0, 0.0), (5.0, 10.0)]
        x, y, _truth = _blobs(centers, seed=2)
        labels, cx, cy = mathops2d.kmeans_cluster(x, y, 3, seed=2)
        assert cx.size == 3 and cy.size == 3
        assert _matched_centroid_error(centers, cx, cy) < 0.3

    def test_labels_partition_the_points_consistently(self):
        """Every point in the same true blob must share a label with the others."""
        x, y, truth = _blobs([(0.0, 0.0), (20.0, 20.0)], seed=3)
        labels, _cx, _cy = mathops2d.kmeans_cluster(x, y, 2, seed=3)
        # Points from truth-blob 0 all share one label, blob 1 all share another,
        # and the two labels differ (blobs are far enough apart that this cannot fail).
        blob0_labels = set(labels[truth == 0].tolist())
        blob1_labels = set(labels[truth == 1].tolist())
        assert len(blob0_labels) == 1
        assert len(blob1_labels) == 1
        assert blob0_labels != blob1_labels

    def test_labels_are_dense_nonnegative_integers(self):
        x, y, _truth = _blobs([(0.0, 0.0), (10.0, 10.0), (0.0, 10.0)], seed=4)
        labels, cx, _cy = mathops2d.kmeans_cluster(x, y, 3, seed=4)
        present = sorted({int(v) for v in labels.tolist()})
        assert present == list(range(cx.size))

    def test_k_equal_to_point_count_gives_one_cluster_per_point(self):
        x = np.array([0.0, 5.0, 10.0])
        y = np.array([0.0, 5.0, 10.0])
        labels, cx, cy = mathops2d.kmeans_cluster(x, y, 3, seed=5)
        assert cx.size == 3
        assert len(set(labels.tolist())) == 3

    def test_deterministic_given_the_same_seed(self):
        x, y, _truth = _blobs([(0.0, 0.0), (10.0, 10.0)], seed=6)
        r1 = mathops2d.kmeans_cluster(x, y, 2, seed=42)
        r2 = mathops2d.kmeans_cluster(x, y, 2, seed=42)
        assert np.array_equal(r1[0], r2[0])
        assert np.allclose(r1[1], r2[1])
        assert np.allclose(r1[2], r2[2])


class TestKmeansClusterNonFinite:
    def test_nonfinite_rows_get_label_negative_one(self):
        x, y, _truth = _blobs([(0.0, 0.0), (10.0, 10.0)], seed=7)
        x = x.copy()
        x[0] = np.nan
        y = y.copy()
        y[1] = np.inf
        labels, _cx, _cy = mathops2d.kmeans_cluster(x, y, 2, seed=7)
        assert labels[0] == -1.0
        assert labels[1] == -1.0
        assert np.all(labels[2:] >= 0.0)

    def test_labels_length_matches_input_length_including_dropped_rows(self):
        x, y, _truth = _blobs([(0.0, 0.0), (10.0, 10.0)], seed=8)
        x = x.copy()
        x[5] = np.nan
        labels, _cx, _cy = mathops2d.kmeans_cluster(x, y, 2, seed=8)
        assert labels.shape == x.shape


class TestKmeansClusterErrors:
    def test_k_less_than_one_raises(self):
        with pytest.raises(ValueError, match="k must be"):
            mathops2d.kmeans_cluster([1.0, 2.0], [1.0, 2.0], 0)

    def test_k_greater_than_finite_points_raises(self):
        with pytest.raises(ValueError, match="cannot exceed"):
            mathops2d.kmeans_cluster([1.0, 2.0], [1.0, 2.0], 5)

    def test_all_nonfinite_raises(self):
        with pytest.raises(ValueError, match="finite"):
            mathops2d.kmeans_cluster([np.nan, np.nan], [np.nan, np.nan], 1)

    def test_mismatched_length_raises(self):
        with pytest.raises(ValueError):
            mathops2d.kmeans_cluster([1.0, 2.0, 3.0], [1.0, 2.0], 1)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            mathops2d.kmeans_cluster([], [], 1)

    def test_non_integer_k_raises(self):
        with pytest.raises(ValueError, match="integer"):
            mathops2d.kmeans_cluster([1.0, 2.0], [1.0, 2.0], "not a number")


class TestKmeansClusterFallback:
    """No scipy.cluster: the pure-numpy Lloyd's-algorithm path must still work, and work
    accurately -- this fallback is documented as honest, not degraded."""

    def test_falls_back_and_still_recovers_centers(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "scipy.cluster", None)
        monkeypatch.setitem(sys.modules, "scipy.cluster.vq", None)
        centers = [(0.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
        x, y, _truth = _blobs(centers, seed=9)
        labels, cx, cy = mathops2d.kmeans_cluster(x, y, 3, seed=9)
        assert cx.size == 3
        assert _matched_centroid_error(centers, cx, cy) < 0.3

    def test_fallback_labels_are_dense_nonnegative_integers(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "scipy.cluster", None)
        monkeypatch.setitem(sys.modules, "scipy.cluster.vq", None)
        x, y, _truth = _blobs([(0.0, 0.0), (10.0, 10.0)], seed=10)
        labels, cx, _cy = mathops2d.kmeans_cluster(x, y, 2, seed=10)
        present = sorted({int(v) for v in labels.tolist()})
        assert present == list(range(cx.size))

    def test_kmeans_cluster_works_again_once_scipy_is_importable(self):
        """The monkeypatched modules above must not leak into the rest of the suite."""
        centers = [(0.0, 0.0), (10.0, 10.0)]
        x, y, _truth = _blobs(centers, seed=11)
        labels, cx, cy = mathops2d.kmeans_cluster(x, y, 2, seed=11)
        assert cx.size == 2
        assert _matched_centroid_error(centers, cx, cy) < 0.3


class TestDensity2dHistogram:
    """mode="histogram" is a thin, exact wrapper around np.histogram2d."""

    def test_matches_np_histogram2d_bin_for_bin(self):
        rng = np.random.default_rng(0)
        x, y = rng.normal(0, 1, 500), rng.normal(0, 1, 500)
        x_edges, y_edges, density = mathops2d.density2d(x, y, mode="histogram", bins=15)
        ref_counts, ref_x, ref_y = np.histogram2d(x, y, bins=15)
        assert np.allclose(x_edges, ref_x)
        assert np.allclose(y_edges, ref_y)
        assert np.array_equal(density, ref_counts)

    def test_edges_length_matches_bins_plus_one(self):
        x = np.linspace(0.0, 1.0, 100)
        y = np.linspace(0.0, 1.0, 100)
        x_edges, y_edges, density = mathops2d.density2d(x, y, mode="histogram", bins=20)
        assert x_edges.size == 21
        assert y_edges.size == 21
        assert density.shape == (20, 20)

    def test_total_count_equals_finite_point_count(self):
        rng = np.random.default_rng(1)
        x, y = rng.normal(0, 1, 300), rng.normal(0, 1, 300)
        _xe, _ye, density = mathops2d.density2d(x, y, mode="histogram", bins=10)
        assert float(density.sum()) == pytest.approx(300.0)

    def test_nonfinite_pairs_are_excluded_from_the_count(self):
        x = np.array([0.0, np.nan, 1.0, 2.0])
        y = np.array([0.0, 1.0, np.inf, 2.0])
        _xe, _ye, density = mathops2d.density2d(x, y, mode="histogram", bins=5)
        assert float(density.sum()) == pytest.approx(2.0)  # only rows 0 and 3 are finite

    def test_bins_less_than_one_raises(self):
        with pytest.raises(ValueError, match="bins"):
            mathops2d.density2d([1.0, 2.0], [1.0, 2.0], mode="histogram", bins=0)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            mathops2d.density2d([], [], mode="histogram")


class TestDensity2dKde:
    """mode="kde": a smooth density surface integrating to ~1 over its own grid."""

    def test_integrates_to_approximately_one(self):
        """sum(density) * cell_area ~ 1 -- most of a KDE's mass lands inside a grid
        padded 5% beyond the data range."""
        rng = np.random.default_rng(2)
        x, y = rng.normal(0, 1, 2000), rng.normal(0, 1, 2000)
        x_edges, y_edges, density = mathops2d.density2d(x, y, mode="kde", grid=80)
        cell_area = (x_edges[1] - x_edges[0]) * (y_edges[1] - y_edges[0])
        total = float(density.sum()) * cell_area
        assert 0.85 < total <= 1.05

    def test_edges_length_matches_grid_plus_one(self):
        x = np.linspace(0.0, 1.0, 100)
        y = np.linspace(0.0, 1.0, 100) + np.sin(np.linspace(0, 10, 100)) * 0.01
        x_edges, y_edges, density = mathops2d.density2d(x, y, mode="kde", grid=25)
        assert x_edges.size == 26
        assert y_edges.size == 26
        assert density.shape == (25, 25)

    def test_density_is_higher_near_a_tight_cluster_than_far_from_it(self):
        rng = np.random.default_rng(3)
        x, y = rng.normal(0, 0.3, 500), rng.normal(0, 0.3, 500)
        x_edges, y_edges, density = mathops2d.density2d(x, y, mode="kde", grid=40)
        x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
        y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
        center_idx = (np.argmin(np.abs(x_centers)), np.argmin(np.abs(y_centers)))
        edge_idx = (0, 0)
        assert density[center_idx] > density[edge_idx]

    def test_bw_method_one_matches_automatic(self):
        """bw_method=1.0 is documented as reproducing the automatic bandwidth exactly."""
        rng = np.random.default_rng(4)
        x, y = rng.normal(0, 1, 300), rng.normal(0, 1, 300)
        _xe1, _ye1, auto = mathops2d.density2d(x, y, mode="kde", grid=20)
        _xe2, _ye2, same = mathops2d.density2d(x, y, mode="kde", grid=20, bw_method=1.0)
        assert np.allclose(auto, same)

    def test_smaller_bw_method_is_sharper(self):
        """A smaller multiplier concentrates density more -- a higher peak."""
        rng = np.random.default_rng(5)
        x, y = rng.normal(0, 1, 500), rng.normal(0, 1, 500)
        _xe1, _ye1, sharp = mathops2d.density2d(x, y, mode="kde", grid=30, bw_method=0.3)
        _xe2, _ye2, auto = mathops2d.density2d(x, y, mode="kde", grid=30)
        assert float(sharp.max()) > float(auto.max())

    def test_larger_bw_method_is_smoother(self):
        rng = np.random.default_rng(6)
        x, y = rng.normal(0, 1, 500), rng.normal(0, 1, 500)
        _xe1, _ye1, smooth = mathops2d.density2d(x, y, mode="kde", grid=30, bw_method=3.0)
        _xe2, _ye2, auto = mathops2d.density2d(x, y, mode="kde", grid=30)
        assert float(smooth.max()) < float(auto.max())

    def test_grid_less_than_one_raises(self):
        rng = np.random.default_rng(7)
        x, y = rng.normal(0, 1, 20), rng.normal(0, 1, 20)
        with pytest.raises(ValueError, match="grid"):
            mathops2d.density2d(x, y, mode="kde", grid=0)

    def test_fewer_than_two_points_raises(self):
        with pytest.raises(ValueError, match="2 finite"):
            mathops2d.density2d([1.0], [1.0], mode="kde")

    def test_unavailable_scipy_raises_kde_unavailable_error(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "scipy.stats", None)
        assert mathops2d.kde2d_available() is False
        x = np.linspace(0.0, 1.0, 50)
        y = np.linspace(0.0, 1.0, 50)
        with pytest.raises(mathops2d.KdeUnavailableError, match="scipy"):
            mathops2d.density2d(x, y, mode="kde")
        assert issubclass(mathops2d.KdeUnavailableError, ValueError)

    def test_kde_works_again_once_scipy_is_importable(self):
        """The monkeypatched module above must not leak into the rest of the suite."""
        assert mathops2d.kde2d_available() is True
        rng = np.random.default_rng(8)
        x, y = rng.normal(0, 1, 50), rng.normal(0, 1, 50)
        _xe, _ye, density = mathops2d.density2d(x, y, mode="kde", grid=10)
        assert density.shape == (10, 10)


class TestDensity2dErrors:
    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="mode"):
            mathops2d.density2d([1.0, 2.0], [1.0, 2.0], mode="bogus")

    def test_all_nonfinite_raises(self):
        with pytest.raises(ValueError, match="finite"):
            mathops2d.density2d([np.nan, np.nan], [np.nan, np.nan], mode="histogram")

    def test_mismatched_length_raises(self):
        with pytest.raises(ValueError):
            mathops2d.density2d([1.0, 2.0, 3.0], [1.0, 2.0], mode="histogram")


class TestHierarchicalClusterRecovery:
    """hierarchical_cluster must recover well-separated cluster centers, any linkage."""

    @pytest.mark.parametrize("method", ["ward", "single", "complete", "average"])
    def test_three_well_separated_blobs(self, method):
        centers = [(0.0, 0.0), (10.0, 0.0), (5.0, 10.0)]
        x, y, _truth = _blobs(centers, seed=20)
        labels, cx, cy = mathops2d.hierarchical_cluster(x, y, 3, method=method)
        assert cx.size == 3 and cy.size == 3
        assert _matched_centroid_error(centers, cx, cy) < 0.3

    def test_labels_partition_the_points_consistently(self):
        x, y, truth = _blobs([(0.0, 0.0), (20.0, 20.0)], seed=21)
        labels, _cx, _cy = mathops2d.hierarchical_cluster(x, y, 2)
        blob0 = set(labels[truth == 0].tolist())
        blob1 = set(labels[truth == 1].tolist())
        assert len(blob0) == 1 and len(blob1) == 1 and blob0 != blob1

    def test_labels_are_dense_nonnegative_integers(self):
        x, y, _truth = _blobs([(0.0, 0.0), (10.0, 10.0), (0.0, 10.0)], seed=22)
        labels, cx, _cy = mathops2d.hierarchical_cluster(x, y, 3)
        present = sorted({int(v) for v in labels.tolist()})
        assert present == list(range(cx.size))

    def test_returns_the_same_shape_contract_as_kmeans(self):
        """The two clustering functions must be interchangeable to a caller."""
        x, y, _truth = _blobs([(0.0, 0.0), (10.0, 10.0)], seed=23)
        labels_h, cx_h, cy_h = mathops2d.hierarchical_cluster(x, y, 2)
        labels_k, cx_k, cy_k = mathops2d.kmeans_cluster(x, y, 2, seed=1)
        assert labels_h.shape == labels_k.shape == x.shape
        assert cx_h.shape == cy_h.shape
        assert cx_k.shape == cy_k.shape


class TestHierarchicalClusterNonFinite:
    def test_nonfinite_rows_get_label_negative_one(self):
        x, y, _truth = _blobs([(0.0, 0.0), (10.0, 10.0)], seed=24)
        x = x.copy()
        x[0] = np.nan
        labels, _cx, _cy = mathops2d.hierarchical_cluster(x, y, 2)
        assert labels[0] == -1.0
        assert np.all(labels[1:] >= 0.0)


class TestHierarchicalClusterErrors:
    def test_k_less_than_one_raises(self):
        with pytest.raises(ValueError, match="k must be"):
            mathops2d.hierarchical_cluster([1.0, 2.0], [1.0, 2.0], 0)

    def test_k_greater_than_finite_points_raises(self):
        with pytest.raises(ValueError, match="cannot exceed"):
            mathops2d.hierarchical_cluster([1.0, 2.0], [1.0, 2.0], 5)

    def test_all_nonfinite_raises(self):
        with pytest.raises(ValueError, match="finite"):
            mathops2d.hierarchical_cluster([np.nan, np.nan], [np.nan, np.nan], 1)

    def test_unknown_method_raises(self):
        x, y, _truth = _blobs([(0.0, 0.0), (10.0, 10.0)], seed=25)
        with pytest.raises(ValueError, match="unknown method"):
            mathops2d.hierarchical_cluster(x, y, 2, method="bogus")

    def test_mismatched_length_raises(self):
        with pytest.raises(ValueError):
            mathops2d.hierarchical_cluster([1.0, 2.0, 3.0], [1.0, 2.0], 1)

    def test_single_point_is_its_own_cluster(self):
        labels, cx, cy = mathops2d.hierarchical_cluster([1.0], [2.0], 1)
        assert labels[0] == 0.0
        assert cx[0] == pytest.approx(1.0)
        assert cy[0] == pytest.approx(2.0)

    def test_unavailable_scipy_raises_hierarchical_unavailable_error(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "scipy.cluster", None)
        monkeypatch.setitem(sys.modules, "scipy.cluster.hierarchy", None)
        assert mathops2d.hierarchical_available() is False
        x, y, _truth = _blobs([(0.0, 0.0), (10.0, 10.0)], seed=26)
        with pytest.raises(mathops2d.HierarchicalUnavailableError, match="scipy"):
            mathops2d.hierarchical_cluster(x, y, 2)
        assert issubclass(mathops2d.HierarchicalUnavailableError, ValueError)

    def test_hierarchical_cluster_works_again_once_scipy_is_importable(self):
        """The monkeypatched modules above must not leak into the rest of the suite."""
        assert mathops2d.hierarchical_available() is True
        centers = [(0.0, 0.0), (10.0, 10.0)]
        x, y, _truth = _blobs(centers, seed=27)
        labels, cx, cy = mathops2d.hierarchical_cluster(x, y, 2)
        assert _matched_centroid_error(centers, cx, cy) < 0.3


class TestSpatialStats:
    """spatial_stats(): nearest-neighbour spacing, bounding box, convex hull."""

    def test_unit_square_hull_area_and_perimeter(self):
        """A unit square (plus an interior point, which must not appear on the hull)."""
        x = np.array([0.0, 1.0, 1.0, 0.0, 0.5])
        y = np.array([0.0, 0.0, 1.0, 1.0, 0.5])
        stats, hull_x, hull_y = mathops2d.spatial_stats(x, y)
        assert stats["hull_area"] == pytest.approx(1.0)
        assert stats["hull_perimeter"] == pytest.approx(4.0)
        assert stats["hull_vertex_count"] == 4.0
        assert hull_x is not None and hull_y is not None

    def test_hull_polygon_is_closed(self):
        x = np.array([0.0, 1.0, 1.0, 0.0])
        y = np.array([0.0, 0.0, 1.0, 1.0])
        _stats, hull_x, hull_y = mathops2d.spatial_stats(x, y)
        assert hull_x[0] == pytest.approx(hull_x[-1])
        assert hull_y[0] == pytest.approx(hull_y[-1])

    def test_bounding_box_is_correct(self):
        x = np.array([1.0, 5.0, 3.0, -2.0])
        y = np.array([10.0, 2.0, 7.0, 4.0])
        stats, _hx, _hy = mathops2d.spatial_stats(x, y)
        assert stats["bbox_width"] == pytest.approx(7.0)  # 5 - (-2)
        assert stats["bbox_height"] == pytest.approx(8.0)  # 10 - 2

    def test_nearest_neighbor_distance_on_a_known_grid(self):
        """A unit grid: every point's nearest neighbour is exactly 1.0 away."""
        xs, ys = np.meshgrid(np.arange(5.0), np.arange(5.0))
        stats, _hx, _hy = mathops2d.spatial_stats(xs.ravel(), ys.ravel())
        assert stats["min_nn_distance"] == pytest.approx(1.0)
        assert stats["mean_nn_distance"] == pytest.approx(1.0)
        assert stats["max_nn_distance"] == pytest.approx(1.0)

    def test_duplicate_points_give_zero_min_nn_distance(self):
        x = np.array([0.0, 0.0, 5.0])
        y = np.array([0.0, 0.0, 5.0])
        stats, _hx, _hy = mathops2d.spatial_stats(x, y)
        assert stats["min_nn_distance"] == pytest.approx(0.0)

    def test_collinear_points_have_no_hull(self):
        x = np.linspace(0.0, 10.0, 20)
        y = np.zeros(20)
        stats, hull_x, hull_y = mathops2d.spatial_stats(x, y)
        assert "hull_area" not in stats
        assert hull_x is None and hull_y is None

    def test_two_points_have_no_hull_but_do_have_nn_distance(self):
        stats, hull_x, hull_y = mathops2d.spatial_stats([0.0, 3.0], [0.0, 4.0])
        assert hull_x is None and hull_y is None
        assert stats["min_nn_distance"] == pytest.approx(5.0)  # 3-4-5 triangle

    def test_single_point_has_no_hull_and_no_nn_distance(self):
        stats, hull_x, hull_y = mathops2d.spatial_stats([1.0], [2.0])
        assert hull_x is None and hull_y is None
        assert "min_nn_distance" not in stats
        assert stats["n"] == 1.0

    def test_nonfinite_pairs_are_excluded(self):
        x = np.array([0.0, np.nan, 1.0, 1.0])
        y = np.array([0.0, 1.0, np.inf, 1.0])
        stats, _hx, _hy = mathops2d.spatial_stats(x, y)
        assert stats["n"] == 2.0  # only rows 0 and 3 are finite

    def test_all_nonfinite_raises(self):
        with pytest.raises(ValueError, match="finite"):
            mathops2d.spatial_stats([np.nan, np.nan], [np.nan, np.nan])

    def test_mismatched_length_raises(self):
        with pytest.raises(ValueError):
            mathops2d.spatial_stats([1.0, 2.0, 3.0], [1.0, 2.0])

    def test_unavailable_scipy_raises_spatial_unavailable_error(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "scipy.spatial", None)
        assert mathops2d.spatial_available() is False
        with pytest.raises(mathops2d.SpatialUnavailableError, match="scipy"):
            mathops2d.spatial_stats([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        assert issubclass(mathops2d.SpatialUnavailableError, ValueError)

    def test_spatial_stats_works_again_once_scipy_is_importable(self):
        """The monkeypatched module above must not leak into the rest of the suite."""
        assert mathops2d.spatial_available() is True
        stats, _hx, _hy = mathops2d.spatial_stats([0.0, 1.0, 2.0], [0.0, 1.0, 0.0])
        assert stats["n"] == 3.0


class TestHeldOutClusterScoring:
    """nearest_centroid/cluster_inertia/silhouette_score -- Phase 1's held-out scoring
    tools for the Cluster tab's train/val/test split."""

    def test_nearest_centroid_assigns_the_closest_one(self):
        cx = np.array([0.0, 10.0])
        cy = np.array([0.0, 10.0])
        x = np.array([0.5, 9.0, -1.0])
        y = np.array([0.5, 11.0, 0.0])
        labels = mathops2d.nearest_centroid(x, y, cx, cy)
        np.testing.assert_array_equal(labels, [0.0, 1.0, 0.0])

    def test_nearest_centroid_marks_nonfinite_rows_as_negative_one(self):
        cx = np.array([0.0, 10.0])
        cy = np.array([0.0, 10.0])
        x = np.array([0.0, np.nan, 10.0])
        y = np.array([0.0, 1.0, np.inf])
        labels = mathops2d.nearest_centroid(x, y, cx, cy)
        assert labels[0] == 0.0
        assert labels[1] == -1.0
        assert labels[2] == -1.0

    def test_cluster_inertia_matches_a_manual_sum_of_squared_distances(self):
        x = np.array([0.0, 1.0, 10.0, 11.0])
        y = np.array([0.0, 1.0, 10.0, 9.0])
        cx = np.array([0.5, 10.0])
        cy = np.array([0.5, 9.5])
        labels = np.array([0.0, 0.0, 1.0, 1.0])
        expected = (
            (x[0] - cx[0]) ** 2
            + (y[0] - cy[0]) ** 2
            + (x[1] - cx[0]) ** 2
            + (y[1] - cy[0]) ** 2
            + (x[2] - cx[1]) ** 2
            + (y[2] - cy[1]) ** 2
            + (x[3] - cx[1]) ** 2
            + (y[3] - cy[1]) ** 2
        )
        got = mathops2d.cluster_inertia(x, y, cx, cy, labels)
        assert got == pytest.approx(float(expected))

    def test_silhouette_is_high_for_well_separated_clusters(self):
        x, y, truth = _blobs([(0.0, 0.0), (50.0, 50.0)], spread=0.2, n_per=60, seed=20)
        score = mathops2d.silhouette_score(x, y, truth.astype(np.float64))
        assert score == pytest.approx(1.0, abs=0.05)

    def test_silhouette_is_nan_with_a_single_effective_cluster(self):
        x = np.array([0.0, 1.0, 2.0, 3.0])
        y = np.array([0.0, 1.0, 2.0, 3.0])
        # Every point in cluster 0 except one lone point in cluster 1: only one cluster
        # has >= 2 members, so the metric is undefined.
        labels = np.array([0.0, 0.0, 0.0, 1.0])
        assert math.isnan(mathops2d.silhouette_score(x, y, labels))

        # All-one-cluster: still just a single valid cluster.
        labels_all_one = np.zeros(4)
        assert math.isnan(mathops2d.silhouette_score(x, y, labels_all_one))

    def test_nearest_centroid_reproduces_kmeans_clusters_own_training_assignment(self):
        x, y, _truth = _blobs([(0.0, 0.0), (10.0, 0.0), (5.0, 10.0)], seed=21)
        labels, cx, cy = mathops2d.kmeans_cluster(x, y, 3, seed=21)
        recomputed = mathops2d.nearest_centroid(x, y, cx, cy)
        np.testing.assert_array_equal(recomputed, labels)
