"""Test the N-dimensional operations in glplot.gui.mathopsnd.

Every function under test takes numpy arrays and returns new numpy arrays, so no OpenGL
context, no window, and no live engine scene is required: these tests run fully headless,
mirroring tests/test_gui_mathops2d.py's conventions (closed-form/synthetic references,
an explicit "unavailable" test for the one hard dependency, UMAP).
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

from glplot.gui import mathopsnd


class TestScaleColumns:
    """scale_columns(): per-column zscore/minmax preprocessing."""

    def test_zscore_gives_zero_mean_unit_std_per_column(self):
        rng = np.random.default_rng(0)
        m = np.column_stack([rng.normal(5.0, 2.0, 300), rng.normal(-3.0, 10.0, 300)])
        out = mathopsnd.scale_columns(m, mode="zscore")
        assert np.allclose(out.mean(axis=0), 0.0, atol=1e-9)
        assert np.allclose(out.std(axis=0), 1.0, atol=1e-9)

    def test_minmax_maps_each_column_onto_zero_one(self):
        m = np.array([[1.0, 100.0], [2.0, 200.0], [3.0, 300.0]])
        out = mathopsnd.scale_columns(m, mode="minmax")
        assert np.allclose(out.min(axis=0), 0.0)
        assert np.allclose(out.max(axis=0), 1.0)

    def test_none_passes_through_unchanged(self):
        m = np.array([[1.0, 2.0], [3.0, 4.0]])
        out = mathopsnd.scale_columns(m, mode="none")
        assert np.array_equal(out, m)
        assert out is not m  # a copy, not the same array

    def test_constant_column_becomes_zero_not_nan(self):
        m = np.array([[5.0, 1.0], [5.0, 2.0], [5.0, 3.0]])
        z = mathopsnd.scale_columns(m, mode="zscore")
        assert np.all(z[:, 0] == 0.0)
        assert np.all(np.isfinite(z))
        mm = mathopsnd.scale_columns(m, mode="minmax")
        assert np.all(mm[:, 0] == 0.0)

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="unknown scale mode"):
            mathopsnd.scale_columns(np.zeros((3, 2)), mode="bogus")


class TestPca:
    """pca(): SVD-based decomposition, verified against sklearn.decomposition.PCA."""

    def test_matches_sklearn_explained_variance_and_scores(self):
        sklearn_decomp = pytest.importorskip("sklearn.decomposition")
        rng = np.random.default_rng(1)
        x = rng.normal(0.0, 1.0, (200, 4))
        x[:, 1] = x[:, 0] * 2.0 + rng.normal(0.0, 0.1, 200)
        x[:, 2] = x[:, 0] * -1.0 + rng.normal(0.0, 0.5, 200)
        cols = [x[:, i] for i in range(4)]

        res = mathopsnd.pca(cols, n_components=2, scale="zscore")
        scaled = (x - x.mean(axis=0)) / x.std(axis=0)
        skl = sklearn_decomp.PCA(n_components=2).fit(scaled)

        assert res["explained_variance_ratio"] == pytest.approx(
            skl.explained_variance_ratio_, abs=1e-6
        )
        skl_scores = skl.transform(scaled)
        for k in range(2):
            same = np.allclose(res["scores"][:, k], skl_scores[:, k], atol=1e-6)
            flipped = np.allclose(res["scores"][:, k], -skl_scores[:, k], atol=1e-6)
            # PCA component sign is arbitrary between implementations (an eigenvector
            # and its negation are equally valid) -- either orientation is correct.
            assert same or flipped, f"component {k} does not match sklearn up to sign"

    def test_first_component_captures_most_variance_for_a_correlated_pair(self):
        """Two near-collinear columns: nearly all variance must land in component 1."""
        rng = np.random.default_rng(2)
        a = rng.normal(0.0, 1.0, 300)
        b = a + rng.normal(0.0, 0.01, 300)
        res = mathopsnd.pca([a, b], n_components=2)
        assert res["explained_variance_ratio"][0] > 0.99

    def test_explained_variance_ratios_sum_to_at_most_one(self):
        rng = np.random.default_rng(3)
        cols = [rng.normal(0.0, 1.0, 100) for _ in range(5)]
        res = mathopsnd.pca(cols, n_components=3)
        assert 0.0 <= float(np.sum(res["explained_variance_ratio"])) <= 1.0 + 1e-9

    def test_scores_shape_and_component_shape(self):
        rng = np.random.default_rng(4)
        cols = [rng.normal(0.0, 1.0, 50) for _ in range(3)]
        res = mathopsnd.pca(cols, n_components=2)
        assert res["scores"].shape == (50, 2)
        assert res["components"].shape == (2, 3)
        assert res["n_samples"] == 50
        assert res["n_features"] == 3

    def test_non_finite_rows_are_dropped(self):
        a = np.array([1.0, 2.0, np.nan, 4.0, 5.0, np.inf])
        b = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        res = mathopsnd.pca([a, b], n_components=1)
        assert res["n_samples"] == 4

    def test_scale_none_differs_from_zscore_on_mismatched_units(self):
        """Without scaling, a column with 100x the spread dominates the first component."""
        rng = np.random.default_rng(5)
        small = rng.normal(0.0, 1.0, 300)
        big = rng.normal(0.0, 100.0, 300)  # unrelated, but huge spread
        res_none = mathopsnd.pca([small, big], n_components=1, scale="none")
        res_z = mathopsnd.pca([small, big], n_components=1, scale="zscore")
        # Unscaled: component 1 is almost pure "big" (component vector ~ [0, 1]).
        assert abs(res_none["components"][0, 1]) > 0.99
        # Both columns have equal variance after zscore, so it need not be so extreme.
        assert abs(res_z["components"][0, 1]) < 0.99

    def test_needs_at_least_two_columns(self):
        with pytest.raises(ValueError, match="at least 2 columns"):
            mathopsnd.pca([[1.0, 2.0, 3.0]], n_components=1)

    def test_mismatched_column_lengths_raise(self):
        with pytest.raises(ValueError, match="same length"):
            mathopsnd.pca([[1.0, 2.0, 3.0], [1.0, 2.0]], n_components=1)

    def test_too_few_finite_rows_raises(self):
        with pytest.raises(ValueError, match="at least 2 finite rows"):
            mathopsnd.pca([[1.0, np.nan], [2.0, np.nan]], n_components=1)

    def test_n_components_out_of_range_raises(self):
        cols = [np.arange(10.0), np.arange(10.0) * 2.0, np.arange(10.0) * 3.0]
        with pytest.raises(ValueError, match="n_components"):
            mathopsnd.pca(cols, n_components=0)
        with pytest.raises(ValueError, match="n_components"):
            mathopsnd.pca(cols, n_components=4)  # > min(n_samples, n_features)=3

    def test_unknown_scale_mode_raises(self):
        cols = [np.arange(10.0), np.arange(10.0) * 2.0]
        with pytest.raises(ValueError, match="unknown scale mode"):
            mathopsnd.pca(cols, n_components=1, scale="bogus")


class TestUmapEmbed:
    """umap_embed(): thin wrapper over umap-learn, plus the soft-dependency contract."""

    def test_embedding_shape(self):
        pytest.importorskip("umap")
        rng = np.random.default_rng(10)
        cols = [rng.normal(0.0, 1.0, 60) for _ in range(3)]
        res = mathopsnd.umap_embed(cols, n_components=2, n_neighbors=10, seed=0)
        assert res["embedding"].shape == (60, 2)
        assert res["n_samples"] == 60
        assert res["n_features"] == 3

    def test_reproducible_with_the_same_seed(self):
        pytest.importorskip("umap")
        rng = np.random.default_rng(11)
        cols = [rng.normal(0.0, 1.0, 60) for _ in range(3)]
        r1 = mathopsnd.umap_embed(cols, n_components=2, seed=42)
        r2 = mathopsnd.umap_embed(cols, n_components=2, seed=42)
        assert np.allclose(r1["embedding"], r2["embedding"])

    def test_well_separated_blobs_stay_separated(self):
        """A coarse sanity check that the embedding preserves cluster structure, not a
        strict numerical reference (UMAP has no closed form to compare against)."""
        pytest.importorskip("umap")
        rng = np.random.default_rng(12)
        a = rng.normal(0.0, 0.2, (40, 4))
        b = rng.normal(20.0, 0.2, (40, 4))
        cols = [np.concatenate([a[:, i], b[:, i]]) for i in range(4)]
        res = mathopsnd.umap_embed(cols, n_components=2, n_neighbors=10, seed=0)
        emb = res["embedding"]
        centroid_a = emb[:40].mean(axis=0)
        centroid_b = emb[40:].mean(axis=0)
        within_a = np.mean(np.linalg.norm(emb[:40] - centroid_a, axis=1))
        between = np.linalg.norm(centroid_a - centroid_b)
        assert between > within_a * 3.0

    def test_n_neighbors_is_clamped_to_the_sample_count(self):
        """UMAP itself requires n_neighbors < n_samples; a small dataset must not crash."""
        pytest.importorskip("umap")
        rng = np.random.default_rng(13)
        cols = [rng.normal(0.0, 1.0, 5) for _ in range(3)]
        res = mathopsnd.umap_embed(cols, n_components=2, n_neighbors=50, seed=0)
        assert res["n_neighbors_used"] <= 4
        assert res["embedding"].shape == (5, 2)

    def test_needs_at_least_two_columns(self):
        pytest.importorskip("umap")
        with pytest.raises(ValueError, match="at least 2 columns"):
            mathopsnd.umap_embed([[1.0, 2.0, 3.0]])

    def test_too_few_finite_rows_raises(self):
        pytest.importorskip("umap")
        with pytest.raises(ValueError, match="at least 3 finite rows"):
            mathopsnd.umap_embed([[1.0, 2.0], [3.0, 4.0]])

    def test_unavailable_umap_raises_umap_unavailable_error(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "umap", None)
        assert mathopsnd.umap_available() is False
        rng = np.random.default_rng(14)
        cols = [rng.normal(0.0, 1.0, 20) for _ in range(3)]
        with pytest.raises(mathopsnd.UmapUnavailableError, match="umap-learn"):
            mathopsnd.umap_embed(cols)
        assert issubclass(mathopsnd.UmapUnavailableError, ValueError)

    def test_umap_works_again_once_importable(self):
        """The monkeypatched module above must not leak into the rest of the suite."""
        pytest.importorskip("umap")
        assert mathopsnd.umap_available() is True
        rng = np.random.default_rng(15)
        cols = [rng.normal(0.0, 1.0, 20) for _ in range(3)]
        res = mathopsnd.umap_embed(cols, n_neighbors=5, seed=0)
        assert res["embedding"].shape == (20, 2)


class TestPcaOutOfSample:
    """Phase 1: pca()'s new mean_/scale_stats_ keys, and pca_transform() -- the
    out-of-sample half a held-out val/test split needs."""

    def test_pca_return_dict_includes_mean_and_scale_stats(self):
        rng = np.random.default_rng(30)
        cols = [rng.normal(0.0, 1.0, 100) for _ in range(3)]
        res = mathopsnd.pca(cols, n_components=2, scale="zscore")
        assert res["mean_"].shape == (3,)
        assert set(res["scale_stats_"].keys()) == {"mean", "std"}
        assert res["scale_stats_"]["mean"].shape == (3,)

    def test_pca_transform_matches_pcas_own_scores_on_the_same_call(self):
        rng = np.random.default_rng(31)
        cols = [rng.normal(0.0, 1.0, 150) for _ in range(4)]
        cols[1] = cols[0] * 2.0 + rng.normal(0.0, 0.05, 150)
        res = mathopsnd.pca(cols, n_components=3, scale="zscore")
        scores = mathopsnd.pca_transform(
            cols,
            mean_=res["mean_"],
            components_=res["components"],
            scale_stats=res["scale_stats_"],
            scale="zscore",
        )
        assert np.allclose(scores, res["scores"], atol=1e-8)

    def test_scale_columns_stats_and_apply_scale_round_trip_scale_columns(self):
        rng = np.random.default_rng(32)
        m = np.column_stack([rng.normal(5.0, 2.0, 80), rng.normal(-1.0, 6.0, 80)])
        for mode in mathopsnd.SCALE_MODES:
            expected = mathopsnd.scale_columns(m, mode=mode)
            stats = mathopsnd.scale_columns_stats(m, mode=mode)
            got = mathopsnd.apply_scale(m, stats, mode=mode)
            assert np.array_equal(got, expected)

    def test_pca_transform_uses_the_fitted_stats_not_a_fresh_refit(self):
        rng = np.random.default_rng(33)
        # Fit on data centered around 0.
        fit_cols = [rng.normal(0.0, 1.0, 200) for _ in range(3)]
        res = mathopsnd.pca(fit_cols, n_components=2, scale="zscore")

        # Transform data with a WILDLY different mean/scale -- if pca_transform re-fit
        # its own stats from this new data instead of using the passed-in ones, the
        # result would be centered near 0 (its own mean subtracted). Using the FITTED
        # stats instead leaves the huge offset largely intact after centering.
        far_cols = [np.full(10, 500.0), np.full(10, 500.0), np.full(10, 500.0)]
        scores = mathopsnd.pca_transform(
            far_cols,
            mean_=res["mean_"],
            components_=res["components"],
            scale_stats=res["scale_stats_"],
            scale="zscore",
        )
        assert np.all(np.abs(scores) > 50.0)


class TestUmapOutOfSample:
    """Phase 1: umap_embed()'s new reducer/scale_stats_ keys -- the out-of-sample half
    a held-out val/test split or a later phase's transfer needs."""

    def test_umap_embed_returns_a_reducer_with_a_working_transform(self):
        pytest.importorskip("umap")
        rng = np.random.default_rng(34)
        cols = [rng.normal(0.0, 1.0, 60) for _ in range(3)]
        res = mathopsnd.umap_embed(cols, n_components=2, n_neighbors=10, seed=0)
        assert hasattr(res["reducer"], "transform")
        assert set(res["scale_stats_"].keys()) == {"mean", "std"}

    def test_reducer_transform_on_held_out_rows_has_the_right_shape(self):
        pytest.importorskip("umap")
        rng = np.random.default_rng(35)
        cols = [rng.normal(0.0, 1.0, 60) for _ in range(3)]
        res = mathopsnd.umap_embed(cols, n_components=2, n_neighbors=10, seed=0)

        holdout = np.column_stack([rng.normal(0.0, 1.0, 12) for _ in range(3)])
        scaled_holdout = mathopsnd.apply_scale(holdout, res["scale_stats_"], mode="zscore")
        projected = res["reducer"].transform(scaled_holdout)
        assert projected.shape == (12, 2)
