"""Test the heuristic "what to try next" recommender in glplot.gui.mathadvise.

Every function here is pure numpy plus glplot.gui.mathops, so this runs fully headless: no
OpenGL context, no window, no live engine scene. Each test builds a signal with one
unambiguous property (noisy, periodic, multi-peaked, trending, clean/flat) and checks that
the corresponding tab surfaces near the top of the ranking -- not that the heuristics are
"correct" in any formal sense, since they are explicitly approximate by design.
"""

from __future__ import annotations

import numpy as np
import pytest

from glplot.gui import mathadvise


def _top_keys(x, y, n=3):
    return [rec.tab_key for rec in mathadvise.recommend(x, y)[:n]]


class TestRecommendReturnShape:
    def test_returns_recommendation_namedtuples(self):
        rng = np.random.default_rng(0)
        x = np.linspace(0.0, 10.0, 300)
        y = np.sin(x) + rng.normal(0.0, 0.5, x.size)
        recs = mathadvise.recommend(x, y)
        assert recs
        for rec in recs:
            assert isinstance(rec.tab_key, str)
            assert isinstance(rec.title, str)
            assert isinstance(rec.reason, str) and rec.reason
            assert isinstance(rec.score, float)
            assert 0.0 <= rec.score <= 1.0

    def test_sorted_by_score_descending(self):
        rng = np.random.default_rng(1)
        x = np.linspace(0.0, 10.0, 300)
        y = np.sin(x) + rng.normal(0.0, 0.5, x.size)
        recs = mathadvise.recommend(x, y)
        scores = [rec.score for rec in recs]
        assert scores == sorted(scores, reverse=True)

    def test_respects_max_results(self):
        rng = np.random.default_rng(2)
        x = np.linspace(0.0, 10.0, 300)
        y = np.sin(x) + rng.normal(0.0, 0.5, x.size)
        assert len(mathadvise.recommend(x, y, max_results=2)) <= 2

    def test_too_few_points_is_empty_not_raising(self):
        assert mathadvise.recommend([1.0, 2.0], [1.0, 2.0]) == []
        assert mathadvise.recommend([], []) == []

    def test_nan_and_inf_are_ignored_not_fatal(self):
        x = np.linspace(0.0, 10.0, 100)
        y = np.sin(x)
        y[5] = np.nan
        y[10] = np.inf
        # Must not raise; a corrupted sample is simply excluded from every heuristic.
        mathadvise.recommend(x, y)


class TestNoiseHeuristic:
    def test_heavily_noisy_signal_suggests_smooth_or_filter(self):
        rng = np.random.default_rng(3)
        x = np.linspace(0.0, 10.0, 800)
        y = np.sin(x) + rng.normal(0.0, 0.8, x.size)  # noise dwarfs the signal's own swing
        top = _top_keys(x, y)
        assert "smooth" in top or "filter" in top

    def test_a_cleanly_sampled_fast_oscillation_is_not_mistaken_for_noise(self):
        """Regression: a window-based estimator flagged a clean, fast sine as noisy.

        Ten periods sampled at 200 points/period have essentially zero curvature between
        adjacent samples; the second-difference noise estimator must see that, where a
        smoothing-window residual (tried first) did not.
        """
        x = np.linspace(0.0, 20.0 * np.pi, 2000)
        y = np.sin(x)
        top = _top_keys(x, y, n=len(mathadvise.recommend(x, y)))
        assert "smooth" not in top
        assert "filter" not in top


class TestPeriodicityHeuristic:
    def test_clean_periodic_signal_suggests_fft_or_autocorr(self):
        x = np.linspace(0.0, 20.0 * np.pi, 2000)
        y = np.sin(x)
        top = _top_keys(x, y)
        assert "fft" in top or "autocorr" in top

    def test_pure_noise_does_not_claim_periodicity(self):
        rng = np.random.default_rng(4)
        x = np.linspace(0.0, 10.0, 1000)
        y = rng.normal(0.0, 1.0, x.size)
        recs = {rec.tab_key: rec.score for rec in mathadvise.recommend(x, y, max_results=10)}
        # White noise has no reliable autocorrelation echo above the 0.3 threshold.
        assert "fft" not in recs and "autocorr" not in recs


class TestPeaksHeuristic:
    def test_a_few_separated_peaks_suggest_peaks_and_fit(self):
        x = np.linspace(0.0, 100.0, 1000)
        y = (
            np.exp(-((x - 20.0) ** 2) / 10.0)
            + np.exp(-((x - 60.0) ** 2) / 8.0)
            + np.exp(-((x - 80.0) ** 2) / 5.0)
        )
        top = _top_keys(x, y, n=len(mathadvise.recommend(x, y)))
        assert "peaks" in top
        assert "fit" in top

    def test_a_single_peak_does_not_trigger_the_peaks_tab(self):
        """One peak is not "peaks" (plural) -- the heuristic requires at least two."""
        x = np.linspace(-5.0, 5.0, 500)
        y = np.exp(-(x**2))
        recs = {rec.tab_key for rec in mathadvise.recommend(x, y, max_results=10)}
        assert "peaks" not in recs


class TestTrendHeuristic:
    def test_strong_linear_trend_suggests_baseline(self):
        rng = np.random.default_rng(5)
        x = np.linspace(0.0, 10.0, 300)
        y = 3.0 * x + 2.0 + rng.normal(0.0, 0.05, x.size)
        top = _top_keys(x, y, n=1)
        assert top == ["detrend"]

    def test_a_flat_signal_around_a_constant_does_not_suggest_baseline(self):
        rng = np.random.default_rng(6)
        x = np.linspace(0.0, 10.0, 300)
        y = 5.0 + rng.normal(0.0, 0.01, x.size)
        recs = {rec.tab_key for rec in mathadvise.recommend(x, y, max_results=10)}
        assert "detrend" not in recs


class TestUnsortedInput:
    def test_unsorted_x_gives_the_same_recommendations_as_sorted(self):
        """The recommender must not inherit the "assumes a time series" bug it was
        written to route users away from -- shuffling x must not change the verdict.
        """
        rng = np.random.default_rng(7)
        x = np.linspace(0.0, 10.0, 400)
        y = np.sin(x) + rng.normal(0.0, 0.5, x.size)
        sorted_recs = [(r.tab_key, round(r.score, 6)) for r in mathadvise.recommend(x, y)]

        order = rng.permutation(x.size)
        shuffled_recs = [
            (r.tab_key, round(r.score, 6)) for r in mathadvise.recommend(x[order], y[order])
        ]
        assert shuffled_recs == sorted_recs


class TestEnvelopeHeuristic:
    def test_amplitude_modulated_signal_suggests_envelope(self):
        t = np.linspace(0.0, 1.0, 2000, endpoint=False)
        mod = 1.0 + 0.8 * np.sin(2.0 * np.pi * 3.0 * t)
        y = np.sin(2.0 * np.pi * 50.0 * t) * mod
        recs = {rec.tab_key for rec in mathadvise.recommend(t, y, max_results=10)}
        assert "envelope" in recs

    def test_steady_unmodulated_oscillation_does_not_suggest_envelope(self):
        t = np.linspace(0.0, 1.0, 2000, endpoint=False)
        y = 2.0 * np.sin(2.0 * np.pi * 20.0 * t)
        recs = {rec.tab_key for rec in mathadvise.recommend(t, y, max_results=10)}
        assert "envelope" not in recs

    def test_short_signal_is_not_flagged(self):
        """Below the length floor, envelope() is not even attempted."""
        recs = {rec.tab_key for rec in mathadvise.recommend([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0])}
        assert "envelope" not in recs


class TestRollingHeuristic:
    def test_heteroskedastic_signal_suggests_rolling(self):
        rng = np.random.default_rng(0)
        n = 1000
        noise_scale = np.linspace(0.02, 3.0, n)
        y = np.sin(np.linspace(0.0, 20.0, n)) + rng.normal(0.0, 1.0, n) * noise_scale
        recs = {rec.tab_key for rec in mathadvise.recommend(np.arange(n, dtype=float), y, max_results=10)}
        assert "rolling" in recs

    def test_stationary_noise_does_not_suggest_rolling(self):
        rng = np.random.default_rng(1)
        n = 1000
        y = np.sin(np.linspace(0.0, 20.0, n)) + rng.normal(0.0, 0.3, n)
        recs = {rec.tab_key for rec in mathadvise.recommend(np.arange(n, dtype=float), y, max_results=10)}
        assert "rolling" not in recs


class TestSpatialHeuristic:
    def test_enough_points_suggests_spatial(self):
        rng = np.random.default_rng(0)
        x = rng.normal(0.0, 1.0, 50)
        y = rng.normal(0.0, 1.0, 50)
        recs = {rec.tab_key for rec in mathadvise.recommend(x, y, max_results=10)}
        assert "spatial" in recs

    def test_too_few_points_does_not_suggest_spatial(self):
        recs = {rec.tab_key for rec in mathadvise.recommend([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0])}
        assert "spatial" not in recs


class TestCompareHeuristic:
    def test_with_dataset_suggests_compare(self):
        rng = np.random.default_rng(0)
        x = np.linspace(0.0, 10.0, 100)
        y = rng.normal(0.0, 1.0, 100)
        recs = {
            rec.tab_key for rec in mathadvise.recommend(x, y, has_dataset=True, max_results=10)
        }
        assert "compare" in recs

    def test_without_dataset_never_suggests_compare(self):
        rng = np.random.default_rng(1)
        x = np.linspace(0.0, 10.0, 100)
        y = rng.normal(0.0, 1.0, 100)
        recs = {
            rec.tab_key for rec in mathadvise.recommend(x, y, has_dataset=False, max_results=10)
        }
        assert "compare" not in recs

    def test_too_few_points_does_not_suggest_compare_even_with_a_dataset(self):
        recs = {
            rec.tab_key
            for rec in mathadvise.recommend([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0], has_dataset=True)
        }
        assert "compare" not in recs


class TestMultivariateHeuristic:
    """PCA/UMAP are suggested only when the dataset has 3+ columns to reduce."""

    def test_three_or_more_columns_suggests_pca_and_umap(self):
        rng = np.random.default_rng(0)
        x = np.linspace(0.0, 10.0, 100)
        y = rng.normal(0.0, 1.0, 100)
        recs = {
            rec.tab_key
            for rec in mathadvise.recommend(x, y, has_dataset=True, n_columns=3, max_results=10)
        }
        assert "pca" in recs
        assert "umap" in recs

    def test_exactly_two_columns_does_not_suggest_either(self):
        """Two columns is just the plotted (x, y) itself -- nothing to reduce."""
        rng = np.random.default_rng(1)
        x = np.linspace(0.0, 10.0, 100)
        y = rng.normal(0.0, 1.0, 100)
        recs = {
            rec.tab_key
            for rec in mathadvise.recommend(x, y, has_dataset=True, n_columns=2, max_results=10)
        }
        assert "pca" not in recs
        assert "umap" not in recs

    def test_default_n_columns_is_zero_and_suggests_neither(self):
        rng = np.random.default_rng(2)
        x = np.linspace(0.0, 10.0, 100)
        y = rng.normal(0.0, 1.0, 100)
        recs = {rec.tab_key for rec in mathadvise.recommend(x, y, max_results=10)}
        assert "pca" not in recs
        assert "umap" not in recs

    def test_a_plotted_layer_with_many_columns_reported_never_suggests_either(self):
        """n_columns without has_dataset should not happen from the real call site, but
        the heuristic must still be gated correctly if it ever does."""
        rng = np.random.default_rng(3)
        x = np.linspace(0.0, 10.0, 100)
        y = rng.normal(0.0, 1.0, 100)
        recs = {
            rec.tab_key
            for rec in mathadvise.recommend(x, y, has_dataset=False, n_columns=0, max_results=10)
        }
        assert "pca" not in recs
        assert "umap" not in recs

    def test_too_few_points_does_not_suggest_either_even_with_enough_columns(self):
        recs = {
            rec.tab_key
            for rec in mathadvise.recommend(
                [1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0], has_dataset=True, n_columns=5
            )
        }
        assert "pca" not in recs
        assert "umap" not in recs
