"""Test the pure numerical operations in glplot.gui.mathops.

Every function under test takes numpy arrays and returns new numpy arrays, so no
OpenGL context, no window, and no live engine scene is required: these tests run
fully headless.

Tolerances here are derived from each method's truncation error rather than a
blanket rtol. A relative tolerance degenerates wherever the true value crosses
zero (comparing d/dx x**3 against 3*x**2 at x == 0 is the classic trap: the code
is correct there, but the relative error is unbounded because the exact answer is
0). Absolute bounds tied to the scheme's order are used instead, and are stated
in each test's docstring.
"""

from __future__ import annotations

import sys
import warnings

import numpy as np
import pytest

from glplot.gui import mathops


def _sin_grid(n=2001):
    """Uniform grid over [0, pi] plus sin(x); the closed-form integral is 2.0."""
    x = np.linspace(0.0, np.pi, n)
    return x, np.sin(x)


def _nonuniform_sin_grid(n=4001, seed=0):
    """Random ascending grid pinned to [0, pi] plus sin(x); integral is still 2.0."""
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(0.0, np.pi, n))
    x[0] = 0.0
    x[-1] = np.pi
    return x, np.sin(x)


class TestIntegrateUniform:
    """Test cumulative integrate() against closed forms on uniform grids."""

    def test_trapezoid_sin_hits_two(self):
        """Cumulative trapezoid of sin over [0, pi] reaches the exact value 2.0.

        The composite trapezoid error is (b - a) * h**2 / 12 * max|f''|, here
        pi * h**2 / 12 with h = pi/2000, i.e. about 6.5e-7. 1e-6 bounds it.
        """
        x, y = _sin_grid()
        out = mathops.integrate(x, y, method="trapezoid")
        assert out.shape == x.shape
        assert np.allclose(out[-1], 2.0, rtol=0.0, atol=1e-6)

    def test_simpson_sin_hits_two(self):
        """Cumulative simpson of sin over [0, pi] reaches 2.0 far tighter than trapezoid.

        The quadratic-panel rule is 3rd order globally, leaving ~1e-12 here.
        """
        x, y = _sin_grid()
        out = mathops.integrate(x, y, method="simpson")
        assert np.allclose(out[-1], 2.0, rtol=0.0, atol=1e-9)

    def test_rectangle_sin_hits_two(self):
        """Cumulative rectangle of sin over [0, pi] reaches 2.0.

        Left-endpoint rectangle is only 1st order, but on sin over a full half
        period the leading error terms cancel by symmetry, so 1e-6 still holds.
        """
        x, y = _sin_grid()
        out = mathops.integrate(x, y, method="rectangle")
        assert np.allclose(out[-1], 2.0, rtol=0.0, atol=1e-6)

    def test_cumulative_curve_matches_closed_form_everywhere(self):
        """out[i] equals the closed form 1 - cos(x[i]), not merely the endpoint."""
        x, y = _sin_grid()
        out = mathops.integrate(x, y, method="trapezoid")
        assert np.allclose(out, 1.0 - np.cos(x), rtol=0.0, atol=1e-6)

    def test_first_element_is_zero(self):
        """The integral from x[0] to x[0] is exactly 0.0 with the default initial."""
        x, y = _sin_grid()
        assert mathops.integrate(x, y)[0] == 0.0

    def test_initial_offsets_every_element(self):
        """initial is a constant of integration added to the whole curve."""
        x, y = _sin_grid(101)
        base = mathops.integrate(x, y)
        shifted = mathops.integrate(x, y, initial=3.5)
        assert shifted[0] == 3.5
        assert np.allclose(shifted - base, 3.5)

    def test_rectangle_is_first_order_on_exp(self):
        """Rectangle is genuinely 1st order: halving h halves the error on exp.

        exp over [0, 1] is asymmetric, so the symmetry cancellation that flatters
        rectangle on sin does not apply and the true order shows.
        """
        exact = np.e - 1.0
        errs = []
        for n in (201, 401):
            x = np.linspace(0.0, 1.0, n)
            errs.append(abs(mathops.definite_integral(x, np.exp(x), method="rectangle") - exact))
        assert np.allclose(errs[0] / errs[1], 2.0, rtol=0.02)


class TestIntegrateNonUniform:
    """Test that integrate() weights each panel by its own width."""

    def test_trapezoid_nonuniform_sin(self):
        """Trapezoid on a random non-uniform grid over [0, pi] still reaches 2.0."""
        x, y = _nonuniform_sin_grid()
        assert np.allclose(mathops.definite_integral(x, y), 2.0, rtol=0.0, atol=1e-5)

    def test_simpson_nonuniform_sin(self):
        """Simpson on a random non-uniform grid over [0, pi] still reaches 2.0."""
        x, y = _nonuniform_sin_grid()
        out = mathops.definite_integral(x, y, method="simpson")
        assert np.allclose(out, 2.0, rtol=0.0, atol=1e-6)

    def test_rectangle_nonuniform_sin(self):
        """Rectangle on a random non-uniform grid over [0, pi] reaches 2.0 coarsely."""
        x, y = _nonuniform_sin_grid()
        out = mathops.definite_integral(x, y, method="rectangle")
        assert np.allclose(out, 2.0, rtol=0.0, atol=1e-3)

    def test_trapezoid_exact_for_linear_on_ragged_grid(self):
        """Trapezoid is exact for a linear integrand at any spacing.

        Integral of 2x + 1 over [0, 5] is 25 + 5 == 30.
        """
        x = np.array([0.0, 0.3, 1.1, 2.0, 5.0])
        assert np.allclose(mathops.definite_integral(x, 2.0 * x + 1.0), 30.0)

    def test_simpson_exact_for_quadratic_on_ragged_grid(self):
        """Simpson's quadratic panels are exact for x**2 at any spacing.

        Integral of x**2 over [0, 5] is 125/3.
        """
        x = np.array([0.0, 0.3, 1.1, 2.0, 5.0])
        out = mathops.definite_integral(x, x**2, method="simpson")
        assert np.allclose(out, 125.0 / 3.0)

    def test_unsorted_x_is_sorted(self):
        """Shuffled input yields the same total as the sorted equivalent."""
        x, y = _sin_grid(501)
        order = np.random.default_rng(4).permutation(x.size)
        assert np.allclose(
            mathops.definite_integral(x[order], y[order]),
            mathops.definite_integral(x, y),
        )

    def test_simpson_degrades_to_trapezoid_below_three_samples(self):
        """With only 2 samples no parabola exists, so simpson equals trapezoid."""
        x = np.array([0.0, 2.0])
        y = np.array([1.0, 3.0])
        assert np.allclose(
            mathops.integrate(x, y, method="simpson"),
            mathops.integrate(x, y, method="trapezoid"),
        )


class TestDefiniteIntegral:
    """Test definite_integral() and its agreement with integrate()."""

    def test_equals_last_element_of_cumulative(self):
        """definite_integral is exactly integrate(...)[-1], so plots cannot disagree."""
        x, y = _sin_grid(301)
        for method in ("trapezoid", "simpson", "rectangle"):
            total = mathops.definite_integral(x, y, method=method)
            assert total == mathops.integrate(x, y, method=method, initial=0.0)[-1]

    def test_returns_python_float(self):
        """The total is a plain float, not a 0-d array."""
        x, y = _sin_grid(51)
        assert isinstance(mathops.definite_integral(x, y), float)

    def test_single_sample_is_zero(self):
        """A single sample spans zero width, so the integral is 0.0."""
        assert mathops.definite_integral([1.0], [5.0]) == 0.0

    def test_cos_over_full_period_is_zero(self):
        """Integral of cos over [0, 2pi] is 0.0; compared with an absolute bound."""
        x = np.linspace(0.0, 2.0 * np.pi, 2001)
        out = mathops.definite_integral(x, np.cos(x), method="simpson")
        assert np.allclose(out, 0.0, rtol=0.0, atol=1e-9)


class TestIntegrateErrors:
    """Test integrate()/definite_integral() input validation."""

    def test_empty_raises(self):
        """Empty input is rejected."""
        with pytest.raises(ValueError, match="empty"):
            mathops.integrate(np.array([]), np.array([]))

    def test_mismatched_length_raises(self):
        """x and y of different lengths are rejected."""
        with pytest.raises(ValueError, match="same length"):
            mathops.integrate(np.arange(5.0), np.arange(4.0))

    def test_definite_integral_empty_raises(self):
        """definite_integral inherits the empty-input rejection."""
        with pytest.raises(ValueError, match="empty"):
            mathops.definite_integral(np.array([]), np.array([]))

    def test_definite_integral_mismatched_raises(self):
        """definite_integral inherits the length check."""
        with pytest.raises(ValueError, match="same length"):
            mathops.definite_integral(np.arange(3.0), np.arange(7.0))

    def test_unknown_method_raises(self):
        """An unknown method name is rejected."""
        with pytest.raises(ValueError, match="unknown integration method"):
            mathops.integrate(np.arange(4.0), np.arange(4.0), method="romberg")

    def test_two_dimensional_raises(self):
        """Genuinely 2-D data is rejected rather than silently flattened."""
        with pytest.raises(ValueError, match="1-D"):
            mathops.integrate(np.zeros((4, 2)), np.zeros((4, 2)))

    def test_simpson_duplicate_x_raises(self):
        """A parabola through coincident abscissae is undefined."""
        x = np.array([0.0, 1.0, 1.0, 2.0])
        with pytest.raises(ValueError, match="strictly increasing"):
            mathops.integrate(x, np.ones(4), method="simpson")

    def test_trapezoid_tolerates_duplicate_x(self):
        """Trapezoid needs no strict increase: a zero-width panel adds nothing."""
        x = np.array([0.0, 1.0, 1.0, 2.0])
        assert np.allclose(mathops.definite_integral(x, np.ones(4)), 2.0)


class TestDerivative:
    """Test derivative() against closed forms with h**2-justified tolerances."""

    def test_central_first_order_cubic_error_is_h_squared(self):
        """d/dx x**3 by central differences has interior error exactly h**2.

        ((x+h)**3 - (x-h)**3) / (2h) == 3x**2 + h**2, so the deviation from the
        closed form 3x**2 is h**2 at every interior point regardless of x -- an
        absolute bound. An rtol here would fail spuriously at x == 0, where the
        true value is 0 but the (correct) code returns h**2.
        """
        x = np.linspace(-1.0, 1.0, 201)
        h = float(x[1] - x[0])
        d = mathops.derivative(x, x**3, method="central")
        assert d.shape == x.shape
        err = np.abs(d[1:-1] - 3.0 * x[1:-1] ** 2)
        assert np.allclose(err, h**2, rtol=0.0, atol=1e-12)
        assert float(np.max(err)) <= h**2 + 1e-12

    def test_central_derivative_at_zero_crossing_is_not_relatively_correct(self):
        """At x == 0 the exact 3x**2 is 0 while the scheme returns h**2, as bounded.

        This pins the behaviour that makes a blanket rtol wrong: the absolute
        error is fine, the relative error is infinite, and the code is correct.
        """
        x = np.linspace(-1.0, 1.0, 201)
        h = float(x[1] - x[0])
        d = mathops.derivative(x, x**3, method="central")
        mid = int(np.argmin(np.abs(x)))
        assert np.allclose(x[mid], 0.0)
        assert np.allclose(d[mid], h**2, rtol=0.0, atol=1e-12)

    def test_central_is_exact_for_a_linear_signal(self):
        """np.gradient reproduces a constant slope exactly, edges included."""
        x = np.linspace(0.0, 4.0, 33)
        assert np.allclose(mathops.derivative(x, 3.0 * x - 1.0), 3.0)

    def test_central_handles_nonuniform_x(self):
        """Central differencing on a ragged grid still recovers d/dx x**2 == 2x."""
        rng = np.random.default_rng(11)
        x = np.sort(rng.uniform(0.0, 3.0, 400))
        d = mathops.derivative(x, x**2, method="central")
        assert np.allclose(d, 2.0 * x, rtol=0.0, atol=1e-9)

    def test_second_order_central_recovers_six_x(self):
        """order=2 on x**3 recovers 6x by applying the scheme twice."""
        x = np.linspace(-1.0, 1.0, 201)
        d2 = mathops.derivative(x, x**3, order=2, method="central")
        assert d2.shape == x.shape
        assert np.allclose(d2[2:-2], 6.0 * x[2:-2], rtol=0.0, atol=1e-9)

    def test_second_order_of_quadratic_is_constant_two(self):
        """order=2 on x**2 is the constant 2 in the interior."""
        x = np.linspace(-2.0, 2.0, 101)
        d2 = mathops.derivative(x, x**2, order=2)
        assert np.allclose(d2[2:-2], 2.0, rtol=0.0, atol=1e-9)

    def test_forward_error_is_exactly_h_on_a_quadratic(self):
        """Forward differencing x**2 gives 2x + h: 1st order, error exactly h.

        ((x+h)**2 - x**2) / h == 2x + h. Again an absolute bound, since 2x is 0
        at the origin.
        """
        x = np.linspace(0.0, 2.0, 201)
        h = float(x[1] - x[0])
        d = mathops.derivative(x, x**2, method="forward")
        assert d.shape == x.shape
        assert np.allclose(d[:-1], 2.0 * x[:-1] + h, rtol=0.0, atol=1e-9)

    def test_forward_repeats_last_difference_to_preserve_length(self):
        """The final forward element repeats the last valid difference."""
        x = np.linspace(0.0, 2.0, 51)
        d = mathops.derivative(x, x**2, method="forward")
        assert d[-1] == d[-2]

    def test_forward_is_exact_for_a_linear_signal(self):
        """A one-sided difference of a straight line is the exact slope."""
        x = np.linspace(0.0, 5.0, 41)
        assert np.allclose(mathops.derivative(x, -2.0 * x + 7.0, method="forward"), -2.0)

    def test_savgol_is_near_exact_for_a_cubic(self):
        """savgol with polyorder=3 differentiates x**3 essentially exactly.

        A cubic lies in the fitted polynomial space, so only float round-off
        (~1e-12) remains -- edges included, thanks to mode='interp'.
        """
        x = np.linspace(-1.0, 1.0, 201)
        d = mathops.derivative(x, x**3, method="savgol", window=11, polyorder=3)
        assert d.shape == x.shape
        assert np.allclose(d, 3.0 * x**2, rtol=0.0, atol=1e-9)

    def test_savgol_second_order_recovers_six_x(self):
        """savgol order=2 on x**3 recovers 6x, edges included."""
        x = np.linspace(-1.0, 1.0, 201)
        d2 = mathops.derivative(x, x**3, order=2, method="savgol", window=11, polyorder=3)
        assert np.allclose(d2, 6.0 * x, rtol=0.0, atol=1e-8)

    def test_savgol_warns_and_falls_back_on_nonuniform_x(self):
        """A non-uniform grid makes savgol inapplicable; it warns and uses central."""
        rng = np.random.default_rng(2)
        x = np.sort(rng.uniform(0.0, 3.0, 200))
        with pytest.warns(RuntimeWarning, match="uniform x grid"):
            d = mathops.derivative(x, x**2, method="savgol", window=11)
        assert np.allclose(d, mathops.derivative(x, x**2, method="central"))

    def test_savgol_falls_back_when_window_too_small(self):
        """A degenerate window cannot host a savgol fit; it warns and uses central."""
        x = np.linspace(0.0, 1.0, 21)
        with pytest.warns(RuntimeWarning):
            d = mathops.derivative(x, x**2, method="savgol", window=2)
        assert np.allclose(d, mathops.derivative(x, x**2, method="central"))

    def test_unsorted_x_is_sorted(self):
        """Shuffled input gives the ascending-x result, matching integrate()."""
        x = np.linspace(0.0, 2.0, 101)
        order = np.random.default_rng(6).permutation(x.size)
        d = mathops.derivative(x[order], (x**2)[order])
        assert np.allclose(d, mathops.derivative(x, x**2))

    def test_single_sample_returns_zero(self):
        """A lone sample has no neighbour, so the slope is reported as 0.0."""
        assert np.allclose(mathops.derivative([1.0], [5.0]), np.zeros(1))


class TestDerivativeErrors:
    """Test derivative() input validation."""

    def test_empty_raises(self):
        """Empty input is rejected."""
        with pytest.raises(ValueError, match="empty"):
            mathops.derivative(np.array([]), np.array([]))

    def test_mismatched_length_raises(self):
        """x and y of different lengths are rejected."""
        with pytest.raises(ValueError, match="same length"):
            mathops.derivative(np.arange(6.0), np.arange(3.0))

    def test_unknown_method_raises(self):
        """An unknown method name is rejected."""
        with pytest.raises(ValueError, match="unknown derivative method"):
            mathops.derivative(np.arange(4.0), np.arange(4.0), method="spectral")

    def test_zero_order_raises(self):
        """order must be at least 1."""
        with pytest.raises(ValueError, match="order must be >= 1"):
            mathops.derivative(np.arange(4.0), np.arange(4.0), order=0)

    def test_negative_order_raises(self):
        """A negative order is rejected."""
        with pytest.raises(ValueError, match="order must be >= 1"):
            mathops.derivative(np.arange(4.0), np.arange(4.0), order=-2)

    def test_duplicate_x_raises(self):
        """Duplicate abscissae give a zero spacing and an undefined slope."""
        x = np.array([0.0, 1.0, 1.0, 2.0])
        with pytest.raises(ValueError, match="strictly increasing"):
            mathops.derivative(x, np.arange(4.0))


class TestDerivativeScipyFallback:
    """Test that derivative(method='savgol') survives a missing scipy."""

    def test_savgol_falls_back_to_central_without_scipy(self, monkeypatch):
        """With scipy.signal unimportable, savgol warns and returns the central result.

        The fallback must still be *sane*: interior error on x**3 stays within
        the central scheme's h**2 bound.
        """
        monkeypatch.setitem(sys.modules, "scipy.signal", None)
        x = np.linspace(-1.0, 1.0, 201)
        h = float(x[1] - x[0])
        with pytest.warns(RuntimeWarning, match="scipy is unavailable"):
            d = mathops.derivative(x, x**3, method="savgol", window=11)
        assert d.shape == x.shape
        assert np.allclose(d, mathops.derivative(x, x**3, method="central"))
        assert float(np.max(np.abs(d[1:-1] - 3.0 * x[1:-1] ** 2))) <= h**2 + 1e-12


class TestSmoothLength:
    """Test that smooth() always preserves the signal length."""

    @pytest.mark.parametrize("method", ["moving_average", "gaussian", "savgol", "median"])
    def test_length_preserved(self, method):
        """Every method returns exactly as many samples as it was given."""
        y = np.sin(np.linspace(0.0, 10.0, 77))
        assert mathops.smooth(y, method=method, window=11).shape == (77,)

    @pytest.mark.parametrize("method", ["moving_average", "gaussian", "savgol", "median"])
    def test_length_preserved_when_window_exceeds_signal(self, method):
        """An oversized window is clamped, not rejected, and length still holds."""
        y = np.sin(np.linspace(0.0, 3.0, 9))
        assert mathops.smooth(y, method=method, window=999).shape == (9,)

    def test_even_window_is_forced_odd(self):
        """An even window would bias the output by half a sample, so it is decremented."""
        y = np.sin(np.linspace(0.0, 10.0, 51))
        assert np.allclose(
            mathops.smooth(y, method="moving_average", window=12),
            mathops.smooth(y, method="moving_average", window=11),
        )

    def test_window_one_returns_unchanged_copy(self):
        """window=1 is a no-op that still returns a distinct array."""
        y = np.sin(np.linspace(0.0, 10.0, 21))
        out = mathops.smooth(y, method="moving_average", window=1)
        assert np.allclose(out, y)
        assert out is not y


class TestSmoothEdges:
    """Test that smooth() does not let edges decay toward zero (the convolve bug)."""

    @pytest.mark.parametrize("method", ["moving_average", "gaussian", "savgol", "median"])
    def test_constant_signal_stays_flat_at_the_edges(self, method):
        """A constant 5.0 signal comes back as 5.0 everywhere, edges included.

        A plain convolve(y, k / k.sum()) treats out-of-range samples as zeros, so
        y[0] would come back near 5 * 6/11 == 2.7 for an 11-wide boxcar. Dividing
        by the actual window overlap keeps the level exact.
        """
        y = np.full(51, 5.0)
        out = mathops.smooth(y, method=method, window=11)
        assert np.allclose(out, 5.0, rtol=0.0, atol=1e-12)
        assert np.allclose(out[0], 5.0, rtol=0.0, atol=1e-12)
        assert np.allclose(out[-1], 5.0, rtol=0.0, atol=1e-12)

    @pytest.mark.parametrize("method", ["moving_average", "gaussian", "savgol", "median"])
    def test_offset_signal_edges_are_not_pulled_toward_zero(self, method):
        """An offset ripple keeps its level at both ends rather than drooping to 0.

        The signal lives near 100; a zero-padded convolve would drag the first
        and last window//2 points down by tens of units.
        """
        t = np.linspace(0.0, 4.0 * np.pi, 121)
        y = 100.0 + np.sin(t)
        out = mathops.smooth(y, method=method, window=11)
        assert np.all(np.abs(out - 100.0) < 2.0)

    def test_moving_average_edge_equals_the_truncated_window_mean(self):
        """The first output is the mean over the samples that actually exist.

        For an 11-wide boxcar at index 0 only indices 0..5 overlap real data, so
        the closed form is mean(y[:6]) -- not a zero-diluted value.
        """
        y = np.arange(31.0)
        out = mathops.smooth(y, method="moving_average", window=11)
        assert np.allclose(out[0], np.mean(y[:6]))
        assert np.allclose(out[-1], np.mean(y[-6:]))

    def test_moving_average_interior_equals_the_window_mean(self):
        """An interior point is the plain mean of its full window."""
        y = np.arange(31.0) ** 2
        out = mathops.smooth(y, method="moving_average", window=11)
        assert np.allclose(out[15], np.mean(y[10:21]))

    def test_moving_average_interpolates_over_an_isolated_nan(self):
        """A lone nan is smoothed over rather than poisoning a whole window."""
        y = np.full(41, 5.0)
        y[20] = np.nan
        out = mathops.smooth(y, method="moving_average", window=11)
        assert np.all(np.isfinite(out))
        assert np.allclose(out, 5.0, rtol=0.0, atol=1e-12)

    def test_savgol_preserves_a_polynomial_exactly(self):
        """savgol with polyorder=3 leaves a cubic untouched, edges included."""
        x = np.linspace(-1.0, 1.0, 61)
        y = 2.0 * x**3 - x + 0.5
        out = mathops.smooth(y, method="savgol", window=11, polyorder=3)
        assert np.allclose(out, y, rtol=0.0, atol=1e-9)

    def test_median_removes_an_isolated_spike(self):
        """A single outlier is rejected outright by the median filter."""
        y = np.full(31, 5.0)
        y[15] = 500.0
        out = mathops.smooth(y, method="median", window=5)
        assert np.allclose(out, 5.0)

    def test_gaussian_accepts_an_explicit_sigma(self):
        """A tiny sigma concentrates the kernel, leaving the signal near-unchanged."""
        y = np.sin(np.linspace(0.0, 10.0, 51))
        out = mathops.smooth(y, method="gaussian", window=11, sigma=0.05)
        assert np.allclose(out, y, rtol=0.0, atol=1e-9)

    def test_gaussian_smooths_more_with_a_larger_sigma(self):
        """A wider kernel must reduce the sample-to-sample variation further."""
        y = np.sin(np.linspace(0.0, 40.0, 201))
        narrow = mathops.smooth(y, method="gaussian", window=11, sigma=0.5)
        wide = mathops.smooth(y, method="gaussian", window=11, sigma=4.0)
        assert np.std(np.diff(wide)) < np.std(np.diff(narrow))


class TestSmoothErrors:
    """Test smooth() input validation."""

    def test_empty_raises(self):
        """Empty input is rejected."""
        with pytest.raises(ValueError, match="empty"):
            mathops.smooth(np.array([]))

    def test_two_dimensional_raises(self):
        """Genuinely 2-D input is rejected rather than flattened."""
        with pytest.raises(ValueError, match="1-D"):
            mathops.smooth(np.zeros((4, 2)))

    def test_unknown_method_raises(self):
        """An unknown method name is rejected."""
        with pytest.raises(ValueError, match="unknown smoothing method"):
            mathops.smooth(np.ones(10), method="lowess")

    def test_unknown_keyword_raises(self):
        """A keyword that the chosen method does not accept is rejected."""
        with pytest.raises(ValueError, match="unknown keyword"):
            mathops.smooth(np.ones(10), method="moving_average", sigma=2.0)

    def test_sigma_rejected_by_savgol(self):
        """sigma belongs to gaussian, not savgol."""
        with pytest.raises(ValueError, match="unknown keyword"):
            mathops.smooth(np.ones(10), method="savgol", sigma=2.0)

    def test_non_positive_sigma_raises(self):
        """A zero or negative sigma has no valid kernel."""
        with pytest.raises(ValueError, match="sigma must be a positive"):
            mathops.smooth(np.ones(21), method="gaussian", window=11, sigma=0.0)

    def test_non_integer_window_raises(self):
        """A window that cannot be interpreted as an int is rejected."""
        with pytest.raises(ValueError, match="window must be an integer"):
            mathops.smooth(np.ones(10), window="wide")


class TestSmoothEma:
    """Test smooth(method="ema"): exponential moving average, pure numpy."""

    def test_matches_pandas_ewm(self):
        """Verified against pandas.Series.ewm(span=N).mean(), the standard reference."""
        pd = pytest.importorskip("pandas")
        rng = np.random.default_rng(0)
        y = rng.normal(0.0, 1.0, 500)
        out = mathops.smooth(y, method="ema", window=21)
        expected = pd.Series(y).ewm(span=21, adjust=False).mean().to_numpy()
        np.testing.assert_allclose(out, expected, atol=1e-9)

    def test_matches_a_hand_written_recursion(self):
        """Independent of pandas: y[0] = x[0]; y[i] = a*x[i] + (1-a)*y[i-1]."""
        rng = np.random.default_rng(1)
        y = rng.normal(0.0, 1.0, 300)
        alpha = 0.3
        out = mathops.smooth(y, method="ema", alpha=alpha)
        expected = np.empty_like(y)
        expected[0] = y[0]
        for i in range(1, y.size):
            expected[i] = alpha * y[i] + (1.0 - alpha) * expected[i - 1]
        np.testing.assert_allclose(out, expected, atol=1e-12)

    def test_default_alpha_is_two_over_window_plus_one(self):
        window = 9
        out_explicit = mathops.smooth(
            np.random.default_rng(2).normal(0.0, 1.0, 100), method="ema", alpha=2.0 / (window + 1)
        )
        out_default = mathops.smooth(
            np.random.default_rng(2).normal(0.0, 1.0, 100), method="ema", window=window
        )
        np.testing.assert_allclose(out_explicit, out_default)

    def test_first_output_equals_first_input(self):
        y = np.array([5.0, 1.0, 9.0, 3.0, 7.0])
        out = mathops.smooth(y, method="ema", alpha=0.5)
        assert out[0] == y[0]

    def test_alpha_one_returns_the_input_unchanged(self):
        """alpha = 1 means "ignore history entirely" -- output equals input."""
        y = np.array([5.0, 1.0, 9.0, 3.0, 7.0])
        out = mathops.smooth(y, method="ema", alpha=1.0)
        np.testing.assert_allclose(out, y)

    def test_smaller_alpha_smooths_more(self):
        rng = np.random.default_rng(3)
        y = np.sin(np.linspace(0.0, 20.0, 400)) + rng.normal(0.0, 0.3, 400)
        smooth_out = mathops.smooth(y, method="ema", alpha=0.05)
        rough_out = mathops.smooth(y, method="ema", alpha=0.8)
        assert np.std(np.diff(smooth_out)) < np.std(np.diff(rough_out))

    def test_is_causal_not_centered(self):
        """A single impulse must only affect samples FROM that point on, never before."""
        y = np.zeros(50)
        y[25] = 1.0
        out = mathops.smooth(y, method="ema", alpha=0.5)
        assert np.all(out[:25] == 0.0)
        assert out[25] > 0.0

    def test_non_finite_samples_are_interpolated_first(self):
        """A causal recursion has no way to recover from a NaN otherwise -- it must not
        poison every value from that point on."""
        y = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        out = mathops.smooth(y, method="ema", alpha=1.0)
        assert np.all(np.isfinite(out))
        # alpha=1 means output tracks the (nan-filled) input exactly.
        np.testing.assert_allclose(out, [1.0, 2.0, 3.0, 4.0, 5.0])

    def test_alpha_out_of_range_raises(self):
        with pytest.raises(ValueError, match="alpha must be in"):
            mathops.smooth(np.ones(10), method="ema", alpha=0.0)
        with pytest.raises(ValueError, match="alpha must be in"):
            mathops.smooth(np.ones(10), method="ema", alpha=1.5)

    def test_unknown_keyword_rejected(self):
        with pytest.raises(ValueError, match="unknown keyword"):
            mathops.smooth(np.ones(10), method="ema", sigma=1.0)

    def test_length_preserved(self):
        out = mathops.smooth(np.arange(37.0), method="ema", window=5)
        assert out.shape == (37,)

    def test_never_needs_scipy(self, monkeypatch):
        """Pure numpy: must work with both scipy and scipy.signal made unimportable."""
        monkeypatch.setitem(sys.modules, "scipy", None)
        monkeypatch.setitem(sys.modules, "scipy.signal", None)
        y = np.random.default_rng(4).normal(0.0, 1.0, 100)
        out = mathops.smooth(y, method="ema", window=11)
        assert np.all(np.isfinite(out))


class TestSmoothScipyFallback:
    """Test that the scipy-backed smoothers degrade to sane numpy fallbacks."""

    def test_savgol_falls_back_to_moving_average_without_scipy(self, monkeypatch):
        """Without scipy.signal, savgol warns and returns the edge-corrected boxcar."""
        monkeypatch.setitem(sys.modules, "scipy.signal", None)
        y = np.full(51, 5.0)
        with pytest.warns(RuntimeWarning, match="scipy is unavailable"):
            out = mathops.smooth(y, method="savgol", window=11)
        assert out.shape == (51,)
        assert np.allclose(out, 5.0, rtol=0.0, atol=1e-12)

    def test_savgol_fallback_still_smooths(self, monkeypatch):
        """The savgol fallback is a real smoother, not a pass-through."""
        monkeypatch.setitem(sys.modules, "scipy.signal", None)
        rng = np.random.default_rng(9)
        y = np.sin(np.linspace(0.0, 10.0, 201)) + rng.normal(0.0, 0.3, 201)
        with pytest.warns(RuntimeWarning):
            out = mathops.smooth(y, method="savgol", window=11)
        assert np.std(np.diff(out)) < np.std(np.diff(y))

    def test_median_falls_back_to_numpy_without_scipy(self, monkeypatch):
        """Without scipy.ndimage the numpy sliding-window median takes over.

        The fallback is exact, not an approximation, so it warns about nothing
        and must match scipy's mode='nearest' result.
        """
        y = np.sin(np.linspace(0.0, 10.0, 41))
        y[20] = 50.0
        reference = mathops.smooth(y, method="median", window=5)
        monkeypatch.setitem(sys.modules, "scipy.ndimage", None)
        out = mathops.smooth(y, method="median", window=5)
        assert out.shape == (41,)
        assert np.allclose(out, reference)

    def test_median_fallback_holds_constant_edges(self, monkeypatch):
        """The numpy median fallback keeps a constant 5.0 flat at the edges too."""
        monkeypatch.setitem(sys.modules, "scipy.ndimage", None)
        out = mathops.smooth(np.full(51, 5.0), method="median", window=11)
        assert np.allclose(out, 5.0, rtol=0.0, atol=1e-12)


class TestAddNoise:
    """Test add_noise() reproducibility, scaling, and per-kind semantics."""

    @pytest.fixture
    def signal(self):
        """A smooth non-constant test signal."""
        t = np.linspace(0.0, 10.0, 101)
        return t + np.sin(t)

    @pytest.mark.parametrize("kind", ["gaussian", "uniform", "poisson", "salt_pepper"])
    def test_same_seed_is_reproducible(self, signal, kind):
        """Two calls with the same seed produce bit-identical output."""
        a = mathops.add_noise(signal, kind=kind, amplitude=0.5, seed=42)
        b = mathops.add_noise(signal, kind=kind, amplitude=0.5, seed=42)
        assert np.array_equal(a, b)

    @pytest.mark.parametrize("kind", ["gaussian", "uniform", "poisson", "salt_pepper"])
    def test_different_seeds_differ(self, signal, kind):
        """Different seeds produce different noise."""
        a = mathops.add_noise(signal, kind=kind, amplitude=0.5, seed=42)
        c = mathops.add_noise(signal, kind=kind, amplitude=0.5, seed=7)
        assert not np.array_equal(a, c)

    @pytest.mark.parametrize("kind", ["gaussian", "uniform", "poisson", "salt_pepper"])
    def test_length_preserved_and_input_untouched(self, signal, kind):
        """Output is a new array of the same length; the input is not mutated."""
        original = signal.copy()
        out = mathops.add_noise(signal, kind=kind, amplitude=0.5, seed=1)
        assert out.shape == signal.shape
        assert out is not signal
        assert np.array_equal(signal, original)

    def test_gaussian_has_the_requested_standard_deviation(self):
        """Absolute gaussian noise has std == amplitude, independent of the data."""
        y = np.zeros(50000)
        out = mathops.add_noise(y, kind="gaussian", amplitude=0.25, seed=3)
        assert np.allclose(np.std(out), 0.25, rtol=0.02)
        assert np.allclose(np.mean(out), 0.0, rtol=0.0, atol=0.01)

    def test_gaussian_relative_scales_with_data_std(self):
        """relative=True multiplies amplitude by nanstd(y), so it tracks the data.

        Scaling the signal by 10 must scale the injected noise by 10.
        """
        base = np.random.default_rng(1).normal(0.0, 1.0, 20000)
        small = mathops.add_noise(base, kind="gaussian", amplitude=0.1, relative=True, seed=3)
        large = mathops.add_noise(
            base * 10.0, kind="gaussian", amplitude=0.1, relative=True, seed=3
        )
        assert np.allclose(np.std(small - base), 0.1 * np.std(base), rtol=0.05)
        assert np.allclose(np.std(large - base * 10.0), 0.1 * np.std(base * 10.0), rtol=0.05)
        assert np.allclose(np.std(large - base * 10.0), 10.0 * np.std(small - base), rtol=1e-9)

    def test_gaussian_relative_on_a_constant_adds_nothing(self):
        """A constant signal has zero spread, so relative noise correctly adds nothing."""
        y = np.full(20, 5.0)
        out = mathops.add_noise(y, kind="gaussian", amplitude=0.5, relative=True, seed=1)
        assert np.allclose(out, 5.0)

    def test_uniform_stays_within_the_amplitude_band(self):
        """Absolute uniform noise never leaves [-amplitude, +amplitude]."""
        y = np.zeros(20000)
        out = mathops.add_noise(y, kind="uniform", amplitude=0.4, seed=5)
        assert np.all(np.abs(out) <= 0.4)
        assert np.max(np.abs(out)) > 0.39

    def test_uniform_relative_scales_with_the_data_range(self):
        """relative=True scales uniform noise by the nan-safe peak-to-peak range."""
        base = np.linspace(0.0, 2.0, 20000)
        span = float(base.max() - base.min())
        out = mathops.add_noise(base, kind="uniform", amplitude=0.1, relative=True, seed=8)
        assert np.all(np.abs(out - base) <= 0.1 * span + 1e-12)
        assert np.max(np.abs(out - base)) > 0.09 * span

    def test_poisson_zero_amplitude_returns_the_signal(self):
        """amplitude=0 means no shot noise at all."""
        y = np.linspace(1.0, 100.0, 50)
        assert np.allclose(mathops.add_noise(y, kind="poisson", amplitude=0.0, seed=1), y)

    def test_poisson_variance_tracks_the_expected_count(self):
        """True shot noise (amplitude=1) has variance equal to the rate lambda."""
        y = np.full(40000, 25.0)
        out = mathops.add_noise(y, kind="poisson", amplitude=1.0, seed=2)
        assert np.allclose(np.var(out), 25.0, rtol=0.05)
        assert np.allclose(np.mean(out), 25.0, rtol=0.02)

    def test_poisson_ignores_relative(self):
        """poisson is already data-scaled, so relative is documented as ignored."""
        y = np.linspace(1.0, 100.0, 60)
        a = mathops.add_noise(y, kind="poisson", amplitude=0.5, relative=False, seed=4)
        b = mathops.add_noise(y, kind="poisson", amplitude=0.5, relative=True, seed=4)
        assert np.array_equal(a, b)

    def test_salt_pepper_replaces_the_requested_fraction(self):
        """amplitude is the fraction of samples replaced by the signal min/max."""
        y = np.linspace(0.0, 1.0, 100)
        out = mathops.add_noise(y, kind="salt_pepper", amplitude=0.2, seed=5)
        changed = out != y
        assert np.allclose(np.count_nonzero(changed), 20, atol=2)
        replaced = set(np.round(out[changed], 12))
        assert replaced <= {round(float(y.min()), 12), round(float(y.max()), 12)}

    def test_salt_pepper_zero_amplitude_returns_the_signal(self):
        """A zero fraction replaces nothing."""
        y = np.linspace(0.0, 1.0, 50)
        assert np.array_equal(mathops.add_noise(y, kind="salt_pepper", amplitude=0.0, seed=1), y)

    def test_salt_pepper_ignores_relative(self):
        """salt_pepper is fraction-valued, so relative is documented as ignored."""
        y = np.linspace(0.0, 1.0, 60)
        a = mathops.add_noise(y, kind="salt_pepper", amplitude=0.3, relative=False, seed=4)
        b = mathops.add_noise(y, kind="salt_pepper", amplitude=0.3, relative=True, seed=4)
        assert np.array_equal(a, b)

    def test_empty_raises(self):
        """Empty input is rejected."""
        with pytest.raises(ValueError, match="empty"):
            mathops.add_noise(np.array([]))

    def test_unknown_kind_raises(self):
        """An unknown noise kind is rejected."""
        with pytest.raises(ValueError, match="unknown noise kind"):
            mathops.add_noise(np.ones(10), kind="pink")

    def test_negative_amplitude_raises(self):
        """A negative magnitude is meaningless."""
        with pytest.raises(ValueError, match="amplitude must be >= 0"):
            mathops.add_noise(np.ones(10), amplitude=-1.0)

    def test_non_finite_amplitude_raises(self):
        """A nan/inf magnitude is rejected."""
        with pytest.raises(ValueError, match="amplitude must be finite"):
            mathops.add_noise(np.ones(10), amplitude=np.nan)


class TestResample:
    """Test resample() endpoint exactness and both interpolation kinds."""

    @pytest.mark.parametrize("kind", ["linear", "cubic"])
    def test_endpoints_are_exact(self, kind):
        """x_new spans exactly [min(x), max(x)] and hits the original endpoint values."""
        x = np.linspace(0.0, 10.0, 50)
        y = np.sin(x)
        x_new, y_new = mathops.resample(x, y, 17, kind=kind)
        assert x_new[0] == float(x[0])
        assert x_new[-1] == float(x[-1])
        assert np.allclose(y_new[0], y[0], rtol=0.0, atol=1e-12)
        assert np.allclose(y_new[-1], y[-1], rtol=0.0, atol=1e-12)

    @pytest.mark.parametrize("kind", ["linear", "cubic"])
    def test_output_length_and_uniform_spacing(self, kind):
        """Both kinds return exactly n evenly spaced abscissae."""
        x = np.linspace(-3.0, 4.0, 40)
        x_new, y_new = mathops.resample(x, np.cos(x), 25, kind=kind)
        assert x_new.shape == (25,)
        assert y_new.shape == (25,)
        assert np.allclose(np.diff(x_new), 7.0 / 24.0)

    def test_linear_reproduces_a_line_exactly(self):
        """Linear interpolation of a straight line is exact at every new abscissa."""
        x = np.linspace(0.0, 6.0, 13)
        x_new, y_new = mathops.resample(x, 2.0 * x + 1.0, 41, kind="linear")
        assert np.allclose(y_new, 2.0 * x_new + 1.0)

    def test_cubic_reproduces_a_cubic_closely(self):
        """A natural cubic spline through cubic data is near-exact in the interior."""
        x = np.linspace(-1.0, 1.0, 41)
        x_new, y_new = mathops.resample(x, x**3, 101, kind="cubic")
        assert np.allclose(y_new[10:-10], x_new[10:-10] ** 3, rtol=0.0, atol=1e-6)

    def test_cubic_beats_linear_on_a_sinusoid(self):
        """Cubic is the higher-order option and must be measurably more accurate."""
        x = np.linspace(0.0, 2.0 * np.pi, 64)
        y = np.sin(x)
        x_new, y_cubic = mathops.resample(x, y, 201, kind="cubic")
        _, y_linear = mathops.resample(x, y, 201, kind="linear")
        exact = np.sin(x_new)
        assert np.max(np.abs(y_cubic - exact)) < np.max(np.abs(y_linear - exact))

    def test_unsorted_x_is_sorted(self):
        """Shuffled input resamples onto the same ascending grid."""
        x = np.linspace(0.0, 5.0, 30)
        y = np.sin(x)
        order = np.random.default_rng(3).permutation(x.size)
        xs, ys = mathops.resample(x[order], y[order], 21, kind="linear")
        xr, yr = mathops.resample(x, y, 21, kind="linear")
        assert np.allclose(xs, xr)
        assert np.allclose(ys, yr)

    def test_upsample_and_downsample_both_work(self):
        """n may be larger or smaller than the input length."""
        x = np.linspace(0.0, 1.0, 50)
        assert mathops.resample(x, x, 500)[0].shape == (500,)
        assert mathops.resample(x, x, 5)[0].shape == (5,)

    def test_empty_raises(self):
        """Empty input is rejected."""
        with pytest.raises(ValueError, match="empty"):
            mathops.resample(np.array([]), np.array([]), 10)

    def test_mismatched_length_raises(self):
        """x and y of different lengths are rejected."""
        with pytest.raises(ValueError, match="same length"):
            mathops.resample(np.arange(5.0), np.arange(4.0), 10)

    def test_unknown_kind_raises(self):
        """An unknown interpolation kind is rejected."""
        with pytest.raises(ValueError, match="unknown resample kind"):
            mathops.resample(np.arange(5.0), np.arange(5.0), 10, kind="nearest")

    def test_n_below_two_raises(self):
        """Fewer than 2 output samples cannot define a grid."""
        with pytest.raises(ValueError, match="n must be >= 2"):
            mathops.resample(np.arange(5.0), np.arange(5.0), 1)

    def test_single_sample_raises(self):
        """Resampling needs at least 2 input samples."""
        with pytest.raises(ValueError, match="at least 2 samples"):
            mathops.resample([1.0], [2.0], 10)

    def test_zero_width_range_raises(self):
        """A degenerate x range cannot be resampled."""
        with pytest.raises(ValueError, match="zero width"):
            mathops.resample(np.full(5, 2.0), np.arange(5.0), 10)


class TestResampleScipyFallback:
    """Test that resample(kind='cubic') survives a missing scipy."""

    def test_cubic_falls_back_to_linear_without_scipy(self, monkeypatch):
        """Without scipy.interpolate, cubic warns and returns the linear result."""
        monkeypatch.setitem(sys.modules, "scipy.interpolate", None)
        x = np.linspace(0.0, 10.0, 50)
        y = np.sin(x)
        with pytest.warns(RuntimeWarning, match="scipy is unavailable"):
            x_new, y_new = mathops.resample(x, y, 21, kind="cubic")
        assert x_new.shape == (21,)
        assert x_new[0] == 0.0
        assert x_new[-1] == 10.0
        assert np.allclose(y_new, mathops.resample(x, y, 21, kind="linear")[1])

    def test_cubic_fallback_is_still_accurate(self, monkeypatch):
        """The linear fallback is a sane interpolant, not garbage."""
        monkeypatch.setitem(sys.modules, "scipy.interpolate", None)
        x = np.linspace(0.0, 2.0 * np.pi, 200)
        with pytest.warns(RuntimeWarning):
            x_new, y_new = mathops.resample(x, np.sin(x), 101, kind="cubic")
        assert np.allclose(y_new, np.sin(x_new), rtol=0.0, atol=1e-3)

    def test_cubic_falls_back_to_linear_on_duplicate_x(self):
        """A spline needs strictly increasing knots; duplicates warn and use linear."""
        x = np.array([0.0, 1.0, 1.0, 2.0, 3.0])
        y = np.array([0.0, 1.0, 1.0, 4.0, 9.0])
        with pytest.warns(RuntimeWarning, match="strictly increasing"):
            x_new, y_new = mathops.resample(x, y, 9, kind="cubic")
        assert np.allclose(y_new, mathops.resample(x, y, 9, kind="linear")[1])


class TestNormalize:
    """Test all four normalize() modes against closed forms."""

    def test_minmax_maps_onto_unit_interval(self):
        """minmax is the affine map onto [0, 1]: (y - min) / (max - min)."""
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        out = mathops.normalize(y, mode="minmax")
        assert np.allclose(out, [0.0, 0.25, 0.5, 0.75, 1.0])
        assert out.min() == 0.0
        assert out.max() == 1.0

    def test_minmax_handles_negative_data(self):
        """Data straddling zero still maps onto [0, 1]."""
        y = np.array([-4.0, 0.0, 4.0])
        assert np.allclose(mathops.normalize(y, mode="minmax"), [0.0, 0.5, 1.0])

    def test_zscore_gives_zero_mean_unit_std(self):
        """zscore subtracts the mean and divides by the population std."""
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        out = mathops.normalize(y, mode="zscore")
        assert np.allclose(np.mean(out), 0.0, rtol=0.0, atol=1e-12)
        assert np.allclose(np.std(out), 1.0)
        assert np.allclose(out, (y - 3.0) / np.std(y))

    def test_max_abs_preserves_sign_and_zero_crossing(self):
        """max_abs divides by max(|y|), mapping onto [-1, 1] without shifting zeros."""
        y = np.array([-4.0, 0.0, 2.0])
        out = mathops.normalize(y, mode="max_abs")
        assert np.allclose(out, [-1.0, 0.0, 0.5])
        assert out[1] == 0.0

    def test_area_gives_unit_absolute_area(self):
        """area divides by the trapezoidal absolute area, so that area becomes 1."""
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        out = mathops.normalize(y, mode="area")
        absy = np.abs(out)
        assert np.allclose(np.sum(0.5 * (absy[1:] + absy[:-1])), 1.0)

    def test_area_uses_absolute_area_for_a_zero_mean_signal(self):
        """A signal straddling zero has ~zero net area; the absolute area saves it."""
        x = np.linspace(0.0, 2.0 * np.pi, 201)
        out = mathops.normalize(np.sin(x), mode="area")
        assert np.all(np.isfinite(out))
        absy = np.abs(out)
        assert np.allclose(np.sum(0.5 * (absy[1:] + absy[:-1])), 1.0)

    @pytest.mark.parametrize("mode", ["minmax", "zscore"])
    def test_constant_signal_yields_zeros(self, mode):
        """A constant signal has zero range/std; zeros beat nan for a UI."""
        assert np.allclose(mathops.normalize(np.full(6, 7.0), mode=mode), np.zeros(6))

    @pytest.mark.parametrize("mode", ["max_abs", "area"])
    def test_all_zero_signal_yields_zeros(self, mode):
        """An all-zero signal has a zero scale; zeros beat inf/nan."""
        assert np.allclose(mathops.normalize(np.zeros(6), mode=mode), np.zeros(6))

    @pytest.mark.parametrize("mode", ["minmax", "zscore", "max_abs", "area"])
    def test_length_preserved(self, mode):
        """Every mode returns as many samples as it was given."""
        assert mathops.normalize(np.linspace(1.0, 9.0, 23), mode=mode).shape == (23,)

    @pytest.mark.parametrize("mode", ["minmax", "zscore", "max_abs"])
    def test_nan_is_ignored_by_the_scale_and_stays_nan(self, mode):
        """Statistics skip nan samples, but those samples remain nan on output."""
        y = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        out = mathops.normalize(y, mode=mode)
        assert np.isnan(out[2])
        assert np.all(np.isfinite(out[[0, 1, 3, 4]]))

    def test_all_nan_returns_a_copy(self):
        """With no finite samples there is no scale to compute, so y is passed back."""
        y = np.full(4, np.nan)
        out = mathops.normalize(y, mode="minmax")
        assert out.shape == (4,)
        assert np.all(np.isnan(out))

    def test_empty_raises(self):
        """Empty input is rejected."""
        with pytest.raises(ValueError, match="empty"):
            mathops.normalize(np.array([]))

    def test_unknown_mode_raises(self):
        """An unknown mode name is rejected."""
        with pytest.raises(ValueError, match="unknown normalize mode"):
            mathops.normalize(np.ones(5), mode="unit")

    def test_two_dimensional_raises(self):
        """Genuinely 2-D input is rejected rather than flattened."""
        with pytest.raises(ValueError, match="1-D"):
            mathops.normalize(np.zeros((4, 2)))


class TestFitPolynomial:
    """Test fit_polynomial() coefficient recovery."""

    def test_recovers_known_cubic_coefficients(self):
        """Fitting noiseless cubic data returns the exact generating coefficients."""
        expected = np.array([2.0, -3.0, 0.5, 1.0])
        x = np.linspace(-2.0, 2.0, 50)
        coeffs, y_fit = mathops.fit_polynomial(x, np.polyval(expected, x), 3)
        assert np.allclose(coeffs, expected, rtol=0.0, atol=1e-9)
        assert np.allclose(y_fit, np.polyval(expected, x), rtol=0.0, atol=1e-9)

    def test_recovers_a_line(self):
        """A degree-1 fit of a line recovers slope and intercept."""
        x = np.linspace(0.0, 10.0, 20)
        coeffs, _ = mathops.fit_polynomial(x, 3.0 * x - 4.0, 1)
        assert np.allclose(coeffs, [3.0, -4.0])

    def test_degree_zero_is_the_mean(self):
        """A degree-0 fit is the least-squares constant, i.e. the mean."""
        y = np.array([1.0, 2.0, 3.0, 10.0])
        coeffs, _ = mathops.fit_polynomial(np.arange(4.0), y, 0)
        assert np.allclose(coeffs, [np.mean(y)])

    def test_coefficients_are_highest_order_first(self):
        """coeffs follow the np.polyfit/np.polyval convention."""
        x = np.linspace(-1.0, 1.0, 30)
        coeffs, y_fit = mathops.fit_polynomial(x, x**2, 2)
        assert len(coeffs) == 3
        assert np.allclose(coeffs[0], 1.0, rtol=0.0, atol=1e-9)
        assert np.allclose(y_fit, np.polyval(coeffs, x))

    def test_y_fit_is_in_input_order(self):
        """No sorting happens, so y_fit pairs with the caller's x directly."""
        x = np.array([3.0, 1.0, 2.0, 0.0])
        coeffs, y_fit = mathops.fit_polynomial(x, 2.0 * x + 1.0, 1)
        assert np.allclose(y_fit, 2.0 * x + 1.0)

    def test_recovers_coefficients_despite_noise(self):
        """A least-squares fit of noisy cubic data stays near the true coefficients."""
        expected = np.array([2.0, -3.0, 0.5, 1.0])
        x = np.linspace(-2.0, 2.0, 500)
        rng = np.random.default_rng(12)
        y = np.polyval(expected, x) + rng.normal(0.0, 0.05, x.size)
        coeffs, _ = mathops.fit_polynomial(x, y, 3)
        assert np.allclose(coeffs, expected, rtol=0.0, atol=0.02)

    def test_nan_pairs_are_dropped(self):
        """Non-finite (x, y) pairs are excluded from the fit but kept in y_fit."""
        x = np.linspace(0.0, 10.0, 21)
        y = 2.0 * x + 1.0
        y[5] = np.nan
        coeffs, y_fit = mathops.fit_polynomial(x, y, 1)
        assert np.allclose(coeffs, [2.0, 1.0])
        assert y_fit.shape == x.shape
        assert np.allclose(y_fit, 2.0 * x + 1.0)

    def test_excessive_degree_is_clamped_not_raised(self):
        """A UI spinner set too high degrades to the highest supportable degree."""
        coeffs, _ = mathops.fit_polynomial(np.array([1.0, 2.0]), np.array([1.0, 2.0]), 15)
        assert len(coeffs) - 1 == 1

    def test_negative_degree_is_clamped_to_zero(self):
        """A negative degree clamps to a constant fit."""
        coeffs, _ = mathops.fit_polynomial(np.arange(5.0), np.full(5, 2.0), -3)
        assert len(coeffs) == 1
        assert np.allclose(coeffs, [2.0])

    def test_empty_raises(self):
        """Empty input is rejected."""
        with pytest.raises(ValueError, match="empty"):
            mathops.fit_polynomial(np.array([]), np.array([]), 1)

    def test_mismatched_length_raises(self):
        """x and y of different lengths are rejected."""
        with pytest.raises(ValueError, match="same length"):
            mathops.fit_polynomial(np.arange(5.0), np.arange(4.0), 1)

    def test_all_nan_raises(self):
        """With no finite pairs there is nothing to fit."""
        with pytest.raises(ValueError, match="at least one finite"):
            mathops.fit_polynomial(np.full(5, np.nan), np.full(5, np.nan), 1)

    def test_non_integer_degree_raises(self):
        """A degree that cannot be interpreted as an int is rejected."""
        with pytest.raises(ValueError, match="degree must be an integer"):
            mathops.fit_polynomial(np.arange(5.0), np.arange(5.0), "cubic")


class TestFftSpectrum:
    """Test fft_spectrum() peak recovery and amplitude scaling."""

    def test_recovers_a_known_frequency_and_amplitude(self):
        """A 50 Hz sinusoid of amplitude 3 gives a 3.0-high peak at exactly 50 Hz.

        1000 samples at 1000 Hz put the bin spacing at exactly 1 Hz, so 50 Hz
        lands on a bin centre and there is no spectral leakage to widen the peak.
        """
        fs, n = 1000.0, 1000
        t = np.arange(n) / fs
        freqs, mag = mathops.fft_spectrum(t, 3.0 * np.sin(2.0 * np.pi * 50.0 * t))
        peak = int(np.argmax(mag[1:])) + 1
        assert np.allclose(freqs[peak], 50.0)
        assert np.allclose(mag[peak], 3.0, rtol=0.0, atol=1e-9)

    def test_output_length_is_the_one_sided_bin_count(self):
        """Both arrays have length n // 2 + 1."""
        t = np.arange(1000) / 1000.0
        freqs, mag = mathops.fft_spectrum(t, np.sin(t))
        assert freqs.shape == (1000 // 2 + 1,)
        assert mag.shape == (1000 // 2 + 1,)

    def test_dc_bin_is_the_mean_and_is_not_doubled(self):
        """A constant offset lands in the DC bin at its true height, undoubled."""
        fs, n = 1000.0, 1000
        t = np.arange(n) / fs
        y = 1.5 + 3.0 * np.sin(2.0 * np.pi * 50.0 * t)
        freqs, mag = mathops.fft_spectrum(t, y)
        assert freqs[0] == 0.0
        assert np.allclose(mag[0], 1.5, rtol=0.0, atol=1e-9)

    def test_nyquist_bin_is_not_doubled(self):
        """The Nyquist bin has no negative-frequency twin, so its doubling is undone."""
        fs, n = 1000.0, 1000
        t = np.arange(n) / fs
        y = 2.0 * np.cos(np.pi * np.arange(n))
        freqs, mag = mathops.fft_spectrum(t, y)
        assert np.allclose(freqs[-1], fs / 2.0)
        assert np.allclose(mag[-1], 2.0, rtol=0.0, atol=1e-9)

    def test_two_tones_are_both_recovered(self):
        """Superposed sinusoids give two independent peaks at their own amplitudes."""
        fs, n = 1000.0, 1000
        t = np.arange(n) / fs
        y = 2.0 * np.sin(2.0 * np.pi * 30.0 * t) + 0.5 * np.sin(2.0 * np.pi * 120.0 * t)
        freqs, mag = mathops.fft_spectrum(t, y)
        assert np.allclose(mag[int(np.argmin(np.abs(freqs - 30.0)))], 2.0, rtol=0.0, atol=1e-9)
        assert np.allclose(mag[int(np.argmin(np.abs(freqs - 120.0)))], 0.5, rtol=0.0, atol=1e-9)

    def test_nan_samples_are_interpolated_away(self):
        """A lone nan is filled in rather than making the whole transform nan."""
        fs, n = 1000.0, 1000
        t = np.arange(n) / fs
        y = 3.0 * np.sin(2.0 * np.pi * 50.0 * t)
        y[10] = np.nan
        freqs, mag = mathops.fft_spectrum(t, y)
        assert np.all(np.isfinite(mag))
        assert np.allclose(freqs[int(np.argmax(mag[1:])) + 1], 50.0)

    def test_nonuniform_x_is_resampled_and_still_finds_the_peak(self):
        """A mildly non-uniform grid is linearly resampled; the peak survives."""
        rng = np.random.default_rng(21)
        n = 2000
        t = np.sort(rng.uniform(0.0, 2.0, n))
        t[0], t[-1] = 0.0, 2.0
        freqs, mag = mathops.fft_spectrum(t, np.sin(2.0 * np.pi * 10.0 * t))
        assert np.allclose(freqs[int(np.argmax(mag[1:])) + 1], 10.0, rtol=0.0, atol=1.0)

    def test_unsorted_x_is_sorted(self):
        """Shuffled input gives the same spectrum as the sorted equivalent."""
        t = np.arange(512) / 512.0
        y = np.sin(2.0 * np.pi * 16.0 * t)
        order = np.random.default_rng(13).permutation(t.size)
        f_a, m_a = mathops.fft_spectrum(t[order], y[order])
        f_b, m_b = mathops.fft_spectrum(t, y)
        assert np.allclose(f_a, f_b)
        assert np.allclose(m_a, m_b)

    def test_empty_raises(self):
        """Empty input is rejected."""
        with pytest.raises(ValueError, match="empty"):
            mathops.fft_spectrum(np.array([]), np.array([]))

    def test_mismatched_length_raises(self):
        """x and y of different lengths are rejected."""
        with pytest.raises(ValueError, match="same length"):
            mathops.fft_spectrum(np.arange(8.0), np.arange(5.0))

    def test_single_sample_raises(self):
        """A spectrum needs at least 2 samples."""
        with pytest.raises(ValueError, match="at least 2 samples"):
            mathops.fft_spectrum([1.0], [2.0])

    def test_zero_width_range_raises(self):
        """A degenerate x range defines no sampling interval."""
        with pytest.raises(ValueError, match="zero width"):
            mathops.fft_spectrum(np.full(8, 3.0), np.arange(8.0))

    def test_all_nan_y_raises(self):
        """With no finite samples there is nothing to transform."""
        with pytest.raises(ValueError, match="no finite samples"):
            mathops.fft_spectrum(np.linspace(0.0, 1.0, 16), np.full(16, np.nan))


class TestEnvelope:
    """Test envelope(): the analytic-signal amplitude, cross-checked against scipy."""

    def test_matches_scipy_signal_hilbert(self):
        pytest.importorskip("scipy.signal")
        from scipy.signal import hilbert

        t = np.linspace(0.0, 1.0, 1000, endpoint=False)
        y = np.sin(2.0 * np.pi * 50.0 * t) * (1.0 + 0.5 * np.sin(2.0 * np.pi * 3.0 * t))
        _x_out, env = mathops.envelope(t, y)
        ref = np.abs(hilbert(y))
        assert np.allclose(env, ref, atol=1e-9)

    def test_tracks_a_known_amplitude_modulation(self):
        """The envelope of a carrier times (1 + 0.5*sin(2*pi*3*t)) must correlate with
        the modulation almost perfectly."""
        t = np.linspace(0.0, 1.0, 2000, endpoint=False)
        mod = 1.0 + 0.5 * np.sin(2.0 * np.pi * 3.0 * t)
        y = np.sin(2.0 * np.pi * 50.0 * t) * mod
        _x_out, env = mathops.envelope(t, y)
        assert np.corrcoef(env, mod)[0, 1] > 0.999

    def test_constant_amplitude_sinusoid_has_a_flat_envelope(self):
        t = np.linspace(0.0, 1.0, 1000, endpoint=False)
        y = 2.5 * np.sin(2.0 * np.pi * 10.0 * t)
        _x_out, env = mathops.envelope(t, y)
        # Edge effects (the FFT assumes periodicity) leave the very ends less accurate;
        # the interior must sit close to the true constant amplitude 2.5.
        interior = env[50:-50]
        assert np.allclose(interior, 2.5, atol=0.05)

    def test_nonuniform_x_is_resampled(self):
        """Same behaviour fft_spectrum documents: non-uniform x is linearly resampled
        onto a uniform grid of the same length first."""
        rng = np.random.default_rng(0)
        t = np.sort(rng.uniform(0.0, 1.0, 500))
        y = np.sin(2.0 * np.pi * 20.0 * t)
        x_out, env = mathops.envelope(t, y)
        assert x_out.size == t.size
        assert np.allclose(np.diff(x_out), np.diff(x_out)[0], atol=1e-9)  # now uniform

    def test_output_length_matches_input(self):
        t = np.linspace(0.0, 1.0, 333, endpoint=False)
        y = np.sin(2.0 * np.pi * 15.0 * t)
        x_out, env = mathops.envelope(t, y)
        assert x_out.size == 333
        assert env.size == 333

    def test_single_sample_raises(self):
        with pytest.raises(ValueError, match="at least 2 samples"):
            mathops.envelope([1.0], [2.0])

    def test_zero_width_range_raises(self):
        with pytest.raises(ValueError, match="zero width"):
            mathops.envelope(np.full(8, 3.0), np.arange(8.0))

    def test_mismatched_length_raises(self):
        with pytest.raises(ValueError, match="same length"):
            mathops.envelope(np.arange(8.0), np.arange(5.0))


class TestRollingStat:
    """Test rolling_stat(): a moving-window statistic, cross-checked against pandas."""

    def test_centered_mean_matches_pandas(self):
        pd = pytest.importorskip("pandas")
        rng = np.random.default_rng(0)
        y = rng.normal(0.0, 1.0, 200)
        out = mathops.rolling_stat(y, 11, stat="mean", center=True)
        ref = pd.Series(y).rolling(11, center=True, min_periods=11).mean().to_numpy()
        assert np.allclose(out, ref, equal_nan=True)

    def test_trailing_mean_matches_pandas(self):
        pd = pytest.importorskip("pandas")
        rng = np.random.default_rng(1)
        y = rng.normal(0.0, 1.0, 200)
        out = mathops.rolling_stat(y, 11, stat="mean", center=False)
        ref = pd.Series(y).rolling(11, center=False, min_periods=11).mean().to_numpy()
        assert np.allclose(out, ref, equal_nan=True)

    def test_std_matches_pandas_population_ddof(self):
        pd = pytest.importorskip("pandas")
        rng = np.random.default_rng(2)
        y = rng.normal(0.0, 1.0, 100)
        out = mathops.rolling_stat(y, 5, stat="std", center=True)
        ref = pd.Series(y).rolling(5, center=True, min_periods=5).std(ddof=0).to_numpy()
        assert np.allclose(out, ref, equal_nan=True)

    def test_median_min_max_are_correct_on_a_known_window(self):
        y = np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0])
        med = mathops.rolling_stat(y, 3, stat="median", center=True)
        mn = mathops.rolling_stat(y, 3, stat="min", center=True)
        mx = mathops.rolling_stat(y, 3, stat="max", center=True)
        # Window centred at index 4 (value 5.0) covers indices 3,4,5 = [1, 5, 9].
        assert med[4] == pytest.approx(5.0)
        assert mn[4] == pytest.approx(1.0)
        assert mx[4] == pytest.approx(9.0)

    def test_output_length_matches_input(self):
        y = np.arange(50.0)
        out = mathops.rolling_stat(y, 7)
        assert out.size == 50

    def test_window_larger_than_signal_is_clamped(self):
        y = np.arange(5.0)
        out = mathops.rolling_stat(y, 1000, stat="mean")
        assert out.size == 5
        assert np.isfinite(out).any()

    def test_window_one_is_the_identity_for_mean(self):
        y = np.array([1.0, 5.0, -3.0, 2.0])
        out = mathops.rolling_stat(y, 1, stat="mean")
        assert np.allclose(out, y)

    def test_nan_sample_excluded_not_propagated(self):
        """A single nan inside a window must not poison the whole window's statistic,
        as long as enough other samples remain (min_periods)."""
        y = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        out = mathops.rolling_stat(y, 3, stat="mean", center=True, min_periods=2)
        # Window at index 2 covers [2.0, nan, 4.0] -> mean of the 2 finite values = 3.0.
        assert out[2] == pytest.approx(3.0)

    def test_min_periods_blanks_a_window_with_too_few_finite_samples(self):
        y = np.array([1.0, np.nan, np.nan, 4.0, 5.0])
        out = mathops.rolling_stat(y, 3, stat="mean", center=True, min_periods=3)
        # Window at index 1 covers [1.0, nan, nan] -> only 1 finite sample, short of 3.
        assert np.isnan(out[1])

    def test_default_min_periods_requires_a_full_window(self):
        """With min_periods unset, the edges (which have no full window) are nan."""
        y = np.arange(10.0)
        out = mathops.rolling_stat(y, 5, stat="mean", center=False)
        assert np.isnan(out[:4]).all()
        assert np.isfinite(out[4:]).all()

    def test_unknown_stat_raises(self):
        with pytest.raises(ValueError, match="unknown stat"):
            mathops.rolling_stat(np.arange(10.0), 3, stat="bogus")

    def test_window_less_than_one_is_clamped_to_one_not_raising(self):
        y = np.arange(5.0)
        out = mathops.rolling_stat(y, 0, stat="mean")
        assert np.allclose(out, y)  # window clamped to 1 -> identity

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            mathops.rolling_stat(np.array([]), 3)


class TestTwoSampleTest:
    """Test two_sample_test(): four methods, cross-checked against scipy directly."""

    def test_welch_matches_scipy_directly(self):
        pytest.importorskip("scipy.stats")
        from scipy import stats as scipy_stats

        rng = np.random.default_rng(0)
        a = rng.normal(0.0, 1.0, 200)
        b = rng.normal(0.5, 1.2, 180)
        result = mathops.two_sample_test(a, b, method="welch")
        ref_stat, ref_p = scipy_stats.ttest_ind(a, b, equal_var=False)
        assert result["statistic"] == pytest.approx(float(ref_stat))
        assert result["p_value"] == pytest.approx(float(ref_p))

    def test_student_matches_scipy_directly(self):
        pytest.importorskip("scipy.stats")
        from scipy import stats as scipy_stats

        rng = np.random.default_rng(1)
        a = rng.normal(0.0, 1.0, 150)
        b = rng.normal(0.3, 1.0, 150)
        result = mathops.two_sample_test(a, b, method="student")
        ref_stat, ref_p = scipy_stats.ttest_ind(a, b, equal_var=True)
        assert result["statistic"] == pytest.approx(float(ref_stat))
        assert result["p_value"] == pytest.approx(float(ref_p))

    def test_mannwhitney_matches_scipy_directly(self):
        pytest.importorskip("scipy.stats")
        from scipy import stats as scipy_stats

        rng = np.random.default_rng(2)
        a = rng.normal(0.0, 1.0, 120)
        b = rng.normal(0.4, 1.0, 100)
        result = mathops.two_sample_test(a, b, method="mannwhitney")
        ref = scipy_stats.mannwhitneyu(a, b, alternative="two-sided")
        assert result["statistic"] == pytest.approx(float(ref.statistic))
        assert result["p_value"] == pytest.approx(float(ref.pvalue))

    def test_ks_matches_scipy_directly(self):
        pytest.importorskip("scipy.stats")
        from scipy import stats as scipy_stats

        rng = np.random.default_rng(3)
        a = rng.normal(0.0, 1.0, 100)
        b = rng.uniform(-2.0, 2.0, 100)
        result = mathops.two_sample_test(a, b, method="ks")
        ref = scipy_stats.ks_2samp(a, b)
        assert result["statistic"] == pytest.approx(float(ref.statistic))
        assert result["p_value"] == pytest.approx(float(ref.pvalue))

    def test_identical_samples_give_a_p_value_of_one(self):
        rng = np.random.default_rng(4)
        a = rng.normal(0.0, 1.0, 100)
        result = mathops.two_sample_test(a, a.copy(), method="welch")
        assert result["p_value"] == pytest.approx(1.0)

    def test_clearly_different_means_are_significant(self):
        rng = np.random.default_rng(5)
        a = rng.normal(0.0, 0.1, 300)
        b = rng.normal(5.0, 0.1, 300)
        result = mathops.two_sample_test(a, b, method="welch")
        assert result["p_value"] < 1e-10

    def test_summary_statistics_are_correct(self):
        a = np.array([1.0, 2.0, 3.0, np.nan])
        b = np.array([10.0, 20.0])
        result = mathops.two_sample_test(a, b, method="welch")
        assert result["n_a"] == 3.0
        assert result["n_b"] == 2.0
        assert result["mean_a"] == pytest.approx(2.0)
        assert result["mean_b"] == pytest.approx(15.0)

    def test_different_sample_sizes_are_allowed(self):
        rng = np.random.default_rng(6)
        a = rng.normal(0.0, 1.0, 50)
        b = rng.normal(0.0, 1.0, 500)
        result = mathops.two_sample_test(a, b, method="welch")
        assert result["n_a"] == 50.0
        assert result["n_b"] == 500.0

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="unknown method"):
            mathops.two_sample_test([1.0, 2.0], [3.0, 4.0], method="bogus")

    def test_too_few_finite_samples_raises(self):
        with pytest.raises(ValueError, match="at least 2 finite"):
            mathops.two_sample_test([1.0], [1.0, 2.0, np.nan])

    def test_unavailable_scipy_raises_two_sample_unavailable_error(self, monkeypatch):
        # scipy.stats alone is not enough: two_sample_test's "from scipy import stats"
        # would still resolve via the already-imported scipy package's own attribute.
        monkeypatch.setitem(sys.modules, "scipy.stats", None)
        monkeypatch.setitem(sys.modules, "scipy", None)
        assert mathops.two_sample_available() is False
        with pytest.raises(mathops.TwoSampleUnavailableError, match="scipy"):
            mathops.two_sample_test([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
        assert issubclass(mathops.TwoSampleUnavailableError, ValueError)

    def test_two_sample_test_works_again_once_scipy_is_importable(self):
        """The monkeypatched module above must not leak into the rest of the suite."""
        assert mathops.two_sample_available() is True
        result = mathops.two_sample_test([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
        assert np.isfinite(result["p_value"])


class TestDescribe:
    """Test describe() summary statistics and nan safety."""

    def test_known_values(self):
        """Every statistic matches its closed form on a simple signal."""
        y = np.array([1.0, 2.0, 3.0, 4.0])
        d = mathops.describe(y)
        assert d["n"] == 4.0
        assert d["min"] == 1.0
        assert d["max"] == 4.0
        assert np.allclose(d["mean"], 2.5)
        assert np.allclose(d["median"], 2.5)
        assert np.allclose(d["sum"], 10.0)
        assert np.allclose(d["var"], 1.25)
        assert np.allclose(d["std"], np.sqrt(1.25))
        assert np.allclose(d["rms"], np.sqrt(30.0 / 4.0))

    def test_all_expected_keys_present_as_floats(self):
        """The dict has exactly the documented keys and only Python floats."""
        d = mathops.describe(np.linspace(0.0, 1.0, 10))
        expected = {"n", "min", "max", "mean", "median", "std", "var", "sum", "rms"}
        assert set(d) == expected
        assert all(isinstance(v, float) for v in d.values())

    def test_std_and_var_are_population_statistics(self):
        """std/var use ddof=0, matching numpy defaults."""
        y = np.array([1.0, 2.0, 3.0, 4.0, 10.0])
        d = mathops.describe(y)
        assert np.allclose(d["std"], np.std(y))
        assert np.allclose(d["var"], np.var(y))

    def test_nan_and_inf_are_excluded_from_every_statistic(self):
        """n counts only finite samples and the rest stay consistent with it."""
        d = mathops.describe(np.array([1.0, 2.0, np.nan, np.inf, -np.inf, 3.0]))
        assert d["n"] == 3.0
        assert d["min"] == 1.0
        assert d["max"] == 3.0
        assert np.allclose(d["mean"], 2.0)
        assert np.allclose(d["sum"], 6.0)
        assert all(np.isfinite(v) for v in d.values())

    def test_mean_equals_sum_over_n(self):
        """The dict is internally consistent even when nans were dropped."""
        d = mathops.describe(np.array([1.0, np.nan, 5.0, 6.0]))
        assert np.allclose(d["mean"], d["sum"] / d["n"])

    def test_all_nan_gives_zero_n_and_nan_elsewhere(self):
        """An all-nan signal reports n == 0 with nan statistics, without raising."""
        d = mathops.describe(np.full(5, np.nan))
        assert d["n"] == 0.0
        for key in ("min", "max", "mean", "median", "std", "var", "sum", "rms"):
            assert np.isnan(d[key])

    def test_single_sample(self):
        """A lone sample has zero spread and equals every location statistic."""
        d = mathops.describe([7.0])
        assert d["n"] == 1.0
        assert d["mean"] == 7.0
        assert d["median"] == 7.0
        assert d["std"] == 0.0
        assert d["var"] == 0.0
        assert d["rms"] == 7.0

    def test_rms_of_a_sinusoid(self):
        """The RMS of a unit sinusoid over whole periods is 1/sqrt(2)."""
        t = np.linspace(0.0, 2.0 * np.pi, 10001)[:-1]
        d = mathops.describe(np.sin(t))
        assert np.allclose(d["rms"], 1.0 / np.sqrt(2.0), rtol=1e-4)

    def test_empty_raises(self):
        """Empty input is rejected."""
        with pytest.raises(ValueError, match="empty"):
            mathops.describe(np.array([]))

    def test_two_dimensional_raises(self):
        """Genuinely 2-D input is rejected rather than flattened."""
        with pytest.raises(ValueError, match="1-D"):
            mathops.describe(np.zeros((4, 2)))


class TestDescribeExtended:
    """Test describe_extended(): describe() plus percentiles/shape/robust statistics."""

    def test_never_diverges_from_describe(self):
        """Every describe() key/value is present unchanged."""
        y = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
        base = mathops.describe(y)
        ext = mathops.describe_extended(y)
        for key, value in base.items():
            assert ext[key] == value

    def test_percentile_keys_match_the_default_request(self):
        y = np.linspace(0.0, 1.0, 1000)
        ext = mathops.describe_extended(y)
        for key in ("percentile_5", "percentile_25", "percentile_75", "percentile_95"):
            assert key in ext

    def test_percentiles_are_correct_on_a_uniform_grid(self):
        """percentile_p of a 0..100 uniform grid is p, by construction."""
        y = np.linspace(0.0, 100.0, 100001)
        ext = mathops.describe_extended(y, percentiles=(10.0, 50.0, 90.0))
        assert np.allclose(ext["percentile_10"], 10.0, atol=0.01)
        assert np.allclose(ext["percentile_50"], 50.0, atol=0.01)
        assert np.allclose(ext["percentile_90"], 90.0, atol=0.01)

    def test_iqr_matches_p75_minus_p25(self):
        y = np.linspace(0.0, 100.0, 10001)
        ext = mathops.describe_extended(y)
        assert np.allclose(ext["iqr"], ext["percentile_75"] - ext["percentile_25"], atol=0.02)

    def test_mad_of_a_symmetric_signal(self):
        """MAD of {-2,-1,0,1,2} about its median (0) is 1."""
        y = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        ext = mathops.describe_extended(y)
        assert np.allclose(ext["mad"], 1.0)

    def test_skewness_and_kurtosis_match_scipy(self):
        """Cross-check against scipy.stats.skew/kurtosis's own (bias=True) defaults."""
        pytest.importorskip("scipy.stats")
        from scipy import stats as scipy_stats

        rng = np.random.default_rng(11)
        y = rng.exponential(2.0, 5000)  # a genuinely skewed distribution
        ext = mathops.describe_extended(y)
        assert np.allclose(ext["skewness"], float(scipy_stats.skew(y)), atol=1e-9)
        assert np.allclose(ext["kurtosis"], float(scipy_stats.kurtosis(y)), atol=1e-9)

    def test_symmetric_distribution_has_near_zero_skewness(self):
        rng = np.random.default_rng(12)
        y = rng.normal(0.0, 1.0, 20000)
        ext = mathops.describe_extended(y)
        assert abs(ext["skewness"]) < 0.05

    def test_constant_signal_has_zero_skewness_and_kurtosis_not_nan(self):
        """std == 0 must not divide by zero into nan."""
        y = np.full(10, 5.0)
        ext = mathops.describe_extended(y)
        assert ext["skewness"] == 0.0
        assert ext["kurtosis"] == 0.0

    def test_nan_count_reflects_dropped_samples(self):
        y = np.array([1.0, np.nan, 2.0, np.inf, 3.0])
        ext = mathops.describe_extended(y)
        assert ext["nan_count"] == 2.0
        assert ext["n"] == 3.0

    def test_outlier_count_flags_a_planted_outlier(self):
        rng = np.random.default_rng(13)
        y = rng.normal(0.0, 1.0, 500)
        y[0] = 1000.0  # unmistakably outside any Tukey fence
        ext = mathops.describe_extended(y)
        assert ext["outlier_count"] >= 1.0

    def test_no_iqr_spread_gives_zero_outliers_not_a_crash(self):
        """A signal with IQR == 0 (e.g. mostly-constant) must not divide by zero."""
        y = np.array([5.0] * 20 + [5.0, 5.0])
        ext = mathops.describe_extended(y)
        assert ext["iqr"] == 0.0
        assert ext["outlier_count"] == 0.0

    def test_all_nan_input_is_nan_throughout_not_raising(self):
        ext = mathops.describe_extended(np.full(5, np.nan))
        assert ext["n"] == 0.0
        for key in ("iqr", "mad", "skewness", "kurtosis", "percentile_25", "outlier_count"):
            assert np.isnan(ext[key])

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            mathops.describe_extended(np.array([]))


class TestCorrelate:
    """Test correlate(): Pearson/Spearman correlation, covariance, a linear fit."""

    def test_perfect_positive_linear_relationship(self):
        x = np.linspace(0.0, 10.0, 200)
        y = 3.0 * x + 2.0
        c = mathops.correlate(x, y)
        assert np.allclose(c["pearson_r"], 1.0, atol=1e-9)
        assert np.allclose(c["spearman_rho"], 1.0, atol=1e-9)
        assert np.allclose(c["slope"], 3.0, atol=1e-9)
        assert np.allclose(c["intercept"], 2.0, atol=1e-9)
        assert np.allclose(c["r_squared"], 1.0, atol=1e-9)

    def test_perfect_negative_linear_relationship(self):
        x = np.linspace(0.0, 10.0, 200)
        y = -2.0 * x + 5.0
        c = mathops.correlate(x, y)
        assert np.allclose(c["pearson_r"], -1.0, atol=1e-9)
        assert np.allclose(c["spearman_rho"], -1.0, atol=1e-9)

    def test_monotonic_nonlinear_relationship_spearman_beats_pearson(self):
        """A curved-but-monotonic relationship: Spearman is exactly 1, Pearson is not."""
        x = np.linspace(0.1, 10.0, 200)
        y = x**3
        c = mathops.correlate(x, y)
        assert np.allclose(c["spearman_rho"], 1.0, atol=1e-9)
        assert c["pearson_r"] < 0.99

    def test_covariance_matches_numpy(self):
        rng = np.random.default_rng(20)
        x = rng.normal(0.0, 1.0, 500)
        y = 2.0 * x + rng.normal(0.0, 0.1, 500)
        c = mathops.correlate(x, y)
        assert np.allclose(c["covariance"], float(np.cov(x, y, ddof=0)[0, 1]))

    def test_uncorrelated_data_is_near_zero(self):
        rng = np.random.default_rng(21)
        x = rng.normal(0.0, 1.0, 5000)
        y = rng.normal(0.0, 1.0, 5000)
        c = mathops.correlate(x, y)
        assert abs(c["pearson_r"]) < 0.05
        assert abs(c["spearman_rho"]) < 0.05

    def test_p_values_are_present_and_small_for_a_strong_relationship(self):
        pytest.importorskip("scipy.stats")
        x = np.linspace(0.0, 10.0, 200)
        y = 3.0 * x + 2.0
        c = mathops.correlate(x, y)
        assert c["pearson_p"] < 1e-6
        assert c["spearman_p"] < 1e-6

    def test_spearman_falls_back_to_ranked_pearson_without_scipy(self, monkeypatch):
        """No scipy.stats: rho degrades to Pearson-of-ranks -- the exact definition of
        Spearman's rho, not an approximation -- and the p-values come back nan."""
        monkeypatch.setitem(sys.modules, "scipy.stats", None)
        monkeypatch.setitem(sys.modules, "scipy", None)
        x = np.linspace(0.1, 10.0, 200)
        y = x**3
        c = mathops.correlate(x, y)
        assert np.allclose(c["spearman_rho"], 1.0, atol=1e-9)
        assert np.isnan(c["pearson_p"])
        assert np.isnan(c["spearman_p"])
        # Pearson r/covariance/slope/intercept/r_squared never touch scipy at all.
        assert np.isfinite(c["pearson_r"])
        assert np.isfinite(c["slope"])

    def test_correlate_works_again_once_scipy_is_importable(self):
        """The monkeypatched module above must not leak into the rest of the suite."""
        x = np.linspace(0.0, 10.0, 200)
        y = 3.0 * x + 2.0
        c = mathops.correlate(x, y)
        assert np.isfinite(c["pearson_p"])

    def test_constant_x_has_nan_slope_but_finite_covariance(self):
        """No spread in x: pearson_r/slope/intercept/r_squared are undefined (nan), but
        covariance (which needs no division) is still a real number: zero."""
        x = np.full(50, 3.0)
        y = np.linspace(0.0, 1.0, 50)
        c = mathops.correlate(x, y)
        assert np.isnan(c["pearson_r"])
        assert np.isnan(c["slope"])
        assert np.allclose(c["covariance"], 0.0, atol=1e-12)

    def test_nan_pairs_are_excluded(self):
        x = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        y = np.array([1.0, 2.0, 3.0, np.nan, 5.0])
        c = mathops.correlate(x, y)
        assert c["n"] == 3.0  # only indices 0, 1, 4 are finite in both

    def test_fewer_than_two_finite_pairs_raises(self):
        with pytest.raises(ValueError, match="2 finite"):
            mathops.correlate([1.0], [1.0])
        with pytest.raises(ValueError, match="2 finite"):
            mathops.correlate([1.0, np.nan], [1.0, 2.0])

    def test_mismatched_length_raises(self):
        with pytest.raises(ValueError):
            mathops.correlate([1.0, 2.0, 3.0], [1.0, 2.0])


class TestNoWarningsOnHappyPath:
    """Test that ordinary scipy-backed calls stay silent."""

    def test_savgol_derivative_on_uniform_grid_does_not_warn(self):
        """A uniform grid with scipy present is the happy path: no fallback warning."""
        x = np.linspace(0.0, 1.0, 101)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            mathops.derivative(x, x**3, method="savgol", window=11)

    def test_cubic_resample_on_strict_grid_does_not_warn(self):
        """Strictly increasing x with scipy present must not warn."""
        x = np.linspace(0.0, 1.0, 50)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            mathops.resample(x, np.sin(x), 25, kind="cubic")


def _noisy(model, truth, x, scale, seed):
    """Samples of ``model`` at ``truth`` plus gaussian noise. The fixture for every fit test."""
    rng = np.random.default_rng(seed)
    spec = mathops.resolve_model(model)
    return spec.fn(x, *truth) + rng.normal(0.0, scale, x.size)


def _assert_recovered(params, cov, truth, tol_sigma=4.0):
    """Every parameter must sit within ``tol_sigma`` of truth *by its own reported error*.

    This is the assertion that matters: a fitter that returns plausible numbers with
    meaningless uncertainties is not reportable. Checking against the error the fit
    itself published tests both at once -- if the errors were nonsense, this fails.
    """
    errors = mathops.fit_errors(cov)
    assert np.all(np.isfinite(errors)), f"non-finite standard errors: {errors}"
    assert np.all(errors > 0.0), f"non-positive standard errors: {errors}"
    deviation = np.abs(np.asarray(params) - np.asarray(truth)) / errors
    assert np.all(
        deviation <= tol_sigma
    ), f"params {params} are {deviation} sigma from truth {truth} (max {tol_sigma})"


class TestFitModelRecovery:
    """fit_model must recover known parameters from noisy data using only its own guess."""

    def test_gaussian_recovers_known_parameters(self):
        """A noisy gaussian is recovered within the reported errors, guess unaided."""
        x = np.linspace(-10.0, 12.0, 400)
        truth = [5.0, 2.0, 1.5, 1.0]
        y = _noisy("gaussian", truth, x, 0.08, seed=1)
        params, cov, y_fit, names = mathops.fit_model(x, y, "gaussian")
        assert names == ["a", "mu", "sigma", "c"]
        assert y_fit.shape == x.shape
        _assert_recovered(params, cov, truth)

    def test_gaussian_dip_recovers_negative_amplitude(self):
        """An absorption dip fits too: the guess picks the extremum that stands out."""
        x = np.linspace(-10.0, 12.0, 400)
        truth = [-4.0, 3.0, 0.9, 7.0]
        y = _noisy("gaussian", truth, x, 0.06, seed=2)
        params, cov, _y_fit, _names = mathops.fit_model(x, y, "gaussian")
        _assert_recovered(params, cov, truth)

    def test_lorentzian_recovers_known_parameters(self):
        """A noisy lorentzian is recovered within the reported errors."""
        x = np.linspace(-8.0, 8.0, 500)
        truth = [3.0, -1.0, 0.8, 0.5]
        y = _noisy("lorentzian", truth, x, 0.03, seed=3)
        params, cov, _y_fit, names = mathops.fit_model(x, y, "lorentzian")
        assert names == ["a", "x0", "gamma", "c"]
        _assert_recovered(params, cov, truth)

    def test_exponential_decay_recovers_known_parameters(self):
        """A decay whose record starts away from x=0: the amplitude is the x=0 intercept.

        Guessing ``a`` as ``y[0]`` instead of the log-fit intercept would be wrong by
        ``exp(x0/tau)``, so this grid deliberately starts at x=2.
        """
        x = np.linspace(2.0, 20.0, 300)
        truth = [8.0, 3.5, 1.2]
        y = _noisy("exp_decay", truth, x, 0.02, seed=4)
        params, cov, _y_fit, names = mathops.fit_model(x, y, "exp_decay")
        assert names == ["a", "tau", "c"]
        _assert_recovered(params, cov, truth)

    def test_exponential_rising_recovers_known_parameters(self):
        """A saturating exponential (negative amplitude) is the same model, fitted."""
        x = np.linspace(0.0, 15.0, 300)
        truth = [-6.0, 2.5, 4.0]
        y = _noisy("exp_decay", truth, x, 0.02, seed=5)
        params, cov, _y_fit, _names = mathops.fit_model(x, y, "exp_decay")
        _assert_recovered(params, cov, truth)

    def test_power_law_recovers_known_parameters(self):
        """A power law with multiplicative noise is recovered from the log-log guess."""
        rng = np.random.default_rng(6)
        x = np.linspace(0.5, 30.0, 300)
        truth = [2.2, 1.7]
        y = mathops.MODELS["power_law"].fn(x, *truth) * (1.0 + rng.normal(0.0, 0.02, x.size))
        params, cov, _y_fit, names = mathops.fit_model(x, y, "power_law")
        assert names == ["a", "b"]
        _assert_recovered(params, cov, truth)

    def test_sigmoid_recovers_known_parameters(self):
        """A rising logistic is recovered from the half-height / 4-over-span guess."""
        x = np.linspace(-6.0, 14.0, 400)
        truth = [9.0, 1.1, 4.0, -2.0]
        y = _noisy("sigmoid", truth, x, 0.06, seed=7)
        params, cov, _y_fit, names = mathops.fit_model(x, y, "sigmoid")
        assert names == ["L", "k", "x0", "c"]
        _assert_recovered(params, cov, truth)

    def test_sigmoid_falling_recovers_negative_rate(self):
        """A falling logistic needs the guess to take its sign from the trend."""
        x = np.linspace(-6.0, 14.0, 400)
        truth = [9.0, -0.8, 3.0, 1.0]
        y = _noisy("sigmoid", truth, x, 0.06, seed=8)
        params, cov, _y_fit, _names = mathops.fit_model(x, y, "sigmoid")
        _assert_recovered(params, cov, truth)

    def test_linear_recovers_known_parameters(self):
        """The linear model is the closed-form line, reported with errors."""
        x = np.linspace(-5.0, 5.0, 300)
        truth = [2.5, -1.25]
        y = _noisy("linear", truth, x, 0.3, seed=9)
        params, cov, _y_fit, names = mathops.fit_model(x, y, "linear")
        assert names == ["m", "b"]
        _assert_recovered(params, cov, truth)

    def test_custom_model_recovers_known_parameters(self):
        """A user-typed f(x) fits its free variables, discovered from the expression."""
        x = np.linspace(0.0, 10.0, 400)
        spec = mathops.custom_model("a*exp(-x/b) + c")
        assert spec.param_names == ("a", "b", "c")
        truth = [4.0, 2.0, 0.5]
        rng = np.random.default_rng(10)
        y = spec.fn(x, *truth) + rng.normal(0.0, 0.03, x.size)
        params, cov, _y_fit, names = mathops.fit_model(x, y, spec)
        assert names == ["a", "b", "c"]
        _assert_recovered(params, cov, truth)

    def test_manual_p0_rescues_a_fit_the_heuristic_cannot_guess(self):
        """A frequency has a local minimum every half period, so p0 must be overridable.

        The custom heuristic is 1.0 per parameter and there is nothing better to infer
        from an arbitrary form -- which is exactly why the API takes p0 and the Fit tab
        exposes it. Both halves are asserted: the flat start fails, the hand-set one works.
        """
        x = np.linspace(0.0, 10.0, 400)
        spec = mathops.custom_model("a*sin(b*x) + c")
        truth = [3.0, 1.4, 0.5]
        rng = np.random.default_rng(11)
        y = spec.fn(x, *truth) + rng.normal(0.0, 0.05, x.size)

        auto_params, _cov, auto_fit, _names = mathops.fit_model(x, y, spec)
        assert mathops.fit_statistics(y, auto_fit, 3)["r_squared"] < 0.5
        assert not np.allclose(auto_params, truth, rtol=0.1)

        params, cov, y_fit, _names = mathops.fit_model(x, y, spec, p0=[1.0, 1.5, 0.0])
        _assert_recovered(params, cov, truth)
        assert mathops.fit_statistics(y, y_fit, 3)["r_squared"] > 0.99

    def test_y_fit_is_in_input_order(self):
        """y_fit lines up with the caller's x even when x is unsorted, like fit_polynomial."""
        rng = np.random.default_rng(12)
        x = np.linspace(-10.0, 12.0, 400)
        y = _noisy("gaussian", [5.0, 2.0, 1.5, 1.0], x, 0.08, seed=13)
        order = rng.permutation(x.size)
        params, _cov, y_fit, _names = mathops.fit_model(x[order], y[order], "gaussian")
        assert np.allclose(y_fit, mathops.MODELS["gaussian"].fn(x[order], *params))

    def test_shuffled_input_gives_the_same_fit(self):
        """Least squares does not depend on sample order; the internal sort must not either."""
        rng = np.random.default_rng(14)
        x = np.linspace(-10.0, 12.0, 400)
        y = _noisy("gaussian", [5.0, 2.0, 1.5, 1.0], x, 0.08, seed=15)
        order = rng.permutation(x.size)
        sorted_params, _c1, _f1, _n1 = mathops.fit_model(x, y, "gaussian")
        shuffled_params, _c2, _f2, _n2 = mathops.fit_model(x[order], y[order], "gaussian")
        assert np.allclose(sorted_params, shuffled_params, rtol=1e-8)

    def test_nonfinite_samples_are_dropped_not_propagated(self):
        """A single nan must not poison the whole fit."""
        x = np.linspace(-10.0, 12.0, 400)
        truth = [5.0, 2.0, 1.5, 1.0]
        y = _noisy("gaussian", truth, x, 0.08, seed=16)
        y[::17] = np.nan
        x = x.copy()
        x[3] = np.inf
        params, cov, y_fit, _names = mathops.fit_model(x, y, "gaussian")
        assert np.all(np.isfinite(params))
        assert y_fit.size == x.size
        _assert_recovered(params, cov, truth)


class TestFitModelGuesses:
    """The initial-guess heuristics: without a decent p0 curve_fit fails on real data."""

    def test_gaussian_guess_is_close_to_truth(self):
        """The peak heuristic lands near truth before any optimisation happens."""
        x = np.linspace(-10.0, 12.0, 400)
        truth = np.array([5.0, 2.0, 1.5, 1.0])
        y = _noisy("gaussian", truth, x, 0.05, seed=17)
        guess = mathops.initial_guess(x, y, "gaussian")
        assert guess.shape == (4,)
        # Amplitude, centre and width within 25% of truth: this is a starting point,
        # not an answer, but a starting point that is already in the right basin.
        assert np.allclose(guess[:3], truth[:3], rtol=0.25)

    def test_exp_decay_guess_recovers_the_rate(self):
        """The log-linear regression puts tau in the right ballpark from the raw data."""
        x = np.linspace(2.0, 20.0, 300)
        y = _noisy("exp_decay", [8.0, 3.5, 1.2], x, 0.01, seed=18)
        guess = mathops.initial_guess(x, y, "exp_decay")
        assert np.isclose(guess[1], 3.5, rtol=0.3)

    def test_power_law_guess_recovers_the_exponent(self):
        """The log-log fit is nearly the answer for a clean power law."""
        x = np.linspace(0.5, 30.0, 300)
        y = mathops.MODELS["power_law"].fn(x, 2.2, 1.7)
        guess = mathops.initial_guess(x, y, "power_law")
        assert np.allclose(guess, [2.2, 1.7], rtol=1e-6)

    def test_guess_is_always_finite(self):
        """A constant signal has no peak, width or rate; the guess must still be usable."""
        x = np.linspace(0.0, 10.0, 50)
        y = np.full(50, 3.0)
        for name in mathops.MODEL_NAMES:
            if name == "power_law":
                continue
            guess = mathops.initial_guess(x, y, name)
            assert np.all(np.isfinite(guess)), f"{name} guessed {guess}"

    def test_fwhm_guess_ignores_a_distant_noise_spike(self):
        """Only the run around the peak counts, else one far spike widens sigma hugely."""
        x = np.linspace(-20.0, 20.0, 801)
        y = mathops.MODELS["gaussian"].fn(x, 5.0, 0.0, 1.0, 0.0)
        y[5] = 4.0  # a lone spike above half height, far from the peak
        guess = mathops.initial_guess(x, y, "gaussian")
        assert np.isclose(guess[2], 1.0, rtol=0.2), f"sigma guess {guess[2]} was dragged out"


class TestFitStatistics:
    """The reportable numbers: standard errors, reduced chi-squared and R squared."""

    def test_r_squared_is_near_one_for_a_good_fit(self):
        """A correct model on low-noise data explains almost all the variance."""
        x = np.linspace(-10.0, 12.0, 400)
        y = _noisy("gaussian", [5.0, 2.0, 1.5, 1.0], x, 0.02, seed=19)
        _p, _c, y_fit, names = mathops.fit_model(x, y, "gaussian")
        stats = mathops.fit_statistics(y, y_fit, len(names))
        assert stats["r_squared"] > 0.999
        assert stats["dof"] == 400 - 4
        assert stats["n"] == 400

    def test_reduced_chi_squared_is_about_one_with_true_errors(self):
        """With real per-point sigma the statistic recovers its textbook meaning."""
        x = np.linspace(-10.0, 12.0, 400)
        noise = 0.08
        y = _noisy("gaussian", [5.0, 2.0, 1.5, 1.0], x, noise, seed=20)
        sigma = np.full(x.size, noise)
        _p, _c, y_fit, names = mathops.fit_model(x, y, "gaussian", sigma=sigma)
        stats = mathops.fit_statistics(y, y_fit, len(names), sigma=sigma)
        assert 0.75 < stats["reduced_chi_squared"] < 1.3

    def test_sigma_weighting_gives_absolute_errors(self):
        """absolute_sigma: supplied errors must set the scale of cov, not just the weights."""
        x = np.linspace(-10.0, 12.0, 400)
        y = _noisy("gaussian", [5.0, 2.0, 1.5, 1.0], x, 0.08, seed=21)
        _p1, cov_small, _f1, _n1 = mathops.fit_model(x, y, "gaussian", sigma=np.full(x.size, 0.08))
        _p2, cov_big, _f2, _n2 = mathops.fit_model(x, y, "gaussian", sigma=np.full(x.size, 0.8))
        # Ten times the assumed error is ten times the parameter error.
        assert np.allclose(
            mathops.fit_errors(cov_big) / mathops.fit_errors(cov_small), 10.0, rtol=1e-3
        )

    def test_errors_are_sqrt_diag_cov(self):
        """fit_errors is exactly sqrt(diag(cov)) for a well-conditioned fit."""
        x = np.linspace(-10.0, 12.0, 400)
        y = _noisy("gaussian", [5.0, 2.0, 1.5, 1.0], x, 0.08, seed=22)
        _p, cov, _f, _n = mathops.fit_model(x, y, "gaussian")
        assert np.allclose(mathops.fit_errors(cov), np.sqrt(np.diag(cov)))

    def test_statistics_of_a_constant_signal(self):
        """No variance to explain: an exact reproduction is 1.0, matching the Math Lab."""
        y = np.full(20, 4.0)
        assert mathops.fit_statistics(y, y, 1)["r_squared"] == 1.0
        assert mathops.fit_statistics(y, np.full(20, 5.0), 1)["r_squared"] == 0.0

    def test_statistics_ignore_nonfinite_pairs(self):
        """n counts the usable pairs, so dof and rmse stay internally consistent."""
        y = np.arange(10.0)
        y_fit = y.copy()
        y_fit[2] = np.nan
        stats = mathops.fit_statistics(y, y_fit, 2)
        assert stats["n"] == 9.0
        assert stats["dof"] == 7.0
        assert stats["rmse"] == 0.0

    def test_all_nonfinite_statistics_are_nan_not_an_error(self):
        """Nothing usable is a result to report, not an exception."""
        stats = mathops.fit_statistics(np.full(5, np.nan), np.zeros(5), 2)
        assert stats["n"] == 0.0
        assert np.isnan(stats["r_squared"])

    def test_fit_errors_rejects_a_non_square_matrix(self):
        """A covariance that is not square is a programming error, not a fit outcome."""
        with pytest.raises(ValueError, match="square"):
            mathops.fit_errors(np.zeros((2, 3)))


class TestFitModelBounds:
    """Bounds switch curve_fit to a trust-region method; p0 must stay feasible."""

    def test_bounds_constrain_the_result(self):
        """A fitted parameter must land inside its box."""
        x = np.linspace(-10.0, 12.0, 400)
        y = _noisy("gaussian", [5.0, 2.0, 1.5, 1.0], x, 0.08, seed=23)
        params, _c, _f, _n = mathops.fit_model(
            x, y, "gaussian", bounds=([0.0, 0.0, 0.0, -10.0], [10.0, 3.0, 2.0, 10.0])
        )
        assert 0.0 <= params[1] <= 3.0
        assert 0.0 <= params[2] <= 2.0

    def test_out_of_box_guess_is_clipped_not_rejected(self):
        """curve_fit rejects an infeasible x0 with an error a user cannot act on."""
        x = np.linspace(-10.0, 12.0, 400)
        y = _noisy("gaussian", [5.0, 2.0, 1.5, 1.0], x, 0.08, seed=24)
        # The heuristic centre is ~2.0, well outside this box for mu.
        params, _c, _f, _n = mathops.fit_model(
            x,
            y,
            "gaussian",
            p0=[5.0, 2.0, 1.5, 1.0],
            bounds=([0.0, -9.0, 0.1, -1.0], [10.0, -5.0, 5.0, 5.0]),
        )
        assert -9.0 <= params[1] <= -5.0

    def test_scalar_bounds_broadcast(self):
        """A scalar bound applies to every parameter."""
        x = np.linspace(-5.0, 5.0, 200)
        y = _noisy("linear", [2.5, -1.25], x, 0.3, seed=25)
        params, _c, _f, _n = mathops.fit_model(x, y, "linear", bounds=(-100.0, 100.0))
        assert np.all(np.abs(params) <= 100.0)

    def test_inverted_bounds_are_rejected(self):
        """A lower bound above its upper bound is caught here, not deep in scipy."""
        x = np.linspace(-5.0, 5.0, 200)
        y = _noisy("linear", [2.5, -1.25], x, 0.3, seed=26)
        with pytest.raises(ValueError, match="strictly below"):
            mathops.fit_model(x, y, "linear", bounds=([5.0, 5.0], [1.0, 1.0]))


class TestFitModelRegistry:
    """The MODELS registry and custom_model."""

    def test_registry_entries_are_fn_names_guess_triples(self):
        """Every entry unpacks as (fn, param_names, guess) and names a formula."""
        for name in mathops.MODEL_NAMES:
            spec = mathops.MODELS[name]
            fn, param_names, guess = spec.fn, spec.param_names, spec.guess
            assert callable(fn) and callable(guess)
            assert param_names and all(isinstance(p, str) for p in param_names)
            assert spec.label and spec.formula

    def test_model_names_covers_the_registry(self):
        """MODEL_NAMES is what a UI iterates; it must not drift from MODELS."""
        assert set(mathops.MODEL_NAMES) == set(mathops.MODELS)
        assert {"gaussian", "lorentzian", "exp_decay", "power_law", "sigmoid", "linear"} <= set(
            mathops.MODEL_NAMES
        )

    def test_resolve_model_passes_a_fitmodel_through(self):
        """A FitModel resolves to itself, so custom models work everywhere a key does."""
        spec = mathops.custom_model("a*x + b")
        assert mathops.resolve_model(spec) is spec
        assert mathops.resolve_model("gaussian") is mathops.MODELS["gaussian"]

    def test_unknown_model_names_the_alternatives(self):
        """An unknown key must tell the user what the valid ones are."""
        with pytest.raises(ValueError, match="unknown fit model"):
            mathops.fit_model([1.0, 2.0], [1.0, 2.0], "quadratic")

    def test_custom_model_free_variables_become_parameters(self):
        """x is the domain; every other name is fitted, sorted."""
        spec = mathops.custom_model("c + a*sin(b*x)")
        assert spec.param_names == ("a", "b", "c")

    def test_custom_model_without_free_parameters_is_rejected(self):
        """There is nothing to fit in an expression of x alone."""
        with pytest.raises(ValueError, match="no free parameters"):
            mathops.custom_model("2*x + 1")

    def test_custom_model_rejects_a_hostile_expression(self):
        """The expressions validator is the gate; fitting must not widen it."""
        with pytest.raises(ValueError):
            mathops.custom_model("__import__('os').system('echo hi')")

    def test_custom_model_rejects_an_empty_expression(self):
        """An empty box gets a sentence telling the user what to type."""
        with pytest.raises(ValueError, match="expression in x"):
            mathops.custom_model("   ")

    def test_custom_model_caps_the_parameter_count(self):
        """A user can type free variables forever; a fit of 20 of them is not a fit."""
        expr = "+".join(f"p{i}*x" for i in range(15))
        with pytest.raises(ValueError, match="free parameters"):
            mathops.custom_model(expr)


class TestMultipeakModel:
    """Test the deconvolution model: a sum of N peaks fit to overlapping data."""

    @staticmethod
    def _overlap(x, truth, shape="gaussian", baseline=0.5):
        out = np.full(x.shape, baseline, dtype=float)
        for a, mu, w in truth:
            if shape == "gaussian":
                out = out + a * np.exp(-((x - mu) ** 2) / (2.0 * w * w))
            else:
                out = out + a * w * w / ((x - mu) ** 2 + w * w)
        return out

    def test_parameter_names_scale_with_peak_count(self):
        """A sum of N peaks has 3N + 1 parameters, grouped a_i, mu_i, width_i, then c."""
        spec = mathops.multipeak_model(3, "gaussian")
        assert spec.param_names == (
            "a1",
            "mu1",
            "sigma1",
            "a2",
            "mu2",
            "sigma2",
            "a3",
            "mu3",
            "sigma3",
            "c",
        )
        assert mathops.multipeak_model(2, "lorentzian").param_names[2] == "gamma1"

    def test_resolves_two_overlapping_gaussians(self):
        """Two Gaussians that never return to baseline between them are separated exactly.

        This is precisely the case find_peaks' valley integration cannot handle: fitting
        the whole envelope recovers each centre, width and amplitude.
        """
        x = np.linspace(0.0, 20.0, 2000)
        truth = [(5.0, 8.0, 0.8), (3.0, 11.0, 0.8)]
        y = self._overlap(x, truth)
        spec = mathops.multipeak_model(2, "gaussian")
        popt, _cov, y_fit, _names = mathops.fit_model(x, y, spec)
        stats = mathops.fit_statistics(y, y_fit, len(spec.param_names))
        assert stats["r_squared"] > 0.999
        # Peaks may come back in either order; compare the recovered centres as a set.
        centres = sorted([popt[1], popt[4]])
        assert centres[0] == pytest.approx(8.0, abs=0.05)
        assert centres[1] == pytest.approx(11.0, abs=0.05)

    def test_deconvolved_areas_are_analytic(self):
        """Each fitted peak's area matches a*sigma*sqrt(2*pi), even overlapping."""
        x = np.linspace(0.0, 20.0, 2000)
        truth = [(5.0, 8.0, 0.8), (3.0, 11.0, 0.8)]
        y = self._overlap(x, truth)
        spec = mathops.multipeak_model(2, "gaussian")
        popt, _cov, _yfit, _names = mathops.fit_model(x, y, spec)
        # Match each fitted peak to its truth by centre, then compare areas.
        fitted = sorted([(popt[1], popt[0], popt[2]), (popt[4], popt[3], popt[5])])
        for (_mu, a, w), (ta, _tmu, ts) in zip(fitted, truth):
            assert mathops.peak_area("gaussian", a, w) == pytest.approx(
                ta * ts * np.sqrt(2.0 * np.pi), rel=0.02
            )

    def test_lorentzian_area_formula(self):
        """A Lorentzian's area is a*gamma*pi."""
        assert mathops.peak_area("lorentzian", 2.0, 1.5) == pytest.approx(2.0 * 1.5 * np.pi)

    def test_peak_count_bounds(self):
        with pytest.raises(ValueError, match="between 1 and"):
            mathops.multipeak_model(0)
        with pytest.raises(ValueError, match="between 1 and"):
            mathops.multipeak_model(99)

    def test_unknown_shape_raises(self):
        with pytest.raises(ValueError, match="unknown peak shape"):
            mathops.multipeak_model(2, "voigt")
        with pytest.raises(ValueError, match="unknown peak shape"):
            mathops.peak_area("voigt", 1.0, 1.0)


class TestFitModelErrors:
    """Every failure is a message for the user, never a traceback."""

    def test_power_law_rejects_non_positive_x(self):
        """x^b is undefined at x <= 0; say so instead of returning nan."""
        with pytest.raises(ValueError, match="x > 0"):
            mathops.fit_model(np.linspace(-1.0, 5.0, 50), np.ones(50), "power_law")

    def test_too_few_samples_names_the_parameter_count(self):
        """Fewer points than parameters is under-determined, and the message says why."""
        with pytest.raises(ValueError, match="4 parameters"):
            mathops.fit_model([1.0, 2.0], [1.0, 2.0], "gaussian")

    def test_all_nan_y_is_rejected(self):
        """No finite pairs survive the mask, so there is nothing to fit."""
        with pytest.raises(ValueError, match="usable"):
            mathops.fit_model(np.arange(50.0), np.full(50, np.nan), "gaussian")

    def test_p0_of_the_wrong_length_names_the_parameters(self):
        """A wrong-length p0 tells the user which parameters the model takes."""
        x = np.linspace(-5.0, 5.0, 100)
        y = _noisy("gaussian", [1.0, 0.0, 1.0, 0.0], x, 0.05, seed=27)
        with pytest.raises(ValueError, match="a, mu, sigma, c"):
            mathops.fit_model(x, y, "gaussian", p0=[1.0, 2.0])

    def test_mismatched_sigma_length_is_rejected(self):
        """Per-point errors must line up with the points."""
        x = np.linspace(-5.0, 5.0, 100)
        y = _noisy("linear", [1.0, 0.0], x, 0.05, seed=28)
        with pytest.raises(ValueError, match="same length"):
            mathops.fit_model(x, y, "linear", sigma=np.ones(10))

    def test_non_positive_sigma_is_rejected(self):
        """A standard deviation of zero divides by zero; catch it with a sentence."""
        x = np.linspace(-5.0, 5.0, 100)
        y = _noisy("linear", [1.0, 0.0], x, 0.05, seed=29)
        with pytest.raises(ValueError, match="strictly positive"):
            mathops.fit_model(x, y, "linear", sigma=np.zeros(100))

    def test_scalar_sigma_broadcasts(self):
        """One error for every point is the common case and must be accepted."""
        x = np.linspace(-5.0, 5.0, 100)
        y = _noisy("linear", [1.0, 0.0], x, 0.05, seed=30)
        params, cov, _f, _n = mathops.fit_model(x, y, "linear", sigma=0.05)
        _assert_recovered(params, cov, [1.0, 0.0])

    def test_a_model_that_cannot_converge_reports_a_message(self):
        """A hopeless fit is a ValueError with advice, not a scipy RuntimeError."""
        x = np.linspace(0.0, 10.0, 30)
        rng = np.random.default_rng(31)
        y = rng.normal(0.0, 1.0, 30)
        try:
            mathops.fit_model(x, y, "gaussian", p0=[1e300, 1e300, 1e-300, 1e300])
        except ValueError:
            pass  # the documented outcome
        except Exception as exc:  # pragma: no cover - the thing being guarded against
            pytest.fail(f"leaked {type(exc).__name__}: {exc}")

    def test_models_do_not_overflow_to_inf(self):
        """exp() is clamped: an extreme parameter saturates rather than poisoning the fit."""
        x = np.linspace(-10.0, 10.0, 100)
        assert np.all(np.isfinite(mathops.MODELS["exp_decay"].fn(x, 1.0, 1e-8, 0.0)))
        assert np.all(np.isfinite(mathops.MODELS["sigmoid"].fn(x, 1.0, 1e9, 0.0, 0.0)))
        assert np.all(np.isfinite(mathops.MODELS["gaussian"].fn(x, 1.0, 0.0, 1e-9, 0.0)))

    def test_fit_available_reports_a_bool(self):
        """The UI asks this to decide whether to offer the feature at all."""
        assert isinstance(mathops.fit_available(), bool)

    def test_unavailable_scipy_degrades_to_a_message(self, monkeypatch):
        """curve_fit has no numpy fallback, so it must report, never raise something odd.

        Unlike every other scipy user in this module there is nothing to fall back *to*:
        an honest "unavailable" beats a worse answer. FitUnavailableError is a ValueError
        subclass precisely so a caller already handling this module's documented failure
        mode shows a message instead of crashing.
        """
        assert issubclass(mathops.FitUnavailableError, ValueError)
        assert "scipy" in mathops.FIT_UNAVAILABLE_MESSAGE

        x = np.linspace(-5.0, 5.0, 100)
        y = _noisy("linear", [1.0, 0.0], x, 0.05, seed=32)
        monkeypatch.setitem(sys.modules, "scipy.optimize", None)
        assert mathops.fit_available() is False
        with pytest.raises(mathops.FitUnavailableError, match="scipy"):
            mathops.fit_model(x, y, "linear")
        with pytest.raises(ValueError):  # the subclass relationship, in practice
            mathops.fit_model(x, y, "linear")

    def test_fit_works_again_once_scipy_is_importable(self):
        """The monkeypatched module above must not leak into the rest of the suite."""
        assert mathops.fit_available() is True
        x = np.linspace(-5.0, 5.0, 100)
        y = _noisy("linear", [1.0, 0.0], x, 0.05, seed=33)
        params, cov, _f, _n = mathops.fit_model(x, y, "linear")
        _assert_recovered(params, cov, [1.0, 0.0])


class TestFitPolynomialUnchanged:
    """fit_polynomial is correct and has callers; the new fitter must not disturb it."""

    def test_polynomial_still_recovers_its_coefficients(self):
        """The pre-existing linear-least-squares path is untouched."""
        x = np.linspace(-3.0, 3.0, 200)
        y = 2.0 * x**2 - 3.0 * x + 1.0
        coeffs, y_fit = mathops.fit_polynomial(x, y, 2)
        assert np.allclose(coeffs, [2.0, -3.0, 1.0])
        assert np.allclose(y_fit, y)


class TestFitPolynomialCovariance:
    """fit_polynomial_covariance: same fit as fit_polynomial, plus a usable covariance."""

    def test_coeffs_and_y_fit_match_fit_polynomial_exactly(self):
        rng = np.random.default_rng(30)
        x = np.linspace(-3.0, 3.0, 200)
        y = 2.0 * x**2 - 3.0 * x + 1.0 + rng.normal(0.0, 0.1, x.size)
        coeffs_a, y_fit_a = mathops.fit_polynomial(x, y, 2)
        coeffs_b, cov, y_fit_b = mathops.fit_polynomial_covariance(x, y, 2)
        assert np.array_equal(coeffs_a, coeffs_b)
        assert np.array_equal(y_fit_a, y_fit_b)
        assert cov.shape == (3, 3)

    def test_covariance_shrinks_with_more_data(self):
        """More samples at the same noise level must narrow the coefficient uncertainty."""
        rng = np.random.default_rng(31)

        def fit_at(n):
            x = np.linspace(-3.0, 3.0, n)
            y = 2.0 * x**2 - 3.0 * x + 1.0 + rng.normal(0.0, 0.2, n)
            _c, cov, _f = mathops.fit_polynomial_covariance(x, y, 2)
            return float(np.sqrt(cov[0, 0]))

        assert fit_at(2000) < fit_at(50)

    def test_exactly_determined_fit_has_infinite_covariance(self):
        """n points == n coefficients: zero residual degrees of freedom, no covariance."""
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([1.0, 3.0, 7.0])  # exactly fits a degree-2 polynomial
        _coeffs, cov, _y_fit = mathops.fit_polynomial_covariance(x, y, 2)
        assert np.all(np.isinf(cov))

    def test_high_degree_on_a_wide_domain_degrades_instead_of_raising(self):
        """REGRESSION: np.polyfit(cov=True) can raise on a rank-deficient Vandermonde
        matrix even with plenty of samples (a wide x range at high degree). The
        coefficients (an unconditional plain polyfit) must still come back; only the
        covariance degrades to all-inf."""
        x = np.linspace(0.0, 10.0, 128)
        y = np.sin(x)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            coeffs, cov, y_fit = mathops.fit_polynomial_covariance(x, y, 20)
        assert coeffs.shape == (21,)
        assert y_fit.shape == x.shape
        assert np.all(np.isfinite(coeffs))
        assert np.all(np.isinf(cov))

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            mathops.fit_polynomial_covariance(np.array([]), np.array([]), 1)


def _three_peaks(n=1500):
    """Three Gaussian peaks of heights 3, 1, 2 at x = 5, 15, 23 on a clean grid."""
    x = np.linspace(0.0, 30.0, n)

    def g(mu, a, s):
        return a * np.exp(-((x - mu) ** 2) / (2.0 * s * s))

    y = g(5.0, 3.0, 0.6) + g(15.0, 1.0, 0.5) + g(23.0, 2.0, 0.8)
    return x, y


class TestFindPeaks:
    """Test find_peaks() locates maxima and honours every filter (scipy path)."""

    def test_finds_the_three_peaks(self):
        """Three well-separated Gaussians yield exactly three peaks, at their centres.

        The grid spacing is 30/1499 ~ 0.02, so a centre recovered to +/-0.05 is within
        a few samples -- the most a discrete maximum can be off by here.
        """
        x, y = _three_peaks()
        px, py, props = mathops.find_peaks(x, y, prominence=0.3)
        assert px.size == 3
        assert np.allclose(px, [5.0, 15.0, 23.0], atol=0.05)
        assert np.allclose(py, [3.0, 1.0, 2.0], atol=0.02)
        assert props["prominences"].shape == (3,)
        assert props["widths"].shape == (3,)

    def test_prominence_rejects_the_smallest_peak(self):
        """A prominence floor above the shortest peak drops it and keeps the tall two."""
        x, y = _three_peaks()
        px, _py, _props = mathops.find_peaks(x, y, prominence=1.5)
        assert np.allclose(px, [5.0, 23.0], atol=0.05)

    def test_ascending_x_order_regardless_of_input(self):
        """Peaks come back sorted by x even when the input is shuffled."""
        x, y = _three_peaks()
        rng = np.random.default_rng(0)
        order = rng.permutation(x.size)
        px, _py, _props = mathops.find_peaks(x[order], y[order], prominence=0.3)
        assert np.all(np.diff(px) > 0.0)
        assert np.allclose(px, [5.0, 15.0, 23.0], atol=0.05)

    def test_distance_drops_the_shorter_of_two_close_peaks(self):
        """With a big minimum separation only the tallest peak survives."""
        x, y = _three_peaks()
        px, _py, _props = mathops.find_peaks(x, y, distance=x.size)
        assert px.size == 1
        assert np.allclose(px, [5.0], atol=0.05)

    def test_no_peaks_returns_empty_aligned_arrays(self):
        """A monotone ramp has no interior maximum: everything comes back empty."""
        x = np.linspace(0.0, 1.0, 100)
        px, py, props = mathops.find_peaks(x, x)
        assert px.size == 0 and py.size == 0
        assert props["prominences"].size == 0
        assert props["widths"].size == 0

    def test_flat_top_counts_once(self):
        """A plateau is a single peak (at its middle), not zero and not one per sample."""
        y = np.array([0, 1, 2, 3, 3, 3, 2, 1, 0], dtype=float)
        x = np.arange(y.size, dtype=float)
        px, _py, _props = mathops.find_peaks(x, y)
        assert px.size == 1
        assert px[0] == 4.0  # (3 + 5) // 2

    def test_width_measured_in_x_units(self):
        """A Gaussian's FWHM is 2.355*sigma; find_peaks reports the width in x, not samples."""
        x = np.linspace(-10.0, 10.0, 4001)
        sigma = 1.3
        y = np.exp(-(x**2) / (2.0 * sigma * sigma))
        _px, _py, props = mathops.find_peaks(x, y, prominence=0.5)
        fwhm = 2.354820045 * sigma
        assert np.allclose(props["widths"][0], fwhm, rtol=0.02)

    @pytest.mark.parametrize("bad", [0, -1, -5])
    def test_distance_below_one_raises(self, bad):
        """A distance of zero or negative is a caller error, not a clamp."""
        x, y = _three_peaks(100)
        with pytest.raises(ValueError, match="distance"):
            mathops.find_peaks(x, y, distance=bad)


class TestFindPeaksNumpyFallback:
    """Test the pure-numpy peak finder still honours every filter without scipy."""

    def test_fallback_matches_scipy_on_clean_peaks(self, monkeypatch):
        """Without scipy the numpy path finds the same three peak locations."""
        x, y = _three_peaks()
        reference, _py, _props = mathops.find_peaks(x, y, prominence=0.3)
        monkeypatch.setitem(sys.modules, "scipy.signal", None)
        px, _py2, props = mathops.find_peaks(x, y, prominence=0.3)
        assert np.allclose(px, reference, atol=0.05)
        assert props["prominences"].size == px.size

    def test_fallback_prominence_filter_applies(self, monkeypatch):
        """The prominence threshold is enforced by the fallback, not silently ignored."""
        monkeypatch.setitem(sys.modules, "scipy.signal", None)
        x, y = _three_peaks()
        px, _py, _props = mathops.find_peaks(x, y, prominence=1.5)
        assert np.allclose(px, [5.0, 23.0], atol=0.05)

    def test_fallback_distance_filter_applies(self, monkeypatch):
        """The greedy distance rule keeps the tallest peak in a wide neighbourhood."""
        monkeypatch.setitem(sys.modules, "scipy.signal", None)
        x, y = _three_peaks()
        px, _py, _props = mathops.find_peaks(x, y, distance=x.size)
        assert np.allclose(px, [5.0], atol=0.05)

    def test_fallback_width_close_to_scipy(self, monkeypatch):
        """The numpy half-prominence width tracks scipy's within a few percent."""
        x = np.linspace(-10.0, 10.0, 4001)
        y = np.exp(-(x**2) / (2.0 * 1.3**2))
        _px, _py, ref = mathops.find_peaks(x, y, prominence=0.5)
        monkeypatch.setitem(sys.modules, "scipy.signal", None)
        _px2, _py2, got = mathops.find_peaks(x, y, prominence=0.5)
        assert np.allclose(got["widths"][0], ref["widths"][0], rtol=0.05)


class TestPeakAreas:
    """Test the valley-to-valley peak area reported by find_peaks()."""

    def test_isolated_gaussian_area_is_analytic(self):
        """An isolated Gaussian's area equals a*sigma*sqrt(2*pi) to integration accuracy."""
        x = np.linspace(-20.0, 20.0, 8000)
        a, sigma = 3.0, 1.5
        y = a * np.exp(-(x**2) / (2.0 * sigma**2))
        _px, _py, props = mathops.find_peaks(x, y, prominence=1.0)
        assert props["areas"].shape == (1,)
        assert props["areas"][0] == pytest.approx(a * sigma * np.sqrt(2.0 * np.pi), rel=1e-3)

    def test_separated_peaks_recover_their_own_areas(self):
        """Two baseline-separated peaks each recover their analytic area independently."""
        x = np.linspace(0.0, 20.0, 4000)

        def g(mu, a, s):
            return a * np.exp(-((x - mu) ** 2) / (2.0 * s * s))

        y = g(5.0, 4.0, 0.4) + g(15.0, 2.0, 0.4)
        _px, _py, props = mathops.find_peaks(x, y, prominence=0.5)
        sqrt2pi = np.sqrt(2.0 * np.pi)
        assert np.allclose(props["areas"], [4.0 * 0.4 * sqrt2pi, 2.0 * 0.4 * sqrt2pi], rtol=0.02)

    def test_area_is_baseline_corrected_on_a_slope(self):
        """A peak on a gentle linear background is measured above that background.

        The valley-to-valley baseline cancels the background as long as the peak stands
        well above the local background variation (the normal quantification regime); a
        gentle 0.05x slope leaves the area within ~1% of the flat-background value.
        """
        x = np.linspace(0.0, 20.0, 4000)
        peak = 3.0 * np.exp(-((x - 10.0) ** 2) / (2.0 * 0.5**2))
        flat_area = mathops.find_peaks(x, peak, prominence=1.0)[2]["areas"][0]
        sloped_area = mathops.find_peaks(x, peak + 0.05 * x + 0.5, prominence=1.0)[2]["areas"][0]
        # The added straight background contributes almost no net area after correction.
        assert sloped_area == pytest.approx(flat_area, rel=0.03)

    def test_fallback_area_matches_scipy(self, monkeypatch):
        """The pure-numpy path computes the same area as the scipy path."""
        x = np.linspace(-10.0, 10.0, 4000)
        y = 2.5 * np.exp(-(x**2) / (2.0 * 0.9**2))
        ref = mathops.find_peaks(x, y, prominence=0.5)[2]["areas"][0]
        monkeypatch.setitem(sys.modules, "scipy.signal", None)
        got = mathops.find_peaks(x, y, prominence=0.5)[2]["areas"][0]
        assert got == pytest.approx(ref, rel=1e-6)

    def test_no_peaks_gives_empty_areas(self):
        """The areas array is present and empty when nothing is detected."""
        x = np.linspace(0.0, 1.0, 100)
        _px, _py, props = mathops.find_peaks(x, x)
        assert props["areas"].size == 0


class TestButterFilter:
    """Test butter_filter(): zero phase, real length, and frequency selectivity."""

    def test_lowpass_kills_the_fast_tone_keeps_the_slow(self):
        """A slow + fast sum, lowpassed between the two, keeps the slow tone intact.

        The 20 Hz-equivalent tone should survive with near-unit amplitude while the 200
        tone is knocked down by more than 20 dB (a factor of 10 in amplitude).
        """
        x = np.linspace(0.0, 1.0, 2000, endpoint=False)
        slow = np.sin(2.0 * np.pi * 5.0 * x)
        fast = np.sin(2.0 * np.pi * 200.0 * x)
        out = mathops.butter_filter(x, slow + fast, 40.0, btype="lowpass", order=4)
        # Compare against the pure slow tone after the same transient region is trimmed.
        core = slice(200, -200)
        assert np.corrcoef(out[core], slow[core])[0, 1] > 0.999
        assert np.std(out[core] - slow[core]) < 0.1

    def test_zero_phase_does_not_shift_a_peak(self):
        """filtfilt is symmetric, so a lowpassed pulse keeps its peak position."""
        x = np.linspace(-10.0, 10.0, 2001)
        y = np.exp(-(x**2) / (2.0 * 0.5**2))
        out = mathops.butter_filter(x, y, 2.0, btype="lowpass", order=4)
        assert abs(x[int(np.argmax(out))] - x[int(np.argmax(y))]) < 0.05

    def test_highpass_removes_a_dc_offset(self):
        """A constant plus a tone, highpassed, loses the constant."""
        x = np.linspace(0.0, 1.0, 2000, endpoint=False)
        y = 7.0 + np.sin(2.0 * np.pi * 50.0 * x)
        out = mathops.butter_filter(x, y, 10.0, btype="highpass", order=4)
        assert abs(float(np.mean(out[100:-100]))) < 0.05

    def test_bandpass_needs_a_pair(self):
        """A single cutoff for a bandpass is a caller error."""
        x = np.linspace(0.0, 1.0, 500, endpoint=False)
        with pytest.raises(ValueError, match="pair"):
            mathops.butter_filter(x, np.sin(x), 5.0, btype="bandpass")

    def test_bandpass_low_below_high(self):
        """A band whose low cutoff is not below its high is rejected."""
        x = np.linspace(0.0, 1.0, 500, endpoint=False)
        with pytest.raises(ValueError, match="below"):
            mathops.butter_filter(x, np.sin(x), (50.0, 20.0), btype="bandpass")

    def test_length_preserved_on_nonuniform_grid(self):
        """A non-uniform grid is resampled internally but the output length matches x."""
        rng = np.random.default_rng(3)
        x = np.sort(rng.uniform(0.0, 1.0, 800))
        y = np.sin(2.0 * np.pi * 5.0 * x)
        out = mathops.butter_filter(x, y, 20.0, btype="lowpass")
        assert out.shape == (800,)
        assert np.all(np.isfinite(out))

    def test_unknown_type_raises(self):
        x = np.linspace(0.0, 1.0, 100, endpoint=False)
        with pytest.raises(ValueError, match="unknown filter type"):
            mathops.butter_filter(x, x, 5.0, btype="allpass")

    def test_unavailable_without_scipy(self, monkeypatch):
        """No scipy.signal means a clean FilterUnavailableError, a ValueError subclass."""
        monkeypatch.setitem(sys.modules, "scipy.signal", None)
        assert not mathops.filter_available()
        x = np.linspace(0.0, 1.0, 200, endpoint=False)
        with pytest.raises(mathops.FilterUnavailableError):
            mathops.butter_filter(x, np.sin(x), 5.0)
        assert issubclass(mathops.FilterUnavailableError, ValueError)


class TestIirFilter:
    """Test iir_filter(): the 4-family generalization butter_filter now wraps."""

    def test_butter_family_matches_butter_filter_exactly(self):
        """family="butter" must be bit-for-bit what the pre-existing function returns --
        it is the same code path (scipy.signal.butter is itself a thin iirfilter call)."""
        rng = np.random.default_rng(0)
        x = np.linspace(0.0, 1.0, 500, endpoint=False)
        y = np.sin(2.0 * np.pi * 5.0 * x) + rng.normal(0.0, 0.1, x.size)
        old = mathops.butter_filter(x, y, 20.0, btype="lowpass", order=4)
        new = mathops.iir_filter(x, y, 20.0, btype="lowpass", family="butter", order=4)
        np.testing.assert_array_equal(old, new)

    @pytest.mark.parametrize("family", mathops.IIR_FAMILIES)
    def test_every_family_matches_scipy_reference(self, family):
        """Each family's output must match calling scipy directly with the same args."""
        stats = pytest.importorskip("scipy.signal")
        x = np.linspace(0.0, 1.0, 800, endpoint=False)
        rng = np.random.default_rng(1)
        y = np.sin(2.0 * np.pi * 5.0 * x) + rng.normal(0.0, 0.05, x.size)

        out = mathops.iir_filter(x, y, 20.0, btype="lowpass", family=family, order=4)

        fs = (x.size - 1) / float(x[-1] - x[0])
        nyquist = 0.5 * fs
        wn = 20.0 / nyquist
        rp = 1.0 if family == "cheby1" else None
        rs = 40.0 if family == "cheby2" else None
        sos = stats.iirfilter(4, wn, rp=rp, rs=rs, btype="lowpass", ftype=family, output="sos")
        expected = stats.sosfiltfilt(sos, y)
        np.testing.assert_allclose(out, expected, atol=1e-9)

    def test_every_family_attenuates_a_fast_tone_in_lowpass(self):
        """A coarse sanity check applicable to all 4 families: whatever their ripple/
        phase trade-offs, a lowpass must still knock down a tone well above cutoff."""
        x = np.linspace(0.0, 1.0, 2000, endpoint=False)
        slow = np.sin(2.0 * np.pi * 5.0 * x)
        fast = np.sin(2.0 * np.pi * 200.0 * x)
        for family in mathops.IIR_FAMILIES:
            out = mathops.iir_filter(x, slow + fast, 40.0, btype="lowpass", family=family, order=4)
            core = slice(300, -300)
            assert np.corrcoef(out[core], slow[core])[0, 1] > 0.99, family

    def test_ripple_and_attenuation_actually_change_the_response(self):
        """cheby1's ripple / cheby2's attenuation are not inert -- a larger value must
        produce a measurably different filtered output at the same order/cutoff."""
        x = np.linspace(0.0, 1.0, 800, endpoint=False)
        rng = np.random.default_rng(2)
        y = np.sin(2.0 * np.pi * 5.0 * x) + rng.normal(0.0, 0.05, x.size)

        cheby1_small = mathops.iir_filter(
            x, y, 20.0, btype="lowpass", family="cheby1", order=4, ripple=0.1
        )
        cheby1_large = mathops.iir_filter(
            x, y, 20.0, btype="lowpass", family="cheby1", order=4, ripple=5.0
        )
        assert not np.allclose(cheby1_small, cheby1_large)

        cheby2_small = mathops.iir_filter(
            x, y, 20.0, btype="lowpass", family="cheby2", order=4, attenuation=20.0
        )
        cheby2_large = mathops.iir_filter(
            x, y, 20.0, btype="lowpass", family="cheby2", order=4, attenuation=60.0
        )
        assert not np.allclose(cheby2_small, cheby2_large)

    def test_unknown_family_raises(self):
        x = np.linspace(0.0, 1.0, 200, endpoint=False)
        with pytest.raises(ValueError, match="unknown filter family"):
            mathops.iir_filter(x, np.sin(x), 5.0, family="elliptic")

    def test_unavailable_without_scipy(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "scipy.signal", None)
        x = np.linspace(0.0, 1.0, 200, endpoint=False)
        with pytest.raises(mathops.FilterUnavailableError):
            mathops.iir_filter(x, np.sin(x), 5.0, family="cheby1")


class TestBaselineAsls:
    """Test the asymmetric-least-squares curved baseline estimator."""

    @staticmethod
    def _peaks_on_curve(n=1000):
        """Two narrow peaks riding on a broad curved background plus an offset."""
        x = np.linspace(0.0, 20.0, n)

        def g(a, mu, s):
            return a * np.exp(-((x - mu) ** 2) / (2.0 * s * s))

        peaks = g(5.0, 6.0, 0.4) + g(4.0, 13.0, 0.4)
        background = 3.0 * np.exp(-((x - 10.0) ** 2) / 40.0) + 0.5
        return x, peaks, background

    def test_baseline_tracks_a_curved_background(self):
        """Off the peaks (and away from the edges), the baseline follows the background.

        The interior mask matters: AsLS, like any penalised smoother, has a weaker second-
        difference constraint at the two ends and drifts there -- a known edge effect, not
        what this test is about.
        """
        x, peaks, background = self._peaks_on_curve()
        baseline = mathops.baseline_asls(peaks + background, lam=1e6, p=0.01)
        interior = np.zeros(x.size, dtype=bool)
        interior[x.size // 10 : -x.size // 10] = True
        off_peak = (np.abs(peaks) < 0.05) & interior
        assert np.max(np.abs(baseline[off_peak] - background[off_peak])) < 0.3

    def test_baseline_stays_below_the_peaks(self):
        """With a small asymmetry the baseline slides under the peaks, not through them."""
        x, peaks, background = self._peaks_on_curve()
        y = peaks + background
        baseline = mathops.baseline_asls(y, lam=1e6, p=0.01)
        # The baseline must not rise above the signal at the peak tops.
        assert baseline[np.argmin(np.abs(x - 6.0))] < y[np.argmin(np.abs(x - 6.0))]
        # Subtracting it recovers most of each peak's height.
        corrected = y - baseline
        assert corrected[np.argmin(np.abs(x - 6.0))] > 0.8 * 5.0

    def test_flat_signal_gives_flat_baseline(self):
        """A constant signal's baseline is that constant."""
        baseline = mathops.baseline_asls(np.full(200, 7.0), lam=1e5)
        assert np.allclose(baseline, 7.0, atol=1e-6)

    def test_short_signal_returns_mean(self):
        """Too short for a second difference: the baseline is the mean."""
        baseline = mathops.baseline_asls(np.array([2.0, 6.0]))
        assert np.allclose(baseline, 4.0)

    def test_returns_baseline_not_corrected_signal(self):
        """The function returns the baseline itself, leaving subtraction to the caller."""
        x, peaks, background = self._peaks_on_curve()
        baseline = mathops.baseline_asls(peaks + background, lam=1e6)
        # A baseline, not a flattened signal: it is close to the background, well above 0.
        assert float(np.mean(baseline)) > 0.4

    @pytest.mark.parametrize("kwargs", [{"p": 0.0}, {"p": 1.0}, {"lam": -1.0}, {"niter": 0}])
    def test_invalid_parameters_raise(self, kwargs):
        with pytest.raises(ValueError):
            mathops.baseline_asls(np.linspace(0.0, 1.0, 50), **kwargs)

    def test_unavailable_without_scipy_sparse(self, monkeypatch):
        """No scipy.sparse means a clean BaselineUnavailableError, a ValueError subclass."""
        monkeypatch.setitem(sys.modules, "scipy.sparse", None)
        monkeypatch.setitem(sys.modules, "scipy.sparse.linalg", None)
        assert not mathops.baseline_available()
        with pytest.raises(mathops.BaselineUnavailableError):
            mathops.baseline_asls(np.linspace(0.0, 1.0, 50))
        assert issubclass(mathops.BaselineUnavailableError, ValueError)


class TestDetrend:
    """Test detrend() removes constant and linear baselines."""

    def test_linear_flattens_a_ramp_plus_signal(self):
        """Subtracting the fitted line leaves the zero-mean tone about a flat baseline."""
        x = np.linspace(0.0, 10.0, 500)
        tone = np.sin(x)
        out = mathops.detrend(x, 3.0 * x + 2.0 + tone, kind="linear")
        assert abs(float(np.mean(out))) < 1e-9
        # The recovered residual is the tone minus its own linear part (sin over a finite
        # window is not orthogonal to a line), so it tracks the tone very closely but not
        # perfectly -- the small shortfall is the removed slope, not an error.
        assert np.corrcoef(out, tone)[0, 1] > 0.99

    def test_constant_subtracts_the_mean(self):
        x = np.linspace(0.0, 1.0, 100)
        y = np.linspace(0.0, 1.0, 100) + 5.0
        out = mathops.detrend(x, y, kind="constant")
        assert abs(float(np.mean(out))) < 1e-12
        # Slope untouched: only the offset moved.
        assert np.allclose(np.diff(out), np.diff(y))

    def test_nan_safe(self):
        """A lone nan does not poison the fit and stays nan in the output."""
        x = np.linspace(0.0, 1.0, 50)
        y = 2.0 * x + 1.0
        y[10] = np.nan
        out = mathops.detrend(x, y, kind="linear")
        assert np.isnan(out[10])
        good = np.isfinite(out)
        assert np.allclose(out[good], 0.0, atol=1e-9)

    def test_unknown_kind_raises(self):
        x = np.linspace(0.0, 1.0, 10)
        with pytest.raises(ValueError, match="unknown detrend kind"):
            mathops.detrend(x, x, kind="quadratic")


class TestHistogram:
    """Test histogram() binning, density normalisation and clamping."""

    def test_counts_sum_to_finite_sample_count(self):
        """Every finite sample lands in exactly one bin, so counts sum to n."""
        rng = np.random.default_rng(0)
        y = rng.normal(0.0, 1.0, 1000)
        edges, heights, centers = mathops.histogram(y, bins=20)
        assert heights.sum() == 1000
        assert edges.size == heights.size + 1
        assert centers.size == heights.size
        assert np.allclose(centers, 0.5 * (edges[:-1] + edges[1:]))

    def test_density_integrates_to_one(self):
        """With density=True the bars integrate (height*width) to 1."""
        rng = np.random.default_rng(1)
        y = rng.normal(0.0, 2.0, 5000)
        edges, heights, _centers = mathops.histogram(y, bins=40, density=True)
        assert np.isclose(float(np.sum(heights * np.diff(edges))), 1.0)

    def test_non_finite_dropped(self):
        """nan and inf are excluded from the binning, not counted or crashed on."""
        y = np.array([1.0, 2.0, np.nan, 3.0, np.inf, 2.0])
        _edges, heights, _centers = mathops.histogram(y, bins=4)
        assert heights.sum() == 4

    def test_bins_clamped(self):
        """A pathological bin count is clamped, not honoured or rejected."""
        y = np.linspace(0.0, 1.0, 100)
        _edges, heights, _centers = mathops.histogram(y, bins=10**9)
        assert heights.size == 10000

    def test_all_nan_raises(self):
        with pytest.raises(ValueError, match="at least one finite"):
            mathops.histogram(np.array([np.nan, np.inf]))


class TestFitDistribution:
    """Test fit_distribution() recovers parameters and scores the fit."""

    def test_normal_recovers_mu_and_sigma(self):
        """A normal sample is fit back to its generating mean and standard deviation."""
        rng = np.random.default_rng(2)
        y = rng.normal(3.0, 1.5, 8000)
        _edges, _heights, centers = mathops.histogram(y, bins=40)
        res = mathops.fit_distribution(y, "normal", centers)
        assert res["labels"] == ("mu", "sigma")
        mu, sigma = res["params"]
        assert mu == pytest.approx(3.0, abs=0.1)
        assert sigma == pytest.approx(1.5, abs=0.1)
        assert res["pdf"].shape == centers.shape

    def test_ks_pvalue_high_for_the_true_distribution(self):
        """The KS test does not reject a normal fit to normal data."""
        rng = np.random.default_rng(3)
        y = rng.normal(0.0, 1.0, 4000)
        res = mathops.fit_distribution(y, "normal", np.linspace(-4.0, 4.0, 100))
        assert res["ks_pvalue"] > 0.05
        assert res["ks_statistic"] < 0.05

    def test_ks_pvalue_low_for_the_wrong_distribution(self):
        """Fitting an exponential to clearly-normal data is rejected by the KS test."""
        rng = np.random.default_rng(4)
        y = rng.normal(10.0, 1.0, 4000)
        res = mathops.fit_distribution(y, "exponential", np.linspace(0.0, 20.0, 100))
        assert res["ks_pvalue"] < 0.01

    def test_gamma_on_gamma_data(self):
        """A positive-support law recovers a good fit on matching data."""
        rng = np.random.default_rng(5)
        y = rng.gamma(2.0, 1.5, 4000)
        _e, _h, centers = mathops.histogram(y, bins=40)
        res = mathops.fit_distribution(y, "gamma", centers)
        assert res["labels"] == ("shape", "loc", "scale")
        assert res["ks_pvalue"] > 0.05

    def test_unknown_distribution_raises(self):
        with pytest.raises(ValueError, match="unknown distribution"):
            mathops.fit_distribution(np.arange(10.0), "cauchy_lorentz", np.arange(10.0))

    def test_too_few_samples_raises(self):
        with pytest.raises(ValueError, match="at least two"):
            mathops.fit_distribution(np.array([1.0]), "normal", np.arange(3.0))

    def test_unavailable_without_scipy(self, monkeypatch):
        """No scipy.stats means a clean DistributionUnavailableError, a ValueError."""
        monkeypatch.setitem(sys.modules, "scipy", None)
        monkeypatch.setitem(sys.modules, "scipy.stats", None)
        assert not mathops.distributions_available()
        with pytest.raises(mathops.DistributionUnavailableError):
            mathops.fit_distribution(np.arange(10.0), "normal", np.arange(10.0))
        assert issubclass(mathops.DistributionUnavailableError, ValueError)


class TestFitDistributionNewFamilies:
    """weibull/chi2/beta/uniform: the histogram tab's newer distribution options."""

    def test_weibull_on_weibull_data(self):
        rng = np.random.default_rng(10)
        y = rng.weibull(2.0, 4000) * 3.0
        _e, _h, centers = mathops.histogram(y, bins=40)
        res = mathops.fit_distribution(y, "weibull", centers)
        assert res["labels"] == ("shape", "loc", "scale")
        assert res["ks_pvalue"] > 0.05
        shape, _loc, scale = res["params"]
        assert shape == pytest.approx(2.0, rel=0.2)
        assert scale == pytest.approx(3.0, rel=0.2)

    def test_chi2_on_chi2_data(self):
        rng = np.random.default_rng(11)
        y = rng.chisquare(4.0, 4000)
        _e, _h, centers = mathops.histogram(y, bins=40)
        res = mathops.fit_distribution(y, "chi2", centers)
        assert res["labels"] == ("df", "loc", "scale")
        assert res["ks_pvalue"] > 0.05

    def test_beta_on_beta_data(self):
        rng = np.random.default_rng(12)
        y = rng.beta(2.0, 5.0, 4000)
        _e, _h, centers = mathops.histogram(y, bins=40)
        res = mathops.fit_distribution(y, "beta", centers)
        assert res["labels"] == ("a", "b", "loc", "scale")
        assert res["ks_pvalue"] > 0.05

    def test_uniform_on_uniform_data(self):
        rng = np.random.default_rng(13)
        y = rng.uniform(2.0, 9.0, 4000)
        _e, _h, centers = mathops.histogram(y, bins=40)
        res = mathops.fit_distribution(y, "uniform", centers)
        assert res["labels"] == ("loc", "scale")
        assert res["ks_pvalue"] > 0.05
        loc, scale = res["params"]
        assert loc == pytest.approx(2.0, abs=0.2)
        assert loc + scale == pytest.approx(9.0, abs=0.2)

    def test_uniform_rejects_clearly_normal_data(self):
        rng = np.random.default_rng(14)
        y = rng.normal(0.0, 1.0, 4000)
        res = mathops.fit_distribution(y, "uniform", np.linspace(-4.0, 4.0, 100))
        assert res["ks_pvalue"] < 0.01

    @pytest.mark.parametrize("name", ["weibull", "chi2", "beta", "uniform"])
    def test_appears_in_the_public_distributions_tuple(self, name):
        assert name in mathops.DISTRIBUTIONS


class TestAutocorrelation:
    """Test autocorrelation() normalisation, period recovery and equivalence."""

    def test_zero_lag_is_one(self):
        """A signal is perfectly correlated with itself at lag 0."""
        rng = np.random.default_rng(0)
        _lags, acf = mathops.autocorrelation(rng.standard_normal(500))
        assert acf[0] == pytest.approx(1.0)
        assert np.all(np.abs(acf) <= 1.0 + 1e-9)

    def test_recovers_a_known_period(self):
        """A sine of period 50 samples shows its first ACF peak at lag 50."""
        n = 2000
        x = np.arange(n, dtype=float)
        y = np.sin(2.0 * np.pi * x / 50.0)
        lags, acf = mathops.autocorrelation(y, max_lag=200)
        px, _py, _props = mathops.find_peaks(lags, acf, prominence=0.2)
        assert px.size >= 1
        assert px[0] == pytest.approx(50.0, abs=1.0)

    def test_matches_direct_correlation(self):
        """The FFT result equals the textbook O(n^2) np.correlate, to rounding."""
        rng = np.random.default_rng(1)
        y = rng.standard_normal(300)
        lags, acf = mathops.autocorrelation(y, max_lag=100)
        z = y - y.mean()
        ref = np.correlate(z, z, mode="full")[y.size - 1 :]
        ref = ref / ref[0]
        assert np.allclose(acf, ref[: lags.size], atol=1e-9)

    def test_constant_signal_decorrelates(self):
        """A constant has no variation, so every non-zero lag is exactly zero."""
        _lags, acf = mathops.autocorrelation(np.full(100, 7.0))
        assert acf[0] == pytest.approx(1.0)
        assert np.allclose(acf[1:], 0.0)

    def test_max_lag_truncates(self):
        """max_lag bounds the returned lags, and is clamped to the signal length."""
        y = np.sin(np.linspace(0.0, 20.0, 400))
        lags, acf = mathops.autocorrelation(y, max_lag=30)
        assert lags.size == 31  # lags 0..30
        assert acf.size == 31
        big_lags, _ = mathops.autocorrelation(y, max_lag=10_000)
        assert big_lags.size == 400  # clamped to len(y)

    def test_nan_interpolated_not_poisoned(self):
        """A lone nan is bridged, not allowed to make the whole ACF nan."""
        y = np.sin(np.linspace(0.0, 20.0, 400))
        y[100] = np.nan
        _lags, acf = mathops.autocorrelation(y, max_lag=50)
        assert np.all(np.isfinite(acf))

    def test_negative_max_lag_raises(self):
        with pytest.raises(ValueError, match="max_lag"):
            mathops.autocorrelation(np.arange(10.0), max_lag=-1)


class TestFitModelRobust:
    """fit_model_robust: same contract as fit_model, but resistant to bad points."""

    def test_return_shape_matches_fit_model(self):
        x = np.linspace(-5.0, 5.0, 100)
        y = _noisy("linear", [1.0, 0.0], x, 0.05, seed=40)
        params, cov, y_fit, names = mathops.fit_model_robust(x, y, "linear")
        assert params.shape == (2,)
        assert cov.shape == (2, 2)
        assert y_fit.shape == x.shape
        assert names == list(mathops.MODELS["linear"].param_names)

    def test_recovers_true_parameters_on_clean_data(self):
        """With no outliers, the robust fit should land close to the plain one."""
        x = np.linspace(-5.0, 5.0, 200)
        y = _noisy("linear", [2.0, -1.0], x, 0.05, seed=41)
        params, cov, _f, _n = mathops.fit_model_robust(x, y, "linear")
        _assert_recovered(params, cov, [2.0, -1.0], tol_sigma=6.0)

    def test_outperforms_plain_least_squares_with_outliers(self):
        """A handful of severe outliers pull the plain fit far off; robust barely moves.

        This is the entire point of the feature -- not "recovers within its own error
        bars" (a robust loss's error bars are themselves approximate) but "closer to
        ground truth than ordinary least squares on the same contaminated data".
        """
        rng = np.random.default_rng(42)
        x = np.linspace(0.0, 10.0, 200)
        truth = [2.0, 1.0]
        y = 2.0 * x + 1.0 + rng.normal(0.0, 0.2, x.size)
        # Contaminate 5% of points with gross outliers.
        bad = rng.choice(x.size, size=10, replace=False)
        y[bad] += rng.choice([-1.0, 1.0], size=10) * rng.uniform(15.0, 30.0, size=10)

        plain_params, _c1, _f1, _n1 = mathops.fit_model(x, y, "linear")
        robust_params, _c2, _f2, _n2 = mathops.fit_model_robust(x, y, "linear")

        plain_error = float(np.hypot(*(np.asarray(plain_params) - truth)))
        robust_error = float(np.hypot(*(np.asarray(robust_params) - truth)))
        assert robust_error < plain_error

    def test_unknown_loss_raises(self):
        x = np.linspace(-5.0, 5.0, 50)
        y = _noisy("linear", [1.0, 0.0], x, 0.05, seed=43)
        with pytest.raises(ValueError, match="loss"):
            mathops.fit_model_robust(x, y, "linear", loss="not_a_real_loss")

    def test_linear_loss_is_ordinary_least_squares(self):
        """loss="linear" is the escape hatch back to plain least squares, for comparison."""
        x = np.linspace(-5.0, 5.0, 200)
        y = _noisy("linear", [3.0, 2.0], x, 0.05, seed=44)
        plain_params, _c1, _f1, _n1 = mathops.fit_model(x, y, "linear")
        robust_linear_params, _c2, _f2, _n2 = mathops.fit_model_robust(x, y, "linear", loss="linear")
        assert np.allclose(plain_params, robust_linear_params, atol=1e-6)

    def test_respects_bounds(self):
        x = np.linspace(0.1, 10.0, 200)
        y = _noisy("power_law", [2.0, 1.5], x, 0.05, seed=45)
        params, _cov, _f, _n = mathops.fit_model_robust(
            x, y, "power_law", bounds=((0.0, 0.0), (1.0, 10.0))
        )
        assert 0.0 <= params[0] <= 1.0

    def test_unavailable_scipy_raises_fit_unavailable_error(self, monkeypatch):
        x = np.linspace(-5.0, 5.0, 50)
        y = _noisy("linear", [1.0, 0.0], x, 0.05, seed=46)
        monkeypatch.setitem(sys.modules, "scipy.optimize", None)
        with pytest.raises(mathops.FitUnavailableError, match="scipy"):
            mathops.fit_model_robust(x, y, "linear")

    def test_works_again_once_scipy_is_importable(self):
        """The monkeypatched module above must not leak into the rest of the suite."""
        x = np.linspace(-5.0, 5.0, 50)
        y = _noisy("linear", [1.0, 0.0], x, 0.05, seed=47)
        params, cov, _f, _n = mathops.fit_model_robust(x, y, "linear")
        _assert_recovered(params, cov, [1.0, 0.0], tol_sigma=6.0)


class TestConfidenceBand:
    """Test confidence_band(): delta-method propagation of a fit's covariance."""

    def _linear_fit(self, seed, n=300, noise=0.2):
        x = np.linspace(0.0, 10.0, n)
        y = _noisy("linear", [2.0, 1.0], x, noise, seed=seed)
        popt, pcov, _y_fit, _names = mathops.fit_model(x, y, "linear")
        return x, popt, pcov

    def test_band_straddles_the_fit(self):
        x, popt, pcov = self._linear_fit(seed=50)
        fn = mathops.MODELS["linear"].fn
        lo, hi = mathops.confidence_band(x, popt, pcov, fn, level=0.95)
        fitted = fn(x, *popt)
        assert np.all(lo <= fitted + 1e-9)
        assert np.all(hi >= fitted - 1e-9)

    def test_wider_level_gives_a_wider_band(self):
        x, popt, pcov = self._linear_fit(seed=51)
        fn = mathops.MODELS["linear"].fn
        lo68, hi68 = mathops.confidence_band(x, popt, pcov, fn, level=0.68)
        lo95, hi95 = mathops.confidence_band(x, popt, pcov, fn, level=0.95)
        lo99, hi99 = mathops.confidence_band(x, popt, pcov, fn, level=0.99)
        mid = x.size // 2
        w68 = hi68[mid] - lo68[mid]
        w95 = hi95[mid] - lo95[mid]
        w99 = hi99[mid] - lo99[mid]
        assert w68 < w95 < w99

    def test_band_narrows_with_more_data_at_the_same_noise(self):
        """More samples at the same noise level pin down the fit tighter."""
        _x_few, popt_few, pcov_few = self._linear_fit(seed=52, n=30)
        _x_many, popt_many, pcov_many = self._linear_fit(seed=52, n=3000)
        fn = mathops.MODELS["linear"].fn
        x_probe = np.array([5.0])
        lo_few, hi_few = mathops.confidence_band(x_probe, popt_few, pcov_few, fn)
        lo_many, hi_many = mathops.confidence_band(x_probe, popt_many, pcov_many, fn)
        assert (hi_many[0] - lo_many[0]) < (hi_few[0] - lo_few[0])

    def test_infinite_covariance_gives_an_all_nan_band(self):
        """An underdetermined fit's band must say "unknown", not draw a false-precision
        strip of some arbitrary width."""
        x = np.linspace(0.0, 10.0, 50)
        popt = np.array([1.0, 0.0])
        pcov = np.full((2, 2), np.inf)
        fn = mathops.MODELS["linear"].fn
        lo, hi = mathops.confidence_band(x, popt, pcov, fn)
        assert np.all(np.isnan(lo))
        assert np.all(np.isnan(hi))

    def test_works_with_a_nonlinear_model(self):
        x = np.linspace(-5.0, 5.0, 300)
        y = _noisy("gaussian", [5.0, 0.0, 1.5, 0.0], x, 0.1, seed=53)
        popt, pcov, _y_fit, _names = mathops.fit_model(x, y, "gaussian")
        fn = mathops.MODELS["gaussian"].fn
        lo, hi = mathops.confidence_band(x, popt, pcov, fn, level=0.95)
        assert lo.shape == x.shape
        assert hi.shape == x.shape
        assert np.all(hi >= lo)

    def test_works_with_a_polynomial_fit(self):
        """confidence_band is model-agnostic: a polynomial's fn takes *coeffs, not a
        fixed named-parameter signature, and that must work too."""
        rng = np.random.default_rng(54)
        x = np.linspace(-3.0, 3.0, 200)
        y = 2.0 * x**2 - 3.0 * x + 1.0 + rng.normal(0.0, 0.2, x.size)
        coeffs, cov, _y_fit = mathops.fit_polynomial_covariance(x, y, 2)
        lo, hi = mathops.confidence_band(x, coeffs, cov, lambda xx, *c: np.polyval(c, xx))
        assert np.all(hi >= lo)
        assert lo.shape == x.shape

    def test_level_out_of_range_raises(self):
        x, popt, pcov = self._linear_fit(seed=55)
        fn = mathops.MODELS["linear"].fn
        with pytest.raises(ValueError, match="level"):
            mathops.confidence_band(x, popt, pcov, fn, level=0.0)
        with pytest.raises(ValueError, match="level"):
            mathops.confidence_band(x, popt, pcov, fn, level=1.0)
        with pytest.raises(ValueError, match="level"):
            mathops.confidence_band(x, popt, pcov, fn, level=1.5)

    def test_falls_back_to_a_normal_quantile_without_scipy_stats(self, monkeypatch):
        """No scipy.stats: the multiplier falls back to a fixed table instead of raising --
        insurance, since fit_model already requires scipy to have produced popt/pcov."""
        x, popt, pcov = self._linear_fit(seed=56)
        fn = mathops.MODELS["linear"].fn
        lo_scipy, hi_scipy = mathops.confidence_band(x, popt, pcov, fn, level=0.95)

        monkeypatch.setitem(sys.modules, "scipy.stats", None)
        lo_fallback, hi_fallback = mathops.confidence_band(x, popt, pcov, fn, level=0.95)
        # Same z ~ 1.96 either way at this sample size (dof is large), so the two bands
        # should be very close -- not identical, since scipy.stats.t vs. the normal table
        # differ slightly at finite dof.
        assert np.allclose(lo_scipy, lo_fallback, rtol=0.02)
        assert np.allclose(hi_scipy, hi_fallback, rtol=0.02)

    def test_confidence_band_works_again_once_scipy_is_importable(self):
        """The monkeypatched module above must not leak into the rest of the suite."""
        x, popt, pcov = self._linear_fit(seed=57)
        fn = mathops.MODELS["linear"].fn
        lo, hi = mathops.confidence_band(x, popt, pcov, fn, level=0.95)
        assert np.all(np.isfinite(lo))
        assert np.all(np.isfinite(hi))


class TestConfidenceIntervalMean:
    """Test confidence_interval_mean(): Student's t interval, verified vs scipy.stats."""

    def test_matches_scipy_t_interval(self):
        stats = pytest.importorskip("scipy.stats")
        rng = np.random.default_rng(60)
        y = rng.normal(5.0, 2.0, 200)
        res = mathops.confidence_interval_mean(y, level=0.95)
        mean, sem = np.mean(y), stats.sem(y)
        lo, hi = stats.t.interval(0.95, y.size - 1, loc=mean, scale=sem)
        assert res["lower"] == pytest.approx(lo)
        assert res["upper"] == pytest.approx(hi)
        assert res["mean"] == pytest.approx(mean)
        assert res["n"] == float(y.size)

    def test_interval_straddles_the_mean(self):
        rng = np.random.default_rng(61)
        y = rng.normal(0.0, 1.0, 500)
        res = mathops.confidence_interval_mean(y, level=0.95)
        assert res["lower"] < res["mean"] < res["upper"]

    def test_wider_level_gives_a_wider_interval(self):
        rng = np.random.default_rng(62)
        y = rng.normal(0.0, 1.0, 200)
        r68 = mathops.confidence_interval_mean(y, level=0.68)
        r95 = mathops.confidence_interval_mean(y, level=0.95)
        r99 = mathops.confidence_interval_mean(y, level=0.99)
        w68 = r68["upper"] - r68["lower"]
        w95 = r95["upper"] - r95["lower"]
        w99 = r99["upper"] - r99["lower"]
        assert w68 < w95 < w99

    def test_more_samples_narrows_the_interval_at_the_same_spread(self):
        rng = np.random.default_rng(63)
        few = rng.normal(0.0, 1.0, 20)
        many = rng.normal(0.0, 1.0, 20000)
        r_few = mathops.confidence_interval_mean(few, level=0.95)
        r_many = mathops.confidence_interval_mean(many, level=0.95)
        assert (r_many["upper"] - r_many["lower"]) < (r_few["upper"] - r_few["lower"])

    def test_non_finite_entries_are_dropped(self):
        y = np.array([1.0, 2.0, 3.0, np.nan, np.inf, -np.inf, 4.0, 5.0])
        res = mathops.confidence_interval_mean(y, level=0.95)
        assert res["n"] == 5.0
        assert res["mean"] == pytest.approx(3.0)

    def test_too_few_samples_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            mathops.confidence_interval_mean([1.0], level=0.95)
        with pytest.raises(ValueError, match="must not be empty"):
            mathops.confidence_interval_mean([], level=0.95)

    def test_level_out_of_range_raises(self):
        with pytest.raises(ValueError, match="level"):
            mathops.confidence_interval_mean([1.0, 2.0, 3.0], level=0.0)
        with pytest.raises(ValueError, match="level"):
            mathops.confidence_interval_mean([1.0, 2.0, 3.0], level=1.0)

    def test_falls_back_to_a_normal_quantile_without_scipy_stats(self, monkeypatch):
        rng = np.random.default_rng(64)
        y = rng.normal(0.0, 1.0, 400)
        with_scipy = mathops.confidence_interval_mean(y, level=0.95)

        monkeypatch.setitem(sys.modules, "scipy.stats", None)
        without_scipy = mathops.confidence_interval_mean(y, level=0.95)
        assert without_scipy["lower"] == pytest.approx(with_scipy["lower"], rel=0.02)
        assert without_scipy["upper"] == pytest.approx(with_scipy["upper"], rel=0.02)


class TestConfidenceIntervalCorrelation:
    """Test confidence_interval_correlation(): Fisher z-transform interval for Pearson r."""

    def test_matches_manual_fisher_z_with_scipy_norm(self):
        stats = pytest.importorskip("scipy.stats")
        rng = np.random.default_rng(70)
        x = rng.normal(0.0, 1.0, 500)
        y = 0.6 * x + rng.normal(0.0, 1.0, 500)
        res = mathops.confidence_interval_correlation(x, y, level=0.95)

        r = np.corrcoef(x, y)[0, 1]
        z = np.arctanh(r)
        se = 1.0 / np.sqrt(x.size - 3)
        zcrit = stats.norm.ppf(0.975)
        lo, hi = np.tanh(z - zcrit * se), np.tanh(z + zcrit * se)
        assert res["lower"] == pytest.approx(lo)
        assert res["upper"] == pytest.approx(hi)
        assert res["r"] == pytest.approx(r)

    def test_interval_straddles_r(self):
        rng = np.random.default_rng(71)
        x = np.linspace(0.0, 10.0, 300)
        y = x + rng.normal(0.0, 3.0, x.size)
        res = mathops.confidence_interval_correlation(x, y, level=0.95)
        assert res["lower"] < res["r"] < res["upper"]

    def test_more_samples_narrows_the_interval_at_the_same_r(self):
        rng = np.random.default_rng(72)

        def sample(n, seed):
            rng2 = np.random.default_rng(seed)
            x = rng2.normal(0.0, 1.0, n)
            y = 0.5 * x + rng2.normal(0.0, 1.0, n)
            return x, y

        r_few = mathops.confidence_interval_correlation(*sample(20, 72), level=0.95)
        r_many = mathops.confidence_interval_correlation(*sample(20000, 72), level=0.95)
        assert (r_many["upper"] - r_many["lower"]) < (r_few["upper"] - r_few["lower"])

    def test_perfect_correlation_clamps_away_from_the_singularity(self):
        """r = 1.0 would send arctanh to +inf; the interval must stay finite."""
        x = np.linspace(0.0, 10.0, 20)
        y = 2.0 * x + 1.0
        res = mathops.confidence_interval_correlation(x, y, level=0.95)
        assert res["r"] == pytest.approx(1.0)
        assert np.isfinite(res["lower"]) and np.isfinite(res["upper"])
        assert 0.9 < res["lower"] <= 1.0
        assert 0.9 < res["upper"] <= 1.0

    def test_constant_input_gives_nan_not_a_crash(self):
        res = mathops.confidence_interval_correlation(np.zeros(10), np.arange(10.0), level=0.95)
        assert np.isnan(res["r"])
        assert np.isnan(res["lower"])
        assert np.isnan(res["upper"])

    def test_too_few_pairs_raises(self):
        with pytest.raises(ValueError, match="at least 4"):
            mathops.confidence_interval_correlation([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], level=0.95)

    def test_level_out_of_range_raises(self):
        x = np.linspace(0.0, 10.0, 10)
        with pytest.raises(ValueError, match="level"):
            mathops.confidence_interval_correlation(x, x, level=0.0)

    def test_falls_back_to_a_normal_quantile_without_scipy_stats(self, monkeypatch):
        """confidence_interval_correlation never needed scipy for r itself, only the
        critical value -- must still work with scipy.stats made unimportable."""
        rng = np.random.default_rng(73)
        x = rng.normal(0.0, 1.0, 400)
        y = 0.4 * x + rng.normal(0.0, 1.0, 400)
        with_scipy = mathops.confidence_interval_correlation(x, y, level=0.95)

        monkeypatch.setitem(sys.modules, "scipy.stats", None)
        without_scipy = mathops.confidence_interval_correlation(x, y, level=0.95)
        assert without_scipy["lower"] == pytest.approx(with_scipy["lower"], rel=0.02)
        assert without_scipy["upper"] == pytest.approx(with_scipy["upper"], rel=0.02)


class TestConfidenceIntervalDifference:
    """Test confidence_interval_difference(): pure-numpy Welch/Student CI on a mean gap."""

    def test_welch_matches_scipy_ttest_ind_confidence_interval(self):
        stats = pytest.importorskip("scipy.stats")
        rng = np.random.default_rng(80)
        a = rng.normal(10.0, 3.0, 150)
        b = rng.normal(8.0, 5.0, 130)
        res = mathops.confidence_interval_difference(a, b, level=0.95, method="welch")
        tt = stats.ttest_ind(a, b, equal_var=False)
        ci = tt.confidence_interval(confidence_level=0.95)
        assert res["lower"] == pytest.approx(ci.low)
        assert res["upper"] == pytest.approx(ci.high)
        assert res["diff"] == pytest.approx(np.mean(a) - np.mean(b))

    def test_student_matches_scipy_ttest_ind_confidence_interval(self):
        stats = pytest.importorskip("scipy.stats")
        rng = np.random.default_rng(81)
        a = rng.normal(10.0, 3.0, 150)
        b = rng.normal(8.0, 3.2, 130)
        res = mathops.confidence_interval_difference(a, b, level=0.95, method="student")
        tt = stats.ttest_ind(a, b, equal_var=True)
        ci = tt.confidence_interval(confidence_level=0.95)
        assert res["lower"] == pytest.approx(ci.low)
        assert res["upper"] == pytest.approx(ci.high)

    def test_clearly_separated_samples_exclude_zero(self):
        rng = np.random.default_rng(82)
        a = rng.normal(100.0, 1.0, 300)
        b = rng.normal(0.0, 1.0, 300)
        res = mathops.confidence_interval_difference(a, b, level=0.95)
        assert res["lower"] > 0.0

    def test_identical_distributions_include_zero_almost_always(self):
        rng = np.random.default_rng(83)
        a = rng.normal(0.0, 1.0, 500)
        b = rng.normal(0.0, 1.0, 500)
        res = mathops.confidence_interval_difference(a, b, level=0.99)
        assert res["lower"] < 0.0 < res["upper"]

    def test_different_length_samples_are_fine(self):
        rng = np.random.default_rng(84)
        a = rng.normal(0.0, 1.0, 40)
        b = rng.normal(0.0, 1.0, 400)
        res = mathops.confidence_interval_difference(a, b, level=0.95)
        assert res["n_a"] == 40.0
        assert res["n_b"] == 400.0

    def test_non_finite_entries_are_dropped(self):
        a = np.array([1.0, 2.0, 3.0, np.nan])
        b = np.array([1.0, np.inf, 2.0, 3.0])
        res = mathops.confidence_interval_difference(a, b, level=0.95)
        assert res["n_a"] == 3.0
        assert res["n_b"] == 3.0

    def test_too_few_samples_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            mathops.confidence_interval_difference([1.0], [1.0, 2.0], level=0.95)

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="unknown method"):
            mathops.confidence_interval_difference(
                [1.0, 2.0], [1.0, 2.0], method="mannwhitney"
            )

    def test_level_out_of_range_raises(self):
        with pytest.raises(ValueError, match="level"):
            mathops.confidence_interval_difference([1.0, 2.0], [1.0, 2.0], level=1.5)

    def test_no_scipy_dependency_at_all(self, monkeypatch):
        """Unlike two_sample_test, this is pure numpy end to end -- must work with both
        scipy and scipy.stats made unimportable, not just degrade gracefully."""
        rng = np.random.default_rng(85)
        a = rng.normal(10.0, 3.0, 150)
        b = rng.normal(8.0, 5.0, 130)
        with_scipy = mathops.confidence_interval_difference(a, b, level=0.95)

        monkeypatch.setitem(sys.modules, "scipy", None)
        monkeypatch.setitem(sys.modules, "scipy.stats", None)
        without_scipy = mathops.confidence_interval_difference(a, b, level=0.95)
        assert without_scipy["lower"] == pytest.approx(with_scipy["lower"], rel=0.02)
        assert without_scipy["upper"] == pytest.approx(with_scipy["upper"], rel=0.02)


class TestCrossCorrelation:
    """Test cross_correlation(): FFT-based, verified against a brute-force reference."""

    @staticmethod
    def _brute_force(a, b, max_lag):
        """R[k] = sum_t a[t] * b[t+k] for k in -max_lag..max_lag, zero outside overlap."""
        n = a.size
        lags = np.arange(-max_lag, max_lag + 1)
        out = np.empty(lags.size)
        for i, k in enumerate(lags.tolist()):
            s = 0.0
            for t in range(n):
                tk = t + k
                if 0 <= tk < n:
                    s += a[t] * b[tk]
            out[i] = s
        return lags.astype(np.float64), out

    def test_matches_brute_force_reference(self):
        rng = np.random.default_rng(0)
        a = rng.normal(0.0, 1.0, 12)
        b = rng.normal(0.0, 1.0, 12)
        lags, xcf = mathops.cross_correlation(a, b, max_lag=5, detrend=False)
        ref_lags, ref = self._brute_force(a, b, 5)
        norm = float(np.sqrt(np.sum(a**2) * np.sum(b**2)))
        assert np.allclose(lags, ref_lags)
        assert np.allclose(xcf, ref / norm, atol=1e-9)

    def test_recovers_a_known_delay(self):
        """b is a delayed by exactly 17 samples -- the peak must land at lag +17."""
        rng = np.random.default_rng(1)
        a = rng.normal(0.0, 1.0, 500)
        delay = 17
        b = np.zeros_like(a)
        b[delay:] = a[: -delay]
        lags, xcf = mathops.cross_correlation(a, b, max_lag=50)
        peak_lag = float(lags[np.argmax(xcf)])
        assert peak_lag == pytest.approx(delay)
        assert float(np.max(xcf)) > 0.9

    def test_negative_lag_when_b_leads(self):
        """b is a ADVANCED by 17 samples (b leads a) -- the peak must land at lag -17."""
        rng = np.random.default_rng(2)
        a = rng.normal(0.0, 1.0, 500)
        lead = 17
        b = np.zeros_like(a)
        b[: -lead] = a[lead:]
        lags, xcf = mathops.cross_correlation(a, b, max_lag=50)
        peak_lag = float(lags[np.argmax(xcf)])
        assert peak_lag == pytest.approx(-lead)

    def test_autocorrelation_is_the_special_case_b_is_a(self):
        """cross_correlation(a, a) at non-negative lags must match autocorrelation(a)."""
        rng = np.random.default_rng(3)
        a = rng.normal(0.0, 1.0, 200)
        lags_x, xcf = mathops.cross_correlation(a, a, max_lag=40)
        lags_a, acf = mathops.autocorrelation(a, max_lag=40)
        nonneg = lags_x >= 0.0
        assert np.allclose(lags_x[nonneg], lags_a)
        assert np.allclose(xcf[nonneg], acf, atol=1e-9)

    def test_bounded_in_minus_one_to_one(self):
        rng = np.random.default_rng(4)
        a = rng.normal(0.0, 1.0, 300)
        b = rng.normal(0.0, 1.0, 300)
        _lags, xcf = mathops.cross_correlation(a, b)
        assert np.all(xcf >= -1.0 - 1e-9)
        assert np.all(xcf <= 1.0 + 1e-9)

    def test_identical_signals_peak_at_exactly_one(self):
        rng = np.random.default_rng(5)
        a = rng.normal(0.0, 1.0, 200)
        lags, xcf = mathops.cross_correlation(a, a, max_lag=0)
        assert lags.size == 1 and lags[0] == 0.0
        assert xcf[0] == pytest.approx(1.0)

    def test_max_lag_truncates_symmetrically(self):
        rng = np.random.default_rng(6)
        a = rng.normal(0.0, 1.0, 400)
        b = rng.normal(0.0, 1.0, 400)
        lags, xcf = mathops.cross_correlation(a, b, max_lag=25)
        assert lags.size == 51  # -25 .. 25
        assert float(lags.min()) == -25.0
        assert float(lags.max()) == 25.0

    def test_constant_signal_gives_zero_correlation(self):
        a = np.full(50, 3.0)
        b = np.linspace(0.0, 1.0, 50)
        _lags, xcf = mathops.cross_correlation(a, b)
        assert np.allclose(xcf, 0.0)

    def test_mismatched_length_raises(self):
        with pytest.raises(ValueError, match="same length"):
            mathops.cross_correlation(np.arange(5.0), np.arange(4.0))

    def test_negative_max_lag_raises(self):
        with pytest.raises(ValueError, match="max_lag"):
            mathops.cross_correlation(np.arange(10.0), np.arange(10.0), max_lag=-1)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            mathops.cross_correlation(np.array([]), np.array([]))
