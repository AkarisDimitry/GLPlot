"""Pure numerical operations for the GLPlot mathematical workstation.

Every function here takes numpy arrays and returns *new* numpy arrays: nothing in
this module touches imgui, OpenGL, or the live engine scene, so it is fully
unit-testable headless.

scipy is an optional accelerator only. Every scipy import is function-local and
wrapped in ``try/except`` with an equivalent pure-numpy fallback, because the CI
matrix (py3.9-3.12 x ubuntu/macos/windows) does not currently prove scipy imports
cleanly. A missing or broken scipy degrades accuracy or speed but never raises.

The **one exception** is nonlinear least squares (:func:`fit_model`), which is
``scipy.optimize.curve_fit`` and has no honest pure-numpy equivalent. It therefore
degrades to a clear "unavailable" report rather than to a worse answer: ask
:func:`fit_available` first, and note that :class:`FitUnavailableError` is a
``ValueError`` subclass so a caller already handling this module's documented
failure mode shows a message instead of crashing.

Quadrature is hand-rolled rather than delegated to ``np.trapezoid``: that name only
exists in numpy >= 2.0 and the package floor is numpy >= 1.23.
"""

from __future__ import annotations

import warnings
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np

from glplot.gui import expressions

__all__ = [
    "integrate",
    "definite_integral",
    "derivative",
    "parametric_derivative",
    "smooth",
    "rolling_stat",
    "ROLLING_STATS",
    "add_noise",
    "resample",
    "normalize",
    "fit_polynomial",
    "fft_spectrum",
    "envelope",
    "find_peaks",
    "butter_filter",
    "iir_filter",
    "IIR_FAMILIES",
    "detrend",
    "baseline_asls",
    "baseline_available",
    "BaselineUnavailableError",
    "BASELINE_UNAVAILABLE_MESSAGE",
    "autocorrelation",
    "cross_correlation",
    "histogram",
    "fit_distribution",
    "distributions_available",
    "describe",
    "FILTER_TYPES",
    "DETREND_KINDS",
    "DISTRIBUTIONS",
    "DIST_UNAVAILABLE_MESSAGE",
    "DistributionUnavailableError",
    "FILTER_UNAVAILABLE_MESSAGE",
    "FilterUnavailableError",
    "filter_available",
    "FIT_UNAVAILABLE_MESSAGE",
    "FitModel",
    "FitUnavailableError",
    "MODELS",
    "MODEL_NAMES",
    "PEAK_SHAPES",
    "multipeak_model",
    "peak_area",
    "custom_model",
    "fit_available",
    "fit_errors",
    "fit_model",
    "fit_statistics",
    "initial_guess",
    "resolve_model",
    "fit_polynomial_covariance",
    "fit_model_robust",
    "ROBUST_LOSS_KINDS",
    "confidence_band",
    "describe_extended",
    "correlate",
    "two_sample_test",
    "two_sample_available",
    "TWO_SAMPLE_METHODS",
    "TWO_SAMPLE_UNAVAILABLE_MESSAGE",
    "TwoSampleUnavailableError",
    "confidence_interval_mean",
    "confidence_interval_correlation",
    "confidence_interval_difference",
    "DIFFERENCE_CI_METHODS",
]

# Relative tolerance used when deciding whether a sample grid is "uniform enough".
_UNIFORM_RTOL = 1e-6


# ---------------------------------------------------------------------------
# input validation helpers
# ---------------------------------------------------------------------------


def _as_1d(a: Any, name: str) -> np.ndarray:
    """Coerce ``a`` to a non-empty 1-D float64 array or raise ValueError.

    Scalars are promoted to length-1 arrays. Shapes such as ``(N, 1)`` that squeeze
    down to 1-D are accepted; genuinely 2-D data is rejected rather than silently
    flattened, since flattening an (N, 2) point array would produce nonsense.
    """
    try:
        arr = np.asarray(a, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric array-like ({exc})") from exc
    if arr.ndim > 1:
        arr = np.squeeze(arr)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got array with shape {np.shape(a)}")
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty")
    return arr


def _as_xy(x: Any, y: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Validate an (x, y) pair: both 1-D, non-empty, and equal length."""
    xa = _as_1d(x, "x")
    ya = _as_1d(y, "y")
    if xa.size != ya.size:
        raise ValueError(f"x and y must have the same length (got {xa.size} and {ya.size})")
    return xa, ya


def _sorted_xy(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (x, y) ordered by ascending x. Already-sorted input is passed through."""
    if x.size < 2 or bool(np.all(np.diff(x) >= 0.0)):
        return x, y
    order = np.argsort(x, kind="stable")
    return x[order], y[order]


def _is_uniform(x: np.ndarray) -> bool:
    """True if ``x`` is (near enough) an evenly spaced ascending grid."""
    if x.size < 3:
        return True
    dx = np.diff(x)
    scale = float(np.max(np.abs(dx)))
    if scale <= 0.0:
        return False
    return bool(np.ptp(dx) <= _UNIFORM_RTOL * scale)


def _clamp_window(window: Any, n: int, *, force_odd: bool) -> int:
    """Clamp a smoothing window into [1, n], optionally forcing it odd.

    Per spec the window is clamped, never rejected: a UI slider must not be able to
    raise just by being dragged to an extreme.
    """
    try:
        w = int(window)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"window must be an integer (got {window!r})") from exc
    w = max(1, min(w, int(n)))
    if force_odd and w % 2 == 0:
        w -= 1
    return max(1, w)


def _fill_nonfinite(y: np.ndarray, name: str) -> np.ndarray:
    """Linearly interpolate over non-finite samples (needed by FFT, which nan-poisons)."""
    good = np.isfinite(y)
    if bool(np.all(good)):
        return y
    if not bool(np.any(good)):
        raise ValueError(f"{name} contains no finite samples")
    idx = np.arange(y.size, dtype=np.float64)
    return np.interp(idx, idx[good], y[good])


# ---------------------------------------------------------------------------
# quadrature primitives
# ---------------------------------------------------------------------------


def _cumulative_trapezoid(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Cumulative trapezoid integral, length ``len(x)``, first element 0.0.

    Correct for non-uniform x: each panel is weighted by its own width.
    """
    out = np.zeros(x.size, dtype=np.float64)
    if x.size < 2:
        return out
    dx = np.diff(x)
    out[1:] = np.cumsum(dx * 0.5 * (y[1:] + y[:-1]))
    return out


def _cumulative_rectangle(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Cumulative left-endpoint rectangle integral, length ``len(x)``.

    First order accurate only; provided for completeness/teaching, not precision.
    """
    out = np.zeros(x.size, dtype=np.float64)
    if x.size < 2:
        return out
    out[1:] = np.cumsum(np.diff(x) * y[:-1])
    return out


def _parabola_panel_integrals(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Integral of the local interpolating parabola across each panel [x_i, x_i+1].

    Returns an array of length ``len(x) - 1``. For panel ``i`` the parabola is fitted
    through the three consecutive samples centred on ``clip(i, 1, n-2)``, so every
    panel is covered by a quadratic that brackets it. Correct for non-uniform x.

    All arithmetic is done in coordinates shifted to the centre knot; integrating
    ``t**3/3`` directly would lose precision catastrophically for x around 1e6.
    """
    n = x.size
    i = np.arange(n - 1)
    j1 = np.clip(i, 1, n - 2)
    j0 = j1 - 1
    j2 = j1 + 1

    xc = x[j1]
    p0 = x[j0] - xc
    p2 = x[j2] - xc
    ua = x[i] - xc
    ub = x[i + 1] - xc

    def anti(u: np.ndarray, p: np.ndarray, q: np.ndarray) -> np.ndarray:
        """Antiderivative of (u - p)(u - q)."""
        return u**3 / 3.0 - (p + q) * u**2 / 2.0 + p * q * u

    zero = np.zeros_like(p0)
    d0 = (p0 - zero) * (p0 - p2)
    d1 = (zero - p0) * (zero - p2)
    d2 = (p2 - p0) * (p2 - zero)

    i0 = (anti(ub, zero, p2) - anti(ua, zero, p2)) / d0
    i1 = (anti(ub, p0, p2) - anti(ua, p0, p2)) / d1
    i2 = (anti(ub, p0, zero) - anti(ua, p0, zero)) / d2

    return y[j0] * i0 + y[j1] * i1 + y[j2] * i2


def _cumulative_simpson(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Cumulative quadratic (Simpson-style) integral, length ``len(x)``.

    Measured convergence is 3rd order globally and exact through quadratics (see the
    note in :func:`integrate` for why this is not classic composite Simpson).
    Falls back to trapezoid for fewer than 3 samples, where no parabola exists.
    """
    if x.size < 3:
        return _cumulative_trapezoid(x, y)
    out = np.zeros(x.size, dtype=np.float64)
    out[1:] = np.cumsum(_parabola_panel_integrals(x, y))
    return out


# ---------------------------------------------------------------------------
# calculus
# ---------------------------------------------------------------------------


def integrate(
    x: Any,
    y: Any,
    *,
    method: str = "trapezoid",
    initial: float = 0.0,
) -> np.ndarray:
    """Cumulative integral of y with respect to x, same length as x.

    This is the primary "plot the integral" operation: ``out[i]`` is the integral
    from ``x[0]`` to ``x[i]``, offset by ``initial`` (so ``out[0] == initial``).

    Args:
        x: Sample abscissae, 1-D.
        y: Sample values, 1-D, same length as x.
        method: One of ``"trapezoid"`` (2nd order, exact for linears, default),
            ``"simpson"`` (quadratic panels, 3rd order, exact for quadratics,
            degrades to trapezoid for n < 3), or ``"rectangle"`` (left endpoint,
            1st order - crude, offered for completeness rather than precision).
            Orders are measured, not merely asserted; see the note below.
        initial: Constant of integration added to every element.

    Returns:
        float64 array of length ``len(x)``.

    Note:
        **Unsorted x is sorted.** The result is in ascending-x order, which is NOT
        the input order if x was unsorted. Callers that need to pair the result with
        their original x must sort x themselves (``order = np.argsort(x)``) before
        calling. All spacings are honoured individually, so non-uniform x is exact.

    Note:
        ``"simpson"`` is a *cumulative* Simpson-family rule, not classic composite
        Simpson, and the two do not agree in general. Classic Simpson is only defined
        over *pairs* of panels, so it cannot produce a value at every sample - which
        is precisely what a cumulative integral must do. Instead each panel is
        integrated under the parabola through the three samples bracketing it, which
        matches the semantics of ``scipy.integrate.cumulative_simpson``. Measured
        consequence: global convergence is 3rd order (not classic Simpson's 4th) and
        the rule is exact through quadratics (not cubics). It remains far more
        accurate than trapezoid at equal sample count - on ``exp(x)`` over [0, 1] it
        beats trapezoid by ~3 orders of magnitude at N=321.
        ``definite_integral`` deliberately shares this rule so that a plotted
        cumulative curve and its reported total can never disagree.

    Raises:
        ValueError: On empty, non-1-D, or mismatched input; on an unknown method; or
            on duplicate x values when ``method="simpson"`` (a parabola through
            coincident abscissae is undefined).
    """
    xa, ya = _as_xy(x, y)
    xs, ys = _sorted_xy(xa, ya)

    if method == "trapezoid":
        out = _cumulative_trapezoid(xs, ys)
    elif method == "rectangle":
        out = _cumulative_rectangle(xs, ys)
    elif method == "simpson":
        if xs.size >= 3 and bool(np.any(np.diff(xs) == 0.0)):
            raise ValueError(
                "simpson integration requires strictly increasing x (duplicate x values found); "
                "use method='trapezoid' instead"
            )
        out = _cumulative_simpson(xs, ys)
    else:
        raise ValueError(
            f"unknown integration method {method!r}; expected 'trapezoid', 'simpson' or 'rectangle'"
        )

    return out + float(initial)


def definite_integral(x: Any, y: Any, *, method: str = "trapezoid") -> float:
    """Definite integral of y over the full span of x.

    Defined as exactly the last element of :func:`integrate` with ``initial=0.0``, so
    a plotted cumulative curve and this reported total can never disagree. Note that
    this makes ``method="simpson"`` the cumulative Simpson-family rule rather than
    classic composite Simpson - see :func:`integrate` for the tradeoff.
    Unsorted x is sorted first (the span is x.min() to x.max()). Returns 0.0 for a
    single sample.

    Args:
        x: Sample abscissae, 1-D.
        y: Sample values, 1-D, same length as x.
        method: Same choices as :func:`integrate`.

    Returns:
        The integral as a Python float.

    Raises:
        ValueError: Same conditions as :func:`integrate`.
    """
    cumulative = integrate(x, y, method=method, initial=0.0)
    return float(cumulative[-1])


def derivative(
    x: Any,
    y: Any,
    *,
    order: int = 1,
    method: str = "central",
    window: int = 11,
    polyorder: int = 3,
) -> np.ndarray:
    """Numerical derivative of y with respect to x, same length as x.

    Args:
        x: Sample abscissae, 1-D, strictly increasing after sorting.
        y: Sample values, 1-D, same length as x.
        order: Derivative order (>= 1). Non-savgol methods apply the scheme
            repeatedly.
        method: One of:

            * ``"central"`` (default) - ``np.gradient``, which is 2nd order and
              handles non-uniform x correctly.
            * ``"forward"`` - one-sided differences; the final element repeats the
              last valid difference to preserve length.
            * ``"savgol"`` - Savitzky-Golay differentiation via scipy. Requires an
              approximately uniform grid; falls back to ``"central"`` if scipy is
              unavailable or x is non-uniform.
        window: Savitzky-Golay window (clamped to [1, len(y)], forced odd). Ignored
            by the other methods.
        polyorder: Savitzky-Golay polynomial order. Clamped below the window and
            raised to at least ``order``. Ignored by the other methods.

    Returns:
        float64 array of length ``len(x)``.

    Note:
        Unsorted x is sorted; the result is in ascending-x order, matching
        :func:`integrate`.

    Raises:
        ValueError: On empty/mismatched input, ``order < 1``, an unknown method, or
            duplicate x values (which would divide by a zero spacing).
    """
    xa, ya = _as_xy(x, y)
    if int(order) < 1:
        raise ValueError(f"order must be >= 1 (got {order})")
    order = int(order)
    if method not in ("central", "forward", "savgol"):
        raise ValueError(
            f"unknown derivative method {method!r}; expected 'central', 'forward' or 'savgol'"
        )

    xs, ys = _sorted_xy(xa, ya)
    if xs.size >= 2 and bool(np.any(np.diff(xs) == 0.0)):
        raise ValueError(
            "derivative requires strictly increasing x (duplicate x values found), "
            "otherwise the spacing is zero and the slope is undefined"
        )
    if xs.size == 1:
        return np.zeros(1, dtype=np.float64)

    if method == "savgol":
        result = _savgol_derivative(xs, ys, order=order, window=window, polyorder=polyorder)
        if result is not None:
            return result
        method = "central"

    out = ys
    for _ in range(order):
        if method == "central":
            out = np.gradient(out, xs, edge_order=1 if xs.size < 3 else 2)
        else:  # forward
            d = np.diff(out) / np.diff(xs)
            out = np.concatenate([d, d[-1:]])
    return np.asarray(out, dtype=np.float64)


def _grad_wrt_index(values: np.ndarray, method: str) -> np.ndarray:
    """d(values)/dt over the sample index t = 0, 1, 2, ... (uniform spacing 1)."""
    if method == "forward":
        d = np.diff(values)
        return np.concatenate([d, d[-1:]])
    return np.gradient(values, edge_order=1 if values.size < 3 else 2)


def parametric_derivative(
    x: Any,
    y: Any,
    *,
    order: int = 1,
    method: str = "central",
) -> np.ndarray:
    """dy/dx of a **parametric** curve ``(x(t), y(t))`` via the chain rule, in sample order.

    Unlike :func:`derivative`, this does not assume ``y`` is a single-valued function of ``x``,
    so it never sorts by ``x``: a parametric curve may loop back on itself (a circle, a spiral,
    a phase portrait), where the same ``x`` has several ``y`` and a sort would scramble it. The
    parameter is the sample index ``t = 0, 1, 2, ...`` and

        dy/dx = (dy/dt) / (dx/dt).

    Higher orders apply the same rule repeatedly -- ``d(prev)/dx = d(prev)/dt / (dx/dt)`` --
    which is the correct parametric second derivative and beyond. At a vertical tangent
    (``dx/dt -> 0``) the slope is ``+/-inf``; that is the honest answer, not an error.

    Args:
        x, y: The parametric coordinates, 1-D and the same length.
        order: Derivative order (>= 1).
        method: ``"central"`` (``np.gradient``) or ``"forward"`` (one-sided).

    Returns:
        float64 array of length ``len(x)``, aligned with the input order (not sorted).
    """
    xa, ya = _as_xy(x, y)
    if int(order) < 1:
        raise ValueError(f"order must be >= 1 (got {order})")
    order = int(order)
    if method not in ("central", "forward"):
        raise ValueError(f"unknown method {method!r}; expected 'central' or 'forward'")
    if xa.size == 1:
        return np.zeros(1, dtype=np.float64)

    # dx/dt is fixed by the curve; only the numerator changes as the order climbs.
    dxdt = _grad_wrt_index(xa, method)
    current = ya.astype(np.float64, copy=True)
    with np.errstate(all="ignore"):
        for _ in range(order):
            current = _grad_wrt_index(current, method) / dxdt
    return np.asarray(current, dtype=np.float64)


def _savgol_derivative(
    x: np.ndarray,
    y: np.ndarray,
    *,
    order: int,
    window: int,
    polyorder: int,
) -> Optional[np.ndarray]:
    """Savitzky-Golay derivative, or None if unavailable/inapplicable (caller falls back)."""
    if not _is_uniform(x):
        warnings.warn(
            "savgol differentiation needs a uniform x grid; falling back to 'central'",
            RuntimeWarning,
            stacklevel=3,
        )
        return None
    try:
        from scipy.signal import savgol_filter
    except Exception:
        warnings.warn(
            "scipy is unavailable; falling back to 'central' differentiation",
            RuntimeWarning,
            stacklevel=3,
        )
        return None

    w = _clamp_window(window, y.size, force_odd=True)
    p = max(int(order), min(int(polyorder), w - 1))
    if w < 3 or p >= w or p < order:
        warnings.warn(
            "not enough samples for a savgol window; falling back to 'central' differentiation",
            RuntimeWarning,
            stacklevel=3,
        )
        return None
    delta = float(x[1] - x[0]) if x.size >= 2 else 1.0
    if delta == 0.0:
        return None
    try:
        return np.asarray(
            savgol_filter(y, w, p, deriv=order, delta=delta, mode="interp"),
            dtype=np.float64,
        )
    except Exception:
        return None


# ---------------------------------------------------------------------------
# smoothing
# ---------------------------------------------------------------------------


def _edge_corrected_convolve(y: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Convolve y with kernel, normalising by the *actual* window overlap.

    This is the crux of edge-correct smoothing. A plain ``convolve(y, k/k.sum())``
    treats out-of-range samples as zeros, so the first and last ``window//2`` points
    decay toward zero and the curve visibly droops at both ends. Dividing by
    ``convolve(ones, k)`` renormalises each output by the weight that actually landed
    on real data, so edges keep their level and a constant signal is reproduced
    exactly everywhere.

    Non-finite samples are excluded from both numerator and denominator, so a lone
    nan is smoothed over rather than poisoning a whole window. Positions with no
    finite support at all come back as nan.
    """
    mask = np.isfinite(y).astype(np.float64)
    filled = np.where(mask > 0.0, y, 0.0)
    num = np.convolve(filled, kernel, mode="same")
    den = np.convolve(mask, kernel, mode="same")
    out = np.full(y.shape, np.nan, dtype=np.float64)
    np.divide(num, den, out=out, where=den > 1e-12)
    return out


def _median_fallback(y: np.ndarray, window: int) -> np.ndarray:
    """Pure-numpy median filter with edge padding, matching scipy's mode='nearest'."""
    half = window // 2
    padded = np.pad(y, (half, half), mode="edge")
    from numpy.lib.stride_tricks import sliding_window_view

    windows = sliding_window_view(padded, window)
    return np.asarray(np.median(windows, axis=-1), dtype=np.float64)


def smooth(y: Any, *, method: str = "moving_average", window: int = 11, **kw: Any) -> np.ndarray:
    """Smooth a 1-D signal, always returning an array of the same length as y.

    Args:
        y: Signal to smooth, 1-D.
        method: One of:

            * ``"moving_average"`` (default) - boxcar, edge-corrected.
            * ``"gaussian"`` - Gaussian kernel, edge-corrected. Accepts ``sigma``
              (defaults to ``window / 6``, i.e. the window spans about +/-3 sigma).
            * ``"savgol"`` - ``scipy.signal.savgol_filter``; falls back to
              ``"moving_average"`` with a warning if scipy is unavailable. Accepts
              ``polyorder`` (default 3, clamped below the window).
            * ``"median"`` - ``scipy.ndimage.median_filter``; falls back to a numpy
              sliding-window median (no warning: the fallback is exact, not an
              approximation).
            * ``"ema"`` - exponential moving average, pure numpy, always available.
              Causal (uses only current and past samples, unlike the other three,
              which are centered) -- what a live/streaming computation could actually
              produce. Accepts ``alpha`` (defaults to ``2 / (window + 1)``, the
              standard "N-period EMA" convention, e.g. matching
              ``pandas.Series.ewm(span=N).mean()``).
        window: Window length. Clamped to ``[1, len(y)]`` and forced odd (an even
            window would bias the output by half a sample). ``window == 1`` returns
            a copy unchanged. For ``"ema"``, only used to derive the default ``alpha``
            when one is not given.
        **kw: ``sigma`` (gaussian), ``polyorder`` (savgol), ``alpha`` (ema) as above.

    Returns:
        float64 array of length ``len(y)``.

    Note:
        The edge-correcting methods (``moving_average``, ``gaussian``) do **not**
        let the signal decay toward zero at the boundaries, and they interpolate
        across isolated nans rather than spreading them across a full window. ``ema``
        also interpolates across non-finite samples first (a causal recursive filter
        has no way to recover from one otherwise: it would poison every later value).

    Raises:
        ValueError: On empty/non-1-D input, an unknown method, an unknown keyword, or
            (``ema`` only) an ``alpha`` outside ``(0, 1]``.
    """
    ya = _as_1d(y, "y")

    allowed_kw = {
        "moving_average": set(),
        "gaussian": {"sigma"},
        "savgol": {"polyorder"},
        "median": set(),
        "ema": {"alpha"},
    }
    if method not in allowed_kw:
        raise ValueError(
            f"unknown smoothing method {method!r}; expected one of {sorted(allowed_kw)}"
        )
    unknown = set(kw) - allowed_kw[method]
    if unknown:
        accepted = sorted(allowed_kw[method]) or "none"
        raise ValueError(
            f"unknown keyword(s) {sorted(unknown)} for method {method!r}; accepted: {accepted}"
        )

    w = _clamp_window(window, ya.size, force_odd=True)
    if w <= 1:
        return ya.copy()

    if method == "moving_average":
        return _edge_corrected_convolve(ya, np.ones(w, dtype=np.float64))

    if method == "gaussian":
        sigma = kw.get("sigma", None)
        sigma = w / 6.0 if sigma is None else float(sigma)
        if not np.isfinite(sigma) or sigma <= 0.0:
            raise ValueError(f"sigma must be a positive finite number (got {sigma!r})")
        half = w // 2
        t = np.arange(-half, half + 1, dtype=np.float64)
        kernel = np.exp(-0.5 * (t / sigma) ** 2)
        return _edge_corrected_convolve(ya, kernel)

    if method == "savgol":
        polyorder = int(kw.get("polyorder", 3))
        p = min(polyorder, w - 1)
        if p < 0:
            raise ValueError(f"polyorder must be >= 0 (got {polyorder})")
        try:
            from scipy.signal import savgol_filter

            return np.asarray(savgol_filter(ya, w, p, mode="interp"), dtype=np.float64)
        except Exception:
            warnings.warn(
                "scipy is unavailable; falling back to 'moving_average' smoothing",
                RuntimeWarning,
                stacklevel=2,
            )
            return _edge_corrected_convolve(ya, np.ones(w, dtype=np.float64))

    if method == "median":
        try:
            from scipy.ndimage import median_filter

            return np.asarray(median_filter(ya, size=w, mode="nearest"), dtype=np.float64)
        except Exception:
            return _median_fallback(ya, w)

    # ema
    alpha = kw.get("alpha", None)
    alpha = 2.0 / (w + 1.0) if alpha is None else float(alpha)
    if not (0.0 < alpha <= 1.0):
        raise ValueError(f"alpha must be in (0, 1] (got {alpha!r})")
    filled = _fill_nonfinite(ya, "y")
    out = np.empty_like(filled)
    out[0] = filled[0]
    for i in range(1, filled.size):
        out[i] = alpha * filled[i] + (1.0 - alpha) * out[i - 1]
    return out


#: rolling_stat's supported reducers, each nan-safe (a non-finite sample inside a window
#: is excluded from that window's statistic, not propagated).
ROLLING_STATS: Tuple[str, ...] = ("mean", "std", "median", "min", "max")


def rolling_stat(
    y: Any,
    window: int,
    *,
    stat: str = "mean",
    center: bool = True,
    min_periods: Optional[int] = None,
) -> np.ndarray:
    """A moving-window statistic over ``y``: mean, std, median, min, or max.

    Unlike :func:`smooth` (which always averages), this reports whichever statistic is
    asked for -- a rolling std is a volatility/noise-level trace, a rolling min/max is a
    moving envelope, useful on its own rather than only as a smoothing step.

    Args:
        y: Signal, 1-D.
        window: Window width in samples. Clamped to ``[1, len(y)]``.
        stat: One of :data:`ROLLING_STATS`.
        center: True centers the window on each sample; False is trailing/causal (the
            window ending at each sample, matching a live computation that cannot see
            the future -- the standard "moving average" a streaming system would report).
        min_periods: Minimum number of finite samples a window must contain to report a
            value there; short of that the result is nan. ``None`` (default) requires a
            full window, so the ``center=False`` case starts with ``window - 1`` leading
            nans rather than a statistic computed from fewer samples than asked for.

    Returns:
        Array the same length as ``y``.

    Raises:
        ValueError: On empty/non-1-D input, ``window < 1``, an unknown ``stat``, or a
            non-positive ``min_periods``.
    """
    ya = _as_1d(y, "y")
    n = ya.size

    try:
        w = int(window)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"window must be an integer (got {window!r})") from exc
    w = max(1, min(w, n))

    reducers: Dict[str, Callable[..., Any]] = {
        "mean": np.nanmean,
        "std": np.nanstd,
        "median": np.nanmedian,
        "min": np.nanmin,
        "max": np.nanmax,
    }
    if stat not in reducers:
        raise ValueError(f"unknown stat {stat!r}; choose one of {', '.join(ROLLING_STATS)}")
    reduce = reducers[stat]

    if min_periods is None:
        required = w
    else:
        try:
            required = int(min_periods)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"min_periods must be an integer (got {min_periods!r})") from exc
        if required < 1:
            raise ValueError(f"min_periods must be >= 1 (got {required})")
        required = min(required, w)

    if center:
        left = w // 2
        right = w - 1 - left
    else:
        left, right = w - 1, 0

    padded = np.concatenate([np.full(left, np.nan), ya, np.full(right, np.nan)])
    windows = np.lib.stride_tricks.sliding_window_view(padded, w)  # (n, w)

    counts = np.sum(np.isfinite(windows), axis=1)
    enough = counts >= required
    out = np.full(n, np.nan, dtype=np.float64)
    if np.any(enough):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # nan* reducers warn on an all-nan slice
            out[enough] = reduce(windows[enough], axis=1)
    return out


# ---------------------------------------------------------------------------
# noise
# ---------------------------------------------------------------------------


def add_noise(
    y: Any,
    *,
    kind: str = "gaussian",
    amplitude: float = 0.1,
    relative: bool = False,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Add synthetic noise to a signal, returning a new array of the same length.

    Args:
        y: Clean signal, 1-D.
        kind: One of:

            * ``"gaussian"`` - additive normal noise of standard deviation
              ``amplitude``.
            * ``"uniform"`` - additive noise drawn uniformly from
              ``[-amplitude, +amplitude]``.
            * ``"poisson"`` - shot noise. ``|y|`` is treated as the expected count;
              the deviation ``poisson(|y|) - |y|`` is scaled by ``amplitude`` and
              added, so ``amplitude=1`` gives true shot noise and ``amplitude=0``
              gives back y.
            * ``"salt_pepper"`` - ``amplitude`` (clipped to ``[0, 1]``) is the
              *fraction* of samples replaced, half with the signal max ("salt"),
              half with the signal min ("pepper").
        amplitude: Noise magnitude; see ``kind`` for the per-kind meaning.
        relative: Scale ``amplitude`` by the data's own spread so it is meaningful
            regardless of units - by ``np.nanstd(y)`` for gaussian, by the nan-safe
            peak-to-peak range for uniform. **Ignored for poisson and salt_pepper**,
            which are already data-scaled and fraction-valued respectively. Note that
            a constant signal has zero spread, so ``relative=True`` correctly adds
            nothing to it.
        seed: Seed for ``np.random.default_rng``. Pass an int for reproducible noise.

    Returns:
        float64 array of length ``len(y)``.

    Raises:
        ValueError: On empty/non-1-D input, an unknown kind, or a non-finite/negative
            amplitude.
    """
    ya = _as_1d(y, "y")
    if kind not in ("gaussian", "uniform", "poisson", "salt_pepper"):
        raise ValueError(
            f"unknown noise kind {kind!r}; expected 'gaussian', 'uniform', 'poisson' "
            f"or 'salt_pepper'"
        )
    amp = float(amplitude)
    if not np.isfinite(amp):
        raise ValueError(f"amplitude must be finite (got {amplitude!r})")
    if amp < 0.0:
        raise ValueError(f"amplitude must be >= 0 (got {amplitude!r})")

    rng = np.random.default_rng(seed)
    n = ya.size

    if kind == "gaussian":
        scale = amp
        if relative:
            scale = amp * float(np.nanstd(ya)) if np.any(np.isfinite(ya)) else 0.0
        if scale == 0.0:
            return ya.copy()
        return ya + rng.normal(0.0, scale, size=n)

    if kind == "uniform":
        scale = amp
        if relative:
            if np.any(np.isfinite(ya)):
                span = float(np.nanmax(ya) - np.nanmin(ya))
            else:
                span = 0.0
            scale = amp * span
        if scale == 0.0:
            return ya.copy()
        return ya + rng.uniform(-scale, scale, size=n)

    if kind == "poisson":
        if amp == 0.0:
            return ya.copy()
        lam = np.abs(np.nan_to_num(ya, nan=0.0, posinf=0.0, neginf=0.0))
        drawn = rng.poisson(lam).astype(np.float64)
        return ya + amp * (drawn - lam)

    # salt_pepper
    frac = min(max(amp, 0.0), 1.0)
    out = ya.copy()
    if frac == 0.0 or not np.any(np.isfinite(ya)):
        return out
    count = int(round(frac * n))
    if count <= 0:
        return out
    idx = rng.choice(n, size=min(count, n), replace=False)
    salt = idx[: idx.size // 2]
    pepper = idx[idx.size // 2 :]
    out[salt] = float(np.nanmax(ya))
    out[pepper] = float(np.nanmin(ya))
    return out


# ---------------------------------------------------------------------------
# resample / transform
# ---------------------------------------------------------------------------


def resample(x: Any, y: Any, n: int, *, kind: str = "linear") -> Tuple[np.ndarray, np.ndarray]:
    """Resample (x, y) onto ``n`` evenly spaced abscissae spanning the same range.

    Args:
        x: Sample abscissae, 1-D, at least 2 samples with a non-zero span.
        y: Sample values, 1-D, same length as x.
        n: Number of output samples (>= 2).
        kind: ``"linear"`` (``np.interp``) or ``"cubic"``
            (``scipy.interpolate.CubicSpline``). Cubic falls back to linear, with a
            warning, if scipy is unavailable or x contains duplicates (a spline needs
            strictly increasing knots).

    Returns:
        ``(x_new, y_new)``, both float64 of length ``n``. The endpoints are exact:
        ``x_new[0] == min(x)`` and ``x_new[-1] == max(x)``, and both interpolants
        pass through the original endpoint values.

    Note:
        Unsorted x is sorted first.

    Raises:
        ValueError: On empty/mismatched input, ``n < 2``, an unknown kind, fewer than
            2 samples, or a zero-width x range.
    """
    xa, ya = _as_xy(x, y)
    if kind not in ("linear", "cubic"):
        raise ValueError(f"unknown resample kind {kind!r}; expected 'linear' or 'cubic'")
    try:
        n_out = int(n)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"n must be an integer (got {n!r})") from exc
    if n_out < 2:
        raise ValueError(f"n must be >= 2 (got {n_out})")
    if xa.size < 2:
        raise ValueError("resample needs at least 2 samples")

    xs, ys = _sorted_xy(xa, ya)
    if float(xs[-1] - xs[0]) == 0.0:
        raise ValueError("x has zero width; cannot resample onto a degenerate range")

    x_new = np.linspace(float(xs[0]), float(xs[-1]), n_out)

    if kind == "cubic":
        has_dupes = bool(np.any(np.diff(xs) == 0.0))
        if has_dupes:
            warnings.warn(
                "cubic resampling needs strictly increasing x (duplicates found); "
                "falling back to linear",
                RuntimeWarning,
                stacklevel=2,
            )
        else:
            try:
                from scipy.interpolate import CubicSpline

                spline = CubicSpline(xs, ys)
                return x_new, np.asarray(spline(x_new), dtype=np.float64)
            except Exception:
                warnings.warn(
                    "scipy is unavailable; falling back to linear resampling",
                    RuntimeWarning,
                    stacklevel=2,
                )

    return x_new, np.asarray(np.interp(x_new, xs, ys), dtype=np.float64)


def normalize(y: Any, *, mode: str = "minmax") -> np.ndarray:
    """Rescale a signal, returning a new array of the same length.

    Args:
        y: Signal to normalise, 1-D.
        mode: One of:

            * ``"minmax"`` (default) - affine map onto ``[0, 1]``.
            * ``"zscore"`` - subtract the mean, divide by the standard deviation.
            * ``"max_abs"`` - divide by ``max(|y|)``, mapping onto ``[-1, 1]`` and
              preserving sign and any zero crossing.
            * ``"area"`` - divide by the total absolute area under the curve
              (trapezoidal, assuming unit spacing since no x is supplied), so
              ``sum(|out|) dx == 1``. Uses the absolute area so that a signal
              straddling zero cannot divide by a near-zero net area.

    Returns:
        float64 array of length ``len(y)``.

    Note:
        Statistics are nan-safe (nan samples are ignored when computing the scale,
        and stay nan in the output). A degenerate scale - a constant signal under
        ``minmax``/``zscore``, an all-zero signal under ``max_abs``/``area`` - yields
        zeros rather than nan/inf, since dividing by zero range is undefined and
        zeros are the least surprising answer for a UI.

    Raises:
        ValueError: On empty/non-1-D input or an unknown mode.
    """
    ya = _as_1d(y, "y")
    if mode not in ("minmax", "zscore", "max_abs", "area"):
        raise ValueError(
            f"unknown normalize mode {mode!r}; expected 'minmax', 'zscore', 'max_abs' or 'area'"
        )
    if not np.any(np.isfinite(ya)):
        return ya.copy()

    if mode == "minmax":
        lo = float(np.nanmin(ya))
        hi = float(np.nanmax(ya))
        span = hi - lo
        if span == 0.0:
            return np.zeros_like(ya)
        return (ya - lo) / span

    if mode == "zscore":
        std = float(np.nanstd(ya))
        if std == 0.0:
            return np.zeros_like(ya)
        return (ya - float(np.nanmean(ya))) / std

    if mode == "max_abs":
        peak = float(np.nanmax(np.abs(ya)))
        if peak == 0.0:
            return np.zeros_like(ya)
        return ya / peak

    # area
    absy = np.abs(np.nan_to_num(ya, nan=0.0, posinf=0.0, neginf=0.0))
    if absy.size < 2:
        area = float(absy[0])
    else:
        area = float(np.sum(0.5 * (absy[1:] + absy[:-1])))
    if area == 0.0:
        return np.zeros_like(ya)
    return ya / area


def fit_polynomial(x: Any, y: Any, degree: int) -> Tuple[np.ndarray, np.ndarray]:
    """Least-squares polynomial fit.

    Args:
        x: Sample abscissae, 1-D.
        y: Sample values, 1-D, same length as x.
        degree: Polynomial degree. Clamped to ``[0, min(20, n_finite - 1)]``, so an
            over-ambitious degree from a UI spinner degrades instead of raising.

    Returns:
        ``(coeffs, y_fit)``. ``coeffs`` is highest-order-first, as
        ``np.polyfit``/``np.polyval`` expect. ``y_fit`` has length ``len(x)`` and is
        in the **input order** (no sorting - polynomial evaluation is order
        independent), so it can be plotted directly against the caller's x.

    Raises:
        ValueError: On empty/mismatched input or when no finite sample pairs remain.
    """
    coeffs, _cov, y_fit = fit_polynomial_covariance(x, y, degree)
    return coeffs, y_fit


def fit_polynomial_covariance(
    x: Any, y: Any, degree: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """:func:`fit_polynomial`, plus the coefficient covariance matrix.

    Identical fit and identical ``coeffs``/``y_fit`` -- this exists only because
    :func:`confidence_band` needs a covariance to propagate, and ``fit_polynomial``'s many
    other callers never do, so it is not computed there.

    Returns:
        ``(coeffs, cov, y_fit)``. ``cov`` is ``(degree + 1, degree + 1)`` and is all-inf
        when the fit is exactly- or under-determined (``np.polyfit(cov=True)`` cannot
        estimate a covariance from zero residual degrees of freedom), matching how an
        undetermined nonlinear fit already reports its covariance elsewhere.

    Raises:
        ValueError: On empty/mismatched input or when no finite sample pairs remain.
    """
    xa, ya = _as_xy(x, y)
    good = np.isfinite(xa) & np.isfinite(ya)
    n_good = int(np.count_nonzero(good))
    if n_good == 0:
        raise ValueError("fit_polynomial needs at least one finite (x, y) pair")

    try:
        deg = int(degree)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"degree must be an integer (got {degree!r})") from exc
    deg = max(0, min(deg, min(20, n_good - 1)))

    xg, yg = xa[good], ya[good]
    n_coef = deg + 1
    coeffs = np.polyfit(xg, yg, deg)
    if n_good > n_coef:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # RankWarning on a poorly conditioned fit
                _coeffs_cov, cov = np.polyfit(xg, yg, deg, cov=True)
        except (ValueError, np.linalg.LinAlgError):
            # A high degree on a wide x range makes the Vandermonde matrix numerically
            # rank-deficient even with n_good > n_coef samples; np.polyfit(cov=True) can
            # then fail outright rather than just warn. The coefficients above are still
            # the ordinary least-squares solution -- only the covariance is unavailable.
            cov = np.full((n_coef, n_coef), np.inf, dtype=np.float64)
    else:
        # np.polyfit(cov=True) requires strictly more points than coefficients; an
        # exactly- or under-determined fit reports coefficients with infinite
        # uncertainty instead of raising.
        cov = np.full((n_coef, n_coef), np.inf, dtype=np.float64)

    y_fit = np.polyval(coeffs, xa)
    return (
        np.asarray(coeffs, dtype=np.float64),
        np.asarray(cov, dtype=np.float64),
        np.asarray(y_fit, dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# nonlinear least squares
# ---------------------------------------------------------------------------

FIT_UNAVAILABLE_MESSAGE = (
    "Nonlinear curve fitting needs scipy.optimize.curve_fit, which could not be "
    "imported. Polynomial fitting still works; install or repair scipy (pip install "
    "-U scipy) to fit the named models."
)

#: Hard cap on a custom model's free parameters. A user can type ``a+b+c+...`` forever;
#: past a dozen parameters curve_fit is not going to converge on anything meaningful and
#: the parameter table stops fitting on screen.
_FIT_MAX_PARAMS = 12

#: exp() argument clamp. np.exp(710.0) overflows to inf and poisons a whole residual
#: vector; curve_fit then reports "Residuals are not finite in the initial point" and the
#: user sees a failure caused by an intermediate step rather than by their data. Clamping
#: keeps the model finite everywhere while leaving the fitted region untouched (the
#: clamp only bites far outside any plausible optimum).
_EXP_CLAMP = 700.0

#: Gaussian FWHM -> sigma. 2*sqrt(2*ln 2).
_FWHM_TO_SIGMA = 2.3548200450309493


class FitUnavailableError(ValueError):
    """scipy.optimize.curve_fit could not be imported, so no nonlinear fit is possible.

    Deliberately a :class:`ValueError` subclass: every entry point in this module
    documents ValueError as its failure mode, so a caller that already renders that as a
    message keeps doing so instead of crashing. Callers that want to hide the feature
    entirely rather than report a failure should test :func:`fit_available` first.
    """


class FitModel(NamedTuple):
    """One fittable model.

    The first three fields are the ``(fn, param_names, guess)`` triple that does the
    work; ``label``/``formula`` exist so a UI can name the model and show its equation
    without hardcoding a second copy of the registry, and ``check`` lets a model reject a
    domain it cannot be evaluated on (power law needs ``x > 0``) with a sentence the user
    can act on, rather than by returning nan and letting curve_fit report something
    opaque.

    Attributes:
        fn: ``fn(x, *params) -> ndarray``. Must be finite over a sane domain and must
            accept a 1-D float64 ``x``; this is handed straight to ``curve_fit``.
        param_names: Parameter names, in the order ``fn`` takes them.
        guess: ``guess(x, y) -> ndarray`` of ``len(param_names)`` initial values. It is
            given **finite, ascending-x** samples, never raw input.
        label: Human name for a combo box.
        formula: The equation, as ASCII, for a caption.
        check: Optional ``check(x) -> Optional[str]``. A returned string is an error
            message; ``None`` means the domain is fine.
    """

    fn: Callable[..., np.ndarray]
    param_names: Tuple[str, ...]
    guess: Callable[[np.ndarray, np.ndarray], np.ndarray]
    label: str = ""
    formula: str = ""
    check: Optional[Callable[[np.ndarray], Optional[str]]] = None


def _safe_exp(z: np.ndarray) -> np.ndarray:
    """exp() that saturates instead of overflowing to inf. See ``_EXP_CLAMP``."""
    return np.exp(np.clip(z, -_EXP_CLAMP, _EXP_CLAMP))


def _span(x: np.ndarray) -> float:
    """Positive width of an ascending grid, with a non-zero fallback for a degenerate one."""
    if x.size < 2:
        return 1.0
    width = float(x[-1] - x[0])
    return width if width > 0.0 else 1.0


def _slope_sign(x: np.ndarray, y: np.ndarray) -> float:
    """+1 if y trends up with x, -1 if it trends down. Never 0."""
    if x.size < 2:
        return 1.0
    with np.errstate(all="ignore"):
        try:
            slope = float(np.polyfit(x, y, 1)[0])
        except (np.linalg.LinAlgError, ValueError):
            slope = float(y[-1] - y[0])
    if not np.isfinite(slope):
        slope = float(y[-1] - y[0])
    return -1.0 if slope < 0.0 else 1.0


def _peak_shape(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float, int]:
    """Baseline, amplitude, centre and centre index of a peak- or dip-shaped signal.

    The textbook heuristic is ``c = min(y), a = max - min, mu = x[argmax]``, which
    silently fits the wrong feature when the interesting thing is a **dip** (absorption
    lines, notch filters -- both are ordinary Math Lab inputs). Comparing each extreme
    against the median picks the extremum that actually stands out, so a dip yields a
    negative amplitude and a baseline at the top instead of a fit anchored on noise.
    """
    median = float(np.median(y))
    y_max = float(np.max(y))
    y_min = float(np.min(y))
    if abs(y_max - median) >= abs(y_min - median):
        index = int(np.argmax(y))
        baseline, amplitude = y_min, y_max - y_min
    else:
        index = int(np.argmin(y))
        baseline, amplitude = y_max, y_min - y_max
    if amplitude == 0.0:
        # A constant signal: any width is as good as any other, so give curve_fit a
        # non-degenerate starting point rather than an exactly-zero amplitude.
        amplitude = 1.0
    return baseline, amplitude, float(x[index]), index


def _fwhm(x: np.ndarray, y: np.ndarray, baseline: float, amplitude: float, index: int) -> float:
    """Full width at half maximum of the feature containing ``index``.

    Only the **contiguous run** of half-height samples around the peak counts. A plain
    ``ptp(x[y >= half])`` would let one noise spike at the far end of the record widen
    the estimate to the whole span, which is exactly the case where a good width guess
    matters most.  ``x`` must be ascending.
    """
    half = baseline + 0.5 * amplitude
    over = y >= half if amplitude >= 0.0 else y <= half
    below = np.flatnonzero(~over)
    left = below[below < index]
    right = below[below > index]
    lo = int(left[-1]) + 1 if left.size else 0
    hi = int(right[0]) - 1 if right.size else int(x.size) - 1
    width = float(x[hi] - x[lo])
    if width > 0.0:
        return width
    # A single-sample peak (or a flat signal): fall back to a tenth of the record, which
    # is narrow enough to resolve a real peak and wide enough not to start at zero.
    return _span(x) / 10.0


def _guess_gaussian(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """``a=peak-baseline, mu=x at the peak, sigma=FWHM/2.355, c=baseline``."""
    baseline, amplitude, centre, index = _peak_shape(x, y)
    sigma = _fwhm(x, y, baseline, amplitude, index) / _FWHM_TO_SIGMA
    return np.array([amplitude, centre, max(sigma, 1e-12), baseline], dtype=np.float64)


def _guess_lorentzian(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """As the gaussian, but the half width is FWHM/2 (a Lorentzian's gamma *is* the HWHM)."""
    baseline, amplitude, centre, index = _peak_shape(x, y)
    gamma = _fwhm(x, y, baseline, amplitude, index) / 2.0
    return np.array([amplitude, centre, max(gamma, 1e-12), baseline], dtype=np.float64)


def _guess_exp_decay(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """``a`` and ``tau`` from a log-linear regression against the estimated plateau.

    ``y = a*exp(-x/tau) + c`` is linear in ``log(y - c)``, so with ``c`` estimated as the
    plateau (the extreme the curve is heading towards, whichever direction that is) a
    single ``polyfit`` on the log gives both the rate **and** the amplitude: the slope is
    ``-1/tau`` and the intercept is ``log|a|``. Using the intercept rather than ``y[0]``
    matters, because ``a`` is the value extrapolated back to ``x = 0``, not the first
    sample, and the two differ by ``exp(x0/tau)`` -- orders of magnitude when the record
    does not start at the origin.
    """
    fallback_tau = _span(x) / 3.0
    if _slope_sign(x, y) <= 0.0:
        baseline, sign = float(np.min(y)), 1.0
        z = y - baseline
    else:
        baseline, sign = float(np.max(y)), -1.0
        z = baseline - y

    peak = float(np.max(z)) if z.size else 0.0
    tau = fallback_tau
    amplitude = sign * (peak if peak > 0.0 else 1.0) * float(np.exp(min(x[0] / tau, _EXP_CLAMP)))
    if peak > 0.0:
        keep = z > 1e-6 * peak
        if int(np.count_nonzero(keep)) >= 2 and float(np.ptp(x[keep])) > 0.0:
            with np.errstate(all="ignore"):
                try:
                    rate, intercept = np.polyfit(x[keep], np.log(z[keep]), 1)
                except (np.linalg.LinAlgError, ValueError):
                    rate, intercept = 0.0, 0.0
            if np.isfinite(rate) and np.isfinite(intercept) and rate < 0.0:
                tau = -1.0 / float(rate)
                amplitude = sign * float(np.exp(min(float(intercept), _EXP_CLAMP)))
    return np.array([amplitude, tau, baseline], dtype=np.float64)


def _guess_power_law(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """A straight line in log-log: ``b`` is the slope, ``a`` is ``exp(intercept)``.

    Only strictly positive pairs can enter a log-log fit. If too few survive (a power law
    on data that dips to zero), fall back to ``a=1, b=1`` and let curve_fit work from
    there -- a degenerate log fit would hand it a nan and fail outright.
    """
    keep = (x > 0.0) & (y > 0.0)
    if int(np.count_nonzero(keep)) >= 2 and float(np.ptp(x[keep])) > 0.0:
        with np.errstate(all="ignore"):
            try:
                exponent, intercept = np.polyfit(np.log(x[keep]), np.log(y[keep]), 1)
            except (np.linalg.LinAlgError, ValueError):
                exponent, intercept = 1.0, 0.0
        if np.isfinite(exponent) and np.isfinite(intercept):
            return np.array(
                [float(np.exp(min(float(intercept), _EXP_CLAMP))), float(exponent)],
                dtype=np.float64,
            )
    return np.array([1.0, 1.0], dtype=np.float64)


def _guess_sigmoid(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """``L=ptp, c=min, x0=x at half height, k=+-4/span``.

    ``4/span`` is the steepness whose transition fills roughly the whole record: the
    logistic's slope at its centre is ``kL/4``, so ``k=4/span`` puts the full rise across
    ``span``. The sign comes from the trend, which is what lets one model serve both a
    rising and a falling sigmoid.
    """
    baseline = float(np.min(y))
    amplitude = float(np.ptp(y))
    if amplitude == 0.0:
        amplitude = 1.0
    centre_index = int(np.argmin(np.abs(y - (baseline + 0.5 * amplitude))))
    rate = _slope_sign(x, y) * 4.0 / _span(x)
    return np.array([amplitude, rate, float(x[centre_index]), baseline], dtype=np.float64)


def _guess_linear(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Closed form: the least-squares line *is* the answer, so curve_fit starts at it."""
    with np.errstate(all="ignore"):
        try:
            slope, intercept = np.polyfit(x, y, 1)
        except (np.linalg.LinAlgError, ValueError):
            slope, intercept = 0.0, float(np.mean(y))
    if not (np.isfinite(slope) and np.isfinite(intercept)):
        slope, intercept = 0.0, float(np.mean(y))
    return np.array([float(slope), float(intercept)], dtype=np.float64)


def _model_gaussian(x: np.ndarray, a: float, mu: float, sigma: float, c: float) -> np.ndarray:
    with np.errstate(all="ignore"):
        return c + a * _safe_exp(-((x - mu) ** 2) / (2.0 * sigma * sigma))


def _model_lorentzian(x: np.ndarray, a: float, x0: float, gamma: float, c: float) -> np.ndarray:
    with np.errstate(all="ignore"):
        g2 = gamma * gamma
        return c + a * g2 / ((x - x0) ** 2 + g2)


def _model_exp_decay(x: np.ndarray, a: float, tau: float, c: float) -> np.ndarray:
    with np.errstate(all="ignore"):
        return c + a * _safe_exp(-x / tau)


def _model_power_law(x: np.ndarray, a: float, b: float) -> np.ndarray:
    with np.errstate(all="ignore"):
        return a * np.power(x, b)


def _model_sigmoid(x: np.ndarray, ell: float, k: float, x0: float, c: float) -> np.ndarray:
    with np.errstate(all="ignore"):
        return c + ell / (1.0 + _safe_exp(-k * (x - x0)))


def _model_linear(x: np.ndarray, m: float, b: float) -> np.ndarray:
    return m * x + b


def _check_positive_x(x: np.ndarray) -> Optional[str]:
    """A power law is undefined at x <= 0 for a non-integer exponent."""
    if bool(np.any(x <= 0.0)):
        return (
            "the power law model needs x > 0 (x^b is undefined at and below zero for a "
            "fractional exponent); shift x, or filter the non-positive samples out"
        )
    return None


#: The model registry. Keys are what a UI stores and what :func:`fit_model` accepts.
MODELS: Dict[str, FitModel] = {
    "linear": FitModel(
        fn=_model_linear,
        param_names=("m", "b"),
        guess=_guess_linear,
        label="Linear",
        formula="y = m*x + b",
    ),
    "gaussian": FitModel(
        fn=_model_gaussian,
        param_names=("a", "mu", "sigma", "c"),
        guess=_guess_gaussian,
        label="Gaussian",
        formula="y = a*exp(-(x - mu)^2 / (2*sigma^2)) + c",
    ),
    "lorentzian": FitModel(
        fn=_model_lorentzian,
        param_names=("a", "x0", "gamma", "c"),
        guess=_guess_lorentzian,
        label="Lorentzian",
        formula="y = a*gamma^2 / ((x - x0)^2 + gamma^2) + c",
    ),
    "exp_decay": FitModel(
        fn=_model_exp_decay,
        param_names=("a", "tau", "c"),
        guess=_guess_exp_decay,
        label="Exponential decay",
        formula="y = a*exp(-x/tau) + c",
    ),
    "power_law": FitModel(
        fn=_model_power_law,
        param_names=("a", "b"),
        guess=_guess_power_law,
        label="Power law",
        formula="y = a*x^b",
        check=_check_positive_x,
    ),
    "sigmoid": FitModel(
        fn=_model_sigmoid,
        param_names=("L", "k", "x0", "c"),
        guess=_guess_sigmoid,
        label="Sigmoid (logistic)",
        formula="y = L / (1 + exp(-k*(x - x0))) + c",
    ),
}

#: Registry keys in presentation order. Iterate this, not ``MODELS``, in a UI: dict order
#: is an implementation detail and this one is chosen (simplest model first).
MODEL_NAMES: Tuple[str, ...] = tuple(MODELS)

#: Peak shapes a multi-peak (deconvolution) model can be built from.
PEAK_SHAPES: Tuple[str, ...] = ("gaussian", "lorentzian")

#: Hard cap on the number of peaks in a deconvolution. Past this, a 3N-parameter fit is
#: unlikely to converge and the parameter table stops fitting on screen.
_MAX_MULTIPEAK = 10


def peak_area(shape: str, amplitude: float, width: float) -> float:
    """Analytic area of one peak of the given shape, from its fitted amplitude and width.

    ``gaussian`` -> ``a*sigma*sqrt(2*pi)``; ``lorentzian`` -> ``a*gamma*pi``. This is the
    exact area of a *fitted* peak, which is how a deconvolution recovers the area of
    overlapping peaks that no valley integration can separate.
    """
    a = float(amplitude)
    w = abs(float(width))
    if shape == "gaussian":
        return a * w * float(np.sqrt(2.0 * np.pi))
    if shape == "lorentzian":
        return a * w * float(np.pi)
    raise ValueError(f"unknown peak shape {shape!r}; expected one of {list(PEAK_SHAPES)}")


def multipeak_model(n_peaks: int, shape: str = "gaussian") -> FitModel:
    """Build a :class:`FitModel` for a sum of ``n_peaks`` identical-shape peaks plus a baseline.

    This is the deconvolution model: overlapping peaks that never return to baseline cannot
    be separated by valley integration (see :func:`find_peaks`), but fitting a sum of peak
    shapes to the whole envelope recovers each one's centre, width and -- through
    :func:`peak_area` -- its area, even where the peaks overlap heavily.

    Args:
        n_peaks: Number of peaks in the sum (1 .. 10).
        shape: ``"gaussian"`` or ``"lorentzian"`` (:data:`PEAK_SHAPES`).

    Returns:
        A :class:`FitModel` with parameters ``a1, mu1, sigma1, a2, mu2, sigma2, ..., c``
        (``gamma`` in place of ``sigma`` for a Lorentzian). Its ``guess`` seeds the peak
        centres and widths from :func:`find_peaks`, which is what makes a fit of so many
        parameters converge -- a flat start almost never does.

    Raises:
        ValueError: On ``n_peaks`` out of range or an unknown shape.
    """
    n = int(n_peaks)
    if n < 1 or n > _MAX_MULTIPEAK:
        raise ValueError(f"n_peaks must be between 1 and {_MAX_MULTIPEAK} (got {n_peaks})")
    if shape not in PEAK_SHAPES:
        raise ValueError(f"unknown peak shape {shape!r}; expected one of {list(PEAK_SHAPES)}")
    width_name = "sigma" if shape == "gaussian" else "gamma"
    names: List[str] = []
    for i in range(1, n + 1):
        names.extend((f"a{i}", f"mu{i}", f"{width_name}{i}"))
    names.append("c")
    param_names = tuple(names)

    def one(x: np.ndarray, a: float, mu: float, w: float) -> np.ndarray:
        if shape == "gaussian":
            return a * _safe_exp(-((x - mu) ** 2) / (2.0 * w * w))
        return a * w * w / ((x - mu) ** 2 + w * w)

    def fn(x: np.ndarray, *params: float) -> np.ndarray:
        xa = np.asarray(x, dtype=np.float64)
        out = np.full(xa.shape, float(params[-1]), dtype=np.float64)
        with np.errstate(all="ignore"):
            for i in range(n):
                out = out + one(xa, params[3 * i], params[3 * i + 1], params[3 * i + 2])
        return out

    def guess(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        baseline = float(np.min(y))
        span = _span(x)
        width_factor = _FWHM_TO_SIGMA if shape == "gaussian" else 2.0
        # Seed from the most prominent local maxima; a low floor so overlapping shoulders
        # that barely show as separate maxima are still picked up.
        try:
            px, py, props = find_peaks(x, y, prominence=0.02 * float(np.ptp(y) or 1.0))
        except ValueError:
            px = np.empty(0)
            py = np.empty(0)
            props = {"widths": np.empty(0), "prominences": np.empty(0)}
        # Keep the most prominent peaks first, so requesting fewer peaks than exist fits
        # the tallest structure rather than an arbitrary subset.
        prom = _prominence_key(props)
        order = np.argsort(-prom) if prom.size == px.size and px.size else np.arange(px.size)
        widths = np.asarray(props.get("widths", np.empty(0)), dtype=np.float64)

        seed: List[float] = []
        for i in range(n):
            if i < order.size:
                k = int(order[i])
                amp = float(py[k] - baseline) or 1.0
                centre = float(px[k])
                usable_w = k < widths.size and widths[k] > 0.0
                w = float(widths[k]) / width_factor if usable_w else span / (4 * n)
            else:
                # More peaks requested than found: spread the extras across the record so
                # the optimiser can pull them onto real structure rather than piling up.
                amp = max(float(np.max(y) - baseline), 1.0) * 0.5
                centre = float(x[0]) + (i + 0.5) / n * span
                w = span / (4 * n)
            seed.extend((amp, centre, max(w, span * 1e-4)))
        seed.append(baseline)
        return np.asarray(seed, dtype=np.float64)

    return FitModel(
        fn=fn,
        param_names=param_names,
        guess=guess,
        label=f"{n} {shape} peaks",
        formula=f"y = sum of {n} {shape} peaks + c",
    )


def _prominence_key(props: Dict[str, np.ndarray]) -> np.ndarray:
    """Prominences from a find_peaks properties dict, or zeros (guess ordering helper)."""
    prom = props.get("prominences")
    return np.asarray(prom, dtype=np.float64) if prom is not None else np.zeros(0)


def custom_model(expr: str, *, label: Optional[str] = None) -> FitModel:
    """Build a :class:`FitModel` from a user-typed expression in ``x``.

    Every name in ``expr`` that is neither ``x`` nor a builtin from
    ``expressions.SAFE_NAMES`` becomes a free parameter, sorted by name: ``a*exp(-x/b)+c``
    fits ``a``, ``b``, ``c``. The expression is validated once here, so a hostile or
    malformed one is rejected at build time rather than from inside curve_fit's inner
    loop. Initial guesses are 1.0 for every parameter -- there is nothing to infer from an
    arbitrary functional form, which is exactly why the custom path is the one that most
    needs the manual-guess controls.

    Args:
        expr: The right-hand side of ``y = ...``, e.g. ``"a*exp(-x/b) + c"``.
        label: Optional display name.

    Returns:
        A :class:`FitModel` ready for :func:`fit_model`.

    Raises:
        ValueError: If ``expr`` is empty, invalid, has no free parameters (there would be
            nothing to fit), or has more than ``_FIT_MAX_PARAMS`` of them.
    """
    if not isinstance(expr, str) or not expr.strip():
        raise ValueError("enter an expression in x, for example: a*exp(-x/b) + c")
    try:
        names = expressions.free_variables(expr, exclude=("x",))
    except expressions.ExpressionError as exc:
        raise ValueError(f"invalid expression: {exc}") from exc
    if not names:
        raise ValueError(
            "the expression has no free parameters besides x, so there is nothing to "
            "fit; use a name such as 'a' for each value you want fitted"
        )
    if len(names) > _FIT_MAX_PARAMS:
        raise ValueError(
            f"the expression has {len(names)} free parameters "
            f"({', '.join(names)}); at most {_FIT_MAX_PARAMS} can be fitted"
        )
    param_names = tuple(names)

    def fn(x: np.ndarray, *params: float) -> np.ndarray:
        bindings = {name: float(value) for name, value in zip(param_names, params)}
        try:
            return expressions.evaluate_1d(
                expr, np.asarray(x, dtype=np.float64), variables=bindings
            )
        except expressions.ExpressionError as exc:
            raise ValueError(f"could not evaluate '{expr}': {exc}") from exc

    def guess(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.ones(len(param_names), dtype=np.float64)

    return FitModel(
        fn=fn,
        param_names=param_names,
        guess=guess,
        label=label or "Custom f(x)",
        formula=f"y = {expr}",
    )


def resolve_model(model: Any) -> FitModel:
    """Return the :class:`FitModel` for a registry key, or pass a FitModel through.

    Raises:
        ValueError: For an unknown key or a value that is neither.
    """
    if isinstance(model, FitModel):
        return model
    if isinstance(model, str):
        try:
            return MODELS[model]
        except KeyError:
            raise ValueError(
                f"unknown fit model {model!r}; choose one of {', '.join(MODEL_NAMES)}, "
                "or build one with custom_model(expr)"
            ) from None
    raise ValueError(f"model must be a MODELS key or a FitModel, got {type(model).__name__}")


def fit_available() -> bool:
    """True if a nonlinear fit is possible, i.e. ``scipy.optimize.curve_fit`` imports.

    Ask this before offering the feature. :func:`fit_model` raises
    :class:`FitUnavailableError` (a ValueError) when it is False; this is the way to show
    ``FIT_UNAVAILABLE_MESSAGE`` up front instead of only after a user presses something.
    """
    try:
        from scipy.optimize import curve_fit  # noqa: F401
    except Exception:  # pragma: no cover - only on a broken/absent scipy
        return False
    return True


def _clean_fit_inputs(
    x: Any, y: Any, sigma: Any, n_params: int, model_label: str
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], np.ndarray]:
    """Validate, drop non-finite rows and sort by x. Returns ``(x, y, sigma, x_full)``."""
    xa, ya = _as_xy(x, y)
    good = np.isfinite(xa) & np.isfinite(ya)

    sig: Optional[np.ndarray] = None
    if sigma is not None:
        sig = _as_1d(sigma, "sigma")
        if sig.size == 1 and xa.size != 1:
            sig = np.full(xa.size, float(sig[0]), dtype=np.float64)
        if sig.size != xa.size:
            raise ValueError(f"sigma must have the same length as y (got {sig.size} and {ya.size})")
        good &= np.isfinite(sig)

    n_good = int(np.count_nonzero(good))
    if n_good < n_params:
        raise ValueError(
            f"the {model_label} model has {n_params} parameters and needs at least that "
            f"many finite (x, y) samples; only {n_good} of {xa.size} are usable"
        )

    xg, yg = _sorted_xy(xa[good], ya[good])
    if sig is None:
        return xg, yg, None, xa

    # _sorted_xy sorts a pair; sigma has to ride the same permutation or the weights end
    # up attached to the wrong samples -- a silently wrong fit, not a crash.
    order = np.argsort(xa[good], kind="stable")
    sg = sig[good][order]
    if bool(np.any(sg <= 0.0)):
        raise ValueError("sigma must be strictly positive: it is a standard deviation per point")
    return xg, yg, sg, xa


def initial_guess(x: Any, y: Any, model: Any) -> np.ndarray:
    """The heuristic starting parameters :func:`fit_model` would use for ``(x, y)``.

    Exposed because a UI needs to *show* the automatic guess before a user can sensibly
    override it, and because "reset to auto" has to come from somewhere. The guess is
    computed on the same cleaned, x-sorted, finite-only samples the fit itself uses.

    Returns:
        float64 array of ``len(model.param_names)``. Always finite: a non-finite value
        from a heuristic is replaced by 1.0, since curve_fit rejects a nan start outright
        and a mediocre number beats no fit at all.

    Raises:
        ValueError: On unusable input or an unknown model.
    """
    spec = resolve_model(model)
    n_params = len(spec.param_names)
    xg, yg, _sig, _xa = _clean_fit_inputs(x, y, None, n_params, spec.label or "custom")
    with np.errstate(all="ignore"):
        p0 = np.asarray(spec.guess(xg, yg), dtype=np.float64).ravel()
    if p0.size != n_params:
        raise ValueError(
            f"the guess for model {spec.label!r} produced {p0.size} values but the model "
            f"takes {n_params}"
        )
    return np.where(np.isfinite(p0), p0, 1.0)


def _normalize_bounds(
    bounds: Any, n_params: int
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Coerce a ``(lower, upper)`` pair to arrays, or ``(None, None)`` for unbounded."""
    if bounds is None:
        return None, None
    try:
        lower, upper = bounds
    except (TypeError, ValueError) as exc:
        raise ValueError("bounds must be a (lower, upper) pair") from exc

    def _side(value: Any, name: str) -> np.ndarray:
        arr = np.asarray(value, dtype=np.float64)
        if arr.ndim == 0:
            arr = np.full(n_params, float(arr), dtype=np.float64)
        arr = arr.ravel()
        if arr.size != n_params:
            raise ValueError(
                f"{name} bound must be a scalar or have {n_params} values, got {arr.size}"
            )
        return arr

    lo = _side(lower, "lower")
    hi = _side(upper, "upper")
    if bool(np.any(lo >= hi)):
        raise ValueError("every lower bound must be strictly below its upper bound")
    return lo, hi


def _prepare_fit(
    x: Any, y: Any, model: Any, sigma: Any
) -> Tuple[FitModel, List[str], int, str, np.ndarray, np.ndarray, Optional[np.ndarray], np.ndarray]:
    """Shared setup for :func:`fit_model` and :func:`fit_model_robust`.

    Resolves the model, cleans/sorts/finite-filters the input (and sigma, riding the same
    permutation), and runs the model's own domain check. Split out so the two fit engines
    -- which differ only in which scipy optimiser call they make -- cannot drift on how
    they read their input.

    Returns:
        ``(spec, names, n_params, label, xg, yg, sg, xa)`` -- the resolved model, its
        parameter names and count, its label, the cleaned data actually fit against, the
        matching sigma (or None), and the *full*, unfiltered x the caller's ``y_fit`` must
        align with.
    """
    spec = resolve_model(model)
    names = list(spec.param_names)
    n_params = len(names)
    label = spec.label or "custom"

    xg, yg, sg, xa = _clean_fit_inputs(x, y, sigma, n_params, label)

    if spec.check is not None:
        message = spec.check(xg)
        if message:
            raise ValueError(message)

    return spec, names, n_params, label, xg, yg, sg, xa


def _resolve_start(
    p0: Optional[Sequence[float]],
    xg: np.ndarray,
    yg: np.ndarray,
    spec: FitModel,
    n_params: int,
    names: List[str],
    label: str,
) -> np.ndarray:
    """The initial parameter vector: the heuristic guess, or a validated user ``p0``."""
    if p0 is None:
        return initial_guess(xg, yg, spec)
    start = _as_1d(p0, "p0")
    if start.size != n_params:
        raise ValueError(
            f"p0 must have {n_params} values for model {label!r} "
            f"({', '.join(names)}), got {start.size}"
        )
    return np.where(np.isfinite(start), start, 1.0)


def _run_curve_fit(
    curve_fit_fn: Any,
    fn: Callable[..., Any],
    xg: np.ndarray,
    yg: np.ndarray,
    start: np.ndarray,
    xa: np.ndarray,
    label: str,
    kwargs: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Call ``curve_fit_fn``, shape its covariance, and evaluate the fit at every ``xa``.

    Shared by :func:`fit_model` and :func:`fit_model_robust`: both are ``curve_fit`` with
    different keyword arguments, and both need identical error handling, covariance
    shaping and full-domain evaluation so a caller cannot tell which one ran except by the
    numbers it gets back.
    """
    with warnings.catch_warnings():
        # OptimizeWarning ("covariance could not be estimated") is reported through cov
        # (which comes back inf) and shown as an inf error; a warning on stderr on top of
        # that is noise in a GUI.
        warnings.simplefilter("ignore")
        try:
            with np.errstate(all="ignore"):
                popt, pcov = curve_fit_fn(fn, xg, yg, p0=start, **kwargs)
        except RuntimeError as exc:
            raise ValueError(
                f"the {label} fit did not converge: {exc}. Try a different model, loss "
                "or initial guess."
            ) from exc
        except (ValueError, TypeError, np.linalg.LinAlgError) as exc:
            raise ValueError(f"the {label} fit failed: {exc}") from exc

    params = np.asarray(popt, dtype=np.float64).ravel()
    n_params = params.size
    cov = np.asarray(pcov, dtype=np.float64)
    if cov.shape != (n_params, n_params):
        cov = np.full((n_params, n_params), np.inf, dtype=np.float64)

    with np.errstate(all="ignore"):
        try:
            fitted = fn(xa, *params)
        except ValueError as exc:
            raise ValueError(f"the {label} model could not be evaluated: {exc}") from exc
    y_fit = np.asarray(fitted, dtype=np.float64)
    if y_fit.shape != xa.shape:
        # A constant model ("a") evaluates to a scalar; broadcast so the caller always
        # gets one value per input sample.
        y_fit = np.broadcast_to(y_fit, xa.shape).astype(np.float64, copy=True)

    return params, cov, y_fit


def fit_model(
    x: Any,
    y: Any,
    model: Any,
    *,
    p0: Optional[Sequence[float]] = None,
    bounds: Any = None,
    sigma: Any = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Nonlinear least-squares fit of ``model`` to ``(x, y)``.

    Args:
        x: Sample abscissae, 1-D.
        y: Sample values, 1-D, same length as x.
        model: A :data:`MODELS` key (``"gaussian"``, ``"exp_decay"``, ...) or a
            :class:`FitModel`, e.g. from :func:`custom_model`.
        p0: Initial parameters. ``None`` (the default) runs the model's heuristic --
            see :func:`initial_guess`. **This is the parameter that decides whether the
            fit works**: curve_fit is a local optimiser and a bad start converges to a
            local minimum or not at all.
        bounds: Optional ``(lower, upper)``, each a scalar or one value per parameter.
            With bounds, curve_fit switches from Levenberg-Marquardt to a trust-region
            method. ``p0`` is clipped into the box, because curve_fit rejects an
            infeasible start with an error a user cannot act on.
        sigma: Optional per-point standard deviation (scalar or per sample). When given,
            ``cov`` is the **absolute** covariance implied by those errors; when omitted,
            curve_fit's default scales ``cov`` by the residual variance, which is the
            right thing for unknown errors.

    Returns:
        ``(params, cov, y_fit, param_names)``. ``cov`` is ``(n_params, n_params)`` and may
        contain inf when a parameter is not determined by the data -- pass it to
        :func:`fit_errors`. ``y_fit`` has length ``len(x)`` and is in the **input order**
        (like :func:`fit_polynomial`), so it plots directly against the caller's x.

    Raises:
        FitUnavailableError: If scipy.optimize.curve_fit cannot be imported. It is a
            ValueError subclass; check :func:`fit_available` to avoid it entirely.
        ValueError: On unusable input, an unknown model, a domain the model rejects, or a
            fit that does not converge. Nothing else escapes: a failure here is a message
            for the user, never a traceback.
    """
    try:
        from scipy.optimize import curve_fit
    except Exception as exc:  # pragma: no cover - only on a broken/absent scipy
        raise FitUnavailableError(FIT_UNAVAILABLE_MESSAGE) from exc

    spec, names, n_params, label, xg, yg, sg, xa = _prepare_fit(x, y, model, sigma)
    start = _resolve_start(p0, xg, yg, spec, n_params, names, label)

    lo, hi = _normalize_bounds(bounds, n_params)
    kwargs: Dict[str, Any] = {}
    if lo is None or hi is None:
        # Unbounded: Levenberg-Marquardt, whose iteration cap is spelled maxfev. The trf
        # path below rejects that name, which is why this is not one shared kwarg.
        kwargs["maxfev"] = max(2000, 200 * (n_params + 1))
    else:
        start = np.clip(start, lo, hi)
        kwargs["bounds"] = (lo, hi)
        kwargs["max_nfev"] = max(2000, 200 * (n_params + 1))
    if sg is not None:
        kwargs["sigma"] = sg
        kwargs["absolute_sigma"] = True

    params, cov, y_fit = _run_curve_fit(curve_fit, spec.fn, xg, yg, start, xa, label, kwargs)
    return params, cov, y_fit, names


#: scipy.optimize.least_squares/curve_fit robust loss kinds, "linear" (plain least squares,
#: for comparison) first.
ROBUST_LOSS_KINDS: Tuple[str, ...] = ("linear", "soft_l1", "huber", "cauchy", "arctan")


def fit_model_robust(
    x: Any,
    y: Any,
    model: Any,
    *,
    p0: Optional[Sequence[float]] = None,
    bounds: Any = None,
    sigma: Any = None,
    loss: str = "soft_l1",
    f_scale: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Outlier-robust nonlinear least-squares fit of ``model`` to ``(x, y)``.

    Same contract and return shape as :func:`fit_model` -- interchangeable with it for
    every downstream consumer (:func:`fit_errors`, :func:`fit_statistics`,
    :func:`confidence_band`) -- but each residual is scored by ``loss`` instead of a plain
    square, so a handful of bad points cannot drag every parameter toward them the way an
    ordinary least-squares fit would.

    Args:
        loss: A robust loss kind (``"soft_l1"``, ``"huber"``, ``"cauchy"``, ``"arctan"``)
            or ``"linear"`` for ordinary least squares (see :data:`ROBUST_LOSS_KINDS`).
            Forwarded verbatim to ``scipy.optimize.least_squares``.
        f_scale: The residual magnitude, in the model's own units, past which a point is
            treated as an outlier and down-weighted rather than fit exactly. Points inside
            this margin are fit close to ordinary least squares.

    Implementation note: a robust ``loss`` is a ``least_squares`` feature, and
    ``scipy.optimize.curve_fit`` forwards unrecognised keyword arguments straight to its
    underlying optimiser once ``method="trf"`` selects that code path (the default method,
    Levenberg-Marquardt, does not support a robust loss at all) -- so this is
    :func:`fit_model` with ``method="trf", loss=loss, f_scale=f_scale`` added, not a
    hand-rolled optimiser. ``pcov`` comes back from the same Jacobian-at-the-solution
    convention curve_fit always uses, so :func:`fit_errors` remains meaningful, just for
    the robust objective rather than the plain sum of squares.

    Raises:
        FitUnavailableError: If scipy.optimize.curve_fit cannot be imported.
        ValueError: On unusable input, an unknown model, a domain the model rejects, an
            unknown ``loss``, or a fit that does not converge.
    """
    try:
        from scipy.optimize import curve_fit
    except Exception as exc:  # pragma: no cover - only on a broken/absent scipy
        raise FitUnavailableError(FIT_UNAVAILABLE_MESSAGE) from exc

    loss_kind = str(loss)
    if loss_kind not in ROBUST_LOSS_KINDS:
        raise ValueError(f"unknown loss {loss!r}; choose one of {', '.join(ROBUST_LOSS_KINDS)}")

    spec, names, n_params, label, xg, yg, sg, xa = _prepare_fit(x, y, model, sigma)
    start = _resolve_start(p0, xg, yg, spec, n_params, names, label)

    lo, hi = _normalize_bounds(bounds, n_params)
    kwargs: Dict[str, Any] = {
        "method": "trf",
        "loss": loss_kind,
        "f_scale": float(f_scale),
        "max_nfev": max(2000, 200 * (n_params + 1)),
    }
    if lo is not None and hi is not None:
        start = np.clip(start, lo, hi)
        kwargs["bounds"] = (lo, hi)
    if sg is not None:
        kwargs["sigma"] = sg
        kwargs["absolute_sigma"] = True

    params, cov, y_fit = _run_curve_fit(
        curve_fit, spec.fn, xg, yg, start, xa, f"robust {label}", kwargs
    )
    return params, cov, y_fit, names


def fit_errors(cov: Any) -> np.ndarray:
    """Per-parameter 1-sigma standard errors, ``sqrt(diag(cov))``.

    A negative diagonal entry cannot happen for a true covariance but can survive a
    numerically hopeless fit; it becomes nan rather than a ValueError, because "this
    parameter has no meaningful error" is a result to report, not a crash. An
    undetermined parameter yields inf, which is what curve_fit already signalled.

    Raises:
        ValueError: If ``cov`` is not square 2-D.
    """
    c = np.asarray(cov, dtype=np.float64)
    if c.ndim != 2 or c.shape[0] != c.shape[1]:
        raise ValueError(f"cov must be a square 2-D matrix, got shape {c.shape}")
    diagonal = np.array(np.diag(c), dtype=np.float64)
    with np.errstate(all="ignore"):
        return np.where(diagonal >= 0.0, np.sqrt(np.abs(diagonal)), np.nan)


def fit_statistics(y: Any, y_fit: Any, n_params: int, *, sigma: Any = None) -> Dict[str, float]:
    """Goodness-of-fit numbers for a fit, computed over the finite pairs only.

    Args:
        y: Observed values.
        y_fit: Model values at the same x, same length.
        n_params: Number of fitted parameters, for the degrees of freedom.
        sigma: Optional per-point standard deviation used for chi-squared.

    Returns:
        Dict with ``n``, ``dof``, ``ss_res``, ``ss_tot``, ``r_squared``,
        ``chi_squared``, ``reduced_chi_squared``, ``rmse``. All Python floats.

    Note:
        **Without ``sigma``, chi-squared uses unit weights**, so ``reduced_chi_squared``
        is the residual variance estimate (in y units squared) and *not* the
        dimensionless "should be about 1" statistic. Only with real per-point errors does
        the usual reading apply. Reporting it as if it were dimensionless when no errors
        were supplied is the classic way to make a fit look authoritative and be wrong;
        the caller must label it accordingly.

        ``r_squared`` of a constant signal (zero variance to explain) is 1.0 for an exact
        reproduction and 0.0 otherwise, matching :mod:`glplot.gui.panels.mathlab`.

    Raises:
        ValueError: On empty/mismatched input.
    """
    ya, yf = _as_xy(y, y_fit)
    good = np.isfinite(ya) & np.isfinite(yf)

    sig: Optional[np.ndarray] = None
    if sigma is not None:
        sig = _as_1d(sigma, "sigma")
        if sig.size == 1 and ya.size != 1:
            sig = np.full(ya.size, float(sig[0]), dtype=np.float64)
        if sig.size != ya.size:
            raise ValueError(f"sigma must have the same length as y (got {sig.size}, {ya.size})")
        good &= np.isfinite(sig) & (sig > 0.0)

    n = int(np.count_nonzero(good))
    nan = float("nan")
    if n == 0:
        return {
            "n": 0.0,
            "dof": nan,
            "ss_res": nan,
            "ss_tot": nan,
            "r_squared": nan,
            "chi_squared": nan,
            "reduced_chi_squared": nan,
            "rmse": nan,
        }

    observed = ya[good]
    predicted = yf[good]
    residual = observed - predicted
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((observed - float(np.mean(observed))) ** 2))
    if ss_tot > 0.0:
        r_squared = 1.0 - ss_res / ss_tot
    else:
        r_squared = 1.0 if ss_res <= 0.0 else 0.0

    if sig is not None:
        chi_squared = float(np.sum((residual / sig[good]) ** 2))
    else:
        chi_squared = ss_res

    dof = n - int(n_params)
    reduced = chi_squared / dof if dof > 0 else nan
    return {
        "n": float(n),
        "dof": float(dof),
        "ss_res": ss_res,
        "ss_tot": ss_tot,
        "r_squared": float(r_squared),
        "chi_squared": chi_squared,
        "reduced_chi_squared": float(reduced),
        "rmse": float(np.sqrt(ss_res / n)),
    }


def fft_spectrum(x: Any, y: Any) -> Tuple[np.ndarray, np.ndarray]:
    """One-sided amplitude spectrum of y sampled at x.

    Args:
        x: Sample abscissae, 1-D, at least 2 samples with a non-zero span.
        y: Sample values, 1-D, same length as x.

    Returns:
        ``(freqs, magnitude)``, both float64 of length ``len(x) // 2 + 1``.
        ``freqs`` is in cycles per unit of x. ``magnitude`` is scaled so a pure
        sinusoid of amplitude A produces a peak of height A: bins are divided by n
        and the non-DC, non-Nyquist bins doubled to fold in the negative
        frequencies.

    Note:
        The FFT assumes uniform sampling. **If x is non-uniform it is linearly
        resampled onto a uniform grid of the same length first**, which is a real
        approximation, not a formality - a badly non-uniform grid will smear the
        spectrum. Unsorted x is sorted. Non-finite y samples are linearly
        interpolated away first, because a single nan would otherwise make the whole
        transform nan.

    Raises:
        ValueError: On empty/mismatched input, fewer than 2 samples, a zero-width x
            range, or an all-nan y.
    """
    xa, ya = _as_xy(x, y)
    if xa.size < 2:
        raise ValueError("fft_spectrum needs at least 2 samples")

    xs, ys = _sorted_xy(xa, ya)
    if float(xs[-1] - xs[0]) == 0.0:
        raise ValueError("x has zero width; cannot compute a spectrum")

    ys = _fill_nonfinite(ys, "y")
    if not _is_uniform(xs):
        xs, ys = resample(xs, ys, xs.size, kind="linear")

    n = xs.size
    dt = float(xs[-1] - xs[0]) / (n - 1)
    if dt <= 0.0:
        raise ValueError("x must be increasing to define a sampling interval")

    spectrum = np.fft.rfft(ys)
    freqs = np.fft.rfftfreq(n, d=dt)
    magnitude = np.abs(spectrum) / n
    if magnitude.size > 1:
        magnitude[1:] *= 2.0
        if n % 2 == 0:
            # The Nyquist bin has no negative-frequency twin, so undo the doubling.
            magnitude[-1] /= 2.0
    return np.asarray(freqs, dtype=np.float64), np.asarray(magnitude, dtype=np.float64)


def envelope(x: Any, y: Any) -> Tuple[np.ndarray, np.ndarray]:
    """The amplitude envelope of an oscillating signal, via the analytic signal.

    Args:
        x: Sample abscissae, 1-D, at least 2 samples with a non-zero span.
        y: Sample values, 1-D, same length as x.

    Returns:
        ``(x_out, env)``. ``env`` is ``|analytic(y)|``, the magnitude of the Hilbert
        transform's analytic signal -- the smooth curve that traces the top of an
        amplitude-modulated or decaying oscillation, exactly what you would draw by eye
        connecting each cycle's peaks. ``x_out`` is ``x`` unless it was resampled (see
        below), in which case it is the uniform grid ``env`` was actually computed on.

    Note:
        Same uniform-sampling requirement and resampling behaviour as
        :func:`fft_spectrum` (this *is* an FFT-domain operation): non-uniform x is
        linearly resampled onto a uniform grid of the same length first. Unsorted x is
        sorted. Non-finite y samples are linearly interpolated away first. Pure numpy --
        the Hilbert transform here is the same length-N Fourier-domain construction
        ``scipy.signal.hilbert`` uses (multiply the positive frequencies by 2, zero the
        negative ones, keep DC and Nyquist unscaled), reimplemented directly since it is
        a fixed ~10-line algorithm with no approximation to it, not something worth a
        scipy dependency for.

    Raises:
        ValueError: On empty/mismatched input, fewer than 2 samples, or a zero-width x
            range.
    """
    xa, ya = _as_xy(x, y)
    if xa.size < 2:
        raise ValueError("envelope needs at least 2 samples")

    xs, ys = _sorted_xy(xa, ya)
    if float(xs[-1] - xs[0]) == 0.0:
        raise ValueError("x has zero width; cannot compute an envelope")

    ys = _fill_nonfinite(ys, "y")
    if not _is_uniform(xs):
        xs, ys = resample(xs, ys, xs.size, kind="linear")

    n = ys.size
    spectrum = np.fft.fft(ys)
    weights = np.zeros(n, dtype=np.float64)
    half = n // 2
    if n % 2 == 0:
        weights[0] = 1.0
        weights[half] = 1.0
        weights[1:half] = 2.0
    else:
        weights[0] = 1.0
        weights[1 : half + 1] = 2.0
    analytic = np.fft.ifft(spectrum * weights)
    env = np.abs(analytic)
    return np.asarray(xs, dtype=np.float64), np.asarray(env, dtype=np.float64)


# ---------------------------------------------------------------------------
# peak detection
# ---------------------------------------------------------------------------


def _local_maxima(y: np.ndarray) -> np.ndarray:
    """Indices of strict-plateau-aware local maxima, matching scipy.signal.find_peaks.

    A sample is a peak if it is strictly greater than its neighbours; a flat top counts
    once, at the middle of the plateau (rounded down), so a boxcar-topped pulse yields one
    peak rather than none. Endpoints are never peaks: a maximum at the very edge has no
    neighbour to rise above and is by convention not reported.
    """
    n = y.size
    if n < 3:
        return np.empty(0, dtype=np.intp)
    peaks: List[int] = []
    i = 1
    while i < n - 1:
        if y[i - 1] < y[i]:
            # Walk across a possible flat top: [i, i_end] all equal and above the left.
            i_end = i
            while i_end < n - 1 and y[i_end + 1] == y[i]:
                i_end += 1
            if y[i_end + 1] < y[i]:
                peaks.append((i + i_end) // 2)
            i = i_end + 1
        else:
            i += 1
    return np.asarray(peaks, dtype=np.intp)


def _prominences(y: np.ndarray, peaks: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Topographic prominence and base indices of each peak, matching scipy.

    For each peak the search runs outward in both directions, stopping at the first sample
    that is higher than the peak (or the array end). The prominence is the peak height
    minus the higher of the two minima found on those two intervals -- the "key col" a
    hiker must cross to reach a taller summit. The two minima positions are the left and
    right *bases* (the valleys bounding the peak), returned so a caller can integrate the
    area under the peak between them.

    Returns ``(prominences, left_bases, right_bases)``. Pure numpy; used by the fallback
    path and as the honest way to apply a prominence threshold when scipy is absent.
    """
    out = np.zeros(peaks.size, dtype=np.float64)
    left_bases = np.zeros(peaks.size, dtype=np.intp)
    right_bases = np.zeros(peaks.size, dtype=np.intp)
    n = y.size
    for k, peak in enumerate(peaks):
        height = y[peak]
        i = peak - 1
        left_min = height
        left_base = peak
        while i >= 0 and y[i] <= height:
            if y[i] < left_min:
                left_min = y[i]
                left_base = i
            i -= 1
        j = peak + 1
        right_min = height
        right_base = peak
        while j < n and y[j] <= height:
            if y[j] < right_min:
                right_min = y[j]
                right_base = j
            j += 1
        out[k] = height - max(left_min, right_min)
        left_bases[k] = left_base
        right_bases[k] = right_base
    return out, left_bases, right_bases


def _enforce_distance(peaks: np.ndarray, priority: np.ndarray, distance: int) -> np.ndarray:
    """Keep the tallest peaks such that no two are within ``distance`` samples.

    Mirrors scipy's greedy rule: peaks are removed in ascending priority (height) order,
    and removing one un-blocks nothing, so the survivors are exactly the tallest in every
    ``distance``-wide neighbourhood. ``peaks`` must be sorted ascending.
    """
    keep = np.ones(peaks.size, dtype=bool)
    order = np.argsort(priority, kind="stable")  # lowest priority first
    for idx in order[::-1]:  # highest first: a tall peak suppresses shorter neighbours
        if not keep[idx]:
            continue
        j = idx - 1
        while j >= 0 and peaks[idx] - peaks[j] < distance:
            keep[j] = False
            j -= 1
        j = idx + 1
        while j < peaks.size and peaks[j] - peaks[idx] < distance:
            keep[j] = False
            j += 1
    return np.flatnonzero(keep)


def find_peaks(
    x: Any,
    y: Any,
    *,
    height: Optional[float] = None,
    prominence: Optional[float] = None,
    distance: Optional[int] = None,
    width: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Locate local maxima of a signal and measure each one.

    This is the "find the peaks" operation: given a sampled curve it returns where the
    peaks are and how big they are, which is the first step of almost every spectroscopy,
    chromatography or event-detection workflow.

    Args:
        x: Sample abscissae, 1-D.
        y: Sample values, 1-D, same length as x.
        height: Keep only peaks whose value is at least this (in y units). ``None``
            disables the filter.
        prominence: Keep only peaks that stand at least this far above the surrounding
            baseline (in y units) -- the single most useful knob for rejecting noise
            wiggles while keeping real, possibly-small peaks. ``None`` disables it.
        distance: Minimum separation between accepted peaks, **in samples**. When two are
            closer, the shorter is dropped. ``None`` disables it. Matches
            ``scipy.signal.find_peaks``; note it is a sample count, not an x distance.
        width: Keep only peaks at least this wide at half prominence, **in x units**.
            ``None`` disables it.

    Returns:
        ``(peak_x, peak_y, properties)``. ``peak_x``/``peak_y`` are the accepted peaks in
        ascending-x order. ``properties`` always carries ``"prominences"``, ``"widths"``
        (x units), ``"heights"``, and ``"areas"``, all aligned with the peaks; empty arrays
        when no peak survives. Peaks are indices into the **x-sorted** signal, so the
        coordinates are the ones a plot would show.

        ``"areas"`` is each peak integrated above the straight baseline joining its two
        bounding valleys (the standard valley-to-valley quantification), with limits pulled
        in to the valley between adjacent peaks so neighbours never double-count a shoulder.
        This is **accurate for baseline-separated peaks** and is exact for an isolated one.
        Heavily *overlapping* peaks do not return to baseline between them, so no valley
        integration can separate their areas correctly (the shared region gets split by a
        straight cut, and a value can even come out negative): those need deconvolution --
        fit a sum of peak models instead.

    Note:
        Unsorted x is sorted first. Non-finite y makes a sample ineligible as a peak (it
        cannot be compared), but is otherwise left in place. scipy powers the width and
        (when scipy is present) the prominence measurement; a pure-numpy fallback computes
        local maxima, prominence and a half-prominence width directly, so every filter
        still applies without scipy -- accuracy of the width estimate is the only thing
        that degrades.

    Raises:
        ValueError: On empty/mismatched input, or a negative distance/width.
    """
    xa, ya = _as_xy(x, y)
    xs, ys = _sorted_xy(xa, ya)

    if distance is not None:
        try:
            distance = int(distance)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"distance must be an integer number of samples (got {distance!r})"
            ) from exc
        if distance < 1:
            raise ValueError(f"distance must be >= 1 sample (got {distance})")
    if width is not None and float(width) < 0.0:
        raise ValueError(f"width must be >= 0 (got {width!r})")

    empty = np.empty(0, dtype=np.float64)
    dx = float(np.median(np.diff(xs))) if xs.size > 1 else 1.0
    if dx <= 0.0:
        dx = 1.0

    scipy_peaks = _scipy_find_peaks(ys, height, prominence, distance, width, dx)
    if scipy_peaks is not None:
        peaks, prom, widths_x, left_bases, right_bases = scipy_peaks
    else:
        peaks, prom, widths_x, left_bases, right_bases = _numpy_find_peaks(
            ys, height, prominence, distance, width, dx
        )

    if peaks.size == 0:
        return (
            empty,
            empty,
            {"prominences": empty, "widths": empty, "heights": empty, "areas": empty},
        )
    int_left, int_right = _integration_bounds(ys, peaks, left_bases, right_bases)
    areas = _peak_areas(xs, ys, int_left, int_right)
    return (
        np.asarray(xs[peaks], dtype=np.float64),
        np.asarray(ys[peaks], dtype=np.float64),
        {
            "prominences": np.asarray(prom, dtype=np.float64),
            "widths": np.asarray(widths_x, dtype=np.float64),
            "heights": np.asarray(ys[peaks], dtype=np.float64),
            "areas": np.asarray(areas, dtype=np.float64),
        },
    )


def _integration_bounds(
    y: np.ndarray, peaks: np.ndarray, left_bases: np.ndarray, right_bases: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Pull each peak's integration limits in to the valley between it and its neighbours.

    A prominence base can run past an adjacent peak entirely (nothing higher stops it
    until well beyond the neighbour), which would double-count a shared shoulder when
    integrating. Clamping the left limit to the minimum between this peak and the previous
    one -- and the right limit to the minimum before the next peak -- is the standard
    valley-to-valley rule: adjacent peaks then meet exactly at their shared valley and
    neither claims the other's area. Isolated peaks keep their prominence bases.
    """
    n = int(peaks.size)
    left = np.array(left_bases, dtype=np.intp)
    right = np.array(right_bases, dtype=np.intp)
    for k in range(n):
        if k > 0:
            lo, hi = int(peaks[k - 1]), int(peaks[k])
            valley = lo + int(np.argmin(y[lo : hi + 1]))
            left[k] = max(int(left[k]), valley)
        if k < n - 1:
            lo, hi = int(peaks[k]), int(peaks[k + 1])
            valley = lo + int(np.argmin(y[lo : hi + 1]))
            right[k] = min(int(right[k]), valley)
    return left, right


def _peak_areas(
    x: np.ndarray, y: np.ndarray, left_bases: np.ndarray, right_bases: np.ndarray
) -> np.ndarray:
    """Trapezoidal area of each peak above the straight baseline joining its two valleys.

    This is the chromatographer's peak area: integrate the signal from the left base to
    the right base and subtract the trapezoid under the line connecting those two valley
    points, so a peak sitting on a sloped or raised background is measured against that
    background rather than against zero. Honours non-uniform x.
    """
    areas = np.zeros(int(left_bases.size), dtype=np.float64)
    for k in range(left_bases.size):
        lb = int(left_bases[k])
        rb = int(right_bases[k])
        if rb <= lb:
            continue
        xs = x[lb : rb + 1]
        ys = y[lb : rb + 1]
        total = float(np.sum(0.5 * (ys[1:] + ys[:-1]) * np.diff(xs)))
        baseline = 0.5 * (float(y[lb]) + float(y[rb])) * (float(x[rb]) - float(x[lb]))
        areas[k] = total - baseline
    return areas


def _scipy_find_peaks(
    ys: np.ndarray,
    height: Optional[float],
    prominence: Optional[float],
    distance: Optional[int],
    width: Optional[float],
    dx: float,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """scipy.signal.find_peaks path, or None if scipy is unavailable (caller falls back).

    Returns ``(peaks, prominences, widths_x, left_bases, right_bases)``.
    """
    try:
        from scipy.signal import find_peaks as _sp_find_peaks
        from scipy.signal import peak_prominences, peak_widths
    except Exception:
        return None
    # width is expressed in x units by the caller; scipy measures it in samples.
    width_samples = None if width is None else float(width) / dx
    kwargs: Dict[str, Any] = {}
    if height is not None:
        kwargs["height"] = float(height)
    if prominence is not None:
        kwargs["prominence"] = float(prominence)
    if distance is not None:
        kwargs["distance"] = int(distance)
    if width_samples is not None:
        kwargs["width"] = width_samples
    try:
        peaks, props = _sp_find_peaks(ys, **kwargs)
        peaks = np.asarray(peaks, dtype=np.intp)
        empty_i = np.empty(0, dtype=np.intp)
        # Bases are needed for the area; take them straight from find_peaks when it already
        # measured prominence, else compute them once with peak_prominences.
        if peaks.size == 0:
            return peaks, np.empty(0), np.empty(0), empty_i, empty_i
        if "prominences" in props and "left_bases" in props:
            prom = np.asarray(props["prominences"], dtype=np.float64)
            left_bases = np.asarray(props["left_bases"], dtype=np.intp)
            right_bases = np.asarray(props["right_bases"], dtype=np.intp)
        else:
            prom_arr, lb, rb = peak_prominences(ys, peaks)
            prom = np.asarray(prom_arr, dtype=np.float64)
            left_bases = np.asarray(lb, dtype=np.intp)
            right_bases = np.asarray(rb, dtype=np.intp)
        if "widths" in props:
            widths = np.asarray(props["widths"], dtype=np.float64)
        else:
            widths = np.asarray(peak_widths(ys, peaks)[0], dtype=np.float64)
        return peaks, prom, widths * dx, left_bases, right_bases
    except Exception:
        return None


def _numpy_find_peaks(
    ys: np.ndarray,
    height: Optional[float],
    prominence: Optional[float],
    distance: Optional[int],
    width: Optional[float],
    dx: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pure-numpy peak finder honouring every filter, used when scipy is absent.

    Returns ``(peaks, prominences, widths_x, left_bases, right_bases)``.
    """
    empty = np.empty(0, dtype=np.float64)
    empty_i = np.empty(0, dtype=np.intp)
    peaks = _local_maxima(ys)
    peaks = peaks[np.isfinite(ys[peaks])] if peaks.size else peaks

    if height is not None and peaks.size:
        peaks = peaks[ys[peaks] >= float(height)]
    if distance is not None and peaks.size:
        peaks = peaks[_enforce_distance(peaks, ys[peaks], int(distance))]

    if not peaks.size:
        return peaks, empty, empty, empty_i, empty_i

    prom, left_bases, right_bases = _prominences(ys, peaks)
    if prominence is not None:
        keep = prom >= float(prominence)
        peaks, prom = peaks[keep], prom[keep]
        left_bases, right_bases = left_bases[keep], right_bases[keep]

    widths = _half_prominence_widths(ys, peaks, prom) * dx if peaks.size else empty
    if width is not None and peaks.size:
        keep = widths >= float(width)
        peaks, prom, widths = peaks[keep], prom[keep], widths[keep]
        left_bases, right_bases = left_bases[keep], right_bases[keep]
    return peaks, prom, widths, left_bases, right_bases


def _half_prominence_widths(y: np.ndarray, peaks: np.ndarray, prom: np.ndarray) -> np.ndarray:
    """Width in samples at half prominence, linearly interpolated at the crossings.

    A numpy stand-in for ``scipy.signal.peak_widths`` at ``rel_height=0.5``: from each peak
    walk out to where the signal first drops to ``height - prominence/2`` and interpolate
    the fractional crossing, so a narrow peak on a coarse grid is not quantised to whole
    samples.
    """
    out = np.zeros(peaks.size, dtype=np.float64)
    n = y.size
    for k, peak in enumerate(peaks):
        level = y[peak] - 0.5 * prom[k]

        i = peak
        while i > 0 and y[i] > level:
            i -= 1
        if y[i] <= level and i < peak:
            left = i + (level - y[i]) / (y[i + 1] - y[i]) if y[i + 1] != y[i] else float(i)
        else:
            left = float(peak)

        j = peak
        while j < n - 1 and y[j] > level:
            j += 1
        if y[j] <= level and j > peak:
            right = (
                (j - 1) + (y[j - 1] - level) / (y[j - 1] - y[j]) if y[j - 1] != y[j] else float(j)
            )
        else:
            right = float(peak)

        out[k] = max(right - left, 0.0)
    return out


# ---------------------------------------------------------------------------
# filtering / detrending
# ---------------------------------------------------------------------------

#: Frequency responses the Butterworth filter offers. "lowpass"/"highpass" take one
#: cutoff; "bandpass"/"bandstop" take a (low, high) pair.
FILTER_TYPES: Tuple[str, ...] = ("lowpass", "highpass", "bandpass", "bandstop")

#: Trend shapes detrend() can remove.
DETREND_KINDS: Tuple[str, ...] = ("constant", "linear")


#: IIR filter families iir_filter() offers, mirroring scipy.signal.iirfilter's ftype.
#: Excludes "ellip": it needs BOTH ripple and attenuation specified together, which
#: would double the UI's parameter surface for a family the other four already cover
#: well (cheby1/cheby2 each isolate one of those two trade-offs on its own).
IIR_FAMILIES: Tuple[str, ...] = ("butter", "cheby1", "cheby2", "bessel")


def iir_filter(
    x: Any,
    y: Any,
    cutoff: Any,
    *,
    btype: str = "lowpass",
    family: str = "butter",
    order: int = 4,
    ripple: float = 1.0,
    attenuation: float = 40.0,
) -> np.ndarray:
    """Zero-phase IIR filter of y, returning a new array of the same length.

    The workhorse of frequency-selective smoothing: applied forwards-and-backwards
    (``filtfilt``) it introduces **no phase shift**, so a filtered peak stays where it
    was, regardless of which family below is chosen.

    Args:
        x: Sample abscissae, 1-D. Only the spacing matters; it sets the sample rate.
        y: Signal to filter, 1-D, same length as x.
        cutoff: Cutoff frequency in cycles per unit of x. A scalar for ``lowpass``/
            ``highpass``; a ``(low, high)`` pair for ``bandpass``/``bandstop``.
        btype: One of :data:`FILTER_TYPES`.
        family: One of :data:`IIR_FAMILIES`:

            * ``"butter"`` (default) -- maximally flat passband, no ripple anywhere.
              The safe, textbook default.
            * ``"cheby1"`` -- sharper roll-off than butter at the same order, at the
              cost of ripple IN THE PASSBAND, sized by ``ripple`` (dB).
            * ``"cheby2"`` -- sharper roll-off with a flat passband like butter, but
              ripple IN THE STOPBAND instead, sized by ``attenuation`` (dB).
            * ``"bessel"`` -- gentler roll-off, but near-linear phase and minimal
              ringing/overshoot -- best when preserving a pulse or waveform's SHAPE
              matters more than a sharp cutoff.
        order: Filter order (>= 1). Higher is a sharper transition (and, for
            cheby1/cheby2/bessel, more pronounced ripple/overshoot); the effective
            order is doubled by the forward-backward pass.
        ripple: Passband ripple in dB, ``cheby1`` only. Ignored otherwise.
        attenuation: Stopband attenuation in dB, ``cheby2`` only. Ignored otherwise.

    Returns:
        float64 array of length ``len(y)``, in ascending-x order.

    Note:
        Unsorted x is sorted first, and a non-uniform grid is linearly resampled onto a
        uniform one before filtering and mapped back afterwards, because a digital filter
        is only defined on evenly spaced samples -- the same honest approximation
        :func:`fft_spectrum` makes. Non-finite y is interpolated across first.
        **Requires scipy**: there is no faithful pure-numpy ``filtfilt``, so this raises
        :class:`FilterUnavailableError` (a ValueError) rather than degrade to a different
        filter. Cutoffs are clamped just inside ``(0, Nyquist)`` so a slider at the extreme
        cannot raise.

    Raises:
        ValueError: On empty/mismatched input, an unknown btype/family, a bad cutoff
            (wrong count for the type, non-positive, or low >= high for a band), order
            < 1, or a zero-width/degenerate x. FilterUnavailableError if scipy is
            missing.
    """
    xa, ya = _as_xy(x, y)
    if btype not in FILTER_TYPES:
        raise ValueError(f"unknown filter type {btype!r}; expected one of {list(FILTER_TYPES)}")
    if family not in IIR_FAMILIES:
        raise ValueError(f"unknown filter family {family!r}; expected one of {list(IIR_FAMILIES)}")
    if int(order) < 1:
        raise ValueError(f"order must be >= 1 (got {order})")

    try:
        from scipy.signal import iirfilter, sosfiltfilt
    except Exception as exc:
        raise FilterUnavailableError(FILTER_UNAVAILABLE_MESSAGE) from exc

    xs, ys = _sorted_xy(xa, ya)
    if xs.size < 4:
        raise ValueError("IIR filtering needs at least 4 samples")
    if float(xs[-1] - xs[0]) == 0.0:
        raise ValueError("x has zero width; cannot define a sample rate")

    ys = _fill_nonfinite(ys, "y")
    resampled = not _is_uniform(xs)
    if resampled:
        xu, yu = resample(xs, ys, xs.size, kind="linear")
    else:
        xu, yu = xs, ys

    n = xu.size
    fs = (n - 1) / float(xu[-1] - xu[0])  # samples per unit x
    nyquist = 0.5 * fs

    cutoffs = np.atleast_1d(np.asarray(cutoff, dtype=np.float64)).ravel()
    needs_pair = btype in ("bandpass", "bandstop")
    if needs_pair and cutoffs.size != 2:
        raise ValueError(f"{btype} needs a (low, high) cutoff pair, got {cutoffs.size} value(s)")
    if not needs_pair and cutoffs.size != 1:
        raise ValueError(f"{btype} needs a single cutoff, got {cutoffs.size} values")
    if bool(np.any(~np.isfinite(cutoffs))) or bool(np.any(cutoffs <= 0.0)):
        raise ValueError("cutoff frequencies must be positive and finite")

    eps = nyquist * 1e-6
    wn = np.clip(cutoffs, eps, nyquist - eps) / nyquist
    if needs_pair:
        if wn[0] >= wn[1]:
            raise ValueError("the low cutoff must be strictly below the high cutoff")
        wn = (float(wn[0]), float(wn[1]))
    else:
        wn = float(wn[0])

    sos = iirfilter(
        int(order),
        wn,
        rp=float(ripple) if family == "cheby1" else None,
        rs=float(attenuation) if family == "cheby2" else None,
        btype=btype,
        ftype=family,
        output="sos",
    )
    filtered = sosfiltfilt(sos, yu)

    if resampled:
        # Map the uniform-grid result back onto the caller's own abscissae.
        filtered = np.interp(xs, xu, filtered)
    return np.asarray(filtered, dtype=np.float64)


def butter_filter(
    x: Any,
    y: Any,
    cutoff: Any,
    *,
    btype: str = "lowpass",
    order: int = 4,
) -> np.ndarray:
    """Zero-phase Butterworth filter of y. A thin :func:`iir_filter` wrapper fixed to
    ``family="butter"`` -- kept as its own function since it predates
    :data:`IIR_FAMILIES` and is the tab's (and callers') long-standing default path.

    See :func:`iir_filter` for the full parameter and error documentation.
    """
    return iir_filter(x, y, cutoff, btype=btype, family="butter", order=order)


FILTER_UNAVAILABLE_MESSAGE = (
    "Butterworth filtering needs scipy.signal, which could not be imported. Install or "
    "repair scipy (pip install -U scipy) to use the Filter tab; the Smooth tab's "
    "moving-average and gaussian methods work without scipy."
)


class FilterUnavailableError(ValueError):
    """scipy.signal could not be imported, so no Butterworth filter is possible.

    A :class:`ValueError` subclass for the same reason as :class:`FitUnavailableError`: a
    caller already rendering this module's documented failure mode shows a message rather
    than crashing.
    """


def filter_available() -> bool:
    """True if :func:`butter_filter` can run, i.e. ``scipy.signal`` imports."""
    try:
        from scipy.signal import butter, sosfiltfilt  # noqa: F401
    except Exception:  # pragma: no cover - only on a broken/absent scipy
        return False
    return True


def detrend(x: Any, y: Any, *, kind: str = "linear") -> np.ndarray:
    """Remove a constant or linear trend from y, returning a new same-length array.

    Args:
        x: Sample abscissae, 1-D. Used by ``"linear"`` so the fitted line honours the real
            spacing; ignored (but validated) by ``"constant"``.
        y: Signal to detrend, 1-D, same length as x.
        kind: ``"constant"`` subtracts the mean; ``"linear"`` subtracts the least-squares
            line, leaving the residual about a flat zero baseline.

    Returns:
        float64 array of length ``len(y)``, in the **input order** (no sorting: subtracting
        a fitted trend is order independent).

    Note:
        nan-safe: the mean/line is fit over the finite samples only, and non-finite inputs
        stay non-finite in the output. Pure numpy; no scipy needed.

    Raises:
        ValueError: On empty/mismatched input or an unknown kind.
    """
    xa, ya = _as_xy(x, y)
    if kind not in DETREND_KINDS:
        raise ValueError(f"unknown detrend kind {kind!r}; expected 'constant' or 'linear'")
    good = np.isfinite(xa) & np.isfinite(ya)
    if not bool(np.any(good)):
        return ya.copy()

    if kind == "constant":
        return ya - float(np.mean(ya[good]))

    with np.errstate(all="ignore"):
        try:
            slope, intercept = np.polyfit(xa[good], ya[good], 1)
        except (np.linalg.LinAlgError, ValueError):
            slope, intercept = 0.0, float(np.mean(ya[good]))
    if not (np.isfinite(slope) and np.isfinite(intercept)):
        slope, intercept = 0.0, float(np.mean(ya[good]))
    return ya - (slope * xa + intercept)


BASELINE_UNAVAILABLE_MESSAGE = (
    "Asymmetric-least-squares baseline estimation needs scipy.sparse, which could not be "
    "imported. The constant and linear detrend modes work without scipy; install or repair "
    "scipy (pip install -U scipy) to remove a curved baseline."
)


class BaselineUnavailableError(ValueError):
    """scipy.sparse could not be imported, so no AsLS baseline can be estimated.

    A :class:`ValueError` subclass, for the same reason as :class:`FilterUnavailableError`.
    """


def baseline_available() -> bool:
    """True if :func:`baseline_asls` can run, i.e. ``scipy.sparse`` imports."""
    try:
        from scipy.sparse.linalg import spsolve  # noqa: F401
    except Exception:  # pragma: no cover - only on a broken/absent scipy
        return False
    return True


def baseline_asls(y: Any, *, lam: float = 1e5, p: float = 0.01, niter: int = 10) -> np.ndarray:
    """Estimate a smooth, curved baseline under a peaky signal (Eilers & Boelens AsLS).

    Asymmetric least squares fits a baseline that is penalised for roughness and for
    following the signal *upward*, so it slides under peaks and tracks a curved background
    (fluorescence, scattering, a drifting detector) that a straight-line detrend cannot.
    Subtract the returned baseline from the signal to flatten it before peak detection,
    integration or fitting.

    Args:
        y: Signal, 1-D. Non-finite samples are interpolated across first.
        lam: Smoothness (lambda). Larger is a stiffer, flatter baseline; typical range
            1e2 .. 1e9. The default 1e5 suits a few hundred to a few thousand samples.
        p: Asymmetry, in ``(0, 1)``. The weight given to points *above* the baseline;
            small (~0.01) keeps the baseline below the peaks. ``1 - p`` weights points
            below it.
        niter: Reweighting iterations. Ten is plenty; the weights settle quickly.

    Returns:
        float64 array of length ``len(y)`` -- the baseline, **not** the corrected signal
        (subtract it yourself, so the caller controls what happens to the result).

    Note:
        **Requires scipy** (``scipy.sparse``): the penalised system is banded and there is
        no fast pure-numpy solve for it at scale, so this raises
        :class:`BaselineUnavailableError` rather than degrade. A signal shorter than 3
        samples has no second difference to penalise and is returned as a flat mean.

    Raises:
        BaselineUnavailableError: If scipy.sparse cannot be imported (a ValueError
            subclass; check :func:`baseline_available` to avoid it).
        ValueError: On empty/non-1-D input, an all-nan signal, ``p`` outside ``(0, 1)``,
            ``lam <= 0``, or ``niter < 1``.
    """
    ya = _as_1d(y, "y")
    p = float(p)
    lam = float(lam)
    if not (0.0 < p < 1.0):
        raise ValueError(f"p must be in the open interval (0, 1) (got {p})")
    if not (lam > 0.0 and np.isfinite(lam)):
        raise ValueError(f"lam must be a positive finite number (got {lam})")
    if int(niter) < 1:
        raise ValueError(f"niter must be >= 1 (got {niter})")

    z = _fill_nonfinite(ya, "y")
    n = z.size
    if n < 3:
        return np.full(n, float(np.mean(z)), dtype=np.float64)

    try:
        from scipy import sparse
        from scipy.sparse.linalg import spsolve
    except Exception as exc:  # pragma: no cover - only on a broken/absent scipy
        raise BaselineUnavailableError(BASELINE_UNAVAILABLE_MESSAGE) from exc

    # Second-difference operator D (n-2 x n); the roughness penalty is lam * D' D.
    diff = sparse.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(n - 2, n))
    penalty = lam * (diff.transpose() @ diff)
    w = np.ones(n, dtype=np.float64)
    baseline = z
    for _ in range(int(niter)):
        weights = sparse.spdiags(w, 0, n, n)
        baseline = spsolve((weights + penalty).tocsc(), w * z)
        w = p * (z > baseline) + (1.0 - p) * (z <= baseline)
    return np.asarray(baseline, dtype=np.float64)


# ---------------------------------------------------------------------------
# autocorrelation
# ---------------------------------------------------------------------------


def autocorrelation(
    y: Any, *, max_lag: Optional[int] = None, detrend: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """Normalised autocorrelation of a signal versus lag.

    Autocorrelation measures how similar a signal is to a delayed copy of itself: a peak
    at lag ``k`` means the signal repeats every ``k`` samples, so this is the tool for
    finding a period that a single glance -- or even an FFT of a short record -- can miss.

    Args:
        y: Signal, 1-D. Non-finite samples are interpolated across first.
        max_lag: Largest lag to return, in samples. ``None`` returns every lag
            ``0 .. len(y) - 1``. Clamped to that range.
        detrend: Subtract the mean before correlating (the standard definition). Without
            it a signal with a large offset looks correlated at every lag simply because
            it never crosses zero.

    Returns:
        ``(lags, acf)``. ``lags`` is ``0, 1, ...`` in samples; ``acf`` is normalised so
        ``acf[0] == 1`` (a signal is perfectly correlated with itself at zero lag). A
        constant signal, which has no variation to correlate, returns ``acf == 0`` for
        every non-zero lag.

    Note:
        Computed by FFT (``O(n log n)``), so it stays cheap on long records. Pure numpy;
        no scipy needed. The lags are sample indices -- a caller with an x axis multiplies
        by the sample spacing to get a lag in x units.

    Raises:
        ValueError: On empty/non-1-D input, a negative ``max_lag``, or an all-nan signal.
    """
    ya = _as_1d(y, "y")
    n = ya.size
    if max_lag is not None:
        try:
            max_lag = int(max_lag)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"max_lag must be an integer (got {max_lag!r})") from exc
        if max_lag < 0:
            raise ValueError(f"max_lag must be >= 0 (got {max_lag})")

    z = _fill_nonfinite(ya, "y")
    if detrend:
        z = z - float(np.mean(z))

    if n < 2:
        return np.zeros(1, dtype=np.float64), np.ones(1, dtype=np.float64)

    # Zero-pad to at least 2n-1 (and up to a power of two, where numpy's FFT is fastest)
    # so the circular correlation the FFT computes equals the linear one.
    size = 1 << int(np.ceil(np.log2(2 * n - 1)))
    spectrum = np.fft.rfft(z, size)
    acf_full = np.fft.irfft(spectrum * np.conjugate(spectrum), size)[:n]

    zero = float(acf_full[0])
    if zero > 0.0:
        acf = acf_full / zero
    else:
        # A constant (or zero) signal after detrending: perfectly correlated only with
        # itself at lag 0, uncorrelated everywhere else.
        acf = np.zeros(n, dtype=np.float64)
        acf[0] = 1.0
    lags = np.arange(n, dtype=np.float64)

    if max_lag is not None:
        keep = min(max_lag, n - 1) + 1
        lags = lags[:keep]
        acf = acf[:keep]
    return lags, np.asarray(acf, dtype=np.float64)


def cross_correlation(
    a: Any, b: Any, *, max_lag: Optional[int] = None, detrend: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """Normalised cross-correlation of two same-length signals versus lag.

    Cross-correlation measures how similar ``a`` is to a shifted copy of ``b`` -- the tool
    for finding a delay between two related signals. :func:`autocorrelation` is the
    special case ``b is a``, restricted to non-negative lags because a signal correlated
    with itself is symmetric; two different signals are not, so this returns both
    directions.

    Args:
        a: First signal, 1-D.
        b: Second signal, 1-D, same length as ``a``.
        max_lag: Largest ``|lag|`` to return, in samples. ``None`` returns every lag from
            ``-(n - 1)`` to ``n - 1``. Clamped to that range.
        detrend: Subtract each signal's own mean before correlating (the standard
            definition -- otherwise two signals that merely share a large offset look
            correlated at every lag simply because neither crosses zero).

    Returns:
        ``(lags, xcf)``. ``lags`` runs ascending from negative to positive. A positive
        lag ``k`` means ``b`` **lags** ``a`` by ``k`` samples (``b`` is a delayed copy of
        ``a`` there); a negative lag means ``b`` leads ``a``. ``xcf`` is normalised by
        ``sqrt(sum(a**2) * sum(b**2))`` (after detrending) so it lies in ``[-1, 1]`` --
        unlike autocorrelation, cross-correlation cannot be normalised to exactly 1 at any
        fixed lag, since ``a`` and ``b`` are different signals.

    Note:
        Computed by FFT (``O(n log n)``). Pure numpy; no scipy needed.

    Raises:
        ValueError: On empty/mismatched-length/non-1-D input, or a negative ``max_lag``.
    """
    xa = _as_1d(a, "a")
    xb = _as_1d(b, "b")
    if xa.size != xb.size:
        raise ValueError(f"a and b must have the same length (got {xa.size} and {xb.size})")
    n = xa.size
    if max_lag is not None:
        try:
            max_lag = int(max_lag)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"max_lag must be an integer (got {max_lag!r})") from exc
        if max_lag < 0:
            raise ValueError(f"max_lag must be >= 0 (got {max_lag})")

    za = _fill_nonfinite(xa, "a")
    zb = _fill_nonfinite(xb, "b")
    if detrend:
        za = za - float(np.mean(za))
        zb = zb - float(np.mean(zb))

    if n < 2:
        return np.zeros(1, dtype=np.float64), np.zeros(1, dtype=np.float64)

    # Zero-pad to at least 2n-1 (and up to a power of two) so the circular correlation
    # the FFT computes equals the linear one, exactly like autocorrelation above.
    size = 1 << int(np.ceil(np.log2(2 * n - 1)))
    spec_a = np.fft.rfft(za, size)
    spec_b = np.fft.rfft(zb, size)
    # conj(A) * B, inverse-transformed, gives R[k] = sum_t a[t] * b[t+k] circularly for
    # k = 0 .. size-1; the negative-lag half wraps around to the tail of the padded
    # buffer, so it is rolled to the front to unwrap into linear, ascending order.
    full = np.fft.irfft(np.conjugate(spec_a) * spec_b, size)
    xcf_full = np.concatenate([full[-(n - 1) :], full[:n]])
    lags_full = np.arange(-(n - 1), n, dtype=np.float64)

    norm = float(np.sqrt(np.sum(za**2) * np.sum(zb**2)))
    if norm > 0.0:
        xcf_full = xcf_full / norm

    if max_lag is not None:
        keep = min(max_lag, n - 1)
        mask = np.abs(lags_full) <= keep
        lags_full = lags_full[mask]
        xcf_full = xcf_full[mask]

    return lags_full, np.asarray(xcf_full, dtype=np.float64)


# ---------------------------------------------------------------------------
# histograms and distribution fitting
# ---------------------------------------------------------------------------


def histogram(
    y: Any, *, bins: int = 50, density: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bin a signal's values into a histogram.

    Args:
        y: Values to bin, 1-D. Non-finite samples are dropped first.
        bins: Number of equal-width bins. Clamped to ``[1, 10000]`` so a UI spinner cannot
            ask for a pathological count.
        density: When True the bars integrate to 1 (a probability density) rather than
            counting samples; this is the form that overlays directly on a fitted PDF.

    Returns:
        ``(edges, heights, centers)``. ``edges`` has ``len(heights) + 1`` entries;
        ``heights`` is counts (or density) per bin; ``centers`` is the bin midpoints,
        aligned with ``heights`` so it can serve as an x column.

    Raises:
        ValueError: On empty/non-1-D input, or when no finite sample remains.
    """
    ya = _as_1d(y, "y")
    good = ya[np.isfinite(ya)]
    if good.size == 0:
        raise ValueError("histogram needs at least one finite sample")
    try:
        nbins = int(bins)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"bins must be an integer (got {bins!r})") from exc
    nbins = max(1, min(nbins, 10000))
    heights, edges = np.histogram(good, bins=nbins, density=bool(density))
    centers = 0.5 * (edges[:-1] + edges[1:])
    return (
        np.asarray(edges, dtype=np.float64),
        np.asarray(heights, dtype=np.float64),
        np.asarray(centers, dtype=np.float64),
    )


#: Fittable distributions. Keys are what a UI stores; each maps to the scipy.stats name
#: and the labels of the parameters ``dist.fit`` returns, in order. Curated to the shapes
#: that actually come up in lab data, simplest first.
_DISTRIBUTIONS: Dict[str, Tuple[str, Tuple[str, ...]]] = {
    "normal": ("norm", ("mu", "sigma")),
    "lognormal": ("lognorm", ("shape", "loc", "scale")),
    "exponential": ("expon", ("loc", "scale")),
    "gamma": ("gamma", ("shape", "loc", "scale")),
    "rayleigh": ("rayleigh", ("loc", "scale")),
    "weibull": ("weibull_min", ("shape", "loc", "scale")),
    "chi2": ("chi2", ("df", "loc", "scale")),
    "beta": ("beta", ("a", "b", "loc", "scale")),
    "uniform": ("uniform", ("loc", "scale")),
}

#: Distribution keys in presentation order (iterate this, not the dict).
DISTRIBUTIONS: Tuple[str, ...] = tuple(_DISTRIBUTIONS)

DIST_UNAVAILABLE_MESSAGE = (
    "Distribution fitting needs scipy.stats, which could not be imported. The histogram "
    "still works without a fit; install or repair scipy (pip install -U scipy) to fit a "
    "distribution and run the goodness-of-fit test."
)


class DistributionUnavailableError(ValueError):
    """scipy.stats could not be imported, so no distribution can be fitted.

    A :class:`ValueError` subclass, for the same reason as :class:`FitUnavailableError`.
    """


def distributions_available() -> bool:
    """True if :func:`fit_distribution` can run, i.e. ``scipy.stats`` imports."""
    try:
        from scipy import stats  # noqa: F401
    except Exception:  # pragma: no cover - only on a broken/absent scipy
        return False
    return True


def fit_distribution(y: Any, name: str, x_grid: Any) -> Dict[str, Any]:
    """Fit a named distribution to a sample by maximum likelihood and score the fit.

    Args:
        y: The sample, 1-D. Non-finite values are dropped.
        name: One of :data:`DISTRIBUTIONS`.
        x_grid: Abscissae to evaluate the fitted probability density on, so a caller can
            overlay the PDF on a histogram without re-importing scipy.

    Returns:
        Dict with ``params`` (tuple of floats), ``labels`` (their names), ``pdf`` (the
        density at ``x_grid``), ``ks_statistic`` and ``ks_pvalue`` (Kolmogorov-Smirnov
        goodness of fit -- a small statistic and a p-value above ~0.05 mean the data is
        consistent with the distribution), and ``log_likelihood``.

    Raises:
        DistributionUnavailableError: If scipy.stats cannot be imported (a ValueError
            subclass; check :func:`distributions_available` to avoid it).
        ValueError: On an unknown name, fewer than two finite samples, or a fit that
            fails on this data (e.g. a positive-support law on data that is not positive).
    """
    if name not in _DISTRIBUTIONS:
        raise ValueError(f"unknown distribution {name!r}; choose one of {', '.join(DISTRIBUTIONS)}")
    try:
        from scipy import stats
    except Exception as exc:  # pragma: no cover - only on a broken/absent scipy
        raise DistributionUnavailableError(DIST_UNAVAILABLE_MESSAGE) from exc

    scipy_name, labels = _DISTRIBUTIONS[name]
    dist = getattr(stats, scipy_name)
    data = _as_1d(y, "y")
    data = data[np.isfinite(data)]
    if data.size < 2:
        raise ValueError(f"fitting {name} needs at least two finite samples")

    grid = np.asarray(x_grid, dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            with np.errstate(all="ignore"):
                params = dist.fit(data)
                pdf = np.asarray(dist.pdf(grid, *params), dtype=np.float64)
                # `stats.kstest(data, scipy_name, args=params)` -- passing the distribution
                # by its string name -- hits a real bug in scipy >= 1.18 for at least
                # `norm` ("ndtr() takes from 1 to 2 positional arguments but 3 were given").
                # Passing the already-resolved `dist.cdf` instead produces the identical
                # result through a code path the bug does not touch.
                ks = stats.kstest(data, dist.cdf, args=params)
                log_likelihood = float(np.sum(dist.logpdf(data, *params)))
        except (ValueError, RuntimeError, TypeError, FloatingPointError) as exc:
            raise ValueError(f"the {name} fit failed on this data: {exc}") from exc

    return {
        "params": tuple(float(p) for p in params),
        "labels": labels,
        "pdf": pdf,
        "ks_statistic": float(ks.statistic),
        "ks_pvalue": float(ks.pvalue),
        "log_likelihood": log_likelihood,
    }


def describe(y: Any) -> Dict[str, float]:
    """Summary statistics for a signal, nan-safe throughout.

    Args:
        y: Signal, 1-D.

    Returns:
        Dict with keys ``n``, ``min``, ``max``, ``mean``, ``median``, ``std``,
        ``var``, ``sum``, ``rms``. All values are Python floats.

    Note:
        ``n`` is the count of **finite** samples, not ``len(y)``. Every other
        statistic ignores nan/inf too, so the dict stays internally consistent
        (``mean == sum / n``). An all-nan input yields ``n == 0`` and nan elsewhere.
        ``std``/``var`` are population (ddof=0) statistics, matching numpy defaults.

    Raises:
        ValueError: On empty or non-1-D input.
    """
    ya = _as_1d(y, "y")
    good = ya[np.isfinite(ya)]
    n = int(good.size)
    if n == 0:
        nan = float("nan")
        return {
            "n": 0.0,
            "min": nan,
            "max": nan,
            "mean": nan,
            "median": nan,
            "std": nan,
            "var": nan,
            "sum": nan,
            "rms": nan,
        }
    return {
        "n": float(n),
        "min": float(np.min(good)),
        "max": float(np.max(good)),
        "mean": float(np.mean(good)),
        "median": float(np.median(good)),
        "std": float(np.std(good)),
        "var": float(np.var(good)),
        "sum": float(np.sum(good)),
        "rms": float(np.sqrt(np.mean(good**2))),
    }


def describe_extended(
    y: Any, *, percentiles: Sequence[float] = (5.0, 25.0, 75.0, 95.0)
) -> Dict[str, float]:
    """:func:`describe` plus percentiles, spread and shape statistics, nan-safe throughout.

    Args:
        y: Signal, 1-D.
        percentiles: Percentile ranks (0-100) to report, each as a ``percentile_<p>`` key
            (e.g. ``percentile_25``). ``iqr`` (P75 - P25) is always computed regardless of
            what is requested here.

    Returns:
        Every key :func:`describe` returns, plus:
          * ``percentile_<p>`` for each requested percentile,
          * ``iqr`` -- the interquartile range, P75 minus P25,
          * ``mad`` -- median absolute deviation from the median, the robust analogue of
            ``std``,
          * ``skewness``/``kurtosis`` -- the population (biased) third and fourth
            standardised moments, matching ``scipy.stats.skew``/``kurtosis``'s defaults
            (``bias=True``, and Fisher's excess convention for kurtosis) numerically --
            computed here in plain numpy since these are closed-form moment formulas, not
            a fitted distribution, so no scipy dependency buys anything,
          * ``nan_count`` -- samples that were not finite,
          * ``outlier_count`` -- samples outside the standard Tukey fence
            ``[P25 - 1.5*iqr, P75 + 1.5*iqr]``.

    Note:
        Never diverges from :func:`describe`: this calls it first and only adds keys, so
        every existing consumer of ``describe()``'s dict keeps working against this one.

    Raises:
        ValueError: On empty or non-1-D input.
    """
    out = dict(describe(y))
    ya = _as_1d(y, "y")
    good = ya[np.isfinite(ya)]
    n = int(good.size)
    out["nan_count"] = float(ya.size - n)

    if n == 0:
        nan = float("nan")
        for p in percentiles:
            out[f"percentile_{float(p):g}"] = nan
        out["iqr"] = nan
        out["mad"] = nan
        out["skewness"] = nan
        out["kurtosis"] = nan
        out["outlier_count"] = nan
        return out

    for p in percentiles:
        out[f"percentile_{float(p):g}"] = float(np.percentile(good, float(p)))

    p25, p75 = (float(v) for v in np.percentile(good, [25.0, 75.0]))
    iqr = p75 - p25
    out["iqr"] = iqr
    out["mad"] = float(np.median(np.abs(good - out["median"])))

    std = out["std"]
    if std > 0.0:
        centered = good - out["mean"]
        out["skewness"] = float(np.mean(centered**3) / std**3)
        out["kurtosis"] = float(np.mean(centered**4) / std**4 - 3.0)
    else:
        out["skewness"] = 0.0
        out["kurtosis"] = 0.0

    if iqr > 0.0:
        lo, hi = p25 - 1.5 * iqr, p75 + 1.5 * iqr
        out["outlier_count"] = float(np.count_nonzero((good < lo) | (good > hi)))
    else:
        out["outlier_count"] = 0.0

    return out


def _rankdata(a: np.ndarray) -> np.ndarray:
    """1-based ranks with ties averaged -- ``scipy.stats.rankdata``'s default convention.

    A small pure-numpy stand-in used only when scipy is unavailable, so
    :func:`correlate`'s Spearman fallback (Pearson correlation of ranks -- the exact
    definition of Spearman's rho, not an approximation) has ranks to correlate.
    """
    order = np.argsort(a, kind="stable")
    sorted_a = a[order]
    ranks = np.empty(a.size, dtype=np.float64)
    i = 0
    while i < a.size:
        j = i
        while j + 1 < a.size and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1
    return ranks


def correlate(x: Any, y: Any) -> Dict[str, float]:
    """Linear relationship between two same-length series: correlation, covariance, a fit.

    Args:
        x: First series, 1-D.
        y: Second series, 1-D, same length as ``x``. ``pearson_r``/``spearman_rho``/
            ``covariance`` are symmetric in x and y; ``slope``/``intercept``/``r_squared``
            treat ``y`` as the dependent variable of a degree-1 fit against ``x``.

    Returns:
        Dict with ``n`` (finite-pair count), ``pearson_r``, ``pearson_p``,
        ``spearman_rho``, ``spearman_p``, ``covariance``, ``slope``, ``intercept``,
        ``r_squared``. Only finite ``(x, y)`` pairs are used throughout.

    Note:
        ``pearson_r``/``covariance``/``slope``/``intercept``/``r_squared`` are pure numpy
        and always available. ``spearman_rho`` degrades to a Pearson correlation of the
        two series' ranks when scipy is unavailable. The two p-values (``pearson_p``,
        ``spearman_p``) have no honest numpy substitute and come back nan without scipy,
        rather than a faked value.

    Raises:
        ValueError: On empty/mismatched input, or fewer than 2 finite pairs.
    """
    xa, ya = _as_xy(x, y)
    good = np.isfinite(xa) & np.isfinite(ya)
    xg, yg = xa[good], ya[good]
    n = int(xg.size)
    if n < 2:
        raise ValueError(f"correlate() needs at least 2 finite (x, y) pairs, got {n}")

    covariance = float(np.cov(xg, yg, ddof=0)[0, 1])
    std_x = float(np.std(xg))
    std_y = float(np.std(yg))
    pearson_r = covariance / (std_x * std_y) if std_x > 0.0 and std_y > 0.0 else float("nan")

    slope = intercept = r_squared = float("nan")
    if std_x > 0.0:
        slope_f, intercept_f = np.polyfit(xg, yg, 1)
        slope, intercept = float(slope_f), float(intercept_f)
        fitted = slope * xg + intercept
        ss_res = float(np.sum((yg - fitted) ** 2))
        ss_tot = float(np.sum((yg - np.mean(yg)) ** 2))
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else float("nan")

    pearson_p = spearman_rho = spearman_p = float("nan")
    try:
        from scipy import stats as _stats

        _, pearson_p = _stats.pearsonr(xg, yg)
        pearson_p = float(pearson_p)
        spearman_rho, spearman_p = _stats.spearmanr(xg, yg)
        spearman_rho, spearman_p = float(spearman_rho), float(spearman_p)
    except Exception:
        rx, ry = _rankdata(xg), _rankdata(yg)
        rx_std, ry_std = float(np.std(rx)), float(np.std(ry))
        if rx_std > 0.0 and ry_std > 0.0:
            spearman_rho = float(np.cov(rx, ry, ddof=0)[0, 1] / (rx_std * ry_std))

    return {
        "n": float(n),
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_rho": spearman_rho,
        "spearman_p": spearman_p,
        "covariance": covariance,
        "slope": slope,
        "intercept": intercept,
        "r_squared": r_squared,
    }


#: Two-sample hypothesis tests two_sample_test() offers. "welch" is the safe default
#: (unlike "student" it does not assume the two samples share a variance, and costs
#: nothing when they do); "mannwhitney"/"ks" make no normality assumption at all.
TWO_SAMPLE_METHODS: Tuple[str, ...] = ("welch", "student", "mannwhitney", "ks")

TWO_SAMPLE_UNAVAILABLE_MESSAGE = (
    "Two-sample hypothesis tests need scipy.stats, which could not be imported. Install "
    "or repair scipy (pip install -U scipy) to compare two samples."
)


class TwoSampleUnavailableError(ValueError):
    """scipy.stats could not be imported, so no two-sample test is possible.

    A :class:`ValueError` subclass, for the same reason as every other
    ``*UnavailableError`` in this module: a caller that already handles ValueError as
    this module's documented failure mode keeps doing so instead of crashing.
    """


def two_sample_available() -> bool:
    """True if a two-sample test is possible, i.e. ``scipy.stats`` imports.

    Ask this before offering the feature. :func:`two_sample_test` raises
    :class:`TwoSampleUnavailableError` when this is False.
    """
    try:
        import scipy.stats  # noqa: F401
    except Exception:  # pragma: no cover - only on a broken/absent scipy
        return False
    return True


def two_sample_test(a: Any, b: Any, *, method: str = "welch") -> Dict[str, float]:
    """Compare two independent samples: are they plausibly drawn from the same population?

    Args:
        a: First sample, 1-D. Non-finite entries are dropped.
        b: Second sample, 1-D. Need not be the same length as ``a`` -- these are two
            independent samples (e.g. two dataset columns of different sizes, or a
            column split by a filter), not a paired ``(x, y)`` relationship the way
            :func:`correlate` reads its input.
        method: One of :data:`TWO_SAMPLE_METHODS`:

            * ``"welch"`` -- Welch's t-test (unequal variances allowed). Tests whether
              the two means differ. The default: Student's equal-variance assumption is
              rarely exactly true, and Welch's costs essentially nothing when it does
              hold.
            * ``"student"`` -- Student's t-test (equal variances assumed), offered for
              comparison / textbook reproducibility.
            * ``"mannwhitney"`` -- Mann-Whitney U, a rank-based test for whether one
              sample tends to have larger values than the other. No normality
              assumption, so it is robust to outliers and skew that would mislead a
              t-test.
            * ``"ks"`` -- two-sample Kolmogorov-Smirnov. Tests whether the two samples'
              full distributions differ (shape and location both), not just their means.

    Returns:
        Dict with ``n_a``, ``n_b``, ``mean_a``, ``mean_b``, ``std_a``, ``std_b``
        (population std, ddof=0, of the finite samples), ``statistic`` and ``p_value``
        for the requested test. The descriptive summary is always included regardless of
        which test was requested, since it is cheap and is what makes the test's verdict
        interpretable (a significant p-value with ``mean_a == mean_b`` is a distribution-
        shape story, not a location one -- only visible from the summary alongside it).

    Raises:
        TwoSampleUnavailableError: ``scipy.stats`` cannot be imported.
        ValueError: On an unknown ``method`` or fewer than 2 finite samples in either
            group.
    """
    if method not in TWO_SAMPLE_METHODS:
        raise ValueError(
            f"unknown method {method!r}; choose one of {', '.join(TWO_SAMPLE_METHODS)}"
        )

    xa = _as_1d(a, "a")
    xb = _as_1d(b, "b")
    ga = xa[np.isfinite(xa)]
    gb = xb[np.isfinite(xb)]
    if ga.size < 2 or gb.size < 2:
        raise ValueError(
            "two_sample_test needs at least 2 finite samples in each group "
            f"(got {ga.size} and {gb.size})"
        )

    try:
        from scipy import stats as _scipy_stats
    except Exception as exc:  # pragma: no cover - only on a broken/absent scipy
        raise TwoSampleUnavailableError(TWO_SAMPLE_UNAVAILABLE_MESSAGE) from exc

    if method == "welch":
        result = _scipy_stats.ttest_ind(ga, gb, equal_var=False)
    elif method == "student":
        result = _scipy_stats.ttest_ind(ga, gb, equal_var=True)
    elif method == "mannwhitney":
        result = _scipy_stats.mannwhitneyu(ga, gb, alternative="two-sided")
    else:  # "ks"
        result = _scipy_stats.ks_2samp(ga, gb)
    statistic, p_value = float(result[0]), float(result[1])

    return {
        "n_a": float(ga.size),
        "n_b": float(gb.size),
        "mean_a": float(np.mean(ga)),
        "mean_b": float(np.mean(gb)),
        "std_a": float(np.std(ga)),
        "std_b": float(np.std(gb)),
        "statistic": statistic,
        "p_value": p_value,
    }


#: Two-tailed standard-normal quantiles for the confidence levels the Fit tab offers,
#: keyed by ``round(level, 2)``. Used only when scipy.stats is unimportable -- insurance,
#: not the main path, since fit_model/fit_model_robust already require scipy to have
#: produced ``popt``/``pcov`` in the first place.
_NORMAL_QUANTILES: Dict[float, float] = {
    0.68: 0.9944579,
    0.80: 1.2815516,
    0.90: 1.6448536,
    0.95: 1.9599640,
    0.99: 2.5758293,
}


def _t_multiplier(level: float, dof: int) -> float:
    """Two-tailed Student's t critical value at ``level`` confidence and ``dof``.

    Falls back to the standard-normal quantile (exact only as ``dof -> inf``, a mild
    over-confidence at small ``dof``) when scipy.stats is unimportable.
    """
    try:
        from scipy import stats as _stats

        return float(_stats.t.ppf(0.5 + float(level) / 2.0, dof))
    except Exception:
        return _NORMAL_QUANTILES.get(round(float(level), 2), 1.9599640)


def _normal_multiplier(level: float) -> float:
    """Two-tailed standard-normal critical value at ``level`` confidence.

    Falls back to the same fixed table as :func:`_t_multiplier` when scipy.stats is
    unimportable; exact either way for the levels the UI offers.
    """
    try:
        from scipy import stats as _stats

        return float(_stats.norm.ppf(0.5 + float(level) / 2.0))
    except Exception:
        return _NORMAL_QUANTILES.get(round(float(level), 2), 1.9599640)


def confidence_band(
    x: Any,
    popt: Any,
    pcov: Any,
    fn: Callable[..., Any],
    *,
    level: float = 0.95,
) -> Tuple[np.ndarray, np.ndarray]:
    """Confidence band for a fitted model, evaluated at every point of ``x``.

    Propagates the fit's parameter covariance (as returned by :func:`fit_model`/
    :func:`fit_model_robust`) through ``fn`` via a central-difference Jacobian -- the
    standard delta method -- and scales it by the two-tailed critical value at ``level``
    confidence and ``dof = len(x) - len(popt)`` degrees of freedom.

    Args:
        x: Points to evaluate the band at, 1-D.
        popt: Fitted parameters, as returned by fit_model/fit_model_robust.
        pcov: Parameter covariance, ``(n_params, n_params)``, same source.
        fn: The model function, called as ``fn(x, *popt)`` -- the same callable a
            :class:`FitModel` carries as ``.fn``.
        level: Confidence level in (0, 1), e.g. 0.95 for a 95% band.

    Returns:
        ``(lower, upper)``, each the same shape as ``x``. Where ``pcov`` is not fully
        finite (an underdetermined fit -- see :func:`fit_errors`) the band is nan at every
        point, matching how an infinite standard error is already reported elsewhere
        rather than drawing a band that claims a precision the fit does not have.

    Raises:
        ValueError: If ``level`` is not strictly between 0 and 1.
    """
    if not (0.0 < float(level) < 1.0):
        raise ValueError(f"level must be in (0, 1), got {level!r}")

    xa = _as_1d(x, "x")
    p = np.asarray(popt, dtype=np.float64).ravel()
    cov = np.asarray(pcov, dtype=np.float64)
    n_params = p.size

    with np.errstate(all="ignore"):
        base = np.broadcast_to(np.asarray(fn(xa, *p), dtype=np.float64), xa.shape).astype(
            np.float64, copy=True
        )

    if cov.shape != (n_params, n_params) or not np.all(np.isfinite(cov)):
        nan = np.full(xa.shape, np.nan, dtype=np.float64)
        return nan, nan

    # Central-difference Jacobian d(fn)/d(param_i) at popt -- the same technique MINPACK
    # itself uses when curve_fit is not given an analytic Jacobian.
    jac = np.empty((xa.size, n_params), dtype=np.float64)
    for i in range(n_params):
        step = max(abs(float(p[i])), 1.0) * 1e-6
        p_hi, p_lo = p.copy(), p.copy()
        p_hi[i] += step
        p_lo[i] -= step
        with np.errstate(all="ignore"):
            f_hi = np.broadcast_to(np.asarray(fn(xa, *p_hi), dtype=np.float64), xa.shape)
            f_lo = np.broadcast_to(np.asarray(fn(xa, *p_lo), dtype=np.float64), xa.shape)
        jac[:, i] = (f_hi - f_lo) / (2.0 * step)

    variance = np.clip(np.einsum("ij,jk,ik->i", jac, cov, jac), 0.0, None)
    se = np.sqrt(variance)

    dof = max(1, xa.size - n_params)
    multiplier = _t_multiplier(level, dof)

    margin = multiplier * se
    return base - margin, base + margin


def confidence_interval_mean(y: Any, *, level: float = 0.95) -> Dict[str, float]:
    """Confidence interval for the population mean of ``y``, via Student's t.

    Args:
        y: Sample, 1-D. Non-finite entries are dropped.
        level: Confidence level in (0, 1), e.g. 0.95 for a 95% interval.

    Returns:
        Dict with ``n``, ``mean``, ``std`` (sample std, ddof=1), ``se`` (standard error
        of the mean), ``dof``, ``lower``, ``upper``, ``level``.

    Raises:
        ValueError: Fewer than 2 finite samples, or ``level`` not in (0, 1).
    """
    if not (0.0 < float(level) < 1.0):
        raise ValueError(f"level must be in (0, 1), got {level!r}")
    ya = _as_1d(y, "y")
    yg = ya[np.isfinite(ya)]
    n = int(yg.size)
    if n < 2:
        raise ValueError(f"confidence_interval_mean needs at least 2 finite samples, got {n}")

    mean = float(np.mean(yg))
    std = float(np.std(yg, ddof=1))
    dof = n - 1
    se = std / np.sqrt(n)
    margin = _t_multiplier(level, dof) * se

    return {
        "n": float(n),
        "mean": mean,
        "std": std,
        "se": se,
        "dof": float(dof),
        "lower": mean - margin,
        "upper": mean + margin,
        "level": float(level),
    }


def confidence_interval_correlation(x: Any, y: Any, *, level: float = 0.95) -> Dict[str, float]:
    """Confidence interval for the Pearson correlation of ``(x, y)``, via Fisher's z.

    The standard transform for a correlation CI: Pearson r has no closed-form normal
    sampling distribution, but ``arctanh(r)`` is approximately normal with standard
    error ``1 / sqrt(n - 3)`` -- build the interval in that space and transform back.

    Args:
        x: First series, 1-D.
        y: Second series, 1-D, same length as ``x``.
        level: Confidence level in (0, 1).

    Returns:
        Dict with ``n`` (finite-pair count), ``r`` (Pearson correlation), ``lower``,
        ``upper``, ``level``. ``r`` of exactly +/-1 is clamped away from the transform's
        singularity before it is applied.

    Raises:
        ValueError: Fewer than 4 finite pairs (the transform's standard error is
            undefined at ``n <= 3``), or ``level`` not in (0, 1).
    """
    if not (0.0 < float(level) < 1.0):
        raise ValueError(f"level must be in (0, 1), got {level!r}")
    xa, ya = _as_xy(x, y)
    good = np.isfinite(xa) & np.isfinite(ya)
    xg, yg = xa[good], ya[good]
    n = int(xg.size)
    if n < 4:
        raise ValueError(
            f"confidence_interval_correlation needs at least 4 finite (x, y) pairs, got {n}"
        )

    std_x, std_y = float(np.std(xg)), float(np.std(yg))
    if std_x == 0.0 or std_y == 0.0:
        r = float("nan")
    else:
        r = float(np.cov(xg, yg, ddof=0)[0, 1] / (std_x * std_y))

    if not np.isfinite(r):
        return {
            "n": float(n),
            "r": r,
            "lower": float("nan"),
            "upper": float("nan"),
            "level": float(level),
        }

    r_clamped = max(-0.9999999, min(0.9999999, r))
    z = np.arctanh(r_clamped)
    se_z = 1.0 / np.sqrt(n - 3)
    margin = _normal_multiplier(level) * se_z

    return {
        "n": float(n),
        "r": r,
        "lower": float(np.tanh(z - margin)),
        "upper": float(np.tanh(z + margin)),
        "level": float(level),
    }


#: Methods confidence_interval_difference() offers for the standard error / degrees of
#: freedom of a two-sample mean difference. A subset of TWO_SAMPLE_METHODS: mannwhitney
#: and ks test the full distribution, not the mean, and have no simple closed-form CI
#: on a mean difference the way a t-based interval does.
DIFFERENCE_CI_METHODS: Tuple[str, ...] = ("welch", "student")


def confidence_interval_difference(
    a: Any, b: Any, *, level: float = 0.95, method: str = "welch"
) -> Dict[str, float]:
    """Confidence interval for the difference of two independent samples' means.

    Pure numpy: unlike :func:`two_sample_test`, no scipy dependency, since both the
    Welch-Satterthwaite and pooled-variance formulas below are closed-form.

    Args:
        a: First sample, 1-D. Non-finite entries are dropped.
        b: Second sample, 1-D. Need not be the same length as ``a``.
        level: Confidence level in (0, 1).
        method: ``"welch"`` (unequal variances allowed, the safe default) or
            ``"student"`` (pooled variance, assumes the two populations share one).

    Returns:
        Dict with ``n_a``, ``n_b``, ``mean_a``, ``mean_b``, ``diff`` (``mean_a -
        mean_b``), ``se``, ``dof``, ``lower``, ``upper``, ``level``, ``method``.

    Raises:
        ValueError: Fewer than 2 finite samples in either group, an unknown ``method``,
            or ``level`` not in (0, 1).
    """
    if method not in DIFFERENCE_CI_METHODS:
        raise ValueError(
            f"unknown method {method!r}; choose one of {', '.join(DIFFERENCE_CI_METHODS)}"
        )
    if not (0.0 < float(level) < 1.0):
        raise ValueError(f"level must be in (0, 1), got {level!r}")

    xa = _as_1d(a, "a")
    xb = _as_1d(b, "b")
    ga = xa[np.isfinite(xa)]
    gb = xb[np.isfinite(xb)]
    n_a, n_b = int(ga.size), int(gb.size)
    if n_a < 2 or n_b < 2:
        raise ValueError(
            f"confidence_interval_difference needs at least 2 finite samples in each "
            f"group (got {n_a} and {n_b})"
        )

    mean_a, mean_b = float(np.mean(ga)), float(np.mean(gb))
    var_a, var_b = float(np.var(ga, ddof=1)), float(np.var(gb, ddof=1))
    diff = mean_a - mean_b

    if method == "welch":
        term_a, term_b = var_a / n_a, var_b / n_b
        se = float(np.sqrt(term_a + term_b))
        denom = (term_a**2) / (n_a - 1) + (term_b**2) / (n_b - 1)
        dof = (term_a + term_b) ** 2 / denom if denom > 0.0 else float(n_a + n_b - 2)
    else:  # "student"
        pooled_var = ((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2)
        se = float(np.sqrt(pooled_var * (1.0 / n_a + 1.0 / n_b)))
        dof = float(n_a + n_b - 2)

    margin = _t_multiplier(level, dof) * se

    return {
        "n_a": float(n_a),
        "n_b": float(n_b),
        "mean_a": mean_a,
        "mean_b": mean_b,
        "diff": diff,
        "se": se,
        "dof": float(dof),
        "lower": diff - margin,
        "upper": diff + margin,
        "level": float(level),
    }
