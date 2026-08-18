"""The Math Lab panel: integrate, smooth, overlay and add noise to a signal.

This is the "plotear la integral de una funcion o de datos, suavizar y sobreponer,
agregar ruido" surface. Every operation tab renders a **live preview** in
:func:`widgets.mini_plot` with the source curve and the result drawn on the same axes
before anything is committed -- that overlay is the "sobreponer".

Structure
---------
A source (a :class:`DataSet` column pair, or the geometry of an existing plotted layer)
feeds a tab bar of pure :mod:`glplot.gui.mathops` operations. Nothing is applied until
the user picks an apply mode and presses Apply, at which point a :class:`Command` is
submitted to the :class:`CommandQueue` (CONTRACT 1.1: a draw callback may never touch
``scene.layers``) and recorded on the :class:`UndoStack`.

Preview caching
---------------
``draw`` runs every frame; ``mathops`` on 1e6 samples does not fit in a 16ms budget
(CONTRACT 1.10). Results are therefore memoised against a key of
``(epoch, source, fingerprint, params)``. The fingerprint is deliberately cheap -- length,
endpoints and the sum of ~64 strided samples -- so it is a *heuristic*: an edit that
leaves all of those invariant will not invalidate the preview. The Recompute button in
the source header exists precisely for that case, and Apply always commits the array the
preview last showed, so what you see is what you get either way.

Ordering
--------
``integrate``/``derivative``/``resample`` return their results in ascending-x order
regardless of the input order. This module reproduces that sort locally (:func:`_sorted_by_x`
mirrors ``mathops._sorted_xy``) so the overlaid original lines up with the result sample
for sample, and so Apply plots the result against the x it was actually computed on.

Text is ASCII only: the default ImGui font is ProggyClean, whose glyph range stops at
U+00FF, so no integral sign and no Greek here (CONTRACT 3).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from glplot.gui import dynamics, icons, layerops, mathops, theme, widgets
from glplot.gui.datasets import Column, DataSet
from glplot.gui.history import Command, snapshot
from glplot.gui.panels.base import Panel

try:
    from imgui_bundle import imgui

    IMGUI_AVAILABLE = True
except (ImportError, Exception):  # pragma: no cover - exercised only on GL-less systems
    IMGUI_AVAILABLE = False
    imgui = None

logger = logging.getLogger(__name__)

__all__ = ["MathLabPanel"]


# Height of every preview plot, in pixels.
_PREVIEW_HEIGHT = 150.0

# Floor on the scrolling body's height. Pinning Apply to the bottom must never squeeze
# the body to nothing: in a panel too short to hold both, the body keeps this much and
# the panel's own scrollbar takes over.
_MIN_BODY_HEIGHT = 80.0

# Samples taken by _fingerprint. Small enough to be free at 1e6 rows, large enough that a
# realistic edit moves the sum.
_FINGERPRINT_SAMPLES = 64

# Geometry attributes a layer may expose, and the default name of each of its columns.
# Order matters: ``ab`` is probed before ``pts`` for the same reason datasets.py does it,
# so a line_family is never mistaken for a scatter.
_LAYER_GEOMETRY: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("ab", ("a", "b")),
    ("pts", ("x", "y")),
    ("vertices", ("x", "y", "z")),
)

# Layer types with no editable tabular form. ``patch`` is named explicitly because it also
# carries a ``vertices`` attribute and would otherwise be caught by the duck-typed probe.
_LAYER_UNSUPPORTED = frozenset({"text", "patch", "semantic_line", "semantic_span"})

# Layer types ``layerops.update_layer_xy`` can rewrite in place. 3D geometry is absent by
# design: its rows are (x, y, z) and a two-column swap would silently drop z.
_REPLACEABLE_LAYER_TYPES = frozenset({"scatter", "polyline", "line_family"})

_INTEGRAL_METHODS = ("trapezoid", "simpson", "rectangle")
_DERIVATIVE_METHODS = ("central", "forward", "savgol")
_SMOOTH_METHODS = ("moving_average", "gaussian", "savgol", "median")
_NOISE_KINDS = ("gaussian", "uniform", "poisson", "salt_pepper")
_RESAMPLE_KINDS = ("linear", "cubic")
_NORMALIZE_MODES = ("minmax", "zscore", "max_abs", "area")
#: Read from mathops so the tab and the operation can never disagree on the vocabulary.
_FILTER_TYPES = mathops.FILTER_TYPES
#: Baseline-removal modes: the two straight-line detrends plus the curved AsLS estimate.
_BASELINE_KINDS = mathops.DETREND_KINDS + ("asls",)
#: "none" (histogram only) plus every distribution mathops can fit.
_HIST_DIST_OPTIONS = ("none",) + mathops.DISTRIBUTIONS

#: The Dynamics tab's analyses. The signal is read as a dynamical system's time series and
#: reconstructed/characterised by :mod:`glplot.gui.dynamics`.
_DYNAMICS_MODES = ("phase", "return_map", "lyapunov")

#: The kinds the "new layer" apply mode may produce. Read from the registry that builds
#: them rather than copied: a pair hardcoded here is a result the user can compute and
#: then cannot plot as anything but a line or a scatter.
_LAYER_KINDS = layerops.KIND_KEYS

# Fit tab models. "polynomial" is linear least squares on the coefficients and keeps the
# old degree spinner (it is correct, it has callers, and it needs no initial guess); the
# rest are mathops.MODELS driven through scipy curve_fit; "custom" compiles a user-typed
# f(x). Driven from mathops.MODEL_NAMES rather than hardcoded so the registry stays the
# single source of truth.
_FIT_MODELS: Tuple[str, ...] = (
    ("polynomial",) + mathops.MODEL_NAMES + ("gaussians", "lorentzians", "custom")
)

#: Fit models that deconvolve a sum of N peaks, and the shape each one uses. Kept apart
#: from the single-shape MODELS registry because they take a peak count, not fixed params.
_MULTIPEAK_MODELS: Dict[str, str] = {"gaussians": "gaussian", "lorentzians": "lorentzian"}

#: Ceiling on the deconvolution peak count, mirroring mathops.multipeak_model's own cap.
_MAX_FIT_PEAKS = 10

# Default custom model. Chosen because its flat 1.0 start actually converges, so the
# first thing a user sees on the custom tab is a fit and not a failure.
_DEFAULT_FIT_EXPR = "a*exp(-x/b) + c"

_APPLY_MODES: Tuple[Tuple[str, str], ...] = (
    ("new_layer", "New layer"),
    ("new_dataset", "New dataset"),
    ("add_column", "Add column"),
    ("replace", "Replace source"),
)

# (state key, tab title, tab body method name)
_TABS: Tuple[Tuple[str, str, str], ...] = (
    ("integral", "Integral", "_tab_integral"),
    ("derivative", "Derivative", "_tab_derivative"),
    ("smooth", "Smooth", "_tab_smooth"),
    ("noise", "Noise", "_tab_noise"),
    ("resample", "Resample", "_tab_resample"),
    ("normalize", "Normalize", "_tab_normalize"),
    ("filter", "Filter", "_tab_filter"),
    ("detrend", "Baseline", "_tab_detrend"),
    ("fit", "Fit", "_tab_fit"),
    ("fft", "FFT", "_tab_fft"),
    ("autocorr", "Autocorr", "_tab_autocorr"),
    ("peaks", "Peaks", "_tab_peaks"),
    ("histogram", "Histogram", "_tab_histogram"),
    ("stats", "Statistics", "_tab_stats"),
    ("dynamics", "Dynamics", "_tab_dynamics"),
)

#: The tabs, grouped for the top-level category bar. Thirteen operations overflow the
#: panel's tab strip at its default width (they scroll out of sight, so a user cannot even
#: see that they exist); grouping them under three headings keeps every operation one
#: glance away. Each key appears in exactly one category (asserted below), and the order
#: here is the presentation order both bars use.
_CATEGORIES: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("Calculus", ("integral", "derivative", "resample", "normalize")),
    ("Signal", ("smooth", "filter", "detrend", "noise", "fft", "autocorr", "peaks")),
    ("Statistics", ("fit", "histogram", "stats")),
    ("Dynamics", ("dynamics",)),
)

#: key -> (title, body method name), so a category can render a tab from its key alone.
_TAB_BY_KEY: Dict[str, Tuple[str, str]] = {key: (title, method) for key, title, method in _TABS}

#: key -> category title, for :meth:`MathLabPanel.show_operation` to jump to the right
#: category before selecting the operation.
_CATEGORY_OF: Dict[str, str] = {key: title for title, keys in _CATEGORIES for key in keys}

# Every operation must live in exactly one category, or a tab would be unreachable (in no
# category) or drawn twice (in two). Cheap to check once at import; a loud failure here
# beats a silently missing tab.
assert set(_CATEGORY_OF) == {key for key, _t, _m in _TABS}, "every tab needs one category"
assert sum(len(keys) for _t, keys in _CATEGORIES) == len(_TABS), "a tab is double-categorised"

_DESCRIBE_ROWS: Tuple[Tuple[str, str], ...] = (
    ("n", "Finite samples"),
    ("min", "Minimum"),
    ("max", "Maximum"),
    ("mean", "Mean"),
    ("median", "Median"),
    ("std", "Std deviation"),
    ("var", "Variance"),
    ("sum", "Sum"),
    ("rms", "RMS"),
)


# ----------------------------------------------------------------------------------
# Pure helpers (no imgui)
# ----------------------------------------------------------------------------------


def _fmt(value: float) -> str:
    """Format a float for a stat row: compact, but never lying about magnitude."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(v):
        return "nan"
    if math.isinf(v):
        return "inf" if v > 0.0 else "-inf"
    magnitude = abs(v)
    if magnitude != 0.0 and (magnitude < 1e-4 or magnitude >= 1e6):
        return f"{v:.4e}"
    return f"{v:.6g}"


def _sorted_by_x(x: Any, y: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(x, y)`` as float64 ordered by ascending x.

    This deliberately mirrors ``mathops._sorted_xy`` (stable argsort, already-ascending
    input passed through unchanged) so that the abscissae this panel plots and overlays
    are exactly the ones ``mathops.integrate``/``derivative`` computed against. Using a
    different tie-break would misalign the overlay on duplicate x values.
    """
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    if xa.size < 2 or bool(np.all(np.diff(xa) >= 0.0)):
        return xa, ya
    order = np.argsort(xa, kind="stable")
    return xa[order], ya[order]


def _histogram_silhouette(edges: np.ndarray, heights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """The staircase outline of a histogram, as an ``(x, y)`` polyline for mini_plot.

    A line renderer cannot draw bars, but the classic histogram *silhouette* -- a riser at
    every bin edge and a flat top across each bar, dropping to zero at both ends -- reads
    as a histogram and, crucially, shares one x grid with a smooth PDF sampled at the same
    points, so the fitted density can be overlaid directly.
    """
    n = int(heights.size)
    x = np.empty(2 * n + 2, dtype=np.float64)
    y = np.empty(2 * n + 2, dtype=np.float64)
    x[0], y[0] = edges[0], 0.0
    x[-1], y[-1] = edges[-1], 0.0
    inner_x = x[1:-1]
    inner_y = y[1:-1]
    inner_x[0::2] = edges[:-1]
    inner_x[1::2] = edges[1:]
    inner_y[0::2] = heights
    inner_y[1::2] = heights
    return x, y


def _fingerprint(*arrays: Optional[np.ndarray]) -> Tuple[Any, ...]:
    """A cheap change-detection key for the preview cache.

    Samples at most ``_FINGERPRINT_SAMPLES`` strided elements per array plus its length
    and endpoints. This is O(1) in the data size, which is the whole point -- hashing a
    1e6-row column every frame would cost more than the operation being previewed. It is
    a heuristic and can miss an edit; see the module docstring.
    """
    parts: List[Any] = []
    for arr in arrays:
        if arr is None:
            parts.append(None)
            continue
        a = np.asarray(arr)
        n = int(a.size)
        parts.append(n)
        if n == 0:
            continue
        step = max(1, n // _FINGERPRINT_SAMPLES)
        with np.errstate(all="ignore"):
            parts.append(float(np.nansum(a[::step])))
        parts.append(float(a[0]))
        parts.append(float(a[-1]))
    return tuple(parts)


def _fit_state_key(model: str, spec: mathops.FitModel) -> str:
    """Identity of a fit model's parameter vector: the model name plus its parameter names.

    The Fit tab remembers an automatic and a manual guess per model. Keying those on the
    model name alone would carry one custom expression's values onto a different
    expression with the same number of parameters, which is a silently wrong start rather
    than a visible one.
    """
    return f"{model}|{','.join(spec.param_names)}"


def _r_squared(y: np.ndarray, y_fit: np.ndarray) -> float:
    """Coefficient of determination, computed over the finite pairs only."""
    good = np.isfinite(y) & np.isfinite(y_fit)
    if not bool(np.any(good)):
        return float("nan")
    observed = y[good]
    predicted = y_fit[good]
    ss_res = float(np.sum((observed - predicted) ** 2))
    ss_tot = float(np.sum((observed - float(np.mean(observed))) ** 2))
    if ss_tot <= 0.0:
        # A constant signal has no variance to explain: a fit that reproduces it exactly
        # is perfect, anything else explains none of it. 0/0 would be nan, which is worse.
        return 1.0 if ss_res <= 0.0 else 0.0
    return 1.0 - ss_res / ss_tot


def _layer_table(layer: Any) -> Optional[Tuple[str, List[str], np.ndarray]]:
    """Resolve ``layer`` to ``(attr, column_names, array)``, or None if not tabular.

    Read-only and copy-free: the returned array is the layer's live geometry. Callers take
    column *views* of it and only cast to float64 inside the memoised compute step.
    """
    layer_type = getattr(layer, "layer_type", "") or ""
    if layer_type in _LAYER_UNSUPPORTED:
        return None
    for attr, default_names in _LAYER_GEOMETRY:
        array = getattr(layer, attr, None)
        if array is None:
            continue
        arr = np.asarray(array)
        if arr.ndim != 2 or arr.shape[1] < 2 or arr.shape[0] < 1:
            continue
        names = list(default_names[: arr.shape[1]])
        names.extend(f"c{i + 1}" for i in range(len(names), arr.shape[1]))
        return attr, names, arr
    return None


def _layer_display(layer: Any) -> str:
    """A stable, unique-per-layer combo entry. Labels alone are not unique."""
    label = getattr(layer, "label", None) or getattr(layer, "layer_type", "layer")
    return f"{label} [{getattr(layer, 'layer_id', '?')}]"


# ----------------------------------------------------------------------------------
# Value objects
# ----------------------------------------------------------------------------------


@dataclass(eq=False)
class _Source:
    """The resolved input of an operation.

    ``eq=False``: a generated ``__eq__`` would compare ndarrays and raise.

    ``x_raw``/``y_raw`` may be float32 strided views into live layer geometry. They are
    never mutated here and are cast to float64 by ``mathops`` at compute time.
    """

    key: Tuple[Any, ...]
    label: str
    x_name: str
    y_name: str
    x_raw: np.ndarray
    y_raw: np.ndarray
    dataset: Optional[DataSet] = None
    x_col: str = ""
    y_col: str = ""
    layer: Optional[Any] = None
    layer_attr: str = ""


@dataclass(eq=False)
class _Result:
    """The output of an operation, plus everything the preview and Apply need."""

    x: np.ndarray
    y: np.ndarray
    x_name: str
    y_name: str
    suffix: str
    plot_label: str
    overlay: Optional[np.ndarray] = None
    overlay_label: str = "source"
    x_changed: bool = False
    stats: List[Tuple[str, str]] = field(default_factory=list)
    allow_apply: bool = True
    allow_replace: bool = True
    #: When set, the preview draws THIS curve as the main line instead of ``x``/``y``.
    #: Peaks needs it: Apply commits the peak coordinates (``x``/``y``), but the preview
    #: must show the whole signal with the peaks marked on it.
    base_x: Optional[np.ndarray] = None
    base_y: Optional[np.ndarray] = None
    #: Points drawn as dots on top of the preview line (the detected peaks).
    markers: Optional[Tuple[np.ndarray, np.ndarray]] = None
    #: Named columns for a multi-column "new dataset" Apply (the full peak table). When
    #: set, it -- not ``x``/``y`` -- is what New dataset writes; None keeps the 2-column
    #: (x, y) default every other operation uses.
    table: Optional[List[Tuple[str, np.ndarray]]] = None


# ----------------------------------------------------------------------------------
# The panel
# ----------------------------------------------------------------------------------


class MathLabPanel(Panel):
    """Integrate / differentiate / smooth / resample / noise, with a live overlay."""

    title = "Math Lab"
    icon = "integral"
    default_open = False

    def __init__(self, ws: Any) -> None:
        super().__init__(ws)

        # -- source selection --
        self._source_kind = "dataset"
        self._ds_name = ""
        self._ds_x_col = ""
        self._ds_y_col = ""
        #: Remembered ``(x_col, y_col)`` pick per dataset name. Without this, switching
        #: dataset and switching back silently resets the columns to the first two, and
        #: a deliberate pick is indistinguishable from a stale one.
        self._ds_columns: Dict[str, Tuple[str, str]] = {}
        #: The dataset names ``_ds_columns`` was last pruned against.
        self._ds_known: List[str] = []
        self._layer_id: Optional[int] = None
        self._layer_x_col = ""
        self._layer_y_col = ""

        # -- operation parameters --
        self._integral_method = "trapezoid"
        self._integral_initial = 0.0

        self._derivative_order = 1
        self._derivative_method = "central"
        self._derivative_window = 11
        self._derivative_polyorder = 3
        #: dy/dx of a parametric curve (x(t), y(t)) via the chain rule, instead of assuming
        #: y is a function of x. Lets a looping/self-intersecting X-Y curve be differentiated.
        self._derivative_parametric = False

        self._smooth_method = "moving_average"
        self._smooth_window = 11
        self._smooth_sigma = 0.0  # 0 => let mathops default it to window / 6
        self._smooth_polyorder = 3

        self._noise_kind = "gaussian"
        self._noise_amplitude = 0.1
        self._noise_relative = True
        self._noise_use_seed = True
        self._noise_seed = 0
        self._noise_nonce = 0

        self._resample_n = 512
        self._resample_kind = "linear"

        self._normalize_mode = "minmax"

        # Cutoffs are stored as a fraction of the Nyquist rate (0, 1), not an absolute
        # frequency: that is unit-free, so the slider needs no per-dataset rescaling, and
        # _compute reports the real cutoff and Nyquist frequency in the stats so the user
        # still sees the value in the data's own units.
        self._filter_type = "lowpass"
        self._filter_order = 4
        self._filter_cutoff = 0.25
        self._filter_cutoff_hi = 0.40

        self._detrend_kind = "linear"  # constant | linear | asls
        # AsLS controls: smoothness as log10(lambda) so a slider spans decades, and the
        # asymmetry p. 1e6 keeps the baseline below moderately sharp peaks by default.
        self._baseline_lam_log10 = 6.0
        self._baseline_p = 0.01

        # Prominence as a fraction of the signal's peak-to-peak range, for the same reason:
        # 0.05 means "at least 5% of the full swing", meaningful at any scale.
        self._peak_prominence = 0.05
        self._peak_distance_on = False
        self._peak_distance = 1

        # Max lag as a fraction of the signal length: the ACF is noisiest at long lags
        # (fewer overlapping samples), so half the record is a sensible default ceiling.
        self._autocorr_maxlag = 0.5

        self._hist_bins = 50
        self._hist_density = False
        self._hist_dist = "normal"

        # Dynamics tab. Delay is in samples (the panel cannot know the signal's period, so
        # a sample count is the honest unit); dim is the embedding dimension used for the
        # Lyapunov reconstruction. Defaults draw a legible phase portrait on a finely
        # sampled ODE trajectory without further tuning.
        self._dyn_mode = "phase"
        self._dyn_delay = 10
        self._dyn_dim = 3

        self._fit_model = "polynomial"
        self._fit_degree = 3
        self._fit_npeaks = 2  # peaks to deconvolve when the model is a sum of peaks
        self._fit_expr = _DEFAULT_FIT_EXPR
        self._fit_manual = False
        #: Manual initial guess per _fit_state_key. Only consulted while _fit_manual.
        self._fit_p0: Dict[str, List[float]] = {}
        #: Heuristic initial guess per _fit_state_key, stashed by _compute. The
        #: heuristics read the source data, which a tab body never sees, so this is the
        #: only way the tab can show the automatic start or seed "Manual" from it.
        self._fit_auto: Dict[str, List[float]] = {}

        # -- apply --
        # Per-tab, keyed by the _TABS key. A name typed on Smooth is about the smoothed
        # curve; carrying it onto Fit (as a single panel-global field did) silently
        # mislabels the output. Same for the mode: "Replace source" makes sense on
        # Smooth and rarely on FFT.
        self._apply_modes: Dict[str, str] = {}
        self._apply_names: Dict[str, str] = {}
        self._layer_kind = "line"
        #: The selected kind's parameters, re-defaulted when the kind changes: a bar's
        #: width means nothing to a histogram, which never declared that option.
        self._layer_opts: Dict[str, Any] = layerops.default_kind_options(self._layer_kind)
        #: Last measured height of each tab's Apply block, keyed by tab. Used to reserve
        #: the space the block needs at the bottom of the panel *before* the scrolling
        #: body is opened; see :meth:`_draw_tab_body`.
        self._apply_heights: Dict[str, float] = {}
        self._notice: Optional[str] = None

        # -- preview cache --
        self._cache_epoch = 0
        self._cache_key: Optional[Tuple[Any, ...]] = None
        self._cache_result: Optional[_Result] = None
        self._cache_error: Optional[str] = None

        #: Operation key requested by :meth:`show_operation`; consumed by the next frame's
        #: tab bar. A pending value cannot be applied immediately -- the tab bar is only
        #: submitted from ``draw``.
        self._pending_tab: Optional[str] = None
        #: Category the pending tab lives in; the top-level bar must switch to it first,
        #: or the operation tab would not be among those drawn and could never be selected.
        self._pending_category: Optional[str] = None

        #: Frames left to show the "Copied" confirmation, and the tab it belongs to, so a
        #: copy on one tab does not flash a confirmation on another.
        self._copied_frames = 0
        self._copied_key = ""

    # -- public API (the workspace's Math actions call this) ------------------------

    def show_operation(self, key: str) -> None:
        """Select the operation tab ``key`` (an id from ``_TABS``) on the next frame.

        The hook the registry's quick verbs -- Integrate, Derivative, Smooth -- need: an
        action can open Math Lab *on the operation the user asked for* instead of on
        whichever tab was last used. Safe to call from a draw callback and from a headless
        process: it only sets state the next frame reads. An unknown key is ignored rather
        than raising, so a stale caller cannot take the frame down.
        """
        if key in _TAB_BY_KEY:
            self._pending_tab = key
            # Jump the top-level bar to the operation's category too, else the operation
            # tab is not among those drawn this frame and the request is silently lost.
            self._pending_category = _CATEGORY_OF[key]

    # -- lifecycle -----------------------------------------------------------------

    def draw(self) -> None:
        """Panel body. The Workspace owns begin/end."""
        if not IMGUI_AVAILABLE:
            return

        self._draw_notice()
        source, source_error = self._draw_source()
        imgui.separator()

        if source is None:
            widgets.error_box(source_error or "No source selected.")
            return

        self._draw_tabs(source)

    def _draw_notice(self) -> None:
        """Show the last deferred message (e.g. "not undoable"), with a dismiss button."""
        if self._notice is None:
            return
        widgets.error_box(self._notice)
        if icons.icon_button("##dismiss_notice", "close", size=18.0, tooltip="Dismiss"):
            self._notice = None

    # -- source --------------------------------------------------------------------

    def _draw_source(self) -> Tuple[Optional[_Source], Optional[str]]:
        """Draw the source selector and resolve it. Returns ``(source, error)``."""
        imgui.push_id("mathlab_source")
        try:
            imgui.text("Source")
            imgui.same_line()
            widgets.help_marker(
                "Operate on two columns of a dataset, or on the geometry of a layer that "
                "is already plotted. Every operation is previewed over the source before "
                "it is applied."
            )
            imgui.same_line()
            if icons.icon_button(
                "##recompute",
                "refresh",
                size=18.0,
                tooltip="Recompute the preview now (the change detector samples the data, "
                "so it can miss an edit)",
            ):
                self._cache_epoch += 1

            # The radios MUST have their own id scope. ImGui derives a widget id from
            # hash(label) ^ id_stack, so radio_button("Dataset") and the enum_combo
            # labelled "Dataset" below resolve to the SAME id, and the combo's popup --
            # which is keyed on that id -- can never open. That made the source selector
            # completely dead: the first control in the panel did nothing. push_id here
            # is what keeps the two apart; do not remove it, and do not rely on the
            # labels staying distinct.
            imgui.push_id("kind")
            try:
                if imgui.radio_button("Dataset", self._source_kind == "dataset"):
                    self._source_kind = "dataset"
                imgui.same_line()
                if imgui.radio_button("Plotted layer", self._source_kind == "layer"):
                    self._source_kind = "layer"
            finally:
                imgui.pop_id()

            imgui.push_item_width(-140.0)
            try:
                if self._source_kind == "dataset":
                    return self._draw_dataset_source()
                return self._draw_layer_source()
            finally:
                imgui.pop_item_width()
        finally:
            imgui.pop_id()

    def _draw_dataset_source(self) -> Tuple[Optional[_Source], Optional[str]]:
        """Dataset selector plus x/y column combos."""
        names = self.store.names()
        if not names:
            return None, "No datasets yet. Paste data or create one in the Data panel."
        if self._ds_name not in names:
            self._ds_name = names[0]
        _changed, self._ds_name = widgets.enum_combo("Dataset", self._ds_name, names)

        dataset = self.store.get(self._ds_name)
        if dataset is None:
            return None, "Select a dataset."
        columns = dataset.column_names()
        if not columns:
            return None, f"Dataset {dataset.name!r} has no columns."
        if dataset.n_rows() == 0:
            return None, f"Dataset {dataset.name!r} has no rows."

        # Per-dataset column memory: recall this dataset's own pick rather than whatever
        # the previously selected dataset happened to be showing.
        x_col, y_col = self._ds_columns.get(self._ds_name, ("", ""))
        if x_col not in columns:
            x_col = columns[0]
        if y_col not in columns:
            y_col = columns[1] if len(columns) > 1 else columns[0]
        _cx, x_col = widgets.enum_combo("X column", x_col, columns)
        _cy, y_col = widgets.enum_combo("Y column", y_col, columns)
        self._ds_columns[self._ds_name] = (x_col, y_col)
        if self._ds_known != names:
            # The set of datasets changed: forget the ones that are gone, so the map
            # cannot grow without bound across a session of create/delete. Comparing two
            # small string lists is cheaper than rebuilding the dict every frame, and a
            # count-based guard would miss a stale key whenever datasets outnumber it.
            self._ds_columns = {k: v for k, v in self._ds_columns.items() if k in names}
            self._ds_known = list(names)
        self._ds_x_col, self._ds_y_col = x_col, y_col

        x = dataset.get(self._ds_x_col)
        y = dataset.get(self._ds_y_col)
        if x is None or y is None:
            return None, "Pick an x and a y column."

        source = _Source(
            key=("dataset", dataset.name, self._ds_x_col, self._ds_y_col),
            label=f"{dataset.name}.{self._ds_y_col}",
            x_name=self._ds_x_col,
            y_name=self._ds_y_col,
            x_raw=x,
            y_raw=y,
            dataset=dataset,
            x_col=self._ds_x_col,
            y_col=self._ds_y_col,
        )
        return source, None

    def _draw_layer_source(self) -> Tuple[Optional[_Source], Optional[str]]:
        """Layer selector plus x/y column combos over the layer's geometry.

        Reading ``plot.scene.layers`` from a draw callback is safe: only *mutation* is
        forbidden (CONTRACT 1.1).
        """
        candidates: List[Tuple[Any, Tuple[str, List[str], np.ndarray]]] = []
        for layer in self.plot.scene.layers:
            table = _layer_table(layer)
            if table is not None:
                candidates.append((layer, table))
        if not candidates:
            return None, "No layer in the scene has tabular x/y data to operate on."

        entries = [_layer_display(layer) for layer, _table in candidates]
        current = ""
        for (layer, _table), entry in zip(candidates, entries):
            if getattr(layer, "layer_id", None) == self._layer_id:
                current = entry
                break
        if not current:
            current = entries[0]
            self._layer_id = getattr(candidates[0][0], "layer_id", None)
        _changed, chosen = widgets.enum_combo("Layer", current, entries)
        index = entries.index(chosen) if chosen in entries else 0
        layer, (attr, columns, array) = candidates[index]
        self._layer_id = getattr(layer, "layer_id", None)

        if self._layer_x_col not in columns:
            self._layer_x_col = columns[0]
        if self._layer_y_col not in columns:
            self._layer_y_col = columns[1]
        _cx, self._layer_x_col = widgets.enum_combo("X column", self._layer_x_col, columns)
        _cy, self._layer_y_col = widgets.enum_combo("Y column", self._layer_y_col, columns)

        # Strided views, not copies: the cast to float64 happens once, inside the cache.
        x_raw = array[:, columns.index(self._layer_x_col)]
        y_raw = array[:, columns.index(self._layer_y_col)]

        source = _Source(
            key=("layer", self._layer_id, attr, self._layer_x_col, self._layer_y_col),
            label=str(getattr(layer, "label", None) or "layer"),
            x_name=self._layer_x_col,
            y_name=self._layer_y_col,
            x_raw=x_raw,
            y_raw=y_raw,
            layer=layer,
            x_col=self._layer_x_col,
            y_col=self._layer_y_col,
            layer_attr=attr,
        )
        return source, None

    # -- tabs ----------------------------------------------------------------------

    def _draw_tabs(self, source: _Source) -> None:
        """Two-level tab bar: a category strip, then that category's operation tabs.

        The categories keep all thirteen operations visible without the tab strip
        overflowing (see ``_CATEGORIES``). The inner operation bar for a category is drawn
        inside that category's tab item, so only one operation bar exists per frame.
        """
        cat_bar_open = imgui.begin_tab_bar("##mathlab_categories")
        if not cat_bar_open:
            return
        try:
            for title, keys in _CATEGORIES:
                pending_here = self._pending_category == title
                flags = imgui.TabItemFlags_.set_selected if pending_here else 0
                selected, _ = imgui.begin_tab_item(title, flags=flags)
                if not selected:
                    continue
                if pending_here:
                    self._pending_category = None
                try:
                    self._draw_operation_tabs(source, keys)
                finally:
                    imgui.end_tab_item()
        finally:
            imgui.end_tab_bar()

    def _draw_operation_tabs(self, source: _Source, keys: Tuple[str, ...]) -> None:
        """The operation tab bar for one category. Each tab draws params, preview, Apply."""
        tab_bar_open = imgui.begin_tab_bar("##mathlab_tabs")
        if not tab_bar_open:
            return
        try:
            for key in keys:
                title, method_name = _TAB_BY_KEY[key]
                flags = imgui.TabItemFlags_.set_selected if self._pending_tab == key else 0
                selected, _ = imgui.begin_tab_item(title, flags=flags)
                if not selected:
                    continue
                if self._pending_tab == key:
                    self._pending_tab = None
                try:
                    imgui.push_id(key)
                    try:
                        self._draw_tab_body(source, key, method_name)
                    finally:
                        imgui.pop_id()
                finally:
                    imgui.end_tab_item()
        finally:
            imgui.end_tab_bar()

    def _draw_tab_body(self, source: _Source, key: str, method_name: str) -> None:
        """Draw one tab: parameters, preview and stats in a scrolling body, then Apply.

        Apply is **pinned to the bottom of the panel** rather than laid out after the
        stats. At the panel's default size it otherwise lands below the fold on 8 of the
        9 tabs (measured: Integral at y=456px in a 423px panel; Fit at y=501px, and it
        moves further down with every coefficient row), so a user could plausibly never
        discover that Apply exists. Everything above it scrolls inside a child region;
        the commit control does not move.

        The reserve is last frame's measured height for *this* tab (tabs differ: the
        Kind combo only exists in "New layer" mode). On the very first frame there is no
        measurement, so an estimate from the current style is used and the layout
        self-corrects on the next frame.
        """
        reserve = self._apply_reserve(key)
        avail = imgui.get_content_region_avail()[1]
        body_height = max(avail - reserve, _MIN_BODY_HEIGHT)

        child_visible = imgui.begin_child("##body", (0.0, body_height))
        result: Optional[_Result] = None
        try:
            if child_visible:
                imgui.spacing()
                body: Callable[[], Tuple[Any, ...]] = getattr(self, method_name)
                imgui.push_item_width(-140.0)
                try:
                    params = body()
                finally:
                    imgui.pop_item_width()

                result, error = self._cached_result(source, params)
                imgui.spacing()
                self._draw_preview(source, result, error)
                if result is not None and result.stats:
                    imgui.spacing()
                    self._draw_report_header(source, result, key)
                    for label, value in result.stats:
                        widgets.stat_row(label, value)
        finally:
            # CONTRACT 2.2: end_child is unconditional.
            imgui.end_child()

        if result is not None and result.allow_apply:
            imgui.separator()
            before = imgui.get_cursor_pos_y()
            self._draw_apply(source, result, key)
            self._apply_heights[key] = max(0.0, imgui.get_cursor_pos_y() - before)
        elif child_visible:
            # The body drew and has nothing to commit -- Statistics, or an op that
            # errored -- so give it the whole panel back. Only when the body actually
            # drew: a culled child must not clobber a good measurement with zero.
            self._apply_heights[key] = 0.0

    def _apply_reserve(self, key: str) -> float:
        """Vertical space to keep free at the panel bottom for ``key``'s Apply block."""
        measured = self._apply_heights.get(key)
        if measured is None:
            # First frame for this tab. Four rows is the "New layer" default: the mode
            # radios, Kind, Name and the Apply button.
            measured = 4.0 * imgui.get_frame_height_with_spacing()
        if measured <= 0.0:
            return 0.0
        # Plus the separator that divides the body from the Apply block.
        return measured + imgui.get_style().item_spacing[1] * 2.0

    def _tab_integral(self) -> Tuple[Any, ...]:
        """Cumulative integral parameters."""
        _c, self._integral_method = widgets.enum_combo(
            "Method",
            self._integral_method,
            _INTEGRAL_METHODS,
            help="trapezoid: exact for straight lines, the safe default. simpson: exact "
            "for parabolas. rectangle: left endpoint, first order, offered for "
            "comparison rather than accuracy.",
        )
        _c, self._integral_initial = widgets.labeled_drag_float(
            "Constant of integration",
            self._integral_initial,
            speed=0.01,
            help="Added to every sample, so the curve starts here instead of at 0.",
        )
        return ("integral", self._integral_method, float(self._integral_initial))

    def _tab_derivative(self) -> Tuple[Any, ...]:
        """Derivative parameters."""
        changed, order = imgui.input_int("Order", int(self._derivative_order))
        if changed:
            self._derivative_order = max(1, min(int(order), 8))
        widgets.help_marker("1 = slope, 2 = curvature. Each order amplifies noise.")

        _c, self._derivative_parametric = imgui.checkbox(
            "Parametric (dy/dx via t)", self._derivative_parametric
        )
        widgets.help_marker(
            "Treat X and Y as a parametric curve (x(t), y(t)) and take dy/dx = (dy/dt)/(dx/dt) "
            "along the sample order. Use this when the curve loops back or crosses itself, so "
            "Y is not a single-valued function of X. Savgol is not available in this mode."
        )

        # Savgol assumes y = f(x) on a uniform x grid; it has no meaning for a parametric
        # curve, so only central/forward are offered there.
        methods = ("central", "forward") if self._derivative_parametric else _DERIVATIVE_METHODS
        if self._derivative_method not in methods:
            self._derivative_method = "central"
        _c, self._derivative_method = widgets.enum_combo(
            "Method",
            self._derivative_method,
            methods,
            help="central: np.gradient, second order and correct on a non-uniform grid. "
            "forward: one-sided. savgol: fits a local polynomial (needs scipy and an "
            "approximately uniform grid; falls back to central otherwise).",
        )
        if self._derivative_method == "savgol":
            changed, window = imgui.input_int("Window", int(self._derivative_window))
            if changed:
                self._derivative_window = max(1, int(window))
            changed, polyorder = imgui.input_int("Poly order", int(self._derivative_polyorder))
            if changed:
                self._derivative_polyorder = max(1, min(int(polyorder), 10))
        return (
            "derivative",
            int(self._derivative_order),
            self._derivative_method,
            int(self._derivative_window),
            int(self._derivative_polyorder),
            bool(self._derivative_parametric),
        )

    def _tab_smooth(self) -> Tuple[Any, ...]:
        """Smoothing parameters."""
        _c, self._smooth_method = widgets.enum_combo(
            "Method",
            self._smooth_method,
            _SMOOTH_METHODS,
            help="moving_average: boxcar, edge-corrected. gaussian: soft kernel. savgol: "
            "preserves peak height and width (needs scipy). median: kills salt-and-pepper "
            "spikes without rounding off edges.",
        )
        changed, window = imgui.input_int("Window", int(self._smooth_window))
        if changed:
            self._smooth_window = max(1, int(window))
        widgets.help_marker(
            "Clamped to the data length and forced odd. 1 leaves the signal untouched."
        )
        if self._smooth_method == "gaussian":
            _c, self._smooth_sigma = widgets.labeled_drag_float(
                "Sigma",
                self._smooth_sigma,
                speed=0.05,
                vmin=0.0,
                vmax=1e6,
                help="0 means window / 6, i.e. the window spans about +/-3 sigma.",
            )
        if self._smooth_method == "savgol":
            changed, polyorder = imgui.input_int("Poly order", int(self._smooth_polyorder))
            if changed:
                self._smooth_polyorder = max(0, min(int(polyorder), 10))
        return (
            "smooth",
            self._smooth_method,
            int(self._smooth_window),
            float(self._smooth_sigma),
            int(self._smooth_polyorder),
        )

    def _tab_noise(self) -> Tuple[Any, ...]:
        """Noise parameters."""
        _c, self._noise_kind = widgets.enum_combo(
            "Kind",
            self._noise_kind,
            _NOISE_KINDS,
            help="gaussian: normal, amplitude is the standard deviation. uniform: flat in "
            "+/-amplitude. poisson: shot noise, amplitude 1 is the true thing. "
            "salt_pepper: amplitude is the fraction of samples replaced by the min/max.",
        )
        _c, self._noise_amplitude = widgets.labeled_drag_float(
            "Amplitude",
            self._noise_amplitude,
            speed=0.005,
            vmin=0.0,
            vmax=1e9,
        )
        if self._noise_kind in ("gaussian", "uniform"):
            _c, self._noise_relative = imgui.checkbox(
                "Relative to data spread", self._noise_relative
            )
            imgui.same_line()
            widgets.help_marker(
                "Scale the amplitude by the signal's own std (gaussian) or peak-to-peak "
                "range (uniform), so 0.1 means 10% whatever the units are. Poisson and "
                "salt_pepper are already data-scaled and ignore this."
            )
        _c, self._noise_use_seed = imgui.checkbox("Fixed seed", self._noise_use_seed)
        if self._noise_use_seed:
            imgui.same_line()
            changed, seed = imgui.input_int("Seed", int(self._noise_seed))
            if changed:
                self._noise_seed = max(0, int(seed))
        imgui.same_line()
        if icons.icon_button("##reroll", "refresh", size=18.0, tooltip="Draw new noise"):
            self._noise_nonce += 1
            if self._noise_use_seed:
                self._noise_seed += 1
        return (
            "noise",
            self._noise_kind,
            float(self._noise_amplitude),
            bool(self._noise_relative),
            int(self._noise_seed) if self._noise_use_seed else None,
            self._noise_nonce,
        )

    def _tab_resample(self) -> Tuple[Any, ...]:
        """Resampling parameters."""
        changed, n = imgui.input_int("Samples", int(self._resample_n))
        if changed:
            self._resample_n = max(2, min(int(n), 10_000_000))
        widgets.help_marker("Evenly spaced over the source's x range; endpoints are exact.")
        _c, self._resample_kind = widgets.enum_combo(
            "Kind",
            self._resample_kind,
            _RESAMPLE_KINDS,
            help="linear: np.interp. cubic: a spline through every knot (needs scipy and "
            "strictly increasing x; falls back to linear otherwise).",
        )
        return ("resample", int(self._resample_n), self._resample_kind)

    def _tab_normalize(self) -> Tuple[Any, ...]:
        """Normalisation parameters."""
        _c, self._normalize_mode = widgets.enum_combo(
            "Mode",
            self._normalize_mode,
            _NORMALIZE_MODES,
            help="minmax: onto [0, 1]. zscore: mean 0, std 1. max_abs: onto [-1, 1], "
            "keeping the sign and any zero crossing. area: unit total absolute area.",
        )
        return ("normalize", self._normalize_mode)

    def _tab_filter(self) -> Tuple[Any, ...]:
        """Butterworth filter parameters. Cutoffs are fractions of the Nyquist rate."""
        if not mathops.filter_available():
            # No pure-numpy filtfilt exists, so there is no fallback to offer: say so and
            # let _compute raise the same message when it runs. Smooth is the alternative.
            widgets.error_box(mathops.FILTER_UNAVAILABLE_MESSAGE)
        _c, self._filter_type = widgets.enum_combo(
            "Type",
            self._filter_type,
            _FILTER_TYPES,
            help="lowpass: keep slow variation, drop fast wiggles. highpass: the reverse. "
            "bandpass: keep a band. bandstop: notch a band out (e.g. mains hum).",
        )
        changed, order = imgui.input_int("Order", int(self._filter_order))
        if changed:
            self._filter_order = max(1, min(int(order), 8))
        widgets.help_marker(
            "Sharper cutoff and more ringing as it rises. The filter is applied forwards "
            "and backwards (zero phase), so a feature is not shifted and the effective "
            "order is doubled."
        )
        is_band = self._filter_type in ("bandpass", "bandstop")
        if is_band:
            _c, self._filter_cutoff = widgets.labeled_slider_float(
                "Low cutoff",
                self._filter_cutoff,
                vmin=0.001,
                vmax=0.999,
                help="Fraction of the Nyquist frequency (half the sample rate). The real "
                "frequency it maps to is shown below the preview.",
            )
            _c, self._filter_cutoff_hi = widgets.labeled_slider_float(
                "High cutoff", self._filter_cutoff_hi, vmin=0.001, vmax=0.999
            )
        else:
            _c, self._filter_cutoff = widgets.labeled_slider_float(
                "Cutoff",
                self._filter_cutoff,
                vmin=0.001,
                vmax=0.999,
                help="Fraction of the Nyquist frequency (half the sample rate). The real "
                "cutoff frequency is shown below the preview.",
            )
        return (
            "filter",
            self._filter_type,
            int(self._filter_order),
            float(self._filter_cutoff),
            float(self._filter_cutoff_hi),
        )

    def _tab_detrend(self) -> Tuple[Any, ...]:
        """Baseline removal: subtract a constant, a line, or a curved AsLS estimate."""
        _c, self._detrend_kind = widgets.enum_combo(
            "Baseline",
            self._detrend_kind,
            _BASELINE_KINDS,
            help="constant: subtract the mean. linear: subtract the least-squares line, "
            "flattening a sloped baseline. asls: a curved baseline that slides under the "
            "peaks -- for a fluorescent or scattering background a straight line cannot "
            "follow. The estimated baseline is drawn over the signal.",
        )
        if self._detrend_kind == "asls":
            if not mathops.baseline_available():
                widgets.error_box(mathops.BASELINE_UNAVAILABLE_MESSAGE)
            _c, self._baseline_lam_log10 = widgets.labeled_slider_float(
                "Smoothness",
                self._baseline_lam_log10,
                vmin=2.0,
                vmax=9.0,
                fmt="1e%.1f",
                help="log10 of lambda. Higher is a stiffer, flatter baseline; lower lets "
                "it bend into the signal. Raise it if the baseline eats into the peaks.",
            )
            _c, self._baseline_p = widgets.labeled_slider_float(
                "Asymmetry",
                self._baseline_p,
                vmin=0.001,
                vmax=0.5,
                fmt="%.3f",
                help="Weight of points above the baseline. Small (~0.01) keeps it below "
                "the peaks; raise it for a signal with dips instead of peaks.",
            )
        return (
            "detrend",
            self._detrend_kind,
            float(self._baseline_lam_log10),
            float(self._baseline_p),
        )

    def _tab_peaks(self) -> Tuple[Any, ...]:
        """Peak-detection parameters. Prominence is a fraction of the signal's range."""
        imgui.text_disabled("Local maxima, marked on the signal below.")
        imgui.same_line()
        widgets.help_marker(
            "Prominence is how far a peak stands above the surrounding baseline, as a "
            "fraction of the signal's full peak-to-peak swing. Raise it to reject noise "
            "wiggles; lower it to catch small real peaks. This is the one knob that "
            "matters most."
        )
        _c, self._peak_prominence = widgets.labeled_slider_float(
            "Min prominence",
            self._peak_prominence,
            vmin=0.0,
            vmax=1.0,
            help="Fraction of the peak-to-peak range. 0 accepts every local maximum.",
        )
        _c, self._peak_distance_on = imgui.checkbox("Min separation", self._peak_distance_on)
        if self._peak_distance_on:
            imgui.same_line()
            changed, distance = imgui.input_int("##peak_distance", int(self._peak_distance))
            if changed:
                self._peak_distance = max(1, int(distance))
            imgui.same_line()
            widgets.help_marker(
                "Minimum gap between accepted peaks, in samples. When two are closer, the "
                "shorter is dropped."
            )
        imgui.text_disabled("Apply as a new dataset for the full peak table.")
        imgui.same_line()
        widgets.help_marker(
            "New dataset writes one row per peak: position, height, prominence, width and "
            "area. The area is integrated between each peak's bounding valleys, accurate "
            "for baseline-separated peaks. Overlapping peaks that never return to baseline "
            "need deconvolution instead -- fit them on the Fit tab."
        )
        return (
            "peaks",
            float(self._peak_prominence),
            int(self._peak_distance) if self._peak_distance_on else None,
        )

    def _tab_histogram(self) -> Tuple[Any, ...]:
        """Histogram parameters, with an optional maximum-likelihood distribution fit."""
        changed, bins = imgui.input_int("Bins", int(self._hist_bins))
        if changed:
            self._hist_bins = max(1, min(int(bins), 10000))
        widgets.help_marker("Equal-width bins over the range of the y values.")
        _c, self._hist_density = imgui.checkbox("Density", self._hist_density)
        imgui.same_line()
        widgets.help_marker(
            "Scale the bars so they integrate to 1 -- a probability density, directly "
            "comparable to the fitted curve. Off counts samples per bin."
        )
        if not mathops.distributions_available():
            widgets.error_box(mathops.DIST_UNAVAILABLE_MESSAGE)
        _c, self._hist_dist = widgets.enum_combo(
            "Fit distribution",
            self._hist_dist,
            _HIST_DIST_OPTIONS,
            help="Overlay a maximum-likelihood fit of this distribution and score it with "
            "a Kolmogorov-Smirnov test (a p-value above ~0.05 means the data is consistent "
            "with it). 'none' shows the histogram alone.",
        )
        return ("histogram", int(self._hist_bins), self._hist_dist, bool(self._hist_density))

    def _fit_spec(self, model: str, expr: str, n_peaks: Optional[int] = None) -> mathops.FitModel:
        """The FitModel for a Fit-tab selection.

        Shared by ``_tab_fit`` and ``_compute`` so the parameter table and the fit can
        never disagree about what is being fitted. ``n_peaks`` defaults to the tab's own
        count; ``_compute`` passes the count captured in the cache key so a recompute uses
        exactly the number the preview was showing.

        Raises:
            ValueError: With a message meant for the user (bad expression, unknown model).
        """
        if model in _MULTIPEAK_MODELS:
            count = self._fit_npeaks if n_peaks is None else int(n_peaks)
            return mathops.multipeak_model(int(count), _MULTIPEAK_MODELS[model])
        if model == "custom":
            return mathops.custom_model(expr)
        return mathops.resolve_model(model)

    def _fit_guess_controls(
        self, model: str, spec: mathops.FitModel
    ) -> Optional[Tuple[float, ...]]:
        """The initial-guess block. Returns the p0 to fit with, or None for automatic.

        ``curve_fit`` is a *local* optimiser: the starting point decides whether it finds
        the right basin at all, so the guess is shown rather than hidden, and can be
        taken over. The automatic values come from ``_compute`` (the heuristics need the
        source data, which a tab body does not receive), so on the very first draw of a
        model there is nothing to show yet and the row says so; from the next frame on it
        is the real start.
        """
        key = _fit_state_key(model, spec)
        names = spec.param_names
        auto = self._fit_auto.get(key)
        seed: Optional[List[float]] = list(auto) if auto and len(auto) == len(names) else None

        _changed, self._fit_manual = imgui.checkbox("Manual initial guess", self._fit_manual)
        widgets.help_marker(
            "Off: each parameter starts from a heuristic read off the data -- peak "
            "position and half width for the peak models, a log-linear rate for the "
            "exponential, log-log for the power law. That is what makes these converge "
            "on real data. Turn it on when the fit lands somewhere obviously wrong: a "
            "frequency, for one, has a local minimum every half period and no heuristic "
            "can guess it for you."
        )

        if not self._fit_manual:
            if seed is None:
                imgui.text_disabled("Start values are chosen from the data.")
            else:
                for name, value in zip(names, seed):
                    widgets.stat_row(f"{name} (auto start)", _fmt(value))
            return None

        values = self._fit_p0.get(key)
        if values is None or len(values) != len(names):
            # Seed a manual guess from the automatic one: editing a good start is the
            # workflow; typing four numbers from nothing is not.
            values = seed if seed is not None else [1.0] * len(names)
            self._fit_p0[key] = values

        for i, name in enumerate(names):
            changed, value = imgui.input_float(
                f"{name} (start)", float(values[i]), 0.0, 0.0, "%.6g"
            )
            if changed and math.isfinite(float(value)):
                values[i] = float(value)
        if imgui.small_button("Reset to auto") and seed is not None:
            values[:] = seed
        return tuple(values)

    def _tab_fit(self) -> Tuple[Any, ...]:
        """Fit parameters: polynomial regression, a named nonlinear model, or custom f(x).

        Polynomial keeps its degree spinner and its closed-form path. Everything else is
        nonlinear least squares, which is what "fitting" means to the scientist this
        panel is for: free parameters, an initial guess, and a standard error on every
        fitted value.
        """
        changed, self._fit_model = widgets.enum_combo(
            "Model",
            self._fit_model,
            _FIT_MODELS,
            help="polynomial: linear least squares on the coefficients, no starting "
            "guess needed. Everything else is nonlinear least squares and reports a "
            "standard error per parameter. custom: type your own f(x).",
        )
        model = self._fit_model
        if changed:
            # A guess is a vector of this model's parameters and means nothing for
            # another one; the per-model _fit_p0/_fit_auto keys already keep the values
            # apart, so only the mode has to be reset.
            self._fit_manual = False

        if model == "polynomial":
            degree_changed, degree = imgui.input_int("Degree", int(self._fit_degree))
            if degree_changed:
                self._fit_degree = max(0, min(int(degree), 20))
            widgets.help_marker(
                "Least squares. Clamped to 20 and to the sample count; a high degree on "
                "few points oscillates wildly between them."
            )
            return ("fit", "polynomial", int(self._fit_degree), "", None)

        if not mathops.fit_available():
            # No numpy fallback exists for curve_fit, so say so plainly instead of
            # offering a control that cannot work. Polynomial is still right there.
            widgets.error_box(mathops.FIT_UNAVAILABLE_MESSAGE)
            return ("fit", model, 0, self._fit_expr, None)

        if model in _MULTIPEAK_MODELS:
            npeaks_changed, npeaks = imgui.input_int("Peaks", int(self._fit_npeaks))
            if npeaks_changed:
                self._fit_npeaks = max(1, min(int(npeaks), _MAX_FIT_PEAKS))
            widgets.help_marker(
                "How many overlapping peaks to separate. Their centres and widths are "
                "seeded from the data and fitted jointly, so this resolves peaks that run "
                "into each other -- the case valley integration on the Peaks tab cannot. "
                "Each peak's area is reported exactly from its fitted amplitude and width."
            )

        if model == "custom":
            _expr_changed, self._fit_expr = imgui.input_text("f(x)", self._fit_expr)
            widgets.help_marker(
                "Every name other than x becomes a fitted parameter: 'a*exp(-x/b) + c' "
                "fits a, b and c. Same functions as the Functions panel (sin, exp, "
                "gauss, ...)."
            )

        # The peak count rides in the tuple's int slot (unused by non-polynomial models) so
        # the preview cache invalidates when it changes.
        int_slot = int(self._fit_npeaks) if model in _MULTIPEAK_MODELS else 0
        try:
            spec = self._fit_spec(model, self._fit_expr, int_slot)
        except ValueError as exc:
            widgets.error_box(str(exc))
            return ("fit", model, int_slot, self._fit_expr, None)

        imgui.text_disabled(spec.formula)
        p0 = self._fit_guess_controls(model, spec)
        return ("fit", model, int_slot, self._fit_expr, p0)

    def _tab_fft(self) -> Tuple[Any, ...]:
        """FFT parameters (there are none: the transform is fully determined)."""
        imgui.text_disabled("One-sided amplitude spectrum.")
        imgui.same_line()
        widgets.help_marker(
            "Frequencies are in cycles per unit of x, and a pure sinusoid of amplitude A "
            "peaks at A. A non-uniform x grid is linearly resampled onto a uniform one "
            "first, which smears the spectrum -- that is an approximation, not a detail."
        )
        return ("fft",)

    def _tab_autocorr(self) -> Tuple[Any, ...]:
        """Autocorrelation parameters: how far out in lag to compute."""
        imgui.text_disabled("Self-similarity vs lag; a peak marks a repeating period.")
        imgui.same_line()
        widgets.help_marker(
            "The autocorrelation is 1 at lag 0 and shows a peak at the lag of any period "
            "the signal repeats over -- often clearer than an FFT on a short record. The "
            "first peak (marked) is reported as the dominant period, in x units."
        )
        _c, self._autocorr_maxlag = widgets.labeled_slider_float(
            "Max lag",
            self._autocorr_maxlag,
            vmin=0.05,
            vmax=1.0,
            help="Largest lag to compute, as a fraction of the record length. The estimate "
            "is noisier at long lags, so half is a good default.",
        )
        return ("autocorr", float(self._autocorr_maxlag))

    def _tab_stats(self) -> Tuple[Any, ...]:
        """Statistics has no parameters; describe() is fully determined by the source."""
        imgui.text_disabled("Summary of the source's y column. Every statistic is nan-safe.")
        return ("stats",)

    def _tab_dynamics(self) -> Tuple[Any, ...]:
        """Dynamical-systems analysis: reconstruct and characterise the flow behind y(t)."""
        imgui.text_disabled("Read the y column as a dynamical system's time series.")
        imgui.same_line()
        widgets.help_marker(
            "These analyses treat the source's y column as one observable of a dynamical "
            "system sampled over x (time). Feed them a trajectory column from the Dynamics "
            "panel, or any time series.\n\n"
            "phase: reconstruct the attractor from the single signal by plotting y(t) "
            "against a time-delayed copy y(t+delay) -- a Takens embedding.\n"
            "return map: plot each local maximum of y against the next; a chaotic flow "
            "like Lorenz collapses onto a sharp curve here.\n"
            "lyapunov: estimate the largest Lyapunov exponent (Rosenstein). Positive means "
            "chaos, ~0 a limit cycle, negative a fixed point."
        )
        _c, self._dyn_mode = widgets.enum_combo("Analysis", self._dyn_mode, _DYNAMICS_MODES)

        if self._dyn_mode in ("phase", "lyapunov"):
            changed, delay = imgui.input_int("Delay (samples)", int(self._dyn_delay))
            if changed:
                self._dyn_delay = max(1, int(delay))
            widgets.help_marker(
                "Time shift for the delay coordinate, in samples. Too small and the "
                "embedding is a thin diagonal line; open the loop by raising it toward a "
                "quarter of the signal's period."
            )
        if self._dyn_mode == "lyapunov":
            changed, dim = imgui.input_int("Embedding dim", int(self._dyn_dim))
            if changed:
                self._dyn_dim = max(2, min(int(dim), 10))
            widgets.help_marker(
                "Dimensions of the reconstructed phase space. 3 suits most 3-D flows; raise "
                "it if the exponent looks unstable."
            )
        return ("dynamics", self._dyn_mode, int(self._dyn_delay), int(self._dyn_dim))

    # -- compute -------------------------------------------------------------------

    def _cached_result(
        self, source: _Source, params: Tuple[Any, ...]
    ) -> Tuple[Optional[_Result], Optional[str]]:
        """Memoised ``_compute``. See the module docstring on the fingerprint heuristic."""
        key = (
            self._cache_epoch,
            source.key,
            _fingerprint(source.x_raw, source.y_raw),
            params,
        )
        if key == self._cache_key:
            return self._cache_result, self._cache_error

        self._cache_key = key
        try:
            self._cache_result = self._compute(source, params)
            self._cache_error = None
        except ValueError as exc:
            # The documented failure mode of every mathops entry point.
            self._cache_result = None
            self._cache_error = str(exc)
        except Exception as exc:  # pragma: no cover - defensive
            # An exception escaping here would kill the imgui frame, taking the whole
            # window with it. Surface it and keep drawing.
            logger.exception("Math Lab preview failed for params %r", params)
            self._cache_result = None
            self._cache_error = f"Unexpected error: {exc}"
        return self._cache_result, self._cache_error

    def _compute(self, source: _Source, params: Tuple[Any, ...]) -> _Result:
        """Run one operation. Pure: reads the source, returns a new _Result."""
        kind = params[0]
        x_raw, y_raw = source.x_raw, source.y_raw
        y_name = source.y_name

        if kind == "integral":
            _, method, initial = params
            xs, ys = _sorted_by_x(x_raw, y_raw)
            out = mathops.integrate(x_raw, y_raw, method=method, initial=initial)
            total = mathops.definite_integral(x_raw, y_raw, method=method)
            return _Result(
                x=xs,
                y=out,
                x_name=source.x_name,
                y_name=f"{y_name} integral",
                suffix="integral",
                plot_label=f"cumulative integral ({method})",
                overlay=ys,
                x_changed=True,
                stats=[
                    ("Definite integral", _fmt(total)),
                    ("Result at x max", _fmt(float(out[-1]))),
                    ("Span of x", _fmt(float(xs[-1] - xs[0]))),
                ],
            )

        if kind == "derivative":
            _, order, method, window, polyorder, parametric = params
            if parametric:
                # A parametric curve keeps its own sample order (it may loop back on itself),
                # so nothing is sorted and the original x is the abscissa of the result.
                par_method = method if method in ("central", "forward") else "central"
                out = mathops.parametric_derivative(x_raw, y_raw, order=order, method=par_method)
                xs, ys = np.asarray(x_raw, dtype=np.float64), np.asarray(y_raw, dtype=np.float64)
                label = f"parametric derivative order {order} ({par_method})"
                suffix = f"d^{order}y/dx^{order}" if order > 1 else "dy/dx"
                x_changed = False
            else:
                xs, ys = _sorted_by_x(x_raw, y_raw)
                out = mathops.derivative(
                    x_raw,
                    y_raw,
                    order=order,
                    method=method,
                    window=window,
                    polyorder=polyorder,
                )
                label = f"derivative order {order} ({method})"
                suffix = f"derivative {order}"
                x_changed = True
            with np.errstate(all="ignore"):
                peak = float(np.nanmax(np.abs(out))) if out.size else float("nan")
            return _Result(
                x=xs,
                y=out,
                x_name=source.x_name,
                y_name=f"{y_name} {suffix}",
                suffix=suffix,
                plot_label=label,
                overlay=ys,
                x_changed=x_changed,
                stats=[
                    ("Peak absolute slope", _fmt(peak)),
                    ("Mean slope", _fmt(float(np.nanmean(out)) if out.size else float("nan"))),
                ],
            )

        if kind == "smooth":
            _, method, window, sigma, polyorder = params
            xa = np.asarray(x_raw, dtype=np.float64)
            ya = np.asarray(y_raw, dtype=np.float64)
            kwargs: Dict[str, Any] = {}
            if method == "gaussian" and sigma > 0.0:
                kwargs["sigma"] = sigma
            if method == "savgol":
                kwargs["polyorder"] = polyorder
            out = mathops.smooth(y_raw, method=method, window=window, **kwargs)
            residual = ya - out
            with np.errstate(all="ignore"):
                rms = float(np.sqrt(np.nanmean(residual**2))) if residual.size else float("nan")
            return _Result(
                x=xa,
                y=out,
                x_name=source.x_name,
                y_name=f"{y_name} smoothed",
                suffix="smoothed",
                plot_label=f"smoothed ({method}, window {window})",
                overlay=ya,
                stats=[
                    ("Removed (RMS)", _fmt(rms)),
                    ("Peak residual", _fmt(_nan_peak(residual))),
                ],
            )

        if kind == "noise":
            _, noise_kind, amplitude, relative, seed, _nonce = params
            xa = np.asarray(x_raw, dtype=np.float64)
            ya = np.asarray(y_raw, dtype=np.float64)
            out = mathops.add_noise(
                y_raw,
                kind=noise_kind,
                amplitude=amplitude,
                relative=relative,
                seed=seed,
            )
            added = out - ya
            with np.errstate(all="ignore"):
                rms = float(np.sqrt(np.nanmean(added**2))) if added.size else float("nan")
                signal_rms = float(np.sqrt(np.nanmean(ya**2))) if ya.size else float("nan")
            stats = [("Added noise (RMS)", _fmt(rms))]
            if math.isfinite(rms) and rms > 0.0 and math.isfinite(signal_rms):
                stats.append(("Signal / noise (dB)", _fmt(20.0 * math.log10(signal_rms / rms))))
            return _Result(
                x=xa,
                y=out,
                x_name=source.x_name,
                y_name=f"{y_name} + noise",
                suffix="+ noise",
                plot_label=f"with {noise_kind} noise",
                overlay=ya,
                stats=stats,
            )

        if kind == "resample":
            _, n, resample_kind = params
            x_new, y_new = mathops.resample(x_raw, y_raw, n, kind=resample_kind)
            xs, ys = _sorted_by_x(x_raw, y_raw)
            # The overlay must live on the RESULT's grid, since mini_plot draws both
            # series against a single x. Linear interpolation of the source onto x_new is
            # the honest way to say "here is where the original went".
            overlay = np.interp(x_new, xs, ys)
            spacing = float(x_new[1] - x_new[0]) if x_new.size > 1 else float("nan")
            return _Result(
                x=x_new,
                y=y_new,
                x_name=source.x_name,
                y_name=f"{y_name} resampled",
                suffix="resampled",
                plot_label=f"resampled to {len(y_new)} ({resample_kind})",
                overlay=overlay,
                overlay_label="source (interpolated)",
                x_changed=True,
                stats=[
                    ("Samples in", _fmt(float(np.asarray(y_raw).size))),
                    ("Samples out", _fmt(float(y_new.size))),
                    ("New spacing", _fmt(spacing)),
                ],
            )

        if kind == "normalize":
            _, mode = params
            xa = np.asarray(x_raw, dtype=np.float64)
            ya = np.asarray(y_raw, dtype=np.float64)
            out = mathops.normalize(y_raw, mode=mode)
            return _Result(
                x=xa,
                y=out,
                x_name=source.x_name,
                y_name=f"{y_name} normalized",
                suffix="normalized",
                plot_label=f"normalized ({mode})",
                overlay=ya,
                stats=[
                    ("Result minimum", _fmt(_nan_min(out))),
                    ("Result maximum", _fmt(_nan_max(out))),
                ],
            )

        if kind == "filter":
            _, btype, order, frac_lo, frac_hi = params
            xs, ys = _sorted_by_x(x_raw, y_raw)
            if xs.size < 2 or float(xs[-1] - xs[0]) <= 0.0:
                raise ValueError("filtering needs at least 2 samples over a non-zero x span")
            nyquist = 0.5 * (xs.size - 1) / float(xs[-1] - xs[0])
            is_band = btype in ("bandpass", "bandstop")
            lo = min(max(frac_lo, 1e-6), 1.0 - 1e-6) * nyquist
            hi = min(max(frac_hi, 1e-6), 1.0 - 1e-6) * nyquist
            cutoff: Any = (lo, hi) if is_band else lo
            out = mathops.butter_filter(x_raw, y_raw, cutoff, btype=btype, order=int(order))
            ya = np.asarray(ys, dtype=np.float64)
            residual = ya - out
            with np.errstate(all="ignore"):
                rms = float(np.sqrt(np.nanmean(residual**2))) if residual.size else float("nan")
            stats = [("Nyquist frequency", _fmt(nyquist))]
            if is_band:
                stats.append(("Low cutoff", _fmt(lo)))
                stats.append(("High cutoff", _fmt(hi)))
            else:
                stats.append(("Cutoff frequency", _fmt(lo)))
            stats.append(("Removed (RMS)", _fmt(rms)))
            return _Result(
                x=xs,
                y=out,
                x_name=source.x_name,
                y_name=f"{y_name} filtered",
                suffix="filtered",
                plot_label=f"{btype} (order {order})",
                overlay=ya,
                stats=stats,
            )

        if kind == "detrend":
            _, detrend_kind, lam_log10, asym = params
            xa = np.asarray(x_raw, dtype=np.float64)
            ya = np.asarray(y_raw, dtype=np.float64)
            if detrend_kind == "asls":
                baseline = mathops.baseline_asls(y_raw, lam=10.0**lam_log10, p=asym)
                out = ya - baseline
                label = f"baseline-corrected (asls, 1e{lam_log10:.1f})"
            else:
                out = mathops.detrend(x_raw, y_raw, kind=detrend_kind)
                baseline = ya - out
                label = f"baseline-corrected ({detrend_kind})"
            return _Result(
                x=xa,
                y=out,
                x_name=source.x_name,
                y_name=f"{y_name} corrected",
                suffix="baseline-corrected",
                plot_label=label,
                # Show the original signal with the estimated baseline drawn through it, so
                # the user can judge whether the baseline tracks the background; Apply
                # commits the corrected signal (x/y).
                base_x=xa,
                base_y=ya,
                overlay=baseline,
                overlay_label="baseline",
                stats=[
                    ("Result mean", _fmt(_nan_reduce(out, np.nanmean))),
                    (
                        "Baseline at x min",
                        _fmt(float(baseline[0]) if baseline.size else float("nan")),
                    ),
                    (
                        "Baseline at x max",
                        _fmt(float(baseline[-1]) if baseline.size else float("nan")),
                    ),
                ],
            )

        if kind == "peaks":
            _, prominence_frac, distance = params
            xs, ys = _sorted_by_x(x_raw, y_raw)
            with np.errstate(all="ignore"):
                span = float(np.nanmax(ys) - np.nanmin(ys)) if ys.size else 0.0
            prominence = prominence_frac * span if span > 0.0 else None
            peak_x, peak_y, props = mathops.find_peaks(
                x_raw, y_raw, prominence=prominence, distance=distance
            )
            prominences = np.asarray(props.get("prominences", np.empty(0)), dtype=np.float64)
            widths = np.asarray(props.get("widths", np.empty(0)), dtype=np.float64)
            areas = np.asarray(props.get("areas", np.empty(0)), dtype=np.float64)
            stats = [("Peaks found", str(int(peak_x.size)))]
            table: Optional[List[Tuple[str, np.ndarray]]] = None
            if peak_x.size:
                tallest = int(np.argmax(peak_y))
                stats.append(("Tallest peak at x", _fmt(float(peak_x[tallest]))))
                stats.append(("Tallest peak height", _fmt(float(peak_y[tallest]))))
                if peak_x.size > 1:
                    stats.append(("Mean spacing", _fmt(float(np.mean(np.diff(peak_x))))))
                if widths.size and bool(np.any(np.isfinite(widths))):
                    stats.append(("Mean width", _fmt(_nan_reduce(widths, np.nanmean))))
                if areas.size and bool(np.any(np.isfinite(areas))):
                    stats.append(("Total area", _fmt(float(np.nansum(areas)))))
                    stats.append(("Largest peak area", _fmt(float(areas[tallest]))))
                # The deliverable of peak analysis: a full table, applied as a new dataset.
                table = [
                    ("position", peak_x),
                    ("height", peak_y),
                    ("prominence", prominences),
                    ("width", widths),
                    ("area", areas),
                ]
            return _Result(
                # x/y are the peak coordinates: that is what a "new layer" Apply plots.
                # base_x/base_y carry the full signal so the preview shows it with the peaks
                # marked; ``table`` is the full per-peak table a "new dataset" Apply writes.
                x=peak_x,
                y=peak_y,
                x_name=source.x_name,
                y_name=f"{y_name} peaks",
                suffix="peaks",
                plot_label="signal",
                base_x=np.asarray(xs, dtype=np.float64),
                base_y=np.asarray(ys, dtype=np.float64),
                markers=(peak_x, peak_y),
                table=table,
                # A peak table cannot overwrite the signal it was measured from.
                allow_replace=False,
                # With no peaks there is nothing to plot or store; guard Apply.
                allow_apply=bool(peak_x.size),
                stats=stats,
            )

        if kind == "histogram":
            _, bins, dist_name, density = params
            ya = np.asarray(y_raw, dtype=np.float64)
            edges, heights, centers = mathops.histogram(y_raw, bins=bins, density=density)
            sil_x, sil_y = _histogram_silhouette(edges, heights)
            n_finite = int(np.count_nonzero(np.isfinite(ya)))
            height_name = "density" if density else "count"
            stats = [("Finite samples", str(n_finite)), ("Bins", str(int(heights.size)))]

            overlay: Optional[np.ndarray] = None
            overlay_label = "source"
            if dist_name != "none":
                try:
                    fit = mathops.fit_distribution(y_raw, dist_name, sil_x)
                    pdf = np.asarray(fit["pdf"], dtype=np.float64)
                    if not density:
                        # The PDF integrates to 1; scale it onto the count axis so it sits
                        # on the bars instead of hugging zero.
                        width = float(edges[1] - edges[0]) if edges.size > 1 else 1.0
                        pdf = pdf * n_finite * width
                    overlay = pdf
                    overlay_label = f"{dist_name} fit"
                    for label, value in zip(fit["labels"], fit["params"]):
                        stats.append((label, _fmt(value)))
                    stats.append(("KS statistic", _fmt(fit["ks_statistic"])))
                    stats.append(("KS p-value", _fmt(fit["ks_pvalue"])))
                    stats.append(("Log likelihood", _fmt(fit["log_likelihood"])))
                except ValueError as exc:
                    # A failed or unavailable fit must not take the histogram down with it.
                    stats.append(("Distribution fit", str(exc)))

            return _Result(
                # Apply commits the histogram table (bin centre, height); the preview draws
                # the silhouette (base_*) with the fitted PDF overlaid on the same grid.
                x=centers,
                y=heights,
                x_name=f"{y_name} value",
                y_name=height_name,
                suffix="histogram",
                plot_label="histogram",
                base_x=sil_x,
                base_y=sil_y,
                overlay=overlay,
                overlay_label=overlay_label,
                x_changed=True,
                # A histogram lives in value/count space, not over the signal's own x.
                allow_replace=False,
                stats=stats,
            )

        if kind == "fit":
            _, model_key, degree, expr, p0 = params
            ya = np.asarray(y_raw, dtype=np.float64)
            xa = np.asarray(x_raw, dtype=np.float64)

            if model_key == "polynomial":
                coeffs, y_fit = mathops.fit_polynomial(x_raw, y_raw, degree)
                stats: List[Tuple[str, str]] = [
                    ("R squared", _fmt(_r_squared(ya, y_fit))),
                    ("Effective degree", _fmt(float(len(coeffs) - 1))),
                ]
                # np.polyfit returns highest order first.
                for i, coeff in enumerate(coeffs):
                    power = len(coeffs) - 1 - i
                    name = "constant" if power == 0 else ("x" if power == 1 else f"x^{power}")
                    stats.append((f"Coefficient of {name}", _fmt(float(coeff))))
                return _Result(
                    x=xa,
                    y=y_fit,
                    x_name=source.x_name,
                    y_name=f"{y_name} fit",
                    suffix="fit",
                    plot_label=f"polynomial fit (degree {len(coeffs) - 1})",
                    overlay=ya,
                    stats=stats,
                )

            if not mathops.fit_available():
                raise ValueError(mathops.FIT_UNAVAILABLE_MESSAGE)
            # ``degree`` carries the peak count for the deconvolution models (0 otherwise).
            spec = self._fit_spec(model_key, expr, degree)

            # Stash the heuristic start before fitting, not after: when the fit fails
            # this is exactly the moment the user needs to see what it started from and
            # take it over.
            key = _fit_state_key(model_key, spec)
            try:
                self._fit_auto[key] = [float(v) for v in mathops.initial_guess(xa, ya, spec)]
            except ValueError:
                self._fit_auto.pop(key, None)

            popt, pcov, y_fit, names = mathops.fit_model(
                xa,
                ya,
                spec,
                p0=(np.asarray(p0, dtype=np.float64) if p0 is not None else None),
            )
            errors = mathops.fit_errors(pcov)
            summary = mathops.fit_statistics(ya, y_fit, len(names))
            stats = [
                ("R squared", _fmt(summary["r_squared"])),
                # No per-point errors are available here, so chi squared carries unit
                # weights and this is the residual variance, in y units squared -- not
                # the dimensionless "should be near 1" statistic. Labelled, not implied.
                ("Reduced chi sq (unit weights)", _fmt(summary["reduced_chi_squared"])),
                ("Degrees of freedom", _fmt(summary["dof"])),
                ("RMS residual", _fmt(summary["rmse"])),
            ]
            # The whole point of the tab: a value AND its standard error, per parameter.
            for name, value, error in zip(names, popt, errors):
                stats.append((name, f"{_fmt(float(value))} +/- {_fmt(float(error))}"))
            if model_key in _MULTIPEAK_MODELS:
                # The deconvolution payoff: each fitted peak's area, exact even where the
                # peaks overlap, computed analytically from its amplitude and width.
                shape = _MULTIPEAK_MODELS[model_key]
                peak_areas = [
                    mathops.peak_area(shape, popt[3 * i], popt[3 * i + 2])
                    for i in range((len(names) - 1) // 3)
                ]
                stats.append(("Total peak area", _fmt(float(sum(peak_areas)))))
                for i, area in enumerate(peak_areas, start=1):
                    stats.append((f"Peak {i} area", _fmt(float(area))))
            return _Result(
                x=xa,
                y=y_fit,
                x_name=source.x_name,
                y_name=f"{y_name} fit",
                suffix="fit",
                plot_label=f"{spec.label} fit",
                overlay=ya,
                stats=stats,
            )

        if kind == "fft":
            freqs, magnitude = mathops.fft_spectrum(x_raw, y_raw)
            stats = [("Bins", _fmt(float(freqs.size)))]
            # Ignore DC: it is the mean, it is almost always the tallest bin, and calling
            # it "the dominant frequency" would be wrong.
            if magnitude.size > 1:
                ac = magnitude[1:]
                if bool(np.any(np.isfinite(ac))):
                    peak_index = int(np.nanargmax(ac)) + 1
                    stats.append(("Peak frequency", _fmt(float(freqs[peak_index]))))
                    stats.append(("Peak amplitude", _fmt(float(magnitude[peak_index]))))
            if freqs.size:
                stats.append(("Nyquist frequency", _fmt(float(freqs[-1]))))
            return _Result(
                x=freqs,
                y=magnitude,
                x_name="frequency",
                y_name="magnitude",
                suffix="spectrum",
                plot_label="amplitude spectrum",
                overlay=None,
                x_changed=True,
                # The spectrum lives in frequency, not in x: writing it back over the
                # signal it came from would be meaningless.
                allow_replace=False,
                stats=stats,
            )

        if kind == "autocorr":
            _, maxlag_frac = params
            xs, ys = _sorted_by_x(x_raw, y_raw)
            n = int(xs.size)
            max_lag = max(1, int(maxlag_frac * (n - 1))) if n > 1 else 0
            lags, acf = mathops.autocorrelation(ys, max_lag=max_lag)
            dx = float(np.median(np.diff(xs))) if n > 1 else 1.0
            if dx <= 0.0:
                dx = 1.0
            lag_x = lags * dx

            # The dominant period is the first ACF peak past lag 0; its harmonics follow.
            # A modest prominence floor keeps noise ripple from being called a period.
            peak_x, peak_y, _props = mathops.find_peaks(lag_x, acf, prominence=0.1)
            stats = [("Lags computed", str(int(lags.size)))]
            markers: Optional[Tuple[np.ndarray, np.ndarray]] = None
            if peak_x.size:
                markers = (peak_x, peak_y)
                stats.append(("Dominant period", _fmt(float(peak_x[0]))))
                stats.append(("Correlation at period", _fmt(float(peak_y[0]))))
            else:
                stats.append(("Dominant period", "no clear period"))
            return _Result(
                x=lag_x,
                y=acf,
                x_name="lag",
                y_name="autocorrelation",
                suffix="autocorrelation",
                plot_label="autocorrelation",
                overlay=None,
                markers=markers,
                x_changed=True,
                # An ACF lives in lag space, not over the signal's own x.
                allow_replace=False,
                stats=stats,
            )

        if kind == "stats":
            xa = np.asarray(x_raw, dtype=np.float64)
            ya = np.asarray(y_raw, dtype=np.float64)
            summary = mathops.describe(y_raw)
            stats = [
                (label, _fmt(summary[key]) if key != "n" else str(int(summary[key])))
                for key, label in _DESCRIBE_ROWS
            ]
            stats.insert(1, ("Total samples", str(int(ya.size))))
            return _Result(
                x=xa,
                y=ya,
                x_name=source.x_name,
                y_name=source.y_name,
                suffix="",
                plot_label=source.label,
                overlay=None,
                allow_apply=False,
                allow_replace=False,
                stats=stats,
            )

        if kind == "dynamics":
            _, mode, delay, dim = params
            xs, ys = _sorted_by_x(x_raw, y_raw)

            if mode == "phase":
                emb = dynamics.delay_embedding(ys, delay, dim=2)
                a, b = emb[:, 0], emb[:, 1]
                return _Result(
                    x=a,
                    y=b,
                    x_name=f"{y_name}(t)",
                    y_name=f"{y_name}(t+{delay})",
                    suffix="phase",
                    plot_label=f"delay embedding (delay {delay})",
                    overlay=None,
                    x_changed=True,
                    # A reconstructed phase portrait lives in delay space, not over x.
                    allow_replace=False,
                    stats=[
                        ("Embedded points", str(int(a.size))),
                        ("Delay (samples)", str(int(delay))),
                    ],
                )

            if mode == "return_map":
                m_n, m_next = dynamics.return_map(ys)
                stats = [("Maxima found", str(int(m_n.size) + (1 if m_n.size else 0)))]
                diag_x = diag_y = np.empty(0, dtype=np.float64)
                if m_n.size:
                    both = np.concatenate([m_n, m_next])
                    lo, hi = float(np.min(both)), float(np.max(both))
                    # The y = x reference line: on a return map, points above it are growing
                    # maxima and points below are shrinking ones.
                    diag_x = np.array([lo, hi], dtype=np.float64)
                    diag_y = diag_x.copy()
                    stats.append(("Map points", str(int(m_n.size))))
                return _Result(
                    # x/y are the map points (what New layer/dataset commits); the preview
                    # draws the y = x reference with the points marked on it.
                    x=m_n,
                    y=m_next,
                    x_name=f"{y_name} max (n)",
                    y_name=f"{y_name} max (n+1)",
                    suffix="return map",
                    plot_label="y = x",
                    base_x=diag_x,
                    base_y=diag_y,
                    markers=(m_n, m_next) if m_n.size else None,
                    table=[("max_n", m_n), ("max_n_plus_1", m_next)] if m_n.size else None,
                    x_changed=True,
                    allow_replace=False,
                    allow_apply=bool(m_n.size),
                    stats=stats,
                )

            if mode == "lyapunov":
                dx = float(np.median(np.diff(xs))) if xs.size > 1 else 1.0
                if not math.isfinite(dx) or dx <= 0.0:
                    dx = 1.0
                res = dynamics.lyapunov_rosenstein(ys, dt=dx, delay=delay, dim=dim)
                times = np.asarray(res["times"], dtype=np.float64)
                divergence = np.asarray(res["divergence"], dtype=np.float64)
                fit = np.asarray(res["fit"], dtype=np.float64)
                lam = float(res["lyapunov"])
                verdict = (
                    "chaotic (sensitive)"
                    if lam > 0.01
                    else "regular (cycle / fixed point)" if math.isfinite(lam) else "undetermined"
                )
                stats = [
                    ("Largest Lyapunov exponent", _fmt(lam)),
                    ("Verdict", verdict),
                    ("Embedding dim", str(int(dim))),
                    ("Delay (samples)", str(int(delay))),
                    ("Embedded points", str(int(res["n_points"]))),
                ]
                if int(res["stride"]) > 1:
                    stats.append(("Decimation stride", str(int(res["stride"]))))
                return _Result(
                    x=times,
                    y=divergence,
                    x_name="divergence time",
                    y_name="mean log divergence",
                    suffix="lyapunov divergence",
                    plot_label="divergence curve",
                    # The fitted line whose slope is the exponent, over the linear region.
                    overlay=fit,
                    overlay_label="linear fit",
                    x_changed=True,
                    allow_replace=False,
                    stats=stats,
                )

            raise ValueError(f"unknown dynamics analysis {mode!r}")

        raise ValueError(f"unknown operation {kind!r}")

    # -- preview -------------------------------------------------------------------

    def _draw_preview(
        self, source: _Source, result: Optional[_Result], error: Optional[str]
    ) -> None:
        """The live overlay: source and result on one set of axes ("sobreponer")."""
        if error:
            widgets.error_box(error)
        if result is None:
            widgets.mini_plot(
                "preview",
                source.y_raw,
                x=source.x_raw,
                height=_PREVIEW_HEIGHT,
                label=source.label,
                color=theme.get_color("muted"),
            )
            return

        if result.allow_apply:
            # Nothing on screen said the curve below is uncommitted. ASCII only: the
            # default font is ProggyClean and stops at U+00FF, so no em dash (CONTRACT 3).
            imgui.text_disabled("Preview -- not applied")
            imgui.same_line()
            widgets.help_marker(
                "This curve is computed but not committed. Nothing is added to the scene "
                "or to your data until you press Apply at the bottom of the panel."
            )

        # base_x/base_y override the drawn line (Peaks shows the signal, not its result).
        line_y = result.base_y if result.base_y is not None else result.y
        line_x = result.base_x if result.base_x is not None else result.x
        widgets.mini_plot(
            "preview",
            line_y,
            x=line_x,
            overlay=result.overlay,
            markers=result.markers,
            marker_color=theme.get_color("warn"),
            height=_PREVIEW_HEIGHT,
            label=result.plot_label,
            color=theme.get_color("accent"),
            overlay_color=theme.get_color("warn"),
        )
        self._draw_legend(result)

    def _draw_legend(self, result: _Result) -> None:
        """A colour key for the preview series. mini_plot draws the overlay/markers warn."""
        imgui.text_colored(theme.get_color("accent"), "--")
        imgui.same_line()
        imgui.text(result.plot_label)
        if result.overlay is not None:
            imgui.same_line()
            imgui.text_colored(theme.get_color("warn"), "--")
            imgui.same_line()
            imgui.text(result.overlay_label)
        if result.markers is not None:
            imgui.same_line()
            imgui.text_colored(theme.get_color("warn"), "o")
            imgui.same_line()
            imgui.text(f"peaks ({int(np.asarray(result.markers[0]).size)})")

    # -- report --------------------------------------------------------------------

    def _report_text(self, source: _Source, result: _Result) -> str:
        """The operation's result as tab-separated text, ready to paste into a spreadsheet.

        A header line names the operation and its source; every stat row follows as
        ``label<TAB>value``. This is the escape hatch the panel was missing: a fit gives
        ``mu = 2.00 +/- 0.03`` on screen, and this is how that number leaves the tool.
        """
        lines = [f"# {result.plot_label}\t(source: {source.label})"]
        lines.extend(f"{label}\t{value}" for label, value in result.stats)
        return "\n".join(lines)

    def _draw_report_header(self, source: _Source, result: _Result, key: str) -> None:
        """A "Copy report" button above the stat rows, with a transient confirmation."""
        if imgui.small_button("Copy report"):
            imgui.set_clipboard_text(self._report_text(source, result))
            # ~1.5 s of confirmation at 60 fps, scoped to this tab.
            self._copied_frames = 90
            self._copied_key = key
        imgui.same_line()
        widgets.help_marker(
            "Copy this operation's statistics as tab-separated text (label and value per "
            "row), ready to paste into a spreadsheet or your notes."
        )
        if self._copied_key == key and self._copied_frames > 0:
            self._copied_frames -= 1
            imgui.same_line()
            imgui.text_disabled("Copied")

    # -- apply ---------------------------------------------------------------------

    def _replace_reason(self, source: _Source, result: _Result) -> Optional[str]:
        """None if "Replace source" is possible, else a human explanation of why not."""
        if not result.allow_replace:
            return "This operation's output is not a replacement for its own input."
        if source.dataset is not None:
            rows = source.dataset.n_rows()
            if int(result.y.size) != rows:
                return (
                    f"Replacing in place needs the source's row count: this result has "
                    f"{result.y.size} rows, the dataset has {rows}. Use New dataset or "
                    f"New layer instead."
                )
            return None
        layer_type = getattr(source.layer, "layer_type", "")
        if layer_type not in _REPLACEABLE_LAYER_TYPES:
            return f"A {layer_type!r} layer has no in-place x/y update path."
        return None

    def _apply_mode_for(self, key: str) -> str:
        """This tab's apply mode. Defaults to "New layer", the non-destructive choice."""
        return self._apply_modes.get(key, "new_layer")

    def _apply_name_for(self, key: str) -> str:
        """This tab's user-typed output name ("" means "use the default")."""
        return self._apply_names.get(key, "")

    def _draw_apply(self, source: _Source, result: _Result, key: str) -> None:
        """The apply-mode radios, the output name, and the Apply button.

        All state here is per-tab (``key``): the mode and the name belong to the
        operation, not to the panel.
        """
        reason = self._replace_reason(source, result)
        mode = self._apply_mode_for(key)

        imgui.text("Apply as")
        imgui.same_line()
        for index, (candidate, label) in enumerate(_APPLY_MODES):
            if index:
                imgui.same_line()
            if imgui.radio_button(label, mode == candidate):
                self._apply_modes[key] = candidate
                mode = candidate
            if candidate == "replace" and reason is not None:
                imgui.same_line()
                widgets.help_marker(f"Unavailable: {reason}")
            if candidate == "add_column" and source.dataset is None:
                imgui.same_line()
                widgets.help_marker("Unavailable: a plotted layer has no table to add a column to.")

        imgui.push_item_width(-140.0)
        try:
            if mode == "new_layer":
                kind_changed, self._layer_kind = widgets.enum_combo(
                    "Kind", self._layer_kind, _LAYER_KINDS
                )
                if kind_changed:
                    self._layer_opts = layerops.default_kind_options(self._layer_kind)
                widgets.kind_options_editor(self._layer_opts)
            if mode in ("new_layer", "new_dataset", "add_column"):
                caption = "Column name" if mode == "add_column" else "Name"
                _c, name = imgui.input_text(caption, self._apply_name_for(key))
                self._apply_names[key] = name
                imgui.same_line()
                widgets.help_marker(
                    f"Leave empty for the default: {self._default_name(source, result)!r}"
                )
        finally:
            imgui.pop_item_width()

        if mode == "replace" and reason is not None:
            imgui.text_disabled("Apply")
            imgui.same_line()
            widgets.help_marker(reason)
            return
        if mode == "add_column" and source.dataset is None:
            imgui.text_disabled("Apply")
            imgui.same_line()
            widgets.help_marker("Switch the source to a dataset to add a column.")
            return

        theme.push_accent()
        clicked = imgui.button("Apply", (130.0, 0.0))
        theme.pop_accent()
        imgui.same_line()
        widgets.help_marker(
            "Queued for the next frame (a draw callback may not touch the scene) and "
            "recorded on the undo stack."
        )
        if clicked:
            self._submit_apply(source, result, key)

    def _default_name(self, source: _Source, result: _Result) -> str:
        """The auto-generated name for a new layer or dataset."""
        return f"{source.label} {result.suffix}".strip()

    def _output_name(self, source: _Source, result: _Result, key: str) -> str:
        """The name typed on *this* tab if there is one, else the default."""
        return self._apply_name_for(key).strip() or self._default_name(source, result)

    def _push(self, cmd: Command) -> None:
        """Record and run ``cmd``. Runs inside the drain, never from a draw callback.

        Deliberately not ``Panel.push_command``: that discards ``UndoStack.push``'s return
        value, and a caller is required to surface "operation not undoable" when it comes
        back False. Everything else about the two is identical.
        """
        if not self.undo.push(cmd):
            self._notice = (
                f"{cmd.label}: applied, but the data is too large to snapshot for undo. "
                f"The undo history was cleared."
            )

    def _submit_apply(self, source: _Source, result: _Result, key: str) -> None:
        """Build the Command for this tab's apply mode and queue it."""
        mode = self._apply_mode_for(key)
        if mode == "new_layer":
            cmd = self._command_new_layer(source, result, key)
        elif mode == "new_dataset":
            cmd = self._command_new_dataset(source, result, key)
        elif mode == "add_column":
            cmd = self._command_add_column(source, result, key)
        elif source.dataset is not None:
            cmd = self._command_replace_dataset(source, result)
        else:
            cmd = self._command_replace_layer(source, result)
        if cmd is None:
            return
        self.submit(lambda: self._push(cmd))

    def _command_new_layer(self, source: _Source, result: _Result, key: str) -> Command:
        """Add the result as a fresh layer; undo removes it with the full teardown."""
        plot = self.plot
        hud = self.hud
        # Detach from the live columns now: the source may be edited between this frame
        # and the drain, and the command must apply what the preview showed.
        x = np.array(result.x, dtype=np.float64)
        y = np.array(result.y, dtype=np.float64)
        kind = self._layer_kind
        # Copied for the same reason as x/y: the command drains later, and the options
        # it applies must be the ones the preview was showing.
        options = dict(self._layer_opts)
        label = self._output_name(source, result, key)
        holder: Dict[str, Any] = {}

        def do() -> None:
            holder["layer"] = layerops.add_xy_layer(
                plot, x, y, kind=kind, label=label, options=options
            )

        def undo() -> None:
            layer = holder.pop("layer", None)
            if layer is not None:
                layerops.remove_layer(plot, hud, layer)

        return Command(label=f"Plot {label!r}", do=do, undo=undo)

    def _command_new_dataset(self, source: _Source, result: _Result, key: str) -> Command:
        """Register the result as a new dataset; undo unregisters it.

        A result carrying a ``table`` (the Peaks tab does) becomes a multi-column dataset
        of that table; everything else is the two-column ``(x, y)`` the operation produced.
        """
        store = self.store
        name = self._output_name(source, result, key)
        if result.table is not None:
            columns = [
                Column(col_name, np.array(values, dtype=np.float64))
                for col_name, values in result.table
            ]
        else:
            x_name = result.x_name
            y_name = result.y_name if result.y_name != x_name else f"{result.y_name} (2)"
            columns = [
                Column(x_name, np.array(result.x, dtype=np.float64)),
                Column(y_name, np.array(result.y, dtype=np.float64)),
            ]
        dataset = DataSet(name, columns)
        dataset.source = "derived"

        def do() -> None:
            store.add(dataset)

        def undo() -> None:
            store.remove(dataset)

        return Command(label=f"Create dataset {name!r}", do=do, undo=undo)

    def _command_add_column(self, source: _Source, result: _Result, key: str) -> Optional[Command]:
        """Append the computed variable as a NEW column of the source dataset.

        This is what turns a derivative / integral / smoothed signal into a column the user can
        then map to colour or size, closing the loop with the encodings picker. The dataset
        keeps every column the same length, so the result is padded with NaN when it is shorter
        (it usually matches: derivative, integral and smooth are length-preserving) and
        truncated when longer (resample, peaks). Undo drops the column again.
        """
        dataset = source.dataset
        if dataset is None:
            self._notice = "Add column needs a dataset source (a plotted layer has no table)."
            return None

        name = self._output_name(source, result, key)
        values = np.asarray(result.y, dtype=np.float64)
        n = dataset.n_rows()
        if n and values.size != n:
            fitted = np.full(n, np.nan, dtype=np.float64)
            fitted[: min(values.size, n)] = values[: min(values.size, n)]
            values = fitted

        added: Dict[str, Any] = {}

        def do() -> None:
            column = dataset.add_column(name, values)
            added["name"] = column.name

        def undo() -> None:
            col_name = added.pop("name", None)
            if col_name is not None:
                dataset.remove_column(col_name)

        return Command(label=f"Add column {name!r} to {dataset.name!r}", do=do, undo=undo)

    def _command_replace_dataset(self, source: _Source, result: _Result) -> Optional[Command]:
        """Overwrite the source columns in place, honouring any live layer link."""
        dataset = source.dataset
        if dataset is None:
            return None

        writes: List[Tuple[str, np.ndarray]] = []
        if result.x_changed and source.x_col != source.y_col:
            writes.append((source.x_col, np.array(result.x, dtype=np.float64)))
        writes.append((source.y_col, np.array(result.y, dtype=np.float64)))

        targets: List[Tuple[Column, Optional[np.ndarray], np.ndarray]] = []
        for col_name, values in writes:
            column = next((c for c in dataset.columns if c.name == col_name), None)
            if column is None:
                self._notice = f"Column {col_name!r} is gone from {dataset.name!r}."
                return None
            if column.values.size != values.size:
                # Re-checked here, not just at the radio: the dataset may have been
                # edited between the frame that enabled Replace and this click.
                self._notice = (
                    f"Column {col_name!r} now has {column.values.size} rows but the result "
                    f"has {values.size}. Nothing was changed."
                )
                return None
            targets.append((column, snapshot(column.values), values))

        label = f"Replace {dataset.name}.{source.y_col}"
        sync = self._make_sync(dataset)

        def do() -> None:
            for column, _old, new in targets:
                column.values[...] = new
                column.invalidate()
            sync()

        if any(old is None for _column, old, _new in targets):
            # snapshot() refused: the column is too large to copy (history.MAX_SNAPSHOT_
            # ELEMENTS) and there is no inverse for an overwrite.
            return Command.not_undoable(label, do)

        def undo() -> None:
            for column, old, _new in targets:
                column.values[...] = old
                column.invalidate()
            sync()

        return Command(label=label, do=do, undo=undo)

    def _make_sync(self, dataset: DataSet) -> Callable[[], None]:
        """Return a callback that pushes ``dataset`` back onto its layer, if it has one.

        This is what keeps the "edicion en vivo" link alive: a dataset built from a layer
        writes back to it, and ``write_back`` sets ``gpu_dirty`` per CONTRACT 1.4.
        Queued commands only.
        """
        plot = self.plot

        def sync() -> None:
            if not dataset.is_bound() or dataset.layer_id is None:
                return
            layer = layerops.find_layer(plot, dataset.layer_id)
            if layer is None:
                return
            dataset.write_back(layer)
            layerops.mark_scene_dirty(plot)

        return sync

    def _command_replace_layer(self, source: _Source, result: _Result) -> Optional[Command]:
        """Swap the layer's geometry for the result; undo restores the old arrays.

        The snapshot is taken from the geometry's own columns 0 and 1, not from the
        user's x/y picks -- those may be swapped, and undo must restore what was there.
        """
        plot = self.plot
        layer = source.layer
        if layer is None:
            return None
        geometry = getattr(layer, source.layer_attr, None)
        if geometry is None:
            self._notice = "The source layer no longer has geometry to replace."
            return None
        geo = np.asarray(geometry)

        x = np.array(result.x, dtype=np.float64)
        y = np.array(result.y, dtype=np.float64)
        old_a = snapshot(np.asarray(geo[:, 0], dtype=np.float64))
        old_b = snapshot(np.asarray(geo[:, 1], dtype=np.float64))
        old_colors = snapshot(getattr(layer, "colors", None))
        had_colors = getattr(layer, "colors", None) is not None
        label = f"Replace layer {getattr(layer, 'label', '?')!r}"

        def do() -> None:
            layerops.update_layer_xy(plot, layer, x, y)

        if old_a is None or old_b is None or (had_colors and old_colors is None):
            return Command.not_undoable(label, do)

        def undo() -> None:
            layerops.update_layer_xy(plot, layer, old_a, old_b)
            if old_colors is not None:
                # update_layer_xy refits colours to the new length; on the way back that
                # refit cannot recover per-vertex colours a shrink discarded, so restore
                # the originals verbatim.
                layer.colors = np.ascontiguousarray(old_colors)
                layer.dirty.gpu_dirty = True
                layerops.mark_scene_dirty(plot)

        return Command(label=label, do=do, undo=undo)


# ----------------------------------------------------------------------------------
# nan-safe reductions used by the stat rows
# ----------------------------------------------------------------------------------


def _nan_reduce(values: np.ndarray, fn: Callable[[np.ndarray], Any]) -> float:
    """Apply a nan-ignoring reduction, returning nan for an all-nan or empty input."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0 or not bool(np.any(np.isfinite(arr))):
        return float("nan")
    with np.errstate(all="ignore"):
        return float(fn(arr))


def _nan_min(values: np.ndarray) -> float:
    """Smallest finite value, or nan."""
    return _nan_reduce(values, np.nanmin)


def _nan_max(values: np.ndarray) -> float:
    """Largest finite value, or nan."""
    return _nan_reduce(values, np.nanmax)


def _nan_peak(values: np.ndarray) -> float:
    """Largest finite absolute value, or nan."""
    return _nan_reduce(values, lambda arr: np.nanmax(np.abs(arr)))
