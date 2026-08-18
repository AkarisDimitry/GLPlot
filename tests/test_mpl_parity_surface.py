"""Sweep the whole ``matplotlib.pyplot`` surface and hold GLPlot to its compat contract.

``test_mpl_compat.py`` pins behaviour one hand-written case at a time, which is the right
shape for "does ``axis('equal')`` produce a round circle". It cannot answer the question
this file exists for: *is there any matplotlib keyword left that GLPlot rejects?* That
question is about the whole surface at once, and a hand-written list of it goes stale the
moment somebody adds a parameter -- the gap it is supposed to catch is exactly the gap
nobody thought to write a test for.

So these tests are generated. They read ``matplotlib.pyplot``'s own signatures at run time
and drive every parameter through GLPlot, which means a new matplotlib release grows the
test suite by itself, and the two rules of the compat contract get enforced by
construction rather than by vigilance:

1. **A matplotlib keyword must not raise.** ``TypeError: unexpected keyword argument`` is
   the failure this surface exists to remove -- the plot may differ, but the script runs.
2. **A keyword that does nothing must say so.** Accepting an argument and silently
   dropping it is worse than rejecting it: the caller reads it back in their own source
   and believes it took effect.

Rule 2 cannot be checked generically -- nothing in a signature says whether a keyword was
honoured -- so the no-ops are enumerated in `DOCUMENTED_NOOPS`. That list is a two-way
ratchet: drop a warning and the test fails, *implement* one of them for real and the test
also fails, which is the reminder to delete the warning along with the limitation.

No OpenGL and no GPU: ``GPULinePlot`` constructs without a window (CONTRACT 5.1) and
nothing here calls ``show()``.
"""

from __future__ import annotations

import inspect
import warnings

import numpy as np
import pytest

import glplot.pyplot as gplt

X = np.linspace(0.5, 10.0, 30)
Y = np.sin(X)
M = np.abs(np.random.default_rng(0).random((12, 12))) + 0.1
GX, GY = np.meshgrid(np.linspace(0.0, 1.0, 12), np.linspace(0.0, 1.0, 12))


@pytest.fixture(autouse=True)
def clean_state():
    gplt._cleanup_pyplot_state()
    gplt.figure(width=400, height=300)
    yield
    gplt._cleanup_pyplot_state()


#: One representative call per function: ``name -> (positional args, base keywords)``.
#:
#: Only the arguments needed to make the call *valid*; the generated tests add one
#: matplotlib keyword at a time on top. Functions absent from this table are not swept --
#: they are listed in `NOT_SWEPT` with the reason, so the coverage gap is visible rather
#: than implied by silence.
CALLS: dict = {
    "annotate": (("hi",), {"xy": (1.0, 1.0), "xytext": (2.0, 2.0)}),
    "acorr": ((X,), {}),
    "angle_spectrum": ((X,), {}),
    "arrow": ((0.0, 0.0, 1.0, 1.0), {}),
    "axhline": ((1.0,), {}),
    "axhspan": ((0.0, 1.0), {}),
    "axis": ((), {}),
    "axline": (((0.0, 0.0),), {"slope": 1.0}),
    "axvline": ((1.0,), {}),
    "axvspan": ((0.0, 1.0), {}),
    "bar": ((X, Y), {}),
    "barbs": ((GX, GY, GX, GY), {}),
    "barh": ((X, Y), {}),
    "boxplot": ((Y,), {}),
    "broken_barh": (([(0.0, 1.0)], (0.0, 1.0)), {}),
    "clabel": ((), {}),
    "clim": ((), {}),
    "cohere": ((X, Y), {}),
    "colorbar": ((), {}),
    "contour": ((M,), {}),
    "contourf": ((M,), {}),
    "csd": ((X, Y), {}),
    "ecdf": ((Y,), {}),
    "errorbar": ((X, Y), {"yerr": 0.1}),
    "eventplot": ((X,), {}),
    "figimage": ((M,), {}),
    "figtext": ((0.5, 0.5, "t"), {}),
    "fill": ((X, Y), {}),
    "fill_between": ((X, Y), {}),
    "fill_betweenx": ((X, Y), {}),
    "grid": ((True,), {}),
    "hexbin": ((X, Y), {}),
    "hist": ((Y,), {}),
    "hist2d": ((X, Y), {}),
    "hlines": ((1.0, 0.0, 2.0), {}),
    "imshow": ((M,), {}),
    "legend": ((), {}),
    "locator_params": ((), {}),
    "loglog": ((X, Y), {}),
    "magnitude_spectrum": ((X,), {}),
    "margins": ((0.1,), {}),
    "matshow": ((M,), {}),
    "minorticks_off": ((), {}),
    "minorticks_on": ((), {}),
    "pcolor": ((M,), {}),
    "pcolormesh": ((M,), {}),
    "phase_spectrum": ((X,), {}),
    "pie": (([1.0, 2.0, 3.0],), {}),
    "plot": ((X, Y), {}),
    "psd": ((X,), {}),
    "quiver": ((GX, GY, GX, GY), {}),
    "scatter": ((X, Y), {}),
    "semilogx": ((X, Y), {}),
    "semilogy": ((X, Y), {}),
    "specgram": ((np.tile(Y, 20),), {}),
    "spy": ((M,), {}),
    "stackplot": ((X, Y), {}),
    "stairs": ((Y[:-1], X), {}),
    "stem": ((X, Y), {}),
    "step": ((X, Y), {}),
    "streamplot": ((np.arange(12.0), np.arange(12.0), np.ones((12, 12)), np.ones((12, 12))), {}),
    "subplots_adjust": ((), {}),
    "text": ((1.0, 1.0, "t"), {}),
    "tick_params": ((), {}),
    "ticklabel_format": ((), {}),
    "tight_layout": ((), {}),
    "title": (("t",), {}),
    "tricontour": ((X, Y, Y), {}),
    "tricontourf": ((X, Y, Y), {}),
    "tripcolor": ((X, Y, Y), {}),
    "triplot": ((X, Y), {}),
    "violinplot": ((Y,), {}),
    "vlines": ((1.0, 0.0, 2.0), {}),
    "xcorr": ((X, Y), {}),
    "xlabel": (("t",), {}),
    "xlim": ((), {}),
    "xscale": (("linear",), {}),
    "xticks": ((), {}),
    "ylabel": (("t",), {}),
    "ylim": ((), {}),
    "yscale": (("linear",), {}),
    "yticks": ((), {}),
}

#: Functions deliberately left out of the sweep, and why. Each would need a live window,
#: an event loop, a filesystem target, or would tear down the state the other cases share.
NOT_SWEPT: dict = {
    "show": "blocks on an event loop",
    "pause": "runs an event loop",
    "ginput": "waits for clicks",
    "waitforbuttonpress": "waits for input",
    "savefig": "writes a file",
    "imsave": "writes a file",
    "imread": "reads a file",
    "close": "tears down the figure the other cases share",
    "clf": "tears down the figure the other cases share",
    "cla": "tears down the figure the other cases share",
    "figure": "creates figures; swept implicitly by every other case",
    "subplots": "creates figures; covered by TestSubplotsGrid in test_mpl_compat",
    "subplot": "changes the active panel under the other cases",
    "subplot2grid": "changes the active panel under the other cases",
    "subplot_mosaic": "changes the active panel under the other cases",
    "axes": "changes the active panel under the other cases",
    "sca": "needs an existing axes object",
    "delaxes": "removes the axes the other cases share",
    "twinx": "creates a second axes",
    "twiny": "creates a second axes",
    "switch_backend": "global backend state",
    "rc_context": "a context manager, not a plotting call",
    "setp": "needs an artist instance",
    "getp": "needs an artist instance",
    "get": "needs an artist instance",
    "sci": "needs a mappable instance",
    "bar_label": "needs a bar container; covered by hand in test_mpl_compat",
    "quiverkey": "needs a quiver instance",
    "table": "whole call is a documented no-op",
    "connect": "needs a live canvas",
    "disconnect": "needs a live canvas",
    "install_repl_displayhook": "global interpreter state",
    "uninstall_repl_displayhook": "global interpreter state",
    "set_loglevel": "global logging state",
    "xkcd": "a context manager over global rcParams",
    "rc": "writes global rcParams",
    "rcdefaults": "writes global rcParams",
    "register_cmap": "writes the global colormap registry",
    "set_cmap": "writes global state; covered by hand",
    "polar": "creates a polar axes",
    "plot_date": "date-axis wrapper over plot",
}

#: A plausible value for each matplotlib parameter name, used to drive the sweep.
#:
#: Values are chosen to be *accepted* rather than meaningful -- rule 1 is about the
#: keyword being recognised, not about what it draws. A parameter with no entry here is
#: skipped and reported by `test_every_parameter_is_covered_or_skipped`, so the sweep
#: cannot quietly shrink.
VALUES: dict = {
    "alpha": 0.5,
    "animated": False,
    "angle": 30.0,
    "angles": [0.0],
    "antialiased": True,
    "align": "center",
    "annotation_clip": True,
    "arrowprops": None,
    "aspect": "auto",
    "autopct": "%d",
    "autorange": False,
    "autoscale_on": True,
    "axis": "both",
    "barsabove": True,
    "baseline": 0.0,
    "basefmt": "k-",
    "bins": 5,
    "bootstrap": None,
    "bottom": 0.0,
    "boxprops": None,
    "c": "C1",
    "capprops": None,
    "capsize": 2.0,
    "capstyle": "butt",
    "capthick": 2.0,
    "capwidths": 0.2,
    "center": (0.0, 0.0),
    "clip_on": True,
    "cmap": "viridis",
    "color": "C2",
    "colorizer": None,
    "colors": "C3",
    "complementary": False,
    "compress": False,
    "conf_intervals": None,
    "container": None,
    "corner_mask": True,
    "counterclock": True,
    "cumulative": False,
    "density": False,
    "detrend": None,
    "dpi": 100,
    "ec": "k",
    "ecolor": "k",
    "edgecolor": "k",
    "edgecolors": "k",
    "elinestyle": "--",
    "elinewidth": 1.0,
    "emit": True,
    "errorevery": 2,
    "explode": None,
    "extend": "neither",
    "extent": None,
    "facecolor": "C4",
    "facecolors": None,
    "fc": "C4",
    "fignum": False,
    "filled": True,
    "filternorm": True,
    "filterrad": 4.0,
    "fmt": "%g",
    "font": None,
    "fontdict": None,
    "fontsize": 9,
    "frame": False,
    "frameon": True,
    "gridsize": 10,
    "hatch": None,
    "head_length": 0.1,
    "head_width": 0.1,
    "height": 1.0,
    "histtype": "bar",
    "hold": None,
    "hspace": 0.2,
    "include_self": False,
    "interpolate": False,
    "integration_max_error_scale": 1.5,
    "integration_max_step_scale": 1.5,
    "interpolation": None,
    "interpolation_stage": None,
    "joinstyle": "miter",
    "label": "L",
    "labels": None,
    "labeldistance": 1.1,
    "label_type": "edge",
    "labelbottom": True,
    "labelleft": True,
    "left": 0.0,
    "length_includes_head": False,
    "levels": 5,
    "linefmt": "C0-",
    "linelengths": 1.0,
    "lineoffsets": 1.0,
    "linestyle": "-",
    "linestyles": "-",
    "linecolor": "C4",
    "linewidth": 1.0,
    "linewidths": 1.0,
    "log": False,
    "lolims": False,
    "loc": "best",
    "ls": "-",
    "lw": 1.0,
    "manage_ticks": True,
    "marginals": False,
    "marker": "o",
    "markerfmt": "C0o",
    "markersize": 5.0,
    "mask": None,
    "maxlags": 5,
    "meanline": False,
    "meanprops": None,
    "medianprops": None,
    "mincnt": None,
    "minlength": 0.1,
    "minor": False,
    "mode": None,
    "nbins": 5,
    "nonpositive": "mask",
    "norm": None,
    "normalize": True,
    "normed": True,
    "notch": False,
    "noverlap": None,
    "NFFT": None,
    "num": None,
    "num_arrows": 3,
    "orientation": "vertical",
    "origin": "upper",
    "pad_to": None,
    "padding": 0.0,
    "patch_artist": False,
    "pctdistance": 0.6,
    "pivot": "tail",
    "plotnonfinite": False,
    "points": 50,
    "positions": None,
    "precision": 0.0,
    "quantiles": None,
    "radius": 1.0,
    "rasterized": False,
    "reduce_C_function": None,
    "resample": True,
    "resize": False,
    "return_line": None,
    "right": None,
    "rotatelabels": False,
    "rwidth": None,
    "scale": None,
    "scale_by_freq": None,
    "scale_units": None,
    "scalex": True,
    "scaley": True,
    "scilimits": None,
    "shading": None,
    "shadow": False,
    "showbox": True,
    "side": "low",
    "showcaps": True,
    "showextrema": True,
    "showfliers": True,
    "showmeans": False,
    "showmedians": False,
    "sides": None,
    "sizes": None,
    "snap": None,
    "sparsify": None,
    "stacked": False,
    "startangle": 0.0,
    "start_points": None,
    "step": None,
    "style": "",
    "sym": None,
    "textcoords": None,
    "textprops": None,
    "tick_label": None,
    "tick_labels": None,
    "tight": None,
    "top": None,
    "transform": None,
    "url": None,
    "usermedians": None,
    "usevlines": True,
    "useLocale": None,
    "useMathText": None,
    "useOffset": None,
    "vert": True,
    "visible": True,
    "vmax": None,
    "vmin": None,
    "weights": None,
    "where": None,
    "whis": 1.5,
    "whiskerprops": None,
    "widths": None,
    "width": 0.5,
    "window": None,
    "wedgeprops": None,
    "wspace": 0.2,
    "x": 1.0,
    "xdate": True,
    "xerr": 0.1,
    "xextent": None,
    "xlolims": False,
    "xuplims": False,
    "xy": (1.0, 1.0),
    "xycoords": None,
    "xytext": None,
    "y": 1.0,
    "yerr": 0.1,
    "ydate": False,
    "zdir": "z",
    "zorder": 1.0,
    "zs": None,
    "tz": None,
    "Fc": None,
    "Fs": None,
    "data": None,
    "fname": None,
    "b": None,
    "which": "major",
    "va": "center",
    "ha": "center",
    "rotation": 0.0,
    "clip_path": None,
    "in_layout": True,
    "path_effects": None,
    "picker": None,
    "gid": None,
    "sketch_params": None,
    "transform_rotates_text": False,
    "wrap": False,
    "multialignment": "center",
    "verticalalignment": "center",
    "horizontalalignment": "center",
    "backgroundcolor": "w",
    # Second half of the two-ended arguments, whose first half the case passes positionally.
    "xmin": 0.0,
    "xmax": 1.0,
    "ymin": 0.0,
    "ymax": 1.0,
    "xy2": (2.0, 2.0),
    "x2": 0.0,
    "y2": 0.0,
    "xo": 0,
    "yo": 0,
    "range": None,
    "C": None,
    "s": 8.0,
    "arg": "auto",
    "CS": None,
    "mappable": None,
    "cax": None,
    "ax": None,
    "flierprops": None,
    "uplims": False,
    "fill": False,
    "xscale": "linear",
    "yscale": "linear",
    "cmin": None,
    "cmax": None,
    "bw_method": None,
    "arrowsize": 1.0,
    "arrowstyle": "-|>",
    "maxlength": 4.0,
    "integration_direction": "both",
    "broken_streamlines": True,
    "pad": 1.0,
    "h_pad": 1.0,
    "w_pad": 1.0,
    "rect": (0.0, 0.0, 1.0, 1.0),
    "labelpad": 4.0,
    "ticks": None,
}


def _mpl_parameters(func) -> list:
    """The named parameters ``matplotlib.pyplot.<func>`` accepts, in declaration order."""
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return []
    return [
        name
        for name, parameter in signature.parameters.items()
        if parameter.kind not in (parameter.VAR_KEYWORD, parameter.VAR_POSITIONAL)
    ]


def _sweep_cases() -> list:
    """``(function, keyword)`` for every matplotlib keyword the table can drive."""
    import matplotlib.pyplot as mpl

    cases = []
    for name, (args, base) in sorted(CALLS.items()):
        mpl_func = getattr(mpl, name, None)
        if mpl_func is None:
            continue
        parameters = _mpl_parameters(mpl_func)
        # The first len(args) parameters are already supplied positionally by the case;
        # naming one again is "got multiple values", which says nothing about parity.
        taken = set(parameters[: len(args)]) | set(base)
        for keyword in parameters:
            if keyword in taken or keyword not in VALUES:
                continue
            cases.append((name, keyword))
    return cases


def _uncovered_parameters() -> dict:
    """matplotlib parameters the sweep skips because `VALUES` has no entry for them."""
    import matplotlib.pyplot as mpl

    gaps = {}
    for name, (args, base) in sorted(CALLS.items()):
        mpl_func = getattr(mpl, name, None)
        if mpl_func is None:
            continue
        parameters = _mpl_parameters(mpl_func)
        taken = set(parameters[: len(args)]) | set(base)
        missing = [k for k in parameters if k not in taken and k not in VALUES]
        if missing:
            gaps[name] = missing
    return gaps


SWEEP = _sweep_cases()


class TestRuleOneNoKeywordRaises:
    """Every matplotlib keyword must be *accepted*. The plot may differ; the script runs.

    This is the test that would have caught all fifteen functions that used to raise
    ``TypeError`` on a documented matplotlib argument -- ``boxplot(patch_artist=True)``,
    ``pie(autopct=...)``, ``psd(Fc=0)``, ``margins(tight=True)`` and the rest. Each was a
    one-line signature omission, and each turned a working matplotlib script into a
    traceback at the point GLPlot was supposed to be a drop-in.
    """

    @pytest.mark.parametrize("func_name, keyword", SWEEP, ids=lambda v: str(v))
    def test_keyword_is_accepted(self, func_name, keyword):
        args, base = CALLS[func_name]
        func = getattr(gplt, func_name)
        with warnings.catch_warnings():
            # A no-op warning is a *pass* here -- rule 2 is a separate test.
            warnings.simplefilter("ignore")
            try:
                func(*args, **{**base, keyword: VALUES[keyword]})
            except TypeError as exc:
                if "unexpected keyword argument" in str(exc) or "got multiple values" in str(exc):
                    pytest.fail(f"{func_name}() rejects matplotlib's {keyword!r}: {exc}")
            except Exception:
                # Any other exception means the *value* did not suit this call -- the
                # keyword was still recognised, which is all rule 1 asks.
                pass

    def test_the_sweep_is_not_empty(self):
        """A refactor that broke `_sweep_cases` would make every test above vacuously pass."""
        assert len(SWEEP) > 300, f"the sweep collapsed to {len(SWEEP)} cases"

    def test_every_parameter_is_covered_or_reported(self):
        """No matplotlib parameter may be skipped for want of a value in `VALUES`.

        Without this the sweep could shrink to nothing one omission at a time and every
        test above would still be green -- the failure mode of any data-driven suite.
        """
        gaps = _uncovered_parameters()
        assert not gaps, "add a plausible value to VALUES for: " + "; ".join(
            f"{f}({', '.join(k)})" for f, k in sorted(gaps.items())
        )


class TestSignatureCompleteness:
    """No function with an explicit signature may be missing a matplotlib parameter.

    Complements rule 1 from the other side. A function with ``**kwargs`` swallows anything
    and passes the sweep by accident; one with a closed signature cannot, so its parameter
    list is compared against matplotlib's directly. This is the check that stays true even
    for the parameter combinations the sweep does not reach.
    """

    def _closed_signature_gaps(self) -> dict:
        import matplotlib.pyplot as mpl

        gaps = {}
        for name in dir(mpl):
            if name.startswith("_"):
                continue
            mpl_func = getattr(mpl, name)
            gl_func = getattr(gplt, name, None)
            if not inspect.isfunction(mpl_func) or not callable(gl_func):
                continue
            try:
                gl_signature = inspect.signature(gl_func)
            except (TypeError, ValueError):
                continue
            if any(p.kind is p.VAR_KEYWORD for p in gl_signature.parameters.values()):
                continue  # **kwargs absorbs everything; rule 1's sweep covers it.
            missing = set(_mpl_parameters(mpl_func)) - set(gl_signature.parameters)
            if missing:
                gaps[name] = sorted(missing)
        return gaps

    def test_no_closed_signature_drops_a_matplotlib_parameter(self):
        gaps = self._closed_signature_gaps()
        assert not gaps, (
            "these functions have no **kwargs, so a matplotlib caller gets a TypeError: "
            + "; ".join(f"{f}({', '.join(k)})" for f, k in sorted(gaps.items()))
        )


#: Keywords GLPlot accepts, cannot honour, and must therefore warn about.
#:
#: Every entry is a limitation of drawing through a GPU surface rather than matplotlib's
#: artist stack, and each is stated in the function's docstring too. The test below is a
#: ratchet in both directions: remove a warning and it fails; *implement* the keyword and
#: it fails as well, which is the prompt to delete the entry and the warning together.
DOCUMENTED_NOOPS: list = [
    ("annotate", {"xy": (1.0, 1.0), "xycoords": "axes fraction"}),
    ("annotate", {"xy": (1.0, 1.0), "xytext": (2.0, 2.0), "textcoords": "offset points"}),
    ("annotate", {"xy": (1.0, 1.0), "annotation_clip": True}),
    ("boxplot", {"zorder": 5.0}),
    ("grid", {"axis": "x"}),
    ("grid", {"which": "minor"}),
    ("hexbin", {"marginals": True}),
    ("hexbin", {"edgecolors": "k"}),
    ("imshow", {"interpolation": "bilinear"}),
    ("imshow", {"resample": True}),
    ("margins", {"tight": False}),
    ("pie", {"rotatelabels": True}),
    ("pie", {"hatch": "//"}),
    ("plot", {"scalex": False}),
    ("plot", {"scaley": False}),
    ("streamplot", {"zorder": 2.0}),
    ("ticklabel_format", {"useOffset": False}),
    ("ticklabel_format", {"useMathText": True}),
    ("ticklabel_format", {"useLocale": True}),
    ("tripcolor", {"shading": "gouraud"}),
]

#: The positional arguments each no-op case needs, keyed by function.
NOOP_ARGS: dict = {
    "annotate": ("hi",),
    "boxplot": (Y,),
    "grid": (True,),
    "hexbin": (X, Y),
    "imshow": (M,),
    "margins": (0.1,),
    "pie": ([1.0, 2.0],),
    "plot": (X, Y),
    "streamplot": (np.arange(12.0), np.arange(12.0), np.ones((12, 12)), np.ones((12, 12))),
    "ticklabel_format": (),
    "tripcolor": (X, Y, Y),
}


class TestRuleTwoNoOpsAnnounceThemselves:
    """A keyword GLPlot cannot honour must warn -- silence is the worse failure.

    Accepting ``imshow(interpolation='bilinear')`` and ignoring it produces a plot that
    *looks* finished and disagrees with the source that describes it. The reader has no
    way to find out. A `MatplotlibCompatWarning` costs one line of output and removes the
    entire class of "why doesn't my plot match" questions.
    """

    @pytest.mark.parametrize("func_name, kwargs", DOCUMENTED_NOOPS, ids=lambda v: str(v))
    def test_documented_noop_warns(self, func_name, kwargs):
        func = getattr(gplt, func_name)
        args = NOOP_ARGS[func_name]
        with pytest.warns(gplt.MatplotlibCompatWarning):
            func(*args, **kwargs)

    def test_a_warning_fires_once_not_once_per_call(self):
        """A no-op inside a loop must not bury the message it exists to send."""
        gplt._WARNED_UNSUPPORTED.clear()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for _ in range(20):
                gplt.grid(True, axis="x")
        compat = [w for w in caught if w.category is gplt.MatplotlibCompatWarning]
        assert len(compat) == 1, f"warned {len(compat)} times for 20 calls"


class TestReExportedObjects:
    """``plt.Rectangle``, ``plt.rcParams``, ``plt.LogNorm`` -- the non-function surface.

    A script that says ``ax.add_patch(plt.Rectangle(...))`` or ``plt.style.use('ggplot')``
    is not asking GLPlot to render anything new; it is asking for geometry, colour-scaling
    and settings objects matplotlib already defines. These are re-exported rather than
    reimplemented, so a `Normalize` built here *is* the one ``scatter(norm=...)`` accepts.
    """

    #: Names in matplotlib.pyplot that are stdlib or typing leakage, not API.
    LEAKED = {
        "cast",
        "overload",
        "Enum",
        "ExitStack",
        "AbstractContextManager",
        "TYPE_CHECKING",
        "functools",
        "importlib",
        "inspect",
        "logging",
        "re",
        "sys",
        "threading",
        "time",
        "Any",
        "Callable",
        "Hashable",
        "Iterable",
        "Sequence",
        "Literal",
        "IO",
    }

    def test_every_public_matplotlib_name_resolves_or_is_explained(self):
        """Nothing in ``dir(plt)`` may be missing without a stated reason."""
        import matplotlib.pyplot as mpl

        missing = []
        for name in dir(mpl):
            if name.startswith("_") or name in self.LEAKED:
                continue
            if not hasattr(gplt, name) and name not in gplt._NOT_EXPORTED:
                missing.append(name)
        assert not missing, f"absent from glplot.pyplot and unexplained: {sorted(missing)}"

    def test_deliberate_omissions_explain_themselves(self):
        """``plt.Slider`` must fail with a reason, not a bare attribute error."""
        for name, reason in gplt._NOT_EXPORTED.items():
            with pytest.raises(AttributeError, match="does not provide"):
                getattr(gplt, name)

    @pytest.mark.parametrize("name", sorted(gplt._LAZY_EXPORTS))
    def test_every_lazy_export_resolves(self, name):
        """The table names a module and an attribute; both must exist in this matplotlib.

        Written per-name rather than as one loop so a version bump that removes a single
        class points at that class instead of failing the whole table.
        """
        assert getattr(gplt, name) is not None

    def test_re_exports_are_matplotlibs_own_objects(self):
        """Not look-alikes: an isinstance check against matplotlib must pass."""
        import matplotlib.colors as mcolors
        import matplotlib.patches as mpatches

        assert gplt.Normalize is mcolors.Normalize
        assert isinstance(gplt.Rectangle((0, 0), 1, 1), mpatches.Patch)

    def test_rcparams_is_matplotlibs_own_mapping(self):
        """Shared, not mirrored -- otherwise rc(), rc_context() and style.use() would
        each write a different dict and disagree."""
        import matplotlib

        assert gplt.rcParams is matplotlib.rcParams


class TestRcParamsReachTheRender:
    """The settings GLPlot claims to honour live must actually change what it draws.

    ``style.use('ggplot')`` accepted and inert would be rule 2's failure at the level of a
    whole API: the call succeeds, the docs say it works, and every line still comes out
    tab10. The colour cycle and line width are read from rcParams on each use for exactly
    this reason, and that is only worth anything if it is pinned.
    """

    def _cycle_colours(self):
        gplt.figure("cycle")
        lines = [gplt.plot(X, Y + i)[0] for i in range(3)]
        return [tuple(round(float(c), 3) for c in line.style.color[:3]) for line in lines]

    def test_style_use_changes_the_colour_cycle(self):
        gplt.rcdefaults()
        before = self._cycle_colours()
        gplt.style.use("grayscale")
        after = self._cycle_colours()
        gplt.rcdefaults()
        assert before != after, "style.use() did not reach the drawn colours"

    def test_a_grayscale_style_actually_draws_grey(self):
        """Not merely 'different' -- every channel of a grayscale colour is equal."""
        gplt.style.use("grayscale")
        colours = self._cycle_colours()
        gplt.rcdefaults()
        assert all(len(set(c)) == 1 for c in colours), colours

    def test_prop_cycle_set_through_rc_is_honoured(self):
        gplt.rcdefaults()
        gplt.rc("axes", prop_cycle=gplt.cycler(color=["#ff0000"]))
        colours = self._cycle_colours()
        gplt.rcdefaults()
        assert all(c == (1.0, 0.0, 0.0) for c in colours), colours

    def test_rc_context_restores_on_exit(self):
        gplt.rcdefaults()
        outside = self._cycle_colours()
        with gplt.rc_context({"axes.prop_cycle": gplt.cycler(color=["#111111"])}):
            inside = self._cycle_colours()
        assert inside != outside
        assert self._cycle_colours() == outside

    def test_lines_linewidth_reaches_the_layer(self):
        gplt.rcdefaults()
        gplt.rcParams["lines.linewidth"] = 6.0
        width = gplt.plot(X, Y)[0].style.line_width
        gplt.rcdefaults()
        assert width == pytest.approx(6.0)


class TestPatchObjects:
    """``add_patch(Rectangle(...))`` -- matplotlib's calling form, not just GLPlot's.

    GLPlot's ``add_patch`` took raw vertices, so the single most common way to draw a shape
    in matplotlib did not work at all. Every Patch subclass describes its own outline
    through ``get_path()``, so one conversion covers all of them -- including ones this
    codebase has never named.
    """

    def _vertices(self):
        return gplt._get_or_create_plot().scene.layers[-1].vertices

    def test_a_rectangle_lands_on_its_own_corners(self):
        gplt.add_patch(gplt.Rectangle((0.0, 0.0), 2.0, 1.0))
        v = self._vertices()
        assert (v[:, 0].min(), v[:, 0].max()) == pytest.approx((0.0, 2.0))
        assert (v[:, 1].min(), v[:, 1].max()) == pytest.approx((0.0, 1.0))

    def test_a_circle_is_round_to_its_stated_radius(self):
        """matplotlib's own ``to_polygons()`` flattens the Bezier so coarsely that the
        vertices sit 2.5% outside the radius. It re-subdivides in Agg; GLPlot uploads
        these vertices once, so the curve is evaluated directly instead."""
        gplt.add_patch(gplt.Circle((0.0, 0.0), 1.5))
        rim = self._vertices()[1:]  # vertex 0 is the fan centre
        radii = np.hypot(rim[:, 0], rim[:, 1])
        assert radii.min() == pytest.approx(1.5, abs=1e-3)
        assert radii.max() == pytest.approx(1.5, abs=1e-3)

    def test_a_circles_area_is_pi_r_squared(self):
        """The tessellation, not just the outline: a fan that missed a wedge would still
        put every vertex on the circle."""
        gplt.add_patch(gplt.Circle((0.0, 0.0), 1.0))
        layer = gplt._get_or_create_plot().scene.layers[-1]
        v, tris = layer.vertices.astype(np.float64), layer.indices.reshape(-1, 3)
        # The 2-D cross product written out: numpy 2 deprecated `np.cross` on 2-vectors.
        e1, e2 = v[tris[:, 1]] - v[tris[:, 0]], v[tris[:, 2]] - v[tris[:, 0]]
        area = float(np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]).sum()) / 2.0
        assert area == pytest.approx(np.pi, rel=1e-3)

    @pytest.mark.parametrize(
        "patch",
        [
            gplt.Ellipse((0.0, 0.0), 2.0, 1.0, angle=30.0),
            gplt.Polygon([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]]),
            gplt.Wedge((0.0, 0.0), 1.0, 0.0, 90.0),
            gplt.RegularPolygon((0.0, 0.0), 6, radius=1.0),
            gplt.FancyBboxPatch((0.0, 0.0), 1.0, 1.0),
        ],
        ids=lambda p: type(p).__name__,
    )
    def test_every_patch_subclass_tessellates(self, patch):
        gplt.add_patch(patch)
        layer = gplt._get_or_create_plot().scene.layers[-1]
        assert len(layer.indices) >= 9 and len(layer.indices) % 3 == 0

    def test_the_patchs_own_colour_is_used(self):
        gplt.add_patch(gplt.Rectangle((0.0, 0.0), 1.0, 1.0, facecolor="red"))
        face = gplt._get_or_create_plot().scene.layers[-1].style.face_color
        assert tuple(round(float(c), 2) for c in face[:3]) == (1.0, 0.0, 0.0)

    def test_the_raw_vertex_form_still_works(self):
        """GLPlot's own signature is not displaced by the matplotlib one."""
        gplt.add_patch([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]], mode="strip", face_color="blue")
        assert len(self._vertices()) == 3


class TestDataKeyword:
    """``plot('height', 'mass', data=df)`` -- matplotlib's labelled-container indirection.

    The pandas idiom, and it raised ``ValueError: could not convert string to float`` on
    eight functions. The variadic plotters are the interesting case: a string may be a key
    *or* a format spec, and only one of them is in ``data``.
    """

    FRAME = {"a": X, "b": Y, "grid": M, "gx": GX, "gy": GY, "one": np.array([1.0])}

    @pytest.mark.parametrize(
        "call",
        [
            lambda d: gplt.plot("a", "b", data=d),
            lambda d: gplt.scatter("a", "b", data=d),
            lambda d: gplt.fill("a", "b", data=d),
            lambda d: gplt.step("a", "b", data=d),
            lambda d: gplt.hlines("one", 0.0, 2.0, data=d),
            lambda d: gplt.vlines("one", 0.0, 2.0, data=d),
            lambda d: gplt.quiver("gx", "gy", "gx", "gy", data=d),
            lambda d: gplt.contour("grid", data=d),
            lambda d: gplt.contourf("grid", data=d),
            lambda d: gplt.tripcolor("a", "b", "b", data=d),
            lambda d: gplt.errorbar("a", "b", yerr=0.1, data=d),
            lambda d: gplt.imshow("grid", data=d),
            lambda d: gplt.psd("a", data=d),
        ],
        ids=lambda f: "case",
    )
    def test_string_arguments_resolve_against_data(self, call):
        call(self.FRAME)

    def test_a_format_string_is_not_treated_as_a_key(self):
        """``plot('a', 'b', 'r--', data=df)`` has two keys and a format spec in the same
        argument list. Resolving every string would turn 'r--' into a KeyError."""
        layers = gplt.plot("a", "b", "r--", data=self.FRAME)
        assert layers
        assert tuple(round(float(c), 2) for c in layers[0].style.color[:3]) == (1.0, 0.0, 0.0)

    def test_a_missing_key_still_raises_for_fixed_arity_functions(self):
        """Where a string cannot be anything but a key, a typo must not pass silently."""
        with pytest.raises((KeyError, ValueError)):
            gplt.hlines("nope", 0.0, 1.0, data=self.FRAME)
