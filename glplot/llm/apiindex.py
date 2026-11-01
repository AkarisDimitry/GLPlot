"""An index of GLPlot's public Python API, built by introspection.

The index is what lets a language model that has only ``pip install glplot`` and no copy of the
repository find out what exists and how to call it: :func:`search` answers "how do I draw a
heatmap", :func:`describe` returns a function's real signature and docstring. Nothing here is
written out by hand except the *categories* and the search *synonyms* -- the signatures and
docs are read from the live objects, so they cannot drift from the code.

Pure Python: importing this module imports :mod:`glplot.pyplot` (for the functions it indexes)
and nothing else heavy, and never needs the optional ``mcp`` package.
"""

from __future__ import annotations

import difflib
import inspect
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

__all__ = [
    "CATEGORIES",
    "ApiEntry",
    "UnknownApiName",
    "categories",
    "describe",
    "entries",
    "search",
]

#: Category id -> title. The order is the order a reader is shown them in.
CATEGORIES: Dict[str, str] = {
    "figure": "Figures, panels and layout",
    "plot2d": "Line, scatter, bar and statistical plots",
    "fields": "Images, 2-D fields and vector fields",
    "plot3d": "3-D plots and camera",
    "axes": "Axes: limits, scales, ticks, labels",
    "annotate": "Annotations, reference lines and text",
    "color": "Colour, colormaps and style presets",
    "controls": "Interactive controls (sliders, buttons, interact)",
    "output": "Showing, saving and animation",
    "signal": "Signal and spectral analysis plots",
    "perf": "Rendering and performance options",
    "interaction": "Events and interactive-mode switches",
    "compat": "matplotlib compatibility helpers",
    "other": "Other",
}

# One line per category: the function names that belong to it. Kept as text so a new function
# is a one-word edit, and so a test can check every public function is placed somewhere.
_MEMBERS = {
    "figure": """
        figure subplots subplot subplot2grid subplot_mosaic axes gca gcf sca cla clf close
        delaxes tight_layout subplots_adjust suptitle figlegend figaspect fignum_exists
        get_fignums get_figlabels get_engine
    """,
    "plot2d": """
        plot scatter lineplot lines points plot_lines step stairs stem bar barh grouped_bar
        broken_barh hist hist2d hexbin errorbar fill fill_between fill_betweenx stackplot pie
        polar boxplot box violinplot eventplot ecdf loglog semilogx semilogy plot_date
    """,
    "fields": """
        imshow matshow pcolor pcolormesh contour contourf tricontour tricontourf tripcolor triplot
        quiver quiverkey barbs streamplot spy imread imsave figimage mandelbrot julia function
    """,
    "plot3d": """
        plot3d scatter3d plot_surface plot_wireframe bar3d quiver3d voxels volume3d mesh3d
        view_init view_preset auto_rotate autoscale3d set_projection get_proj_type
        set_proj_type set_box_aspect get_box_aspect set_xlim3d set_ylim3d set_zlim set_zlim3d
        zlabel zlim zscale zticks zticklabels get_zlabel get_zticks invert_zaxis dist3d grid3d
        set_axis3d_on set_axis3d_off
    """,
    "axes": """
        xlim ylim xlabel ylabel title xscale yscale axis grid xticks yticks xticklabels
        yticklabels tick_params ticklabel_format locator_params margins autoscale set_aspect
        twinx twiny secondary_xaxis secondary_yaxis minorticks_on minorticks_off rgrids
        thetagrids set_axis_on set_axis_off get_scale_names legend home reset_view
    """,
    "annotate": """
        text annotate arrow add_patch axhline axvline axhspan axvspan axline hlines vlines
        bar_label pie_label clabel figtext table
    """,
    "color": """
        colorbar clim set_cmap get_cmap register_cmap sci gci inset_colorbar_bounds plot_style
        plot_styles rc rc_context rcdefaults xkcd alpha set_global_alpha viridis plasma inferno
        magma jet hot cool gray bone copper pink spring summer autumn winter flag hsv prism
        nipy_spectral
    """,
    "controls": """
        slider range_slider checkbox dropdown radio textbox number color_picker readout button
        interact control_group controls
    """,
    "output": "show savefig export draw draw_all draw_if_interactive pause",
    "signal": """
        acorr xcorr cohere csd psd specgram magnitude_spectrum angle_spectrum phase_spectrum
    """,
    "perf": """
        density density_gain lod set_lod hud options profile stats blending
        set_layer_compositing ssao
    """,
    "interaction": "connect disconnect ginput waitforbuttonpress ion ioff isinteractive",
    "compat": """
        findobj get getp setp get_backend switch_backend get_current_fig_manager
        new_figure_manager get_plot_commands install_repl_displayhook uninstall_repl_displayhook
        interactive set_loglevel subplot_tool
    """,
}

_CATEGORY_OF: Dict[str, str] = {
    name: category for category, names in _MEMBERS.items() for name in names.split()
}

#: Words a model (or a person) says that are not the function's own name. Each maps to the
#: functions that do the job, most relevant first. Matched against whole query words.
SYNONYMS: Dict[str, Tuple[str, ...]] = {
    "heatmap": ("imshow", "pcolormesh", "hist2d", "matshow"),
    "heat": ("imshow", "pcolormesh", "hist2d"),
    "histogram": ("hist", "hist2d", "hexbin"),
    "density": ("hist2d", "hexbin", "density"),
    "boxplot": ("boxplot", "violinplot"),
    "violin": ("violinplot",),
    "barchart": ("bar", "barh", "grouped_bar"),
    "bars": ("bar", "barh", "grouped_bar", "bar3d"),
    "grouped": ("grouped_bar",),
    "pie": ("pie",),
    "donut": ("pie",),
    "line": ("plot", "step", "hlines", "vlines", "axhline"),
    "curve": ("plot", "fill_between"),
    "points": ("scatter", "scatter3d"),
    "bubble": ("scatter",),
    "band": ("fill_between",),
    "confidence": ("fill_between", "errorbar"),
    "uncertainty": ("fill_between", "errorbar"),
    "interval": ("fill_between", "errorbar"),
    "shade": ("fill_between", "axvspan", "axhspan"),
    "area": ("fill_between", "stackplot", "fill"),
    "stacked": ("stackplot", "bar"),
    "log": ("xscale", "yscale", "loglog", "semilogx", "semilogy"),
    "logarithmic": ("xscale", "yscale", "loglog"),
    "scale": ("xscale", "yscale", "set_aspect"),
    "twin": ("twinx", "twiny", "secondary_xaxis", "secondary_yaxis"),
    "second": ("twinx", "twiny", "secondary_yaxis", "secondary_xaxis"),
    "right": ("twinx", "secondary_yaxis", "legend"),
    "yaxis": ("twinx", "ylim", "ylabel", "yscale"),
    "xaxis": ("twiny", "xlim", "xlabel", "xscale"),
    "secondary": ("twinx", "twiny", "secondary_xaxis", "secondary_yaxis"),
    "dual": ("twinx", "twiny"),
    "range": ("xlim", "ylim", "set_zlim"),
    "limits": ("xlim", "ylim", "set_zlim", "autoscale"),
    "zoom": ("xlim", "ylim", "reset_view"),
    "ticks": ("xticks", "yticks", "tick_params", "locator_params"),
    "label": ("xlabel", "ylabel", "title", "text", "legend"),
    "caption": ("title", "suptitle", "figtext"),
    "legend": ("legend", "figlegend"),
    "colorbar": ("colorbar", "clim"),
    "colormap": ("set_cmap", "get_cmap", "colorbar", "plot_style"),
    "cmap": ("set_cmap", "get_cmap", "colorbar"),
    "colour": ("set_cmap", "plot_style", "colorbar", "scatter"),
    "color": ("set_cmap", "plot_style", "colorbar", "scatter"),
    "theme": ("plot_style", "plot_styles", "rc_context"),
    "style": ("plot_style", "plot_styles", "rc_context"),
    "dark": ("plot_style", "plot_styles"),
    "font": ("rc", "rc_context", "text"),
    "grid": ("grid", "subplots", "subplot_mosaic", "grid3d"),
    "layout": ("subplots", "subplot_mosaic", "tight_layout", "subplots_adjust"),
    "panels": ("subplots", "subplot", "subplot_mosaic", "subplot2grid"),
    "subplot": ("subplots", "subplot", "subplot2grid", "subplot_mosaic"),
    "multiple": ("subplots", "subplot_mosaic"),
    "inset": ("colorbar", "subplots"),
    "save": ("savefig", "export"),
    "export": ("savefig", "export"),
    "png": ("savefig",),
    "pdf": ("savefig",),
    "svg": ("savefig",),
    "image": ("imshow", "imread", "imsave", "savefig"),
    "show": ("show",),
    "window": ("show", "figure"),
    "display": ("show",),
    "animate": ("FuncAnimation", "ArtistAnimation", "slider", "interact"),
    "animation": ("FuncAnimation", "ArtistAnimation", "PillowWriter", "FFMpegWriter"),
    "gif": ("FuncAnimation", "PillowWriter"),
    "movie": ("FuncAnimation", "FFMpegWriter"),
    "video": ("FuncAnimation", "FFMpegWriter"),
    "mp4": ("FuncAnimation", "FFMpegWriter"),
    "slider": ("slider", "range_slider", "interact"),
    "widget": ("slider", "button", "checkbox", "dropdown", "interact"),
    "interactive": ("interact", "slider", "show"),
    "control": ("slider", "button", "checkbox", "dropdown", "interact", "control_group"),
    "input": ("slider", "textbox", "number", "dropdown"),
    "contour": ("contour", "contourf", "tricontour", "clabel"),
    "field": ("imshow", "contourf", "quiver", "streamplot", "pcolormesh"),
    "vector": ("quiver", "streamplot", "quiver3d", "barbs"),
    "arrows": ("quiver", "arrow", "quiver3d", "annotate"),
    "flow": ("streamplot", "quiver"),
    "surface": ("plot_surface", "plot_wireframe", "mesh3d", "pcolormesh"),
    "wireframe": ("plot_wireframe",),
    "3d": ("scatter3d", "plot3d", "plot_surface", "view_init", "bar3d", "volume3d"),
    "camera": ("view_init", "view_preset", "auto_rotate"),
    "rotate": ("view_init", "auto_rotate", "view_preset"),
    "volume": ("volume3d", "voxels"),
    "cloud": ("scatter3d", "volume3d", "scatter"),
    "fractal": ("mandelbrot", "julia"),
    "annotation": ("text", "annotate", "arrow", "axhline", "axvline"),
    "annotate": ("annotate", "text", "arrow"),
    "text": ("text", "annotate", "figtext", "title"),
    "arrow": ("arrow", "annotate", "quiver"),
    "reference": ("axhline", "axvline", "axhspan", "axvspan", "axline", "hlines", "vlines"),
    "horizontal": ("axhline", "hlines", "barh", "axhspan"),
    "vertical": ("axvline", "vlines", "bar", "axvspan"),
    "region": ("axvspan", "axhspan", "fill_between"),
    "highlight": ("axvspan", "axhspan", "annotate"),
    "fft": ("psd", "specgram", "magnitude_spectrum", "phase_spectrum"),
    "spectrum": ("psd", "specgram", "magnitude_spectrum", "angle_spectrum"),
    "spectrogram": ("specgram",),
    "power": ("psd", "magnitude_spectrum"),
    "correlation": ("acorr", "xcorr", "cohere"),
    "empirical": ("ecdf",),
    "cdf": ("ecdf",),
    "cumulative": ("ecdf", "stackplot"),
    "step": ("step", "stairs"),
    "stem": ("stem",),
    "polar": ("polar", "rgrids", "thetagrids"),
    "radar": ("polar",),
    "date": ("plot_date",),
    "time": ("plot_date", "FuncAnimation"),
    "categorical": ("bar", "barh", "grouped_bar", "boxplot"),
    "error": ("errorbar", "fill_between"),
    "errorbars": ("errorbar",),
    "scatterplot": ("scatter",),
    "performance": ("density", "lod", "set_lod", "options", "profile", "hist2d"),
    "fast": ("density", "lod", "hist2d", "hexbin"),
    "million": ("density", "lod", "hist2d", "scatter"),
    "large": ("density", "lod", "hist2d", "hexbin"),
    "big": ("density", "lod", "hist2d", "hexbin"),
    "opacity": ("set_global_alpha", "alpha"),
    "transparent": ("savefig", "set_global_alpha"),
    "aspect": ("set_aspect", "axis", "set_box_aspect"),
    "equal": ("set_aspect", "axis"),
    "close": ("close", "clf", "cla"),
    "clear": ("clf", "cla", "close"),
    "current": ("gca", "gcf", "sca"),
    "pick": ("ginput", "connect"),
    "click": ("connect", "ginput", "button"),
    "event": ("connect", "disconnect", "ginput"),
}

#: Words that carry no meaning for a search over function descriptions.
_STOP = frozenset(
    "a an the of for to in on and or with how do i can you make draw create show me my is it "
    "use using want need get what which that this from into as at by be".split()
)

#: Extra entries for the parts of the API that live outside ``glplot.pyplot``.
_EXTERNAL = (
    ("glplot.animation", "FuncAnimation", "output"),
    ("glplot.animation", "ArtistAnimation", "output"),
    ("glplot.animation", "PillowWriter", "output"),
    ("glplot.animation", "FFMpegWriter", "output"),
    ("glplot.animation", "HTMLWriter", "output"),
)


class UnknownApiName(KeyError):
    """``describe`` was asked for a name that is not in the index."""

    def __str__(self) -> str:  # KeyError would repr() the message with its quotes
        return str(self.args[0])


@dataclass(frozen=True)
class ApiEntry:
    """One public function or class."""

    name: str
    category: str
    module: str
    kind: str  # "function" | "class"
    signature: str
    summary: str
    doc: str = field(repr=False)
    brief: str = "(...)"

    @property
    def usage(self) -> str:
        """How to reach it from a script, e.g. ``plt.scatter(...)``."""
        if self.module == "glplot.pyplot":
            return f"plt.{self.name}{self.brief}"
        return f"from {self.module} import {self.name}  # {self.name}{self.brief}"

    def as_dict(self, *, full: bool = False) -> Dict[str, object]:
        out: Dict[str, object] = {
            "name": self.name,
            "category": self.category,
            "kind": self.kind,
            "module": self.module,
            "signature": self.signature,
            "brief": self.brief,
            "summary": self.summary,
            "usage": self.usage,
        }
        if full:
            out["doc"] = self.doc
        return out


def _summary(doc: str) -> str:
    """The first sentence of a docstring, on one line."""
    if not doc:
        return ""
    first = doc.strip().split("\n\n", 1)[0].replace("\n", " ")
    first = re.sub(r"\s+", " ", first).strip()
    match = re.match(r"(.+?[.!?])(\s|$)", first)
    text = match.group(1) if match else first
    return text if len(text) <= 160 else text[:157].rstrip() + "..."


def _signature(obj: object) -> str:
    try:
        text = str(inspect.signature(obj))
    except (TypeError, ValueError):
        return "(...)"
    # `from __future__ import annotations` leaves annotations as quoted strings.
    return text.replace("'", "") if "'" in text and '"' not in text else text


def _brief(obj: object, limit: int = 130) -> str:
    """``(x, y, color=None, ...)``: parameter names and defaults only, for listings.

    The full signature carries every annotation and runs to several hundred characters for the
    big plotting functions; a list of search hits needs the shape of the call, not the types.
    """
    try:
        parameters = list(inspect.signature(obj).parameters.values())
    except (TypeError, ValueError):
        return "(...)"
    parts: List[str] = []
    keyword_marker_done = False
    for parameter in parameters:
        if parameter.kind is inspect.Parameter.VAR_POSITIONAL:
            parts.append(f"*{parameter.name}")
            keyword_marker_done = True
            continue
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            parts.append(f"**{parameter.name}")
            continue
        if parameter.kind is inspect.Parameter.KEYWORD_ONLY and not keyword_marker_done:
            parts.append("*")
            keyword_marker_done = True
        text = parameter.name
        if parameter.default is not inspect.Parameter.empty:
            shown = repr(parameter.default)
            text += "=" + (shown if len(shown) <= 18 else shown[:15] + "...")
        parts.append(text)
    out = "(" + ", ".join(parts) + ")"
    if len(out) > limit:
        cut = out[: limit - 6].rsplit(",", 1)[0]
        out = cut + ", ...)"
    return out


def _category(name: str, module: str, preferred: Optional[str] = None) -> str:
    if preferred:
        return preferred
    if name in _CATEGORY_OF:
        return _CATEGORY_OF[name]
    if name.endswith("3d"):
        return "plot3d"
    return "other"


@lru_cache(maxsize=1)
def entries() -> Tuple[ApiEntry, ...]:
    """Every indexed function and class, sorted by category then name."""
    import importlib

    from .. import pyplot

    found: List[ApiEntry] = []
    for name in sorted(vars(pyplot)):
        if name.startswith("_"):
            continue
        obj = getattr(pyplot, name)
        if not inspect.isfunction(obj) or obj.__module__ != pyplot.__name__:
            continue
        doc = inspect.getdoc(obj) or ""
        found.append(
            ApiEntry(
                name=name,
                category=_category(name, pyplot.__name__),
                module=pyplot.__name__,
                kind="function",
                signature=_signature(obj),
                summary=_summary(doc),
                doc=doc,
                brief=_brief(obj),
            )
        )
    for module_name, name, category in _EXTERNAL:
        obj = getattr(importlib.import_module(module_name), name)
        doc = inspect.getdoc(obj) or ""
        found.append(
            ApiEntry(
                name=name,
                category=_category(name, module_name, category),
                module=module_name,
                kind="class" if inspect.isclass(obj) else "function",
                signature=_signature(obj),
                summary=_summary(doc),
                doc=doc,
                brief=_brief(obj),
            )
        )
    order = list(CATEGORIES)
    found.sort(key=lambda e: (order.index(e.category), e.name))
    return tuple(found)


def _by_name() -> Dict[str, ApiEntry]:
    return {entry.name: entry for entry in entries()}


def categories() -> List[Dict[str, object]]:
    """Each category with its title and function names, in display order."""
    grouped: Dict[str, List[str]] = {key: [] for key in CATEGORIES}
    for entry in entries():
        grouped[entry.category].append(entry.name)
    return [
        {"id": key, "title": CATEGORIES[key], "names": names}
        for key, names in grouped.items()
        if names
    ]


def describe(name: str) -> ApiEntry:
    """The entry for ``name`` (``"scatter"``, ``"plt.scatter"`` and ``"pyplot.scatter"`` all work).

    Raises :class:`UnknownApiName` with near-miss suggestions, so a model that typed
    ``scater`` is told about ``scatter`` instead of getting a bare failure.
    """
    key = str(name).strip()
    for prefix in ("glplot.pyplot.", "pyplot.", "plt.", "gplt.", "glplot.animation."):
        if key.startswith(prefix):
            key = key[len(prefix) :]
            break
    key = key.split("(", 1)[0].strip()
    table = _by_name()
    if key in table:
        return table[key]
    lowered = {n.lower(): n for n in table}
    if key.lower() in lowered:
        return table[lowered[key.lower()]]
    close = difflib.get_close_matches(key, list(table), n=4, cutoff=0.6)
    hits = [e.name for e in search(key, limit=4)]
    hint = list(dict.fromkeys(close + hits))[:5]
    tail = f" Did you mean: {', '.join(hint)}?" if hint else ""
    raise UnknownApiName(f"No function named {name!r} in the GLPlot API.{tail}")


def _tokens(text: str) -> List[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [w for w in words if w not in _STOP]


def search(query: str, limit: int = 8, category: Optional[str] = None) -> List[ApiEntry]:
    """The entries that best match ``query``, best first. Empty if nothing is plausible.

    Scoring, most to least decisive: the exact name, a synonym ("heatmap" -> ``imshow``), a name
    that starts with / contains a query word, the category, then words in the one-line summary
    and the docstring's opening. Plain and predictable on purpose: a model reading the result
    can see why each hit is there.
    """
    pool = [e for e in entries() if category in (None, "", e.category)]
    words = _tokens(query)
    if not words:
        return []
    names = {e.name for e in pool}
    scores: Dict[str, float] = {}

    def add(name: str, amount: float) -> None:
        if name in names:
            scores[name] = scores.get(name, 0.0) + amount

    whole = query.strip().lower().replace(" ", "_")
    add(whole, 100.0)
    # A word that happens to be a function's whole name ("axis" in "second y axis") is decisive
    # only when it is the whole query; inside a phrase it is one clue among several.
    exact = 100.0 if len(words) == 1 else 30.0
    for word in words:
        add(word, exact)
        for rank, target in enumerate(SYNONYMS.get(word, ())):
            add(target, 40.0 - 4.0 * rank)
    for entry in pool:
        lower = entry.name.lower()
        head = f"{entry.summary} {' '.join(entry.doc.split()[:60])}".lower()
        for word in words:
            if lower.startswith(word):
                add(entry.name, 20.0)
            elif word in lower:
                add(entry.name, 10.0)
            if word in CATEGORIES[entry.category].lower():
                add(entry.name, 4.0)
            if re.search(rf"\b{re.escape(word)}\w*", entry.summary.lower()):
                add(entry.name, 6.0)
            elif re.search(rf"\b{re.escape(word)}\w*", head):
                add(entry.name, 1.5)
    ranked = sorted(
        (e for e in pool if scores.get(e.name, 0.0) > 0.0),
        key=lambda e: (-scores[e.name], e.name),
    )
    return ranked[: max(1, int(limit))]


def names_in(category: str) -> Sequence[str]:
    """Function names in one category (empty for an unknown one)."""
    return [e.name for e in entries() if e.category == category]
