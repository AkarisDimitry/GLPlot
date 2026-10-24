"""Recipes: the gallery examples, distilled into something you can point at your own data.

``examples/gallery`` shows what GLPlot can draw, as scripts. A script is the wrong shape for
"I have a table and I want it to look like *that*": it hard-codes its own data, and the look is
tangled up with the code that makes the numbers. A **recipe** keeps only the part worth
reusing -- *what to draw, from which kinds of column, and in which look* -- and drops the data.

A :class:`Recipe` is declarative leaf data, like a :class:`~glplot.gui.styles.PlotStyle`:

* a **kind** (a 2-D or 3-D kind from the layer registries, or the two field shapes that are not
  tables, ``image`` and ``field``) and the **slots** that kind reads (``x``, ``y``, ``c``...);
* the **look** it was designed in -- a style preset, a colormap, line width, point size,
  opacity, a 3-D camera, per-layer compositing;
* a deterministic **sample** table, so every recipe can be previewed (and tried) before the
  user has any data, and so its gallery thumbnail is rendered from the very numbers the
  "Try with sample data" button creates.

Everything here is pure: no imgui, no GL, no window. :func:`apply_recipe` builds the layers
through :mod:`glplot.gui.layerops` / :mod:`glplot.gui.layerops3d` -- the same functions the Data
panel's Plot button calls -- so a recipe's layer is an ordinary tagged layer that the Scene and
Style panels can edit afterwards, and :func:`recipe_command` wraps it as one undo step.
:func:`apply_recipe` mutates the live scene, so (like every ``layerops`` function) it is
**queued-command only** (CONTRACT §1.1); the Gallery panel only ever calls it through
:func:`recipe_command`.

Two things a recipe deliberately is not:

* **Not a live link.** The layers are independent copies of the data. Editing a cell does not
  move them; re-apply the recipe. Binding a derived layer to its table is the Data panel's job
  and has sharp edges (see ``layerops.bind_dataset``) that a one-click gallery should not
  inherit.
* **Not a style.** Applying one *does* set a look -- that is half the point -- but only the page
  (background, grid, blending, effects) and the colours of the layers it creates. Layers the
  user already had are left alone.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .. import colormaps as colormaps_module
from ..core.camera3d import SYSTEM_3D_ARTISTS
from ..options import BlendMode
from . import layerops, layerops3d, styles
from .history import Command

logger = logging.getLogger(__name__)

__all__ = [
    "CATEGORIES",
    "KEEP_LOOK",
    "RECIPES",
    "RECIPE_KEYS",
    "RECIPE_LOOK",
    "Recipe",
    "RecipeError",
    "Slot",
    "apply_recipe",
    "assign_columns",
    "build_preview_plot",
    "get_recipe",
    "recipe_command",
    "render_preview_rgb",
    "slot_arrays",
    "slot_labels",
    "sample_assignment",
    "sample_data",
    "sample_labels",
    "resolve_look",
    "sample_table",
    "validate_data",
]

#: The two non-preset answers to "which look?": the recipe's own, or leave the page alone.
RECIPE_LOOK = "recipe"
KEEP_LOOK = "keep"

#: Gallery categories, in the order the panel's filter chips show them.
CATEGORIES: Tuple[str, ...] = ("Lines", "Scatter", "Density", "Bars", "Fields", "3D")

#: The sentinel the Data panel offers next to real column names: the row's own position.
INDEX = "(index)"

#: Kinds that are not rows of a table. ``image`` is a matrix (every column is a matrix column);
#: ``field`` is scattered ``(x, y, z)`` samples gridded onto an image.
_SPECIAL_KINDS = ("image", "field")

#: Resolution a scattered field is gridded to. 200 x 200 is as fine as an on-screen heat map
#: needs and keeps the interpolation (``scipy.interpolate.griddata``) well under a second.
FIELD_RESOLUTION = 200

#: Rows beyond which ``line_family`` and the other per-row-geometry kinds are refused with a
#: readable message rather than left to run the GPU out of memory.
MAX_RECIPE_ROWS = 20_000_000


class RecipeError(ValueError):
    """The data cannot be drawn with this recipe; the message says what to change."""


# ----------------------------------------------------------------------------------
# The data model
# ----------------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class Slot:
    """One input a recipe reads: a named role a column can be assigned to."""

    key: str
    label: str
    #: A required slot must be assigned before the recipe can be applied.
    required: bool = True
    #: One line of help, shown as a tooltip beside the picker.
    hint: str = ""
    #: The sample table's column for this slot (what the "Try with sample data" table calls it).
    sample: str = ""
    #: Lower-case column names that auto-assignment prefers for this slot.
    aliases: Tuple[str, ...] = ()
    #: A multi-column slot takes *every* column of the dataset (the matrix recipe's cells).
    multi: bool = False


@dataclass(frozen=True, eq=False)
class Recipe:
    """One gallery card: what to draw, from which columns, and how it should look.

    ``eq=False`` on purpose: ``sample`` is a function and ``options`` a dict, so a generated
    ``__eq__``/``__hash__`` would be wrong at best and raise at worst; identity is the
    right notion for registry entries.
    """

    key: str
    name: str
    description: str
    category: str
    #: A key of :data:`layerops.KIND_KEYS`, :data:`layerops3d.KIND3D_KEYS`, or ``image``/``field``.
    kind: str
    slots: Tuple[Slot, ...]
    #: Key of the :class:`~glplot.gui.styles.PlotStyle` the recipe was designed in.
    style: str
    #: Colormap for value-coloured layers; ``None`` means "the style's own".
    cmap: Optional[str] = None
    #: Kind options (``bins``, ``gridsize``, ``x_lo``...). Validated by ``layerops``.
    options: Mapping[str, Any] = field(default_factory=dict)
    #: Overrides for what the style would give: ``None`` = take the style's.
    width: Optional[float] = None
    size: Optional[float] = None
    #: Extra per-layer ``LayerStyle`` fields (``alpha``, ``blend_mode``, ``auto_alpha``...).
    layer_style: Mapping[str, Any] = field(default_factory=dict)
    #: ``(elevation, azimuth)`` in degrees for a 3-D recipe.
    camera: Optional[Tuple[float, float]] = None
    #: Proportions of the 3-D axis box (x, y, z). The default box is a cube; a flat structure
    #: -- a galaxy disc, a thin sheet -- floats in the middle of it, so such a recipe flattens
    #: the box to what the data actually fills.
    box_aspect: Optional[Tuple[float, float, float]] = None
    #: Colour the layer with the style's palette instead of mapping a value through the
    #: colormap. For kinds drawn as *thin lines* (stems, arrows) a ramp that starts at the
    #: page colour -- the ramps made for a light page do -- would make the low values vanish.
    solid: bool = False
    #: What names the horizontal and vertical axis: a slot key (the column assigned to it) or
    #: ``"=text"`` for a fixed name. The default is the ``x`` and ``y`` slots; it differs where a
    #: slot is not an axis -- a histogram's values run along x and its y is a count, a line
    #: family's slots are slope and intercept while its axes are plain x and y.
    axes: Tuple[str, str] = ("x", "y")
    #: The gallery script this was distilled from, relative to ``examples/gallery``.
    example: str = ""
    #: A title for the sample render. User data gets none unless asked: the title is theirs.
    title: str = ""
    #: ``sample(n, rng)`` -> ordered ``{column name: 1-D array}``; deterministic given ``rng``.
    sample: Optional[Callable[[int, np.random.Generator], Dict[str, np.ndarray]]] = None
    #: Rows the sample table has.
    sample_rows: int = 2000
    #: Search words beyond the name and description.
    tags: Tuple[str, ...] = ()

    @property
    def is_3d(self) -> bool:
        return self.kind in layerops3d.KIND3D_KEYS

    @property
    def dimension(self) -> str:
        return "3D" if self.is_3d else "2D"

    @property
    def is_special(self) -> bool:
        return self.kind in _SPECIAL_KINDS

    def slot(self, key: str) -> Slot:
        for slot in self.slots:
            if slot.key == key:
                return slot
        raise KeyError(key)

    def required_slots(self) -> Tuple[Slot, ...]:
        return tuple(slot for slot in self.slots if slot.required)


# ----------------------------------------------------------------------------------
# Sample data: one deterministic generator per recipe
# ----------------------------------------------------------------------------------


def _signal(n: int, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    t = np.linspace(0.0, 8.0, n)
    envelope = 0.55 + 0.45 * np.exp(-((t - 3.2) ** 2) / 5.5)
    signal = envelope * (
        0.9 * np.sin(2 * np.pi * 1.3 * t) + 0.45 * np.sin(2 * np.pi * 4.1 * t + 0.7)
    ) + rng.normal(0.0, 0.06, n)
    return {"time": t, "signal": signal}


def _step(n: int, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    t = np.linspace(0.0, 6.0, n)
    response = 1.0 - np.exp(-t / 1.1) * np.cos(2.4 * t)
    return {"time": t, "level": np.round(response * 4.0) / 4.0 + rng.normal(0.0, 0.0, n)}


def _stems(n: int, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    times = np.sort(rng.uniform(0.0, 10.0, n))
    amplitude = np.abs(rng.normal(1.0, 0.7, n)) * np.exp(-times / 14.0)
    return {"time": times, "amplitude": amplitude}


def _area(n: int, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    hours = np.linspace(0.0, 24.0, n)
    load = (
        0.35
        + 0.9 * np.exp(-((hours - 9.0) ** 2) / 6.0)
        + 1.2 * np.exp(-((hours - 19.0) ** 2) / 8.0)
        + rng.normal(0.0, 0.015, n)
    )
    return {"hour": hours, "load": np.clip(load, 0.0, None)}


def _line_family(n: int, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    return {
        "slope": rng.normal(0.35, 0.85, n),
        "intercept": rng.normal(0.0, 0.5, n),
    }


def _starfield(n: int, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    theta = rng.uniform(0.0, 18.0 * np.pi, n)
    radius = 0.025 * theta + rng.gamma(1.5, 0.006, n)
    x = radius * np.cos(theta) + rng.normal(scale=0.012, size=n)
    y = radius * np.sin(theta) + rng.normal(scale=0.012, size=n)
    return {"x_pc": x, "y_pc": y, "age": theta + 6.0 * radius}


def _bubbles(n: int, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    income = rng.uniform(1.0, 9.0, n)
    life = 48.0 + 4.2 * np.log(income * 4.0) * 3.0 + rng.normal(0.0, 2.2, n)
    population = np.exp(rng.normal(2.4, 1.0, n))
    return {"income": income, "life_expectancy": life, "population": population}


def _blobs(n: int, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    centres = np.array([(-1.4, -0.6), (1.2, 0.9), (0.3, -1.5)])
    spread = np.array([(0.55, 0.35), (0.40, 0.55), (0.30, 0.30)])
    which = rng.choice(3, size=n, p=(0.45, 0.35, 0.20))
    x = centres[which, 0] + rng.normal(size=n) * spread[which, 0]
    y = centres[which, 1] + rng.normal(size=n) * spread[which, 1]
    return {"x": x, "y": y}


def _bimodal(n: int, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    left = rng.normal(-1.1, 0.55, n // 2)
    right = rng.normal(1.3, 0.8, n - n // 2)
    values = np.concatenate([left, right])
    rng.shuffle(values)
    return {"value": values}


def _bars(n: int, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    month = np.arange(1, n + 1, dtype=float)
    sales = 40.0 + 25.0 * np.sin(month / 12.0 * 2 * np.pi - 1.0) + rng.uniform(0.0, 14.0, n)
    return {"month": month, "sales": sales}


def _matrix(n: int, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    cols = 48
    rows = np.arange(n, dtype=float)[:, None]
    cols_axis = np.arange(cols, dtype=float)[None, :]
    field_ = (
        np.sin(rows / 7.0) * np.cos(cols_axis / 5.0)
        + 0.6 * np.exp(-(((rows - 0.55 * n) / (0.18 * n)) ** 2 + ((cols_axis - 30.0) / 7.0) ** 2))
        + rng.normal(0.0, 0.04, (n, cols))
    )
    return {f"c{i:02d}": field_[:, i] for i in range(cols)}


def _scattered_field(n: int, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    x = rng.uniform(-3.0, 3.0, n)
    y = rng.uniform(-3.0, 3.0, n)
    z = (
        np.exp(-((x - 1.0) ** 2 + (y - 0.6) ** 2) * 1.1)
        + 0.8 * np.exp(-((x + 1.2) ** 2 + (y + 0.8) ** 2) * 0.8)
        + 0.18 * np.sin(2.2 * x) * np.cos(2.2 * y)
    )
    return {"x": x, "y": y, "intensity": z}


def _cloud(n: int, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    """A flared disc inside a diffuse halo, with a thin jet along each pole."""
    kind = rng.choice(3, size=n, p=(0.30, 0.58, 0.12))
    x = np.empty(n)
    y = np.empty(n)
    z = np.empty(n)

    halo = kind == 0
    r = rng.gamma(3.0, 0.45, halo.sum())
    cos_t = rng.uniform(-1.0, 1.0, r.size)
    phi = rng.uniform(0.0, 2 * np.pi, r.size)
    sin_t = np.sqrt(1.0 - cos_t**2)
    x[halo], y[halo], z[halo] = r * sin_t * np.cos(phi), r * sin_t * np.sin(phi), r * cos_t * 0.9

    disc = kind == 1
    rd = rng.gamma(2.4, 0.6, disc.sum())
    pd = rng.uniform(0.0, 2 * np.pi, rd.size)
    x[disc], y[disc] = rd * np.cos(pd), rd * np.sin(pd)
    z[disc] = rng.normal(0.0, 0.05 + 0.05 * rd)

    jet = kind == 2
    zj = rng.uniform(0.3, 3.0, jet.sum()) * rng.choice([-1.0, 1.0], jet.sum())
    spread = 0.04 + 0.06 * np.abs(zj)
    x[jet], y[jet], z[jet] = rng.normal(0.0, spread), rng.normal(0.0, spread), zj

    radius = np.sqrt(x**2 + y**2 + z**2)
    # A few far-flung points would stretch the axis box to twice the size of the structure
    # that fills it; pull the stragglers in so the cloud reads at the size it is.
    far = radius > 3.4
    scale = np.where(far, 3.4 / np.maximum(radius, 1e-9), 1.0)
    x, y, z = x * scale, y * scale, z * scale
    return {"x": x, "y": y, "z": z, "radius": np.sqrt(x**2 + y**2 + z**2)}


def _nebula(n: int, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    """Three spiral arms around a bright bulge, thickest where they are dimmest."""
    bulge = rng.random(n) < 0.06
    arm = rng.integers(0, 3, n)
    # sqrt: uniform in *area*, not in radius, or the centre is a hundred times denser than the
    # arms and an additive render of it is a white dot with a faint ring.
    t = np.sqrt(rng.uniform(0.01, 1.0, n))
    angle = t * 2.6 * np.pi + arm * (2 * np.pi / 3)
    radius = 0.35 + 2.6 * t
    x = radius * np.cos(angle) + rng.normal(0.0, 0.10 + 0.14 * t, n)
    y = radius * np.sin(angle) + rng.normal(0.0, 0.10 + 0.14 * t, n)
    z = rng.normal(0.0, 0.10 + 0.34 * t, n)
    count = int(bulge.sum())
    x[bulge] = rng.normal(0.0, 0.42, count)
    y[bulge] = rng.normal(0.0, 0.42, count)
    z[bulge] = rng.normal(0.0, 0.34, count)
    emission = np.exp(-(np.hypot(x, y) ** 2) / 1.6) + 0.5 * np.exp(-np.abs(z) * 2.5)
    return {"x": x, "y": y, "z": z, "emission": emission}


def _grid_xyz(n: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    side = max(4, int(round(np.sqrt(n))))
    axis = np.linspace(-4.0, 4.0, side)
    gx, gy = np.meshgrid(axis, axis)
    return gx.ravel(), gy.ravel()


def _surface(n: int, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    x, y = _grid_xyz(n, rng)
    radius = np.hypot(x, y)
    z = 1.5 * np.sin(1.6 * radius) * np.exp(-0.05 * radius**2) + 0.35 * np.cos(0.9 * x) * np.sin(
        0.9 * y
    )
    return {"x": x, "y": y, "height": z}


def _city(n: int, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    side = max(4, int(round(np.sqrt(n))))
    axis = np.arange(side, dtype=float)
    gx, gy = np.meshgrid(axis, axis)
    centre = (side - 1) / 2.0
    mound = np.exp(-(((gx - centre) ** 2 + (gy - centre) ** 2) / (0.22 * side**2)))
    heights = 0.4 + 5.0 * mound * rng.uniform(0.35, 1.0, gx.shape) + rng.uniform(0.0, 0.6, gx.shape)
    return {"block_x": gx.ravel(), "block_y": gy.ravel(), "floors": heights.ravel()}


def _flow(n: int, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    axis = np.linspace(-1.0, 1.0, 8)
    z_axis = np.linspace(-0.8, 0.8, 5)
    gx, gy, gz = np.meshgrid(axis, axis, z_axis)
    x, y, z = gx.ravel(), gy.ravel(), gz.ravel()
    swirl = np.exp(-(x**2 + y**2) / 1.8)
    return {
        "x": x,
        "y": y,
        "z": z,
        "u": -y * swirl + 0.08 * rng.normal(size=x.size),
        "v": x * swirl + 0.08 * rng.normal(size=x.size),
        "w": 0.9 * np.cos(2.0 * z) * swirl + 0.05 * rng.normal(size=x.size),
    }


def _orbit(n: int, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    """A Lorenz trajectory, integrated with RK4 (a path that fills space and never closes)."""
    sigma, rho, beta, dt = 10.0, 28.0, 8.0 / 3.0, 0.006

    def rate(p: np.ndarray) -> np.ndarray:
        return np.array(
            [sigma * (p[1] - p[0]), p[0] * (rho - p[2]) - p[1], p[0] * p[1] - beta * p[2]]
        )

    point = np.array([1.0, 1.0, 1.05]) + rng.normal(0.0, 0.01, 3)
    path = np.empty((n, 3))
    for i in range(n):
        k1 = rate(point)
        k2 = rate(point + 0.5 * dt * k1)
        k3 = rate(point + 0.5 * dt * k2)
        k4 = rate(point + dt * k3)
        point = point + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        path[i] = point
    return {"x": path[:, 0], "y": path[:, 1], "z": path[:, 2], "time": np.arange(n) * dt}


def _scattered_heights(n: int, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    x = rng.uniform(-3.0, 3.0, n)
    y = rng.uniform(-3.0, 3.0, n)
    z = 3.2 * np.exp(-((x - 0.8) ** 2 + (y + 0.4) ** 2) / 1.3) + 2.2 * np.exp(
        -((x + 1.2) ** 2 + (y - 1.0) ** 2) / 1.0
    )
    return {"x": x, "y": y, "height": z + rng.uniform(0.0, 0.25, n)}


def _ribbon(n: int, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    t = np.linspace(0.0, 4.0 * np.pi, n)
    return {
        "x": np.cos(t) * (1.0 + 0.25 * np.sin(3 * t)),
        "y": np.sin(t) * (1.0 + 0.25 * np.sin(3 * t)),
        "z": 0.5 + 0.4 * np.sin(2.0 * t) + t / (4.0 * np.pi),
    }


# ----------------------------------------------------------------------------------
# The catalogue
# ----------------------------------------------------------------------------------


def _xy(
    x_sample: str,
    y_sample: str,
    x_label: str = "X",
    y_label: str = "Y",
    x_aliases: Tuple[str, ...] = ("x", "time", "t", "index", "date", "hour"),
    y_aliases: Tuple[str, ...] = ("y", "value", "signal", "v"),
) -> Tuple[Slot, Slot]:
    return (
        Slot("x", x_label, hint="Horizontal position.", sample=x_sample, aliases=x_aliases),
        Slot("y", y_label, hint="Vertical position.", sample=y_sample, aliases=y_aliases),
    )


def _xyz(
    x_sample: str = "x",
    y_sample: str = "y",
    z_sample: str = "z",
    z_label: str = "Z",
    z_hint: str = "Height.",
) -> Tuple[Slot, Slot, Slot]:
    return (
        Slot(
            "x", "X", hint="First horizontal axis.", sample=x_sample, aliases=("x", "lon", "east")
        ),
        Slot(
            "y", "Y", hint="Second horizontal axis.", sample=y_sample, aliases=("y", "lat", "north")
        ),
        Slot(
            "z",
            z_label,
            hint=z_hint,
            sample=z_sample,
            aliases=("z", "height", "depth", "value", "intensity"),
        ),
    )


def _colour(sample: str = "", hint: str = "Colour each point by this column.") -> Slot:
    return Slot(
        "c",
        "Colour by",
        required=False,
        hint=hint,
        sample=sample,
        aliases=("c", "color", "colour", "value", "age", "time", "radius", "emission"),
    )


RECIPES: Tuple[Recipe, ...] = (
    # -- Lines -------------------------------------------------------------------------
    Recipe(
        key="signal_trace",
        name="Signal trace",
        description="A clean line through a measured signal, coloured chalk-on-slate.",
        category="Lines",
        kind="line",
        slots=_xy("time", "signal", "X (e.g. time)", "Y (the signal)") + (_colour(),),
        style="chalk",
        width=1.8,
        example="01_line_plot.py",
        title="Signal trace",
        sample=_signal,
        sample_rows=4000,
        tags=("oscilloscope", "time series", "waveform"),
    ),
    Recipe(
        key="step_response",
        name="Step response",
        description="A staircase that holds each value until the next sample.",
        category="Lines",
        kind="step",
        slots=_xy("time", "level", "X (e.g. time)", "Y (the level)"),
        style="blueprint",
        options={"where": "mid"},
        width=2.0,
        example="06_signal_tools.py",
        title="Step response",
        sample=_step,
        sample_rows=48,
        tags=("digital", "quantised", "control"),
    ),
    Recipe(
        key="event_stems",
        name="Event stems",
        description="A vertical stem per event, rising from the baseline.",
        category="Lines",
        kind="stem",
        slots=_xy("time", "amplitude", "X (when)", "Y (how big)"),
        style="journal",
        width=1.4,
        example="06_signal_tools.py",
        title="Events",
        sample=_stems,
        sample_rows=36,
        tags=("spikes", "impulses", "discrete"),
    ),
    Recipe(
        key="filled_area",
        name="Filled area",
        description="The area under a curve, filled down to the baseline.",
        category="Lines",
        kind="area",
        slots=_xy("hour", "load", "X (e.g. hour)", "Y (the quantity)"),
        style="ocean",
        example="02_scatter_fill.py",
        title="Daily load",
        sample=_area,
        sample_rows=240,
        tags=("fill", "load", "profile"),
    ),
    Recipe(
        key="line_family",
        name="Line family",
        description="One line per row (slope and intercept): the density of lines is the picture.",
        category="Lines",
        kind="line_family",
        slots=(
            Slot(
                "x",
                "Slope",
                hint="Each row is a line y = a*x + b; this is a.",
                sample="slope",
                aliases=("slope", "a", "m"),
            ),
            Slot(
                "y",
                "Intercept",
                hint="...and this is b.",
                sample="intercept",
                aliases=("intercept", "b", "offset"),
            ),
        ),
        style="neon",
        options={"x_lo": -2.0, "x_hi": 2.0},
        axes=("=x", "=y"),
        width=1.0,
        layer_style={"alpha": 0.5},
        example="04_line_family_density.py",
        title="A family of 3,000 lines",
        sample=_line_family,
        sample_rows=3000,
        tags=("density", "many lines", "regression"),
    ),
    # -- Scatter -----------------------------------------------------------------------
    Recipe(
        key="starfield",
        name="Colour-mapped scatter",
        description="Every point coloured by a third column: structure you can see at a glance.",
        category="Scatter",
        kind="scatter",
        slots=_xy("x_pc", "y_pc")
        + (
            Slot(
                "c",
                "Colour by",
                hint="A column mapped through the colormap.",
                sample="age",
                aliases=("c", "color", "value", "age", "time"),
            ),
            Slot(
                "s",
                "Size by",
                required=False,
                hint="Optional: scale each marker by this column.",
                aliases=("s", "size", "weight"),
            ),
        ),
        style="synthwave",
        size=2.6,
        layer_style={"alpha": 0.8},
        example="02_scatter_fill.py",
        title="A spiral of 60,000 stars",
        sample=_starfield,
        sample_rows=60000,
        tags=("colormap", "turbo", "spiral", "points"),
    ),
    Recipe(
        key="bubble_chart",
        name="Bubble chart",
        description="Position, size and colour: three columns on one plot.",
        category="Scatter",
        kind="scatter",
        slots=_xy("income", "life_expectancy")
        + (
            Slot(
                "s",
                "Size by",
                hint="Marker size follows this column.",
                sample="population",
                aliases=("s", "size", "population", "weight"),
            ),
            _colour("population", "Optional: colour by this column too."),
        ),
        style="candy",
        size=10.0,
        layer_style={"alpha": 0.85},
        example="02_scatter_fill.py",
        title="Income and life expectancy",
        sample=_bubbles,
        sample_rows=90,
        tags=("bubbles", "gapminder", "size"),
    ),
    # -- Density -----------------------------------------------------------------------
    Recipe(
        key="hist2d_heat",
        name="2-D density heat map",
        description="Bin a cloud of points onto a grid and colour each cell by its count.",
        category="Density",
        kind="hist2d",
        slots=_xy("x", "y"),
        style="sunset",
        options={"bins": 90},
        example="10_massive_hist2d_density.py",
        title="Where the points pile up",
        sample=_blobs,
        sample_rows=200000,
        tags=("heatmap", "histogram", "bins"),
    ),
    Recipe(
        key="hexbin_density",
        name="Hexagon density",
        description="Hexagonal bins: the honest alternative to a square grid for dense points.",
        category="Density",
        kind="hexbin",
        slots=_xy("x", "y"),
        style="ocean",
        options={"gridsize": 34},
        example="10_massive_hist2d_density.py",
        title="Hexagonal bins",
        sample=_blobs,
        sample_rows=120000,
        tags=("hexbin", "hexagon", "bins"),
    ),
    Recipe(
        key="histogram",
        name="Histogram",
        description="How often each range of values occurs, as bars.",
        category="Density",
        kind="hist",
        slots=(
            Slot(
                "y",
                "Values",
                hint="The column to count.",
                sample="value",
                aliases=("value", "values", "y", "x"),
            ),
        ),
        style="vintage",
        options={"bins": 48},
        axes=("y", "=count"),
        example="03_bar_hist.py",
        title="Distribution",
        sample=_bimodal,
        sample_rows=50000,
        tags=("distribution", "counts", "frequency"),
    ),
    # -- Bars --------------------------------------------------------------------------
    Recipe(
        key="bar_chart",
        name="Bar chart",
        description="One bar per row: compare a quantity across categories.",
        category="Bars",
        kind="bar",
        slots=_xy("month", "sales", "X (category)", "Y (the quantity)"),
        style="slide",
        example="20_categorical_bar_and_spines.py",
        title="Monthly sales",
        sample=_bars,
        sample_rows=12,
        tags=("categories", "compare", "columns"),
    ),
    # -- Fields ------------------------------------------------------------------------
    Recipe(
        key="heatmap_matrix",
        name="Matrix heat map",
        description="Read the whole table as a matrix and colour every cell by its value.",
        category="Fields",
        kind="image",
        slots=(
            Slot(
                "matrix",
                "Matrix",
                hint="Every numeric column becomes a column of the image.",
                multi=True,
            ),
        ),
        style="aurora",
        axes=("=column", "=row"),
        example="09_large_matrix_heatmap.py",
        title="A matrix as an image",
        sample=_matrix,
        sample_rows=64,
        tags=("image", "grid", "correlation"),
    ),
    Recipe(
        key="scattered_field",
        name="Field from scattered samples",
        description="Interpolate (x, y, value) samples onto a smooth, coloured field.",
        category="Fields",
        kind="field",
        slots=_xyz(
            z_label="Value", z_hint="The quantity measured at each (x, y).", z_sample="intensity"
        ),
        style="dracula",
        example="11_contour_pcolormesh_field.py",
        title="Interpolated field",
        sample=_scattered_field,
        sample_rows=2600,
        tags=("interpolate", "contour", "map", "survey"),
    ),
    # -- 3D ----------------------------------------------------------------------------
    Recipe(
        key="cloud3d",
        name="3-D point cloud",
        description="A cloud of points in space, shaded by a fourth column.",
        category="3D",
        kind="scatter3d",
        slots=_xyz() + (_colour("radius", "Colour each point by this column."),),
        style="stage",
        size=2.2,
        camera=(22.0, -50.0),
        layer_style={"alpha": 0.85},
        example="07_projected_3d_cloud.py",
        title="A halo of 40,000 grains",
        sample=_cloud,
        sample_rows=60000,
        tags=("scatter 3d", "halo", "points"),
    ),
    Recipe(
        key="nebula3d",
        name="Volumetric cloud",
        description="Translucent, additive points: density becomes light.",
        category="3D",
        kind="volume3d",
        slots=_xyz() + (_colour("emission", "Colour by emission (or any column)."),),
        style="sunset",
        size=2.0,
        camera=(34.0, -42.0),
        box_aspect=(1.0, 1.0, 0.4),
        layer_style={
            "alpha": 0.9,
            "blend_mode": BlendMode.ADDITIVE,
            "depth_write": False,
            "auto_alpha": 0.42,
        },
        example="13_volumetric_nebula.py",
        title="A spiral nebula",
        sample=_nebula,
        sample_rows=120000,
        tags=("volume", "nebula", "density", "additive"),
    ),
    Recipe(
        key="surface3d",
        name="Surface",
        description="A height field z = f(x, y), triangulated and coloured by height.",
        category="3D",
        kind="surface3d",
        slots=_xyz(z_label="Height", z_sample="height", z_hint="Height at each grid point."),
        style="sunset",
        camera=(30.0, -58.0),
        example="12_surface_wireframe_bar3d.py",
        title="Interference surface",
        sample=_surface,
        sample_rows=5184,
        tags=("height field", "mesh", "landscape"),
    ),
    Recipe(
        key="wireframe3d",
        name="Wireframe",
        description="The grid lines of a surface, for seeing its shape without a fill.",
        category="3D",
        kind="wireframe3d",
        slots=_xyz(z_label="Height", z_sample="height", z_hint="Height at each grid point."),
        style="blueprint",
        options={"stride": 2},
        width=1.4,
        camera=(30.0, -58.0),
        example="12_surface_wireframe_bar3d.py",
        title="Wireframe",
        sample=_surface,
        sample_rows=3364,
        tags=("mesh", "lines", "grid"),
    ),
    Recipe(
        key="bars3d_city",
        name="3-D bars",
        description="A solid column per grid cell, as tall as its value: a skyline.",
        category="3D",
        kind="bar3d",
        slots=_xyz(
            "block_x", "block_y", "floors", z_label="Height", z_hint="How tall each bar stands."
        ),
        style="synthwave",
        camera=(34.0, -48.0),
        example="14_bar3d_hex_box_city.py",
        title="A downtown skyline",
        sample=_city,
        sample_rows=324,
        tags=("bars", "skyline", "columns"),
    ),
    Recipe(
        key="quiver3d_flow",
        name="Vector field",
        description="An arrow at every point along (u, v, w): flow, force, gradient.",
        category="3D",
        kind="quiver3d",
        slots=_xyz()
        + (
            Slot(
                "u",
                "U (x component)",
                hint="Arrow x component.",
                sample="u",
                aliases=("u", "vx", "dx"),
            ),
            Slot(
                "v",
                "V (y component)",
                hint="Arrow y component.",
                sample="v",
                aliases=("v", "vy", "dy"),
            ),
            Slot(
                "w",
                "W (z component)",
                hint="Arrow z component.",
                sample="w",
                aliases=("w", "vz", "dz"),
            ),
        ),
        style="ocean",
        options={"scale": 0.5, "head": 0.3},
        width=2.2,
        solid=True,
        layer_style={"outline_enabled": True, "outline_width": 0.6},
        camera=(26.0, -45.0),
        example="15_vector_field_3d.py",
        title="A swirling jet",
        sample=_flow,
        sample_rows=320,
        tags=("quiver", "arrows", "flow", "velocity"),
    ),
    Recipe(
        key="trajectory3d",
        name="Trajectory in space",
        description="A path through three coordinates, coloured along its length.",
        category="3D",
        kind="line3d",
        slots=_xyz() + (_colour("time", "Colour by time (or any column)."),),
        style="aurora",
        width=2.4,
        camera=(18.0, -62.0),
        example="animations/16_pes_particle_dynamics.py",
        title="A chaotic orbit",
        sample=_orbit,
        sample_rows=6000,
        tags=("path", "orbit", "attractor", "line 3d"),
    ),
    Recipe(
        key="stems3d",
        name="Stems in 3-D",
        description="A drop line from each point to the floor: heights you can read.",
        category="3D",
        kind="stem3d",
        slots=_xyz(z_label="Height", z_sample="height"),
        style="vintage",
        width=1.6,
        size=4.0,
        solid=True,
        layer_style={"outline_enabled": True, "outline_width": 0.7},
        camera=(24.0, -50.0),
        example="12_surface_wireframe_bar3d.py",
        title="Scattered heights",
        sample=_scattered_heights,
        sample_rows=110,
        tags=("lollipop", "drop lines", "samples"),
    ),
    Recipe(
        key="ribbon3d",
        name="Ribbon",
        description="A band hanging from a path to the floor: a fill under a 3-D curve.",
        category="3D",
        kind="ribbon3d",
        slots=_xyz(),
        style="dracula",
        camera=(28.0, -50.0),
        example="12_surface_wireframe_bar3d.py",
        title="A ribbon",
        sample=_ribbon,
        sample_rows=400,
        tags=("band", "fill", "curve"),
    ),
)

RECIPE_KEYS: Tuple[str, ...] = tuple(r.key for r in RECIPES)
_BY_KEY: Dict[str, Recipe] = {r.key: r for r in RECIPES}


def get_recipe(key: str) -> Recipe:
    """The recipe called ``key``. Raises ``KeyError`` on an unknown key -- never guesses."""
    return _BY_KEY[str(key)]


# ----------------------------------------------------------------------------------
# Sample tables
# ----------------------------------------------------------------------------------

#: Seed of every recipe's sample table. Fixed so a thumbnail, the sample dataset and a test
#: all see the same numbers -- and so regenerating assets is a no-op unless a recipe changed.
SAMPLE_SEED = 20240611


def sample_table(recipe: Recipe, rows: Optional[int] = None) -> Dict[str, np.ndarray]:
    """The recipe's sample data as ``{column name: float64 array}``, in column order."""
    if recipe.sample is None:
        raise RecipeError(f"recipe {recipe.key!r} has no sample data")
    rng = np.random.default_rng(SAMPLE_SEED)
    table = recipe.sample(int(rows or recipe.sample_rows), rng)
    return {name: np.ascontiguousarray(values, dtype=np.float64) for name, values in table.items()}


def sample_assignment(recipe: Recipe) -> Dict[str, str]:
    """``{slot key: sample column}`` for the slots the sample table fills."""
    return {slot.key: slot.sample for slot in recipe.slots if slot.sample and not slot.multi}


# ----------------------------------------------------------------------------------
# Assigning columns
# ----------------------------------------------------------------------------------


def assign_columns(recipe: Recipe, columns: Sequence[str]) -> Dict[str, Optional[str]]:
    """Guess which column goes in which slot. ``None`` = unassigned (optional, or no column left).

    The rule is the one a person would use, in order: a column whose name is one of the slot's
    aliases wins; then the remaining columns fill the *required* slots left to right; a recipe
    that needs two columns on a one-column table plots the column against its row number
    (:data:`INDEX`). Optional slots are filled only by an alias match -- guessing a "colour by"
    column nobody asked for would change the picture for no reason.
    """
    names = list(columns)
    lowered = {name.lower(): name for name in names}
    result: Dict[str, Optional[str]] = {}
    used: set = set()

    for slot in recipe.slots:
        if slot.multi:
            result[slot.key] = None
            continue
        match = next(
            (lowered[a] for a in slot.aliases if a in lowered and lowered[a] not in used), None
        )
        result[slot.key] = match
        if match is not None:
            used.add(match)

    free = [name for name in names if name not in used]
    for slot in recipe.slots:
        if slot.multi or not slot.required or result[slot.key] is not None:
            continue
        if free:
            result[slot.key] = free.pop(0)

    # One column and a recipe wanting x and y: it is the thing being measured, so it goes on y
    # and x becomes the row number -- whichever way the rules above happened to place it (on x
    # by position, or on y by a name like "v" with nothing left for x).
    if len(names) == 1 and "x" in result and "y" in result and recipe.slot("x").required:
        only = names[0]
        if result["x"] in (None, only) and result["y"] in (None, only):
            result["x"], result["y"] = INDEX, only
    return result


def slot_arrays(
    recipe: Recipe,
    table: Mapping[str, np.ndarray],
    assignment: Mapping[str, Optional[str]],
    n_rows: int,
) -> Dict[str, np.ndarray]:
    """The arrays :func:`apply_recipe` draws, resolved from a column assignment.

    ``table`` maps column names to their values and ``assignment`` maps slot keys to column
    names (or :data:`INDEX`, the row number). The matrix recipe ignores the assignment and takes
    every column. A slot whose column is missing from the table is left out, and the caller's
    :func:`validate_data` says which required one it was.

    The one place a gallery panel, the MCP server and a script turn "which column goes where"
    into data, so they cannot disagree about it.
    """
    if recipe.kind == "image":
        columns = list(table.values())
        return {"matrix": np.column_stack(columns)} if columns else {}
    data: Dict[str, np.ndarray] = {}
    for slot in recipe.slots:
        column = assignment.get(slot.key)
        if column is None:
            continue
        values = np.arange(int(n_rows), dtype=np.float64) if column == INDEX else table.get(column)
        if values is not None:
            data[slot.key] = values
    return data


def slot_labels(recipe: Recipe, assignment: Mapping[str, Optional[str]]) -> Dict[str, str]:
    """Slot key -> the column name it came from (``"row"`` for :data:`INDEX`), for axis titles."""
    names: Dict[str, str] = {}
    for slot in recipe.slots:
        column = assignment.get(slot.key)
        if column:
            names[slot.key] = "row" if column == INDEX else column
    return names


# ----------------------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------------------


def _finite_rows(arrays: Mapping[str, np.ndarray]) -> np.ndarray:
    """Boolean mask of the rows where every array is finite."""
    mask: Optional[np.ndarray] = None
    for values in arrays.values():
        finite = np.isfinite(values)
        mask = finite if mask is None else (mask & finite)
    return mask if mask is not None else np.ones(0, dtype=bool)


def prepare_data(recipe: Recipe, data: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Check ``data`` and return the arrays :func:`apply_recipe` draws, rows with a gap dropped.

    Raises :class:`RecipeError` with a message fit for the panel. A row with a ``nan`` in any
    *used* slot is dropped for the whole recipe (a gap cannot be drawn through by most of
    these kinds, and dropping it from one column but not another would misalign them).
    """
    if recipe.kind == "image":
        matrix = data.get("matrix")
        if matrix is None or np.ndim(matrix) != 2 or min(np.shape(matrix)) < 2:
            raise RecipeError("A matrix needs at least 2 rows and 2 columns of numbers.")
        return {"matrix": np.asarray(matrix, dtype=np.float64)}

    used = {slot.key: data[slot.key] for slot in recipe.slots if data.get(slot.key) is not None}
    missing = [slot.label for slot in recipe.required_slots() if slot.key not in used]
    if missing:
        raise RecipeError("Choose a column for: " + ", ".join(missing) + ".")

    arrays = {k: np.asarray(v, dtype=np.float64).ravel() for k, v in used.items()}
    lengths = {k: a.size for k, a in arrays.items()}
    if len(set(lengths.values())) > 1:
        detail = ", ".join(f"{k}: {n}" for k, n in lengths.items())
        raise RecipeError(f"The chosen columns have different lengths ({detail}).")
    n = next(iter(lengths.values()))
    if n == 0:
        raise RecipeError("The chosen columns are empty.")
    if n > MAX_RECIPE_ROWS:
        raise RecipeError(f"{n:,} rows is more than the {MAX_RECIPE_ROWS:,} a recipe will draw.")

    keep = _finite_rows(arrays)
    if not bool(keep.any()):
        raise RecipeError("Every row has a missing (NaN) value in the chosen columns.")
    if not bool(keep.all()):
        arrays = {k: a[keep] for k, a in arrays.items()}

    rows = int(next(iter(arrays.values())).size)
    minimum = {"hist2d": 10, "hexbin": 10, "boxplot": 5}.get(recipe.kind, 2)
    if rows < minimum:
        raise RecipeError(f"{recipe.name} needs at least {minimum} rows; {rows} remain.")

    if recipe.kind in ("surface3d", "wireframe3d"):
        try:
            layerops3d.resolve_grid(arrays["x"], arrays["y"], arrays["z"], {})
        except layerops3d.Grid3DError as exc:
            raise RecipeError(
                f"{recipe.name} needs x and y on a full grid (every combination once): {exc}"
            ) from exc
    if recipe.kind == "quiver3d" and rows > layerops3d.MAX_ARROWS:
        raise RecipeError(f"A vector field draws at most {layerops3d.MAX_ARROWS:,} arrows.")
    if recipe.kind == "field":
        if rows < 4:
            raise RecipeError("A field needs at least 4 scattered samples to interpolate.")
        if float(np.ptp(arrays["x"])) == 0.0 or float(np.ptp(arrays["y"])) == 0.0:
            raise RecipeError("The x and y samples must each cover a range, not a single value.")
    return arrays


def validate_data(recipe: Recipe, data: Mapping[str, np.ndarray]) -> Optional[str]:
    """``None`` if ``data`` can be drawn, else the reason, for display next to the Apply button."""
    try:
        prepare_data(recipe, data)
    except RecipeError as exc:
        return str(exc)
    return None


# ----------------------------------------------------------------------------------
# Looks
# ----------------------------------------------------------------------------------


def resolve_look(plot: Any, recipe: Recipe, look: str) -> Tuple[styles.PlotStyle, bool]:
    """``(style, page)``: the style whose palette/colormap colour the layers, and whether to
    also apply it to the page.

    ``look`` is :data:`RECIPE_LOOK` (the recipe's own style, applied to the page), :data:`KEEP_LOOK`
    (the figure's current style for the colours, the page untouched) or a preset key (that
    preset, applied to the page).
    """
    if look == KEEP_LOOK:
        current = str(getattr(plot, "_style_key", "") or "")
        try:
            return styles.get_style(current), False
        except KeyError:
            return styles.default_style(), False
    if look == RECIPE_LOOK or not look:
        return styles.get_style(recipe.style), True
    return styles.get_style(look), True


# ----------------------------------------------------------------------------------
# Applying
# ----------------------------------------------------------------------------------


def _grid_field(
    x: np.ndarray, y: np.ndarray, z: np.ndarray
) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
    """Interpolate scattered ``(x, y, z)`` onto a regular ``FIELD_RESOLUTION`` grid.

    Linear inside the convex hull of the samples, nearest outside it, so the image has no holes
    and no extrapolated overshoot. ``scipy`` is a hard dependency (``pyproject.toml``).
    """
    from scipy.interpolate import griddata

    n = FIELD_RESOLUTION
    x0, x1 = float(x.min()), float(x.max())
    y0, y1 = float(y.min()), float(y.max())
    gx, gy = np.meshgrid(np.linspace(x0, x1, n), np.linspace(y0, y1, n))
    points = np.column_stack([x, y])
    grid = griddata(points, z, (gx, gy), method="linear")
    holes = ~np.isfinite(grid)
    if holes.any():
        grid[holes] = griddata(points, z, (gx[holes], gy[holes]), method="nearest")
    return grid, (x0, x1, y0, y1)


def _size_pixels(values: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Rescale a column into the marker-size band ``[lo, hi]`` pixels, as the Data panel does."""
    finite = values[np.isfinite(values)]
    if finite.size == 0 or float(np.ptp(finite)) < 1e-12:
        return np.full(values.shape, 0.5 * (lo + hi), dtype=np.float32)
    vmin, vmax = float(finite.min()), float(finite.max())
    unit = np.clip((values - vmin) / (vmax - vmin), 0.0, 1.0)
    return (lo + unit * (hi - lo)).astype(np.float32)


def _data_layer_count(plot: Any) -> int:
    return sum(
        1
        for layer in getattr(plot.scene, "layers", [])
        if layer.metadata.get("artist") not in SYSTEM_3D_ARTISTS
    )


def _short(name: str, limit: int = 10) -> str:
    """``name`` cut to ``limit`` characters for an axis title that must fit beside the plot."""
    name = str(name)
    return name if len(name) <= limit else name[: limit - 1] + "\u2026"


def axis_names(recipe: Recipe, columns: Mapping[str, str]) -> Dict[str, str]:
    """The axis titles for ``recipe`` given ``{slot key: column name}``.

    Resolves :attr:`Recipe.axes` -- a slot's column name, or a fixed ``"=text"`` -- and, for a
    3-D recipe, the ``z`` title from its ``z`` slot.
    """
    names: Dict[str, str] = {}
    for axis, source in zip(("x", "y"), recipe.axes, strict=True):
        if source.startswith("="):
            names[axis] = source[1:]
        elif columns.get(source):
            names[axis] = str(columns[source])
    if recipe.is_3d:
        for axis in ("x", "y", "z"):
            if columns.get(axis):
                names[axis] = str(columns[axis])
    return names


def _set_labels(plot: Any, names: Mapping[str, str], title: Optional[str]) -> None:
    """Title and axis names (3-D names go to the axis box, shortened to fit beside it)."""
    if title:
        plot.set_title(title)
    if plot.is_3d_scene():
        axes = plot.axes3d
        for key in ("x", "y", "z"):
            if names.get(key) and hasattr(axes, f"{key}label"):
                setattr(axes, f"{key}label", _short(names[key]))
    else:
        if names.get("x"):
            plot.xlabel = str(names["x"])
        if names.get("y"):
            plot.ylabel = str(names["y"])


def apply_recipe(
    plot: Any,
    recipe: Recipe,
    data: Mapping[str, np.ndarray],
    *,
    name: str = "",
    look: str = RECIPE_LOOK,
    labels: Optional[Mapping[str, str]] = None,
    title: Optional[str] = None,
    frame: bool = True,
) -> List[Any]:
    """Draw ``recipe`` from ``data`` onto ``plot`` and return the layers it created.

    ``data`` maps slot keys to equal-length 1-D arrays (the matrix recipe takes ``"matrix"``,
    a 2-D array); ``labels`` maps the same slot keys to the column names they came from, which
    become the axis titles. Raises :class:`RecipeError` for data it cannot draw, *before*
    touching the scene, so a refusal leaves nothing half built.

    **Queued commands only.** On a figure that is not live yet (a thumbnail being built, a
    headless preview) the scene is not being drawn, so it may be called directly.
    """
    arrays = prepare_data(recipe, data)
    style, page = resolve_look(plot, recipe, look)
    palette = style.palette or (style.ink,)
    color = palette[_data_layer_count(plot) % len(palette)]
    cmap = colormaps_module.resolve(recipe.cmap or style.cmap)
    label = f"{name}: {recipe.name}" if name else recipe.name
    width = float(recipe.width if recipe.width is not None else style.line_width)
    size = float(recipe.size if recipe.size is not None else style.point_size)

    layers_before = {id(layer) for layer in plot.scene.layers}
    kind = recipe.kind
    options = dict(recipe.options)

    if page:
        styles.apply_style(plot, style, layers=False)

    if kind == "image":
        matrix = arrays["matrix"]
        rows, cols = matrix.shape
        layerops.add_image_layer(
            plot, matrix, (0.0, float(cols), 0.0, float(rows)), label=label, cmap=cmap
        )
    elif kind == "field":
        grid, extent = _grid_field(arrays["x"], arrays["y"], arrays["z"])
        layerops.add_image_layer(plot, grid, extent, label=label, cmap=cmap)
    elif recipe.is_3d:
        if not plot.active_panel.is_3d():
            plot.set_ndim(3)
        c = arrays.get("c")
        s = arrays.get("s")
        if kind == "quiver3d":
            options.update({"u": arrays.get("u"), "v": arrays.get("v"), "w": arrays.get("w")})
        layerops3d.add_xyz_layer(
            plot,
            arrays["x"],
            arrays["y"],
            arrays["z"],
            kind=kind,
            label=label,
            color=color,
            width=width,
            size=size,
            options=options,
            c=c,
            s=None if s is None else _size_pixels(s, 2.0, 14.0),
            cmap=cmap,
            colormap_by_z=not recipe.solid,
        )
    else:
        y = arrays["y"]
        # ``hist``/``boxplot`` read only y; the layer builder still wants an x of the same length.
        x = arrays["x"] if "x" in arrays else np.arange(y.size, dtype=np.float64)
        c, s = arrays.get("c"), arrays.get("s")
        if "cmap" in layerops.default_kind_options(kind):
            # The colormapped kinds (hist2d, hexbin) keep their ramp in their options.
            options.setdefault("cmap", cmap)
        layerops.add_xy_layer(
            plot,
            x,
            y,
            kind=kind,
            label=label,
            color=color,
            width=width,
            size=size,
            options=options,
            c=c,
            s=None if s is None else _size_pixels(s, 4.0, 22.0),
            cmap=cmap,
        )

    # The 3-D box, floor, grid and ticks are layers too, and the engine rebuilds them whenever a
    # 3-D layer is added: they are not the recipe's, so they must not receive its styling (an
    # outline or additive blend meant for the data would fatten and fog the axis box) and undo
    # must not remove them -- the engine owns them.
    created = [
        layer
        for layer in plot.scene.layers
        if id(layer) not in layers_before and layer.metadata.get("artist") not in SYSTEM_3D_ARTISTS
    ]
    if recipe.layer_style:
        extra = dict(recipe.layer_style)
        if extra.get("outline_enabled") and "outline_color" not in extra:
            # 3-D lines are one pixel wide in GL whatever ``width`` says; the way to widen one is
            # an outline in its own colour (see ``Recipe.layer_style`` in the catalogue).
            extra["outline_color"] = (color[0], color[1], color[2], 1.0)
        for layer in created:
            layerops.set_layer_style(plot, layer, **extra)
    if style.hand_drawn:
        styles.hand_drawn_layers(plot, style)

    _set_labels(plot, axis_names(recipe, labels or {}), title)

    if frame:
        if recipe.is_3d:
            if recipe.camera is not None:
                plot.set_3d_view(elev=recipe.camera[0], azim=recipe.camera[1])
            if recipe.box_aspect is not None:
                plot.set_3d_view(box_aspect=recipe.box_aspect)
            plot.frame_3d_view()
        else:
            plot.autoscale()
    layerops.mark_scene_dirty(plot)
    return created


# ----------------------------------------------------------------------------------
# Undo
# ----------------------------------------------------------------------------------


def _capture_labels(plot: Any) -> Dict[str, Any]:
    axes = plot.axes3d
    return {
        "title": getattr(plot, "title", ""),
        "options_title": getattr(plot.options, "title", None),
        "panel_title": getattr(plot.active_panel, "title", ""),
        "xlabel": plot.xlabel,
        "ylabel": plot.ylabel,
        "axes3d": {k: getattr(axes, f"{k}label", None) for k in ("x", "y", "z")},
        "ndim": plot.active_panel.ndim,
        "camera": (plot.camera3d.elev, plot.camera3d.azim),
        "box_aspect": plot.camera3d.box_aspect,  # None = the default cube
    }


def _restore_labels(plot: Any, state: Mapping[str, Any]) -> None:
    plot.title = state["title"]
    if state["options_title"] is not None:
        plot.options.title = state["options_title"]
    plot.active_panel.title = state["panel_title"]
    plot.xlabel = state["xlabel"]
    plot.ylabel = state["ylabel"]
    axes = plot.axes3d
    for key, value in state["axes3d"].items():
        if value is not None and hasattr(axes, f"{key}label"):
            setattr(axes, f"{key}label", value)
    panel = plot.active_panel
    previous = state["ndim"]
    still_3d_data = any(
        layer.metadata.get("artist") not in SYSTEM_3D_ARTISTS
        and str(getattr(layer, "layer_type", "")).endswith("3d")
        for layer in plot.scene.layers
    )
    if previous is None and not still_3d_data:
        # "Infer from the layers" is not enough on its own: the 3-D box the recipe's layer
        # brought in is itself made of 3-D layers, so the panel would infer 3-D from the very
        # decoration it should be rid of. Pin to 2-D (which strips the box), then go back to
        # inferring.
        plot.set_ndim(2)
        panel.set_ndim(None)
    else:
        plot.set_ndim(previous)
    # The camera, but only where there is a 3-D view to put it back on: ``set_3d_view`` also
    # rebuilds the axis box, which on a 2-D figure would bring the box straight back.
    elev, azim = state["camera"]
    # ``set_3d_view(box_aspect=None)`` means "leave it", not "back to the cube", so the saved
    # value is written straight onto the camera and the view re-synced from it.
    plot.camera3d.box_aspect = state["box_aspect"]
    if panel.is_3d():
        plot.set_3d_view(elev=elev, azim=azim)
    else:
        plot.camera3d.elev, plot.camera3d.azim = elev, azim
    plot.frame.dirty_ui = True


def recipe_command(
    plot: Any,
    hud: Any,
    recipe: Recipe,
    data: Mapping[str, np.ndarray],
    *,
    name: str = "",
    look: str = RECIPE_LOOK,
    labels: Optional[Mapping[str, str]] = None,
    title: Optional[str] = None,
) -> Command:
    """One undoable "draw this recipe", for ``Panel.push_command``.

    The scene is captured when ``do`` first *runs* (inside the drain), not when the command is
    built (inside a draw callback), exactly as :func:`glplot.gui.styles.style_command` does.
    Undo removes the layers the recipe created and puts the page, the axis names and the
    2-D/3-D mode back as they were; layers the user already had are never touched.
    """
    saved: Dict[str, Any] = {}
    created: List[Any] = []

    def do() -> None:
        if not saved:
            saved["style"] = styles.capture_state(plot, layers=False)
            saved["labels"] = _capture_labels(plot)
        created.clear()
        created.extend(
            apply_recipe(plot, recipe, data, name=name, look=look, labels=labels, title=title)
        )

    def undo() -> None:
        for layer in created:
            layerops.remove_layer(plot, hud, layer)
        created.clear()
        if saved:
            styles.restore_state(plot, saved["style"])
            _restore_labels(plot, saved["labels"])
        layerops.mark_scene_dirty(plot)

    return Command(label=f"Recipe: {recipe.name}", do=do, undo=undo)


# ----------------------------------------------------------------------------------
# Rendering a recipe without a window
# ----------------------------------------------------------------------------------

#: matplotlib's pyplot state is process-global and not thread-safe; the headless renders (live
#: previews on a worker thread, a test) take turns.
_RENDER_LOCK = threading.Lock()


def sample_data(recipe: Recipe) -> Dict[str, np.ndarray]:
    table = sample_table(recipe)
    if recipe.kind == "image":
        return {"matrix": np.column_stack(list(table.values()))}
    return {slot: table[column] for slot, column in sample_assignment(recipe).items()}


def build_preview_plot(
    recipe: Recipe,
    *,
    width: int = 480,
    height: int = 300,
    new_plot: Optional[Callable[[], Any]] = None,
    data: Optional[Mapping[str, np.ndarray]] = None,
    look: str = RECIPE_LOOK,
    labels: Optional[Mapping[str, str]] = None,
    title: Optional[str] = None,
) -> Any:
    """A windowless figure with ``recipe`` drawn on it: from its sample table, or ``data``.

    The one function the gallery thumbnails (``tools/build_gallery_assets.py``, which hands it a
    ``new_plot`` that creates the figure it is about to capture) and the in-app live preview
    both go through, so a card and the preview of the same recipe cannot drift apart.
    """
    if new_plot is None:
        from ..engine import GPULinePlot

        def new_plot() -> Any:  # type: ignore[misc]
            return GPULinePlot(width=width, height=height, title="GLPlot")

    plot = new_plot()
    if data is None:
        data = sample_data(recipe)
        labels = labels if labels is not None else sample_labels(recipe)
        title = recipe.title if title is None else title
    apply_recipe(plot, recipe, data, look=look, labels=labels, title=title)
    return plot


def sample_labels(recipe: Recipe) -> Dict[str, str]:
    """Slot -> column names for the sample's axis titles.

    A 3-D thumbnail uses the bare ``x``/``y``/``z``: its titles sit beside the box, in a
    480-pixel window, and a descriptive column name would run off the edge of it.
    """
    if recipe.is_3d:
        return {"x": "x", "y": "y", "z": "z"}
    return dict(sample_assignment(recipe))


def render_preview_rgb(
    recipe: Recipe,
    data: Optional[Mapping[str, np.ndarray]] = None,
    *,
    look: str = RECIPE_LOOK,
    labels: Optional[Mapping[str, str]] = None,
    title: Optional[str] = None,
    width: int = 480,
    height: int = 300,
) -> np.ndarray:
    """Render ``recipe`` (with ``data``, or its sample) through the headless export.

    Returns an ``(H, W, 3)`` uint8 array. This is the matplotlib-based export, not the OpenGL
    window, so it is a *preview* of the data's shape and the look's palette -- fine for "does
    this mapping make sense" and not a substitute for the real render the Apply button draws.
    """
    from ..utils.preview import render_preview_array

    with _RENDER_LOCK:
        plot = build_preview_plot(
            recipe, width=width, height=height, data=data, look=look, labels=labels, title=title
        )
        # The layer carries the recipe's name so the Scene panel can list it, and the export
        # reads any explicit label as "put me in the legend": a one-entry legend naming the
        # recipe, on top of the plot, in a preview whose title already says what it is. Marking
        # it a bookkeeping label (as the engine does for unlabeled layers) keeps it out of the
        # legend on this throwaway figure only.
        for layer in plot.scene.layers:
            layer.metadata["auto_label"] = True
        return render_preview_array(plot, 1.0)
