"""Live-scene mutation helpers — the only module allowed to touch ``plot.scene``.

Every function here assumes it is running **inside a queued command** (see
``glplot.gui.commands.CommandQueue``), i.e. at the top of ``_main_loop`` on the GL
thread. Calling any of them from an imgui draw callback loses the dirty-flag latch
(``engine.py:1238-1239`` clears it later in the same frame) and, for anything that
releases GPU handles, issues GL calls off the GL thread.

The module imports cleanly headless and deliberately does not import imgui at all. The
only GL it performs is ``glDelete*``, which is both import-guarded and gated on a proven
current context — with no context those calls segfault rather than raise, so they are
skipped entirely (see ``_has_gl_context``) and a context-less test leaks instead of dying.
"""

from __future__ import annotations

import copy
import logging
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from ..core.layers import FunctionLayer, LayerDirtyState
from ..core.legacy import LineDataset
from ..utils.shaders import DENSITY_SCHEMES

if TYPE_CHECKING:
    from ..engine import GPULinePlot

try:
    from OpenGL import platform as _gl_platform
    from OpenGL.GL import (
        glDeleteBuffers,
        glDeleteFramebuffers,
        glDeleteRenderbuffers,
        glDeleteTextures,
        glDeleteVertexArrays,
    )

    GL_AVAILABLE = True
except (ImportError, Exception):  # pragma: no cover - PyOpenGL missing or GL-less host
    GL_AVAILABLE = False
    _gl_platform = None
    glDeleteBuffers = None
    glDeleteFramebuffers = None
    glDeleteRenderbuffers = None
    glDeleteTextures = None
    glDeleteVertexArrays = None

logger = logging.getLogger(__name__)

#: Fallback RGBA for layers created without an explicit colour. Matches the
#: ``geometry3d.py:130`` fallback rather than ``add_line_strip``'s black default,
#: which is invisible on the default dark background.
DEFAULT_LAYER_COLOR: Tuple[float, float, float, float] = (0.1, 0.45, 1.0, 1.0)

#: Fields that live on the layer rather than on ``layer.style``.
_LAYER_LEVEL_FIELDS = frozenset({"label", "translation"})

#: ``_gl`` attribute names, by the GL object kind each holds.
_VAO_NAMES = frozenset({"vao"})
#: Buffer handles whose names the `vbo` prefix does not catch. `cbo` is the patch
#: renderer's per-vertex colour buffer (`renderers.patch.GLPatchBuffers`); a name missing
#: from here is not a cosmetic omission -- `_iter_gl_handles` would never yield it and the
#: buffer would outlive every layer that ever held one.
_BUFFER_NAMES = frozenset({"ebo", "cbo"})
_BUFFER_PREFIXES = ("vbo",)
_TEXTURE_NAMES = frozenset({"tex", "texture"})
_FRAMEBUFFER_NAMES = frozenset({"fbo"})
_RENDERBUFFER_SUFFIXES = ("rbo",)

#: Array attributes copied when duplicating a layer, so the clone is independent.
_DATA_ARRAY_FIELDS = ("pts", "ab", "colors", "vertices", "indices")


def _index_of(seq: Sequence[Any], obj: Any) -> int:
    """Identity-based ``list.index``. Returns -1 if absent.

    ``list.remove``/``.index``/``in`` are **unusable** on ``scene.layers`` and on the
    legacy mirrors. They compare with ``==`` against every element up to the match, and
    the layer/mirror classes are dataclasses holding ndarrays, so their generated
    ``__eq__`` returns an *array* and the comparison raises
    ``ValueError: truth value of an array ... is ambiguous`` as soon as an earlier
    element is of the same class. (Different classes are safe only by accident:
    dataclass ``__eq__`` returns NotImplemented and Python falls back to identity.)

    Layers are identities, never values, so identity is also the semantically right
    test: two scatter layers with equal arrays are still two distinct layers.
    """
    for index, item in enumerate(seq):
        if item is obj:
            return index
    return -1


def _remove_identity(seq: List[Any], obj: Any) -> bool:
    """Remove ``obj`` from ``seq`` by identity. Returns True if it was there."""
    index = _index_of(seq, obj)
    if index < 0:
        return False
    del seq[index]
    return True


# ----------------------------------------------------------------------------------
# Dirty flags
# ----------------------------------------------------------------------------------


def mark_scene_dirty(plot: GPULinePlot) -> None:
    """The CONTRACT §1.4 incantation, verbatim.

    ``capture_window = None`` kills the stale interaction impostor, which otherwise
    keeps drawing the pre-mutation scene; ``refresh_requested`` is the one flag the
    engine does not clear at ``engine.py:1238``, so it always survives to rebuild it.
    """
    plot.frame.dirty_scene = True
    plot.cache.capture_window = None
    plot.cache.refresh_requested = True


# ----------------------------------------------------------------------------------
# GPU teardown
# ----------------------------------------------------------------------------------


def _iter_gl_handles(gl_obj: Any) -> Iterator[Tuple[str, str, int]]:
    """Yield ``(kind, name, handle)`` for every non-zero GL handle held by ``gl_obj``.

    Handles the two shapes the renderers actually produce: the ``GL*Buffers``
    dataclasses/objects (``vao``/``vbo_*``/``ebo``, plus non-handle fields such as
    ``count``, ``size``, ``has_color`` that must not be deleted) and the plain dict
    used by the imshow quad (``scatter.py:236``: ``vao``/``vbo``/``ebo``/``tex``).
    """
    if isinstance(gl_obj, dict):
        items = list(gl_obj.items())
    else:
        namespace = getattr(gl_obj, "__dict__", None)
        if not namespace:
            return
        items = list(namespace.items())

    for name, value in items:
        # bool is an int subclass: has_color=True would otherwise read as handle 1.
        if isinstance(value, bool) or not isinstance(value, int) or value == 0:
            continue
        if name in _VAO_NAMES:
            yield ("vao", name, value)
        elif name in _BUFFER_NAMES or name.startswith(_BUFFER_PREFIXES):
            yield ("buffer", name, value)
        elif name in _TEXTURE_NAMES:
            yield ("texture", name, value)
        elif name in _FRAMEBUFFER_NAMES:
            yield ("framebuffer", name, value)
        elif name.endswith(_RENDERBUFFER_SUFFIXES):
            yield ("renderbuffer", name, value)


def _zero_handle(gl_obj: Any, name: str) -> None:
    """Reset a released handle to 0 so a second release cannot double-delete it.

    GL recycles names aggressively: deleting a stale handle a second time can destroy
    a *different* layer's buffer. Zeroing makes ``release_gl`` idempotent.
    """
    try:
        if isinstance(gl_obj, dict):
            gl_obj[name] = 0
        else:
            setattr(gl_obj, name, 0)
    except Exception:  # pragma: no cover - frozen/exotic buffer object
        pass


def _has_gl_context() -> bool:
    """True only if a GL context is provably current on this thread.

    ``glDelete*`` with no current context does not raise — it **segfaults the process**
    (verified: PyOpenGL exits 139 with no traceback), so try/except cannot make it safe
    and the check has to happen up front. ``PLATFORM.GetCurrentContext()`` is PyOpenGL's
    own probe (it backs the contextdata cache), returns 0 when there is no context, and
    needs neither glfw nor an initialized window system.

    Deliberately conservative: if the probe cannot answer, assume no context and leak.
    A leaked VAO in a headless test is recoverable; a SIGSEGV in the user's process
    is not.
    """
    if not GL_AVAILABLE or _gl_platform is None:
        return False
    try:
        get_current = getattr(_gl_platform.PLATFORM, "GetCurrentContext", None)
        if get_current is None:
            return False
        return bool(get_current())
    except Exception:  # pragma: no cover - exotic PyOpenGL platform plugin
        return False


def release_gl(gl_obj: Any) -> None:
    """Delete every GL handle held by ``gl_obj``. **GL thread only** (i.e. in the drain).

    Nothing in the engine ever calls ``glDelete*`` for layer buffers, so dropping a
    layer without this leaks its VAO/VBOs for the process lifetime.

    Safe to call twice (released handles are zeroed) and safe with no GL context — with
    no context it deletes *nothing* rather than crashing (see ``_has_gl_context``), so a
    headless test that fabricates a ``_gl`` leaks instead of taking down pytest.
    """
    if gl_obj is None or not GL_AVAILABLE:
        return
    if not _has_gl_context():
        logger.debug("release_gl: no current GL context; skipping delete (handles leak)")
        return

    deleters = {
        "vao": glDeleteVertexArrays,
        "buffer": glDeleteBuffers,
        "texture": glDeleteTextures,
        "framebuffer": glDeleteFramebuffers,
        "renderbuffer": glDeleteRenderbuffers,
    }
    for kind, name, handle in list(_iter_gl_handles(gl_obj)):
        deleter = deleters.get(kind)
        if deleter is None:  # pragma: no cover - defensive
            continue
        try:
            deleter(1, [handle])
        except Exception:
            # No context (headless teardown) or an already-dead handle. Zero it anyway:
            # retrying later is more dangerous than leaking, because the name may
            # have been recycled by then.
            logger.debug("release_gl: %s %s=%s could not be deleted", kind, name, handle)
        _zero_handle(gl_obj, name)


def _release_object_gl(obj: Any) -> None:
    """Release ``_gl`` and ``_image_gl`` on a layer or a legacy mirror, then unbind them.

    ``_image_gl`` is the imshow textured-quad path (``scatter.py:240``); it owns a
    texture on top of the usual VAO/VBO/EBO and leaks just as hard.
    """
    for attr in ("_gl", "_image_gl"):
        handle = getattr(obj, attr, None)
        if handle is None:
            continue
        release_gl(handle)
        try:
            setattr(obj, attr, None)
        except Exception:  # pragma: no cover - defensive
            pass


# ----------------------------------------------------------------------------------
# Legacy mirrors
# ----------------------------------------------------------------------------------


def _purge_mirror_list(mirrors: Optional[List[Any]], array: Optional[np.ndarray]) -> None:
    """Drop mirrors whose ``pts`` *is* ``array``, releasing any GL they picked up.

    Identity is the only available link: ``add_scatter``/``add_line_strip`` hand the
    same ndarray object to both the layer and the mirror. ``update_layer_xy`` keeps
    that link intact by re-pointing the mirror when it swaps the layer's array.
    """
    if not mirrors or array is None:
        return
    for mirror in [m for m in mirrors if getattr(m, "pts", None) is array]:
        _release_object_gl(mirror)
        _remove_identity(mirrors, mirror)


def _purge_legacy_mirrors(plot: GPULinePlot, layer: Any) -> None:
    """Drop the write-only ``scene.scatters``/``strips``/``texts``/``lines`` entries.

    No renderer reads these (``ExactLineRenderer.draw``, their only consumer, is never
    called), but they pin the layer's arrays against GC forever — for a 1e6-point
    scatter that is tens of MB retained per "deleted" layer.
    """
    scene = plot.scene
    layer_type = getattr(layer, "layer_type", "")

    if layer_type == "scatter":
        _purge_mirror_list(getattr(scene, "scatters", None), getattr(layer, "pts", None))
    elif layer_type == "polyline":
        _purge_mirror_list(getattr(scene, "strips", None), getattr(layer, "pts", None))
    elif layer_type == "text":
        texts = getattr(scene, "texts", None)
        if texts:
            for index, entry in enumerate(texts):
                # Matched on the identifying scalars only: "color" may be an ndarray,
                # which would make a whole-dict == raise.
                if (
                    entry.get("x") == getattr(layer, "x", None)
                    and entry.get("y") == getattr(layer, "y", None)
                    and entry.get("str") == getattr(layer, "text", None)
                ):
                    del texts[index]
                    break
    elif layer_type == "line_family":
        # scene.lines is the line family's mirror. It still feeds
        # _get_adaptive_alpha(scene.lines.count) (engine.py:941,1030,1672), so leaving
        # it populated makes a removed layer keep dimming the survivors.
        if getattr(plot, "_primary_line_layer", None) is layer:
            scene.lines = LineDataset()
            plot._cpu_line_copy = None


# ----------------------------------------------------------------------------------
# Selection
# ----------------------------------------------------------------------------------


def _clear_dangling_selection(plot: GPULinePlot, hud: Any, layer: Any) -> None:
    """Drop every reference to ``layer`` held by the selection/pick state.

    Dangling ids do not crash (the lookups are ``next(..., None)``-guarded) but they
    silently misbehave, and ``picked_info["layer"]`` is a hard reference that would
    keep the whole layer alive.
    """
    layer_id = getattr(layer, "layer_id", None)

    interaction = getattr(plot, "interaction", None)
    if interaction is not None and getattr(interaction, "selected_layer_id", None) == layer_id:
        interaction.selected_layer_id = None
        interaction.selected_idx = -1
        interaction.selected_type = None

    picked = getattr(plot, "picked_info", None)
    if isinstance(picked, dict) and (
        picked.get("layer") is layer or picked.get("layer_id") == layer_id
    ):
        plot.picked_info = None

    state = getattr(hud, "state", None)
    if state is not None:
        if getattr(state, "selected_layer_id", None) == layer_id:
            state.selected_layer_id = None
        selected_object = getattr(state, "selected_object", None)
        if isinstance(selected_object, dict) and (
            selected_object.get("layer") is layer or selected_object.get("layer_id") == layer_id
        ):
            state.selected_object = None


# ----------------------------------------------------------------------------------
# Lookup
# ----------------------------------------------------------------------------------


def find_layer(plot: GPULinePlot, layer_id: Optional[int]) -> Optional[Any]:
    """Return the layer with ``layer_id``, or None. Read-only; safe from a draw callback."""
    if layer_id is None:
        return None
    return next((layer for layer in plot.scene.layers if layer.layer_id == layer_id), None)


# ----------------------------------------------------------------------------------
# Removal
# ----------------------------------------------------------------------------------


def remove_layer(plot: GPULinePlot, hud: Any, layer: Any) -> None:
    """Remove ``layer`` from the live scene — the full CONTRACT §1.6 ritual.

    All four steps are mandatory and none of them is implied by the others:
    GL teardown (nothing else ever frees the buffers), list removal, legacy-mirror
    purge (they pin the arrays), selection clearing (dangling ids misbehave silently).

    **Queued commands only** — ``release_gl`` must run on the GL thread. ``hud`` may
    be None.
    """
    if layer is None:
        return

    # 1. GPU teardown — must happen before we drop our last handle on the layer.
    _release_object_gl(layer)

    # 2. Drop from the live render graph. Identity-based: list.remove would raise on
    #    any earlier same-class layer (see _index_of) and a swallowed raise here would
    #    leave the layer on screen with its GL freed and its selection cleared.
    if not _remove_identity(plot.scene.layers, layer):
        # Already detached: finish the teardown anyway rather than leaking the rest.
        logger.debug("remove_layer: layer %r was not in scene.layers", getattr(layer, "label", "?"))

    # 3. Drop the legacy mirrors that pin the arrays against GC.
    _purge_legacy_mirrors(plot, layer)

    # 4. Clear dangling selection.
    _clear_dangling_selection(plot, hud, layer)

    # A detached _primary_line_layer is worse than none: a later set_lines_ab would
    # mutate the orphan and never re-insert it (CONTRACT §1.6).
    if getattr(plot, "_primary_line_layer", None) is layer:
        plot._primary_line_layer = None

    mark_scene_dirty(plot)


def gui_clear(plot: GPULinePlot, hud: Any = None) -> None:
    """Empty the scene properly. Use instead of ``plot.clear()``, which is broken (§1.8).

    ``plot.clear()`` rebinds a fresh SceneData: it leaks every GPU buffer, dangles
    ``_primary_line_layer``, never invalidates the cache and strands the selection.
    """
    for layer in list(plot.scene.layers):
        remove_layer(plot, hud, layer)

    scene = plot.scene
    # Anything the identity-based purge missed (e.g. a layer whose array was swapped
    # by code outside update_layer_xy) still owns GL and still pins its arrays.
    for mirrors in (getattr(scene, "scatters", None), getattr(scene, "strips", None)):
        if mirrors:
            for mirror in mirrors:
                _release_object_gl(mirror)
            mirrors.clear()
    if getattr(scene, "texts", None):
        scene.texts.clear()

    scene.layers.clear()
    scene.lines = LineDataset()
    plot._cpu_line_copy = None
    plot._primary_line_layer = None
    plot.picked_info = None

    mark_scene_dirty(plot)


# ----------------------------------------------------------------------------------
# Coercion helpers
# ----------------------------------------------------------------------------------


def _coerce_1d(values: Any, name: str) -> np.ndarray:
    """Coerce to a contiguous 1-D float32 array — the dtype every 2D renderer expects."""
    arr = np.ascontiguousarray(np.asarray(values, dtype=np.float32))
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr.reshape(-1)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {arr.shape}")
    return arr


def _coerce_xy(x: Any, y: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Coerce and cross-validate an (x, y) pair."""
    xs = _coerce_1d(x, "x")
    ys = _coerce_1d(y, "y")
    if len(xs) != len(ys):
        raise ValueError(f"x and y must be the same length, got {len(xs)} and {len(ys)}")
    return xs, ys


def _normalize_color(
    color: Optional[Sequence[float]],
    default: Tuple[float, float, float, float] = DEFAULT_LAYER_COLOR,
) -> Tuple[float, float, float, float]:
    """Normalize an RGB/RGBA sequence to a 4-float tuple. None -> ``default``."""
    if color is None:
        return default
    values = [float(c) for c in np.asarray(color, dtype=np.float64).reshape(-1)]
    if len(values) == 3:
        values.append(1.0)
    if len(values) != 4:
        raise ValueError(f"color must have 3 or 4 components, got {len(values)}")
    return (values[0], values[1], values[2], values[3])


def _fit_colors(colors: Optional[np.ndarray], n: int) -> Optional[np.ndarray]:
    """Resize a per-element colour array to ``n`` rows, preserving its appearance.

    This is load-bearing, not cosmetic. ``scatter.py:123`` sets ``bufs.count`` from
    ``len(layer.pts)`` but uploads ``layer.colors`` at whatever length it happens to
    be (``line_family.py:149-157`` likewise) — so growing the data without growing the
    colours leaves the colour VBO short and the draw call fetches past its end.

    A uniform array is re-tiled exactly; a per-element gradient is resampled
    nearest-neighbour so the mapping still spans the data.
    """
    if colors is None or n <= 0:
        return colors

    arr = np.ascontiguousarray(np.asarray(colors, dtype=np.float32))
    if arr.ndim == 1 and arr.size == 1 and not bool(np.isfinite(arr).any()):
        # NOT a colour: ``engine.add_scatter(x, y, None)`` stores
        # ``np.ascontiguousarray(None, np.float32)``, which is a one-element NaN array
        # meaning "this layer has no per-point colours" — and that is the commonest scatter
        # there is. Passed straight through, because there is nothing to fit and raising
        # here made every geometry-moving animation (``Create``, ``Transform``, any
        # ``draw_fraction`` track) fail on exactly those layers.
        return colors
    if arr.ndim == 1:
        if len(arr) not in (3, 4):
            raise ValueError(f"colors must be (N,4) or a single RGB/RGBA, got shape {arr.shape}")
        if len(arr) == 3:
            arr = np.concatenate([arr, np.ones(1, dtype=np.float32)])
        return np.ascontiguousarray(np.tile(arr, (n, 1)), dtype=np.float32)

    if arr.ndim != 2 or arr.shape[1] != 4:
        raise ValueError(f"colors must have shape (N,4), got {arr.shape}")
    if len(arr) == n:
        return arr
    if len(arr) == 0:
        return None
    if len(arr) == 1 or bool(np.all(arr == arr[0])):
        return np.ascontiguousarray(np.tile(arr[0], (n, 1)), dtype=np.float32)

    idx = np.clip(np.rint(np.linspace(0, len(arr) - 1, n)).astype(np.intp), 0, len(arr) - 1)
    return np.ascontiguousarray(arr[idx], dtype=np.float32)


def _is_3d(layer: Any) -> bool:
    """Match ``hud._is_3d_layer``: geometry3d / scatter3d / volume3d / wireframe3d."""
    return str(getattr(layer, "layer_type", "")).endswith("3d")


def _differs(old: Any, new: Any) -> bool:
    """Value-compare two style values, tolerating None, tuples, strings and ndarrays."""
    if old is new:
        return False
    try:
        return not bool(np.array_equal(old, new))
    except Exception:
        return True


# ----------------------------------------------------------------------------------
# Plot kinds — what is reachable from an (x, y) table
# ----------------------------------------------------------------------------------

#: Plot types the engine can draw that a two-column table **cannot** reach as a single
#: layer, with the reason. Surfaced verbatim in the Data panel rather than hidden in a
#: docstring: a picker that silently omits ``errorbar`` reads as an oversight, and a
#: picker that offers it and produces something else is worse.
#:
#: The shared cause is the engine's one-layer-per-primitive shape. ``pyplot.errorbar``
#: (``pyplot.py:2762``) emits 2N+ ``vlines``/``hlines`` layers and ``pyplot.stem``
#: (``:2817``) emits N+2; a dataset binds exactly one ``layer_id``, so an artist group
#: has no single layer to update, and re-plotting it would orphan every layer but one —
#: which is precisely the bug (10c) this module exists to prevent.
UNSUPPORTED_KINDS: Dict[str, str] = {
    "errorbar": (
        "needs a third column for the error, and draws one layer per error bar "
        "(pyplot.errorbar emits 2N+ vlines/hlines layers). A dataset binds one layer, "
        "so the group could not be updated as a unit. Plot the series, then add the "
        "error band as a second 'area' layer from (y-err, y+err) columns."
    ),
    "imshow / pcolormesh / contour": (
        "needs a 2-D grid, not a 2-column table. Build it from code "
        "(gplt.imshow) — there is no table shape that means 'image'. The 'hist2d' kind "
        "is the heat map two columns *can* express: the density of the rows themselves."
    ),
    "pie": (
        "no primitive draws a wedge, and a pie is a composition (N wedges + labels) "
        "rather than one layer, so a bound dataset could not own it."
    ),
    "polar / pie-like composites": (
        "a wedge, a radial grid and a polar tick ring are a composition of several "
        "artists, not one layer, so a bound dataset would have nothing single to own. "
        "Build them from code (gplt.polar, gplt.pie)."
    ),
}

#: 3D is **not** in :data:`UNSUPPORTED_KINDS` any more. It used to be, with the reason
#: "a 3-D layer needs the scene's 3-D camera and axes, which the Data panel does not
#: own" — which stopped being true when the camera became panel state
#: (:class:`glplot.core.camera3d.Camera3D`) and the axis box became a rebuildable
#: decoration. Three columns now reach nine 3D kinds through
#: :mod:`glplot.gui.layerops3d`; the Data panel's 2D/3D toggle chooses which registry
#: the picker offers.

#: Default per-kind parameters. A kind's options dict is validated against these keys,
#: so an unknown option is a caller bug rather than a silently ignored typo.
_KIND_DEFAULT_OPTIONS: Dict[str, Dict[str, Any]] = {
    "line": {},
    "scatter": {},
    "step": {"where": "pre"},
    "stem": {"baseline": 0.0},
    "area": {"baseline": 0.0},
    "bar": {"baseline": 0.0, "bar_width": 0.0, "align": "center"},
    "hist": {"bins": 0, "baseline": 0.0, "density": False, "cumulative": False},
    "boxplot": {"whis": 1.5, "box_width": 0.0},
    "hist2d": {"bins": 0, "cmap": "magma"},
    # The span each line is drawn across. Unlike every other kind's options these are not
    # a styling choice -- they are half the geometry, since a slope and an intercept do
    # not say where the line starts or stops. `layer_kind_options` recovers the real pair
    # off the layer rather than handing back this default, which exists only for a table
    # that has never been a line family.
    "line_family": {"x_lo": -1.0, "x_hi": 1.0},
}

#: ``where`` values ``step`` accepts, matching ``pyplot.step`` (``pyplot.py:2740``).
STEP_WHERE_OPTIONS: Tuple[str, ...] = ("pre", "post", "mid")

#: ``align`` values ``bar`` accepts, matching ``matplotlib.pyplot.bar``.
BAR_ALIGN_OPTIONS: Tuple[str, ...] = ("center", "edge")

#: Widest a ``hist`` will bin. Above this, ``np.histogram``'s output stops being a plot
#: and starts being a memory allocation.
MAX_HIST_BINS = 5000


@dataclass(frozen=True)
class PlotKind:
    """One entry of the kind registry: how a table becomes a layer, and back.

    ``geom`` is the whole design. It maps ``(x, y, options)`` to renderer-ready
    geometry, and it is called by *both* the builder and the in-place updater, so a
    kind cannot drift between "what Plot draws" and "what a cell edit redraws".

    Attributes
    ----------
    key
        The stable identifier stored in ``metadata[KIND_METADATA_KEY]``.
    label
        What the picker shows.
    layer_type
        The ``layer_type`` this kind compiles to. Several kinds share one (``step``,
        ``stem`` and ``line`` are all polylines) — which is exactly why the kind tag
        exists and ``layer_type`` cannot be used as the identity.
    derived
        True when the geometry is *computed* from the columns rather than being them.
        A derived layer's ``pts``/``vertices`` do not line up row-for-row with the
        table, so ``DataSet.write_back`` must never be pointed at it (it would write
        N table rows over 4N bar vertices); the data link re-derives instead.
    icon
        An ``icons.ICON_SHAPES`` name for the picker.
    colormapped
        True when the kind colours itself *from the data* rather than from the caller's
        one colour. Such a kind's ``geom`` returns ``(points, values)`` instead of bare
        points, and the builder maps the values through ``options["cmap"]`` into the
        per-vertex colour VBO — the only way a scatter can carry more than one colour.
        It also retains the scalars as ``metadata["cvalues"]``, which is what lets the
        Scene panel's colormap picker re-map the layer afterwards rather than showing a
        control with no numbers left to work from (see :func:`layer_colormap_kind`).
    """

    key: str
    label: str
    layer_type: str
    derived: bool
    icon: str
    geom: Callable[[np.ndarray, np.ndarray, Dict[str, Any]], Any]
    description: str
    colormapped: bool = False


def _step_pts(xs: np.ndarray, ys: np.ndarray, where: str) -> np.ndarray:
    """The staircase through ``(xs, ys)`` as one polyline. Mirrors ``pyplot.step``."""
    if where not in STEP_WHERE_OPTIONS:
        raise ValueError(f"step 'where' must be one of {STEP_WHERE_OPTIONS}, got {where!r}")
    if len(xs) < 2:
        return np.ascontiguousarray(np.column_stack([xs, ys]), dtype=np.float32)

    if where == "post":
        sx = np.repeat(xs, 2)[1:]
        sy = np.repeat(ys, 2)[:-1]
    elif where == "pre":
        sx = np.repeat(xs, 2)[:-1]
        sy = np.repeat(ys, 2)[1:]
    else:
        mid = 0.5 * (xs[:-1] + xs[1:])
        sx = np.concatenate([xs[:1], np.repeat(mid, 2), xs[-1:]])
        sy = np.repeat(ys, 2)
    return np.ascontiguousarray(np.column_stack([sx, sy]), dtype=np.float32)


def _stem_pts(xs: np.ndarray, ys: np.ndarray, baseline: float) -> np.ndarray:
    """Stems plus their baseline as one polyline: ``(x,b) -> (x,y) -> (x,b) -> ...``.

    One layer, by construction: ``pyplot.stem`` spends N+2 layers on the same picture
    (a ``vlines`` per point, a scatter for the heads, a line for the base), and a
    dataset can only bind one of them.

    The trade the shape makes, stated plainly because it is visible: each stem is
    traced twice (up, then back down), so at ``style.alpha < 1`` the stems read darker
    than the baseline. The marker heads ``pyplot.stem`` draws are not here — they are a
    scatter, i.e. a second layer. Plot the same columns as a second ``scatter`` layer
    if you want them.
    """
    n = len(xs)
    if n == 0:
        return np.zeros((0, 2), dtype=np.float32)
    pts = np.empty((3 * n, 2), dtype=np.float32)
    pts[0::3, 0] = xs
    pts[0::3, 1] = baseline
    pts[1::3, 0] = xs
    pts[1::3, 1] = ys
    pts[2::3, 0] = xs
    pts[2::3, 1] = baseline
    return pts


def _area_verts(xs: np.ndarray, ys: np.ndarray, baseline: float) -> np.ndarray:
    """The band between ``ys`` and ``baseline`` as a ``GL_TRIANGLE_STRIP``.

    Interleaved top/bottom exactly as ``pyplot.fill_between`` (``pyplot.py:1186-1191``)
    builds it, so the two produce the same picture from the same numbers.
    """
    n = len(xs)
    verts = np.empty((2 * n, 2), dtype=np.float32)
    verts[0::2, 0] = xs
    verts[0::2, 1] = ys
    verts[1::2, 0] = xs
    verts[1::2, 1] = baseline
    return verts


def auto_bar_width(xs: np.ndarray) -> float:
    """A bar width that leaves a visible gap: 80% of the typical x spacing.

    The *median* gap, not the mean: one far-away outlier x must not shrink every bar to
    a hairline. Degenerate input (0 or 1 bars, all-equal x, non-finite x) falls back to
    1.0 rather than 0.0, because a zero-width bar is invisible and reads as a bug.
    """
    finite = np.asarray(xs, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if len(finite) < 2:
        return 1.0
    gaps = np.diff(np.sort(finite))
    gaps = gaps[gaps > 0.0]
    if len(gaps) == 0:
        return 1.0
    return float(np.median(gaps)) * 0.8


def _rect_geometry(
    left: np.ndarray, right: np.ndarray, bottom: np.ndarray, top: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """``(vertices, indices)`` drawing one quad per rect as ``GL_TRIANGLES``.

    One layer for N bars, where ``pyplot.bar`` (``pyplot.py:1266-1281``) spends one
    patch *per bar*. dtypes are explicit because ``add_patch`` does no coercion
    (CONTRACT §1.5): float32 vertices, uint32 indices.
    """
    n = len(left)
    verts = np.empty((4 * n, 2), dtype=np.float32)
    verts[0::4, 0] = left
    verts[0::4, 1] = bottom
    verts[1::4, 0] = left
    verts[1::4, 1] = top
    verts[2::4, 0] = right
    verts[2::4, 1] = top
    verts[3::4, 0] = right
    verts[3::4, 1] = bottom

    base = (np.arange(n, dtype=np.uint32) * 4).reshape(n, 1)
    quad = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32).reshape(1, 6)
    indices = np.ascontiguousarray((base + quad).reshape(-1), dtype=np.uint32)
    return verts, indices


def _bar_geometry(
    xs: np.ndarray, ys: np.ndarray, baseline: float, bar_width: float, align: str = "center"
) -> Tuple[np.ndarray, np.ndarray]:
    """Bars from x to ``ys``, sitting on ``baseline``.

    ``align`` places each bar relative to its x, matching ``matplotlib.pyplot.bar``:
    ``"center"`` centres the bar on x, ``"edge"`` puts the bar's left edge at x.
    """
    width = float(bar_width) if bar_width and bar_width > 0.0 else auto_bar_width(xs)
    if str(align) == "edge":
        left, right = xs, xs + width
    else:  # center
        half = width / 2.0
        left, right = xs - half, xs + half
    return _rect_geometry(left, right, np.full(len(xs), baseline, np.float64), ys)


def _hist_bin_count(n: int, bins: int) -> int:
    """Resolve the bin count: an explicit value, else Sturges-ish, always bounded."""
    if bins and int(bins) > 0:
        return int(min(int(bins), MAX_HIST_BINS))
    if n <= 1:
        return 1
    return int(min(max(int(np.sqrt(n)), 5), 50))


def _hist_geometry(
    ys: np.ndarray,
    bins: int,
    baseline: float,
    *,
    density: bool = False,
    cumulative: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Bars of ``np.histogram(y)``. **``x`` is ignored** — a histogram has no x column.

    ``density`` normalises so the bars integrate to 1 (a probability density); ``cumulative``
    runs a running total, giving a CDF (or a running count when ``density`` is off). Matches
    ``matplotlib.pyplot.hist``'s ``density``/``cumulative`` keywords.

    Non-finite values are dropped, as ``np.histogram`` cannot bin them. An empty (or
    all-nan) column yields empty geometry, which every renderer and the bounds path
    already treat as "draw nothing" rather than raising.
    """
    finite = np.asarray(ys, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros(0, dtype=np.uint32)
    counts, edges = np.histogram(
        finite, bins=_hist_bin_count(len(finite), bins), density=bool(density)
    )
    counts = counts.astype(np.float64)
    if cumulative:
        # A density CDF sums bar areas (height * bin width) so it ends at 1; a plain
        # cumulative sums the raw heights so it ends at the sample count.
        counts = np.cumsum(counts * np.diff(edges)) if density else np.cumsum(counts)
    return _rect_geometry(
        edges[:-1],
        edges[1:],
        np.full(len(counts), baseline, np.float64),
        counts + baseline,
    )


def _boxplot_geometry(xs: np.ndarray, ys: np.ndarray, whis: float, box_width: float) -> np.ndarray:
    """A Tukey box plot of the ``y`` column, traced as one polyline.

    **``x`` positions the box; it is not plotted.** Like ``hist``, this summarises the y
    column alone -- there is no per-row geometry. The box sits at the middle of the x
    range and defaults to half its span, so it lands on top of the data and survives
    autoscale instead of hiding at the origin at width 1.

    A polyline, not a patch, because a patch has exactly one ``face_color``: a filled box
    would bury the median line inside it in its own colour, and the whiskers would need a
    thickness in *data* units, which stretches with every zoom. Traced as lines they keep
    a pixel width, which is what makes the picture read at any scale. The cost is stated
    plainly because it shows: the path retraces itself to reach each piece (as ``stem``
    does), so at ``alpha < 1`` the retraced segments read darker.

    Tukey's definition: the box spans Q1..Q3, the whiskers reach the furthest points still
    within ``whis`` * IQR of the box, and the caps mark them. **Outliers are not drawn** --
    fliers are points, i.e. a second layer, and a dataset binds one. Plot the same column
    as a ``scatter`` beside it to see them.
    """
    finite = np.asarray(ys, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return np.zeros((0, 2), dtype=np.float32)

    q1, med, q3 = (float(v) for v in np.percentile(finite, [25.0, 50.0, 75.0]))
    iqr = q3 - q1
    within = finite[(finite >= q1 - whis * iqr) & (finite <= q3 + whis * iqr)]
    lo, hi = (float(within.min()), float(within.max())) if len(within) else (q1, q3)

    xa = np.asarray(xs, dtype=np.float64)
    xa = xa[np.isfinite(xa)]
    if len(xa):
        centre = 0.5 * (float(xa.min()) + float(xa.max()))
        span = float(xa.max()) - float(xa.min())
    else:
        centre, span = 0.0, 0.0
    width = float(box_width) if box_width > 0.0 else (0.5 * span if span > 0.0 else 1.0)
    h = 0.5 * width
    c = 0.25 * width  # cap half-width

    path = [
        (centre - c, lo),  # bottom cap
        (centre + c, lo),
        (centre, lo),  # retrace to the spine
        (centre, q1),  # lower whisker
        (centre - h, q1),  # around the box, closing it
        (centre - h, q3),
        (centre + h, q3),
        (centre + h, q1),
        (centre - h, q1),
        (centre - h, med),  # the median, drawn across the box
        (centre + h, med),
        (centre + h, q3),  # back to the top edge
        (centre, q3),
        (centre, hi),  # upper whisker
        (centre - c, hi),  # top cap
        (centre + c, hi),
    ]
    return np.ascontiguousarray(path, dtype=np.float32)


def _hist2d_geometry(xs: np.ndarray, ys: np.ndarray, bins: int) -> Tuple[np.ndarray, np.ndarray]:
    """The joint density of ``(xs, ys)`` as ``(cell centres, counts)``.

    Mirrors ``pyplot.hist2d`` (``pyplot.py:1638-1652``) exactly -- same ``histogram2d``,
    same centres, same "drop the empty cells" mask -- so the two draw the same picture
    from the same numbers. This is the one heat map a two-column table can express: it is
    a *density of the rows*, not an image. An image needs a 2-D grid, and no table shape
    means that (see :data:`UNSUPPORTED_KINDS`).

    Empty cells are dropped rather than drawn at zero: a full grid of mostly-zero squares
    paints the background solid at the colormap's low end and hides the data.

    Rows where either column is non-finite are dropped -- ``histogram2d`` cannot bin them.
    """
    xa = np.asarray(xs, dtype=np.float64)
    ya = np.asarray(ys, dtype=np.float64)
    keep = np.isfinite(xa) & np.isfinite(ya)
    xa, ya = xa[keep], ya[keep]
    if len(xa) == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros(0, dtype=np.float64)

    n = _hist_bin_count(len(xa), bins)
    counts, xedges, yedges = np.histogram2d(xa, ya, bins=n)
    xc = 0.5 * (xedges[:-1] + xedges[1:])
    yc = 0.5 * (yedges[:-1] + yedges[1:])
    xx, yy = np.meshgrid(xc, yc, indexing="ij")
    values = counts.ravel()
    mask = values > 0
    pts = np.ascontiguousarray(np.column_stack([xx.ravel()[mask], yy.ravel()[mask]]), np.float32)
    return pts, values[mask]


def colormap_colors(
    values: np.ndarray,
    cmap: str,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> np.ndarray:
    """Map ``values`` through ``cmap`` to an ``(N, 4)`` float32 RGBA array.

    The same normalise-then-look-up ``pyplot.scatter(c=...)`` performs
    (``pyplot.py:1249-1253``), kept identical on purpose: a heat map built here and one
    built from code must be the same picture, and a second implementation of the
    normalisation is a second place for the two to drift apart.
    """
    from matplotlib import colormaps

    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        return np.zeros((0, 4), dtype=np.float32)
    lo = float(np.nanmin(arr)) if vmin is None else float(vmin)
    hi = float(np.nanmax(arr)) if vmax is None else float(vmax)
    denom = max(hi - lo, 1e-12)
    normed = np.clip((arr.astype(np.float32) - lo) / denom, 0.0, 1.0)
    return np.asarray(colormaps.get_cmap(cmap)(normed), dtype=np.float32)


_KIND_REGISTRY: Dict[str, PlotKind] = {
    "line": PlotKind(
        key="line",
        label="line",
        layer_type="polyline",
        derived=False,
        icon="chart_line",
        geom=lambda xs, ys, opts: np.ascontiguousarray(np.column_stack([xs, ys]), dtype=np.float32),
        description="Connect the rows in table order.",
    ),
    "scatter": PlotKind(
        key="scatter",
        label="scatter",
        layer_type="scatter",
        derived=False,
        icon="chart_scatter",
        geom=lambda xs, ys, opts: np.ascontiguousarray(np.column_stack([xs, ys]), dtype=np.float32),
        description="One point per row.",
    ),
    "step": PlotKind(
        key="step",
        label="step",
        layer_type="polyline",
        derived=True,
        icon="chart_line",
        geom=lambda xs, ys, opts: _step_pts(xs, ys, str(opts.get("where", "pre"))),
        description="Staircase — holds each y until the next x.",
    ),
    "stem": PlotKind(
        key="stem",
        label="stem",
        layer_type="polyline",
        derived=True,
        icon="chart_bar",
        geom=lambda xs, ys, opts: _stem_pts(xs, ys, float(opts.get("baseline", 0.0))),
        description="A vertical stem per row, plus the baseline. No marker heads "
        "(those would be a second layer); each stem is traced twice, which shows at "
        "alpha < 1.",
    ),
    "area": PlotKind(
        key="area",
        label="area (fill_between)",
        layer_type="patch",
        derived=True,
        icon="chart_bar",
        geom=lambda xs, ys, opts: (
            _area_verts(xs, ys, float(opts.get("baseline", 0.0))),
            None,
            "strip",
        ),
        description="Fill between y and the baseline.",
    ),
    "bar": PlotKind(
        key="bar",
        label="bar",
        layer_type="patch",
        derived=True,
        icon="chart_bar",
        geom=lambda xs, ys, opts: (
            *_bar_geometry(
                xs,
                ys,
                float(opts.get("baseline", 0.0)),
                float(opts.get("bar_width", 0.0)),
                str(opts.get("align", "center")),
            ),
            "triangles",
        ),
        description="One bar per row, on x. Width 0 = auto (80% of the median x spacing). "
        "align: center (on x) or edge (left edge at x).",
    ),
    "hist": PlotKind(
        key="hist",
        label="histogram (y only)",
        layer_type="patch",
        derived=True,
        icon="chart_bar",
        geom=lambda xs, ys, opts: (
            *_hist_geometry(
                ys,
                int(opts.get("bins", 0)),
                float(opts.get("baseline", 0.0)),
                density=bool(opts.get("density", False)),
                cumulative=bool(opts.get("cumulative", False)),
            ),
            "triangles",
        ),
        description="Bin the y column and draw the counts. The x column is ignored, "
        "and the bars do not line up with the table's rows.",
    ),
    "boxplot": PlotKind(
        key="boxplot",
        label="box plot (y only)",
        layer_type="polyline",
        derived=True,
        icon="chart_bar",
        geom=lambda xs, ys, opts: _boxplot_geometry(
            xs, ys, float(opts.get("whis", 1.5)), float(opts.get("box_width", 0.0))
        ),
        description="Quartiles of the y column as a Tukey box, positioned on the x "
        "range. Width 0 = auto (half the x span). Outliers are not drawn (they are "
        "points, i.e. a second layer) — plot the column as a scatter beside it to see "
        "them. The path retraces itself, which shows at alpha < 1.",
    ),
    "line_family": PlotKind(
        key="line_family",
        label="line family (y = ax + b)",
        layer_type="line_family",
        # NOT derived, despite the picture being nothing like the numbers. `derived` asks
        # whether the layer's stored array lines up row-for-row with the table, and `ab`
        # does: one row, one line. The expansion into segments happens on the GPU and
        # never lands in an array here -- unlike a bar's 4N corners or a step's staircase,
        # which is what `derived` exists to keep `write_back` away from. `datasets.py`
        # already binds this layer's `ab` as its (a, b) columns for exactly that reason.
        derived=False,
        icon="chart_line",
        geom=lambda xs, ys, opts: np.ascontiguousarray(np.column_stack([xs, ys]), dtype=np.float32),
        description="Read the two columns as slope and intercept and draw one line per "
        "row across the x range. This is what gplt.plot_lines builds, and the kind a "
        "density view is for: the picture is where the lines pile up, not any one of "
        "them. The rows are the lines' parameters, not points on the plot.",
    ),
    "hist2d": PlotKind(
        key="hist2d",
        label="heat map (hist2d)",
        layer_type="scatter",
        derived=True,
        icon="chart_scatter",
        colormapped=True,
        geom=lambda xs, ys, opts: _hist2d_geometry(xs, ys, int(opts.get("bins", 0))),
        description="Bin the rows onto a 2-D grid and colour each cell by how many "
        "landed in it. Empty cells are not drawn, and the cells do not line up with the "
        "table's rows. Cells draw as round points sized in pixels (the engine has no "
        "marker shapes), so they read as a dot grid rather than a tiled image, and Point "
        "size — not the bin count — is what closes the gaps at a given zoom.",
    ),
}

#: Every kind key, in picker order.
KIND_KEYS: Tuple[str, ...] = tuple(_KIND_REGISTRY)


def kind_spec(kind: str) -> PlotKind:
    """Return the :class:`PlotKind` for ``kind``. Raises ValueError on an unknown key."""
    spec = _KIND_REGISTRY.get(str(kind).strip().lower())
    if spec is None:
        raise ValueError(f"kind must be one of {', '.join(KIND_KEYS)}, got {kind!r}")
    return spec


def kind_is_derived(kind: Optional[str]) -> bool:
    """True when ``kind``'s geometry is computed from the columns, not equal to them.

    The question every data-binding site must ask before writing columns onto a layer:
    a derived layer's arrays have a different row count and a different meaning.
    Unknown kinds answer True — refusing to write back is the safe side of the bet.
    """
    if kind is None:
        return True
    spec = _KIND_REGISTRY.get(str(kind).strip().lower())
    return True if spec is None else spec.derived


def default_kind_options(kind: str) -> Dict[str, Any]:
    """A fresh, mutable options dict for ``kind`` — the panel's starting state."""
    return dict(_KIND_DEFAULT_OPTIONS.get(str(kind).strip().lower(), {}))


def _resolve_options(kind: str, options: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge ``options`` over the kind's defaults, rejecting keys the kind has no use for.

    Rejecting rather than ignoring: a silently dropped ``bins`` on a bar chart is a bug
    the user would debug by staring at the plot.
    """
    resolved = default_kind_options(kind)
    if not options:
        return resolved
    unknown = [key for key in options if key not in resolved]
    if unknown:
        raise ValueError(
            f"kind {kind!r} has no option(s) {', '.join(sorted(unknown))}; "
            f"it accepts {', '.join(sorted(resolved)) or '(none)'}"
        )
    resolved.update(options)
    return resolved


# ----------------------------------------------------------------------------------
# Creation / update
# ----------------------------------------------------------------------------------


def add_xy_layer(
    plot: GPULinePlot,
    x: Any,
    y: Any,
    *,
    kind: str,
    label: str,
    color: Optional[Sequence[float]] = None,
    width: float = 1.0,
    size: float = 6.0,
    options: Optional[Dict[str, Any]] = None,
    c: Optional[Any] = None,
    s: Optional[Any] = None,
    cmap: str = "viridis",
) -> Any:
    """Add a layer of ``kind`` from x/y data and return it, tagged with its kind.

    ``c`` and ``s`` make colour and marker size data-driven dimensions for ``scatter``
    layers: ``c`` is a per-point numeric array mapped through the ``cmap`` option (like
    ``pyplot.scatter(c=...)``), and ``s`` a per-point size array (``pyplot.scatter(s=...)``).
    Both are length N (one per point) or ``None`` for a flat colour / uniform size.

    ``kind`` is any key of :data:`KIND_KEYS` — ``line``/``scatter`` compile to the
    table's own points; ``step``/``stem``/``area``/``bar``/``hist`` compile to geometry
    *derived* from it (see :class:`PlotKind`). :data:`UNSUPPORTED_KINDS` records what a
    two-column table cannot reach, and why.

    **Queued commands only.** Routes through the public ``add_*`` API (§1.5) and always
    passes an explicit ``label`` — the engine's default is ``f"Scatter {len(layers)}"``,
    which renames itself as the scene changes and cannot be used to find the layer again.

    Freshly constructed layers need no dirty flags: ``LayerDirtyState`` defaults to all
    True and ``_gl`` does not exist yet, so the first ``draw()`` creates and uploads.
    """
    spec = kind_spec(kind)
    if not label or not str(label).strip():
        raise ValueError("label is required: the engine's index-based default is unstable")

    xs, ys = _coerce_xy(x, y)
    opts = _resolve_options(spec.key, options)
    rgba = _normalize_color(color)
    geometry = spec.geom(xs, ys, opts)
    # The scalars an explicit ``c`` colour encoding came from, retained for the Scene panel's
    # colormap picker. Set by the scatter and bar branches; None everywhere else.
    cvalues: Optional[np.ndarray] = None

    if spec.layer_type == "polyline":
        # Colour-by-column for a line: one colour per vertex, mapped through the colormap.
        # The renderer gradient-interpolates each segment between its endpoints' colours.
        line_colors = None
        if c is not None:
            cvals = np.asarray(c, dtype=np.float64).ravel()
            if len(cvals) == len(geometry):
                line_colors = colormap_colors(cvals, str(cmap))
                cvalues = cvals
        plot.add_line_strip(
            geometry[:, 0],
            geometry[:, 1],
            color=rgba,
            width=float(width),
            label=str(label),
            colors=line_colors,
        )
    elif spec.layer_type == "scatter":
        # A colormapped kind hands geometry back as (points, values); unpack first so the
        # colour source below (explicit ``c``, the kind's own values, or a flat colour) all
        # see the same points array.
        if spec.colormapped:
            geometry, kind_values = geometry
        else:
            kind_values = None
        # The colormapped *kind* keeps its cmap in options; an explicit ``c`` column uses the
        # ``cmap`` parameter, since the plain "scatter" kind has no cmap option to read.
        if c is not None:
            cvalues = np.asarray(c, dtype=np.float64).ravel()
            colors = colormap_colors(cvalues, str(cmap))
        elif kind_values is not None:
            cvalues = np.asarray(kind_values)
            colors = colormap_colors(cvalues, str(opts.get("cmap", "viridis")))
        else:
            colors = _fit_colors(np.asarray(rgba, dtype=np.float32), len(geometry))
        sizes_arr = None if s is None else np.asarray(s, dtype=np.float32).ravel()
        plot.add_scatter(
            geometry[:, 0],
            geometry[:, 1],
            colors,
            size=float(size),
            label=str(label),
            sizes=sizes_arr,
        )
    elif spec.layer_type == "line_family":
        # `set_lines_ab` rather than an `add_*` call: the family is the engine's singleton
        # (`_primary_line_layer`), and the expansion from (a, b) into segments happens on
        # the GPU, so there is no vertex array to hand to add_*.
        x_lo, x_hi = float(opts["x_lo"]), float(opts["x_hi"])
        if not x_hi > x_lo:
            raise ValueError(f"line_family needs x_hi > x_lo, got ({x_lo}, {x_hi})")
        plot.set_lines_ab(
            geometry,
            x_range=(x_lo, x_hi),
            colors=_fit_colors(np.asarray(rgba, dtype=np.float32), len(geometry)),
            label=str(label),
        )
    else:
        verts, indices, mode = geometry
        # Colour-by-column for bars: one value per bar mapped through the colormap, then
        # broadcast to that bar's vertices (bars are a fixed number of verts each, so the
        # repeat factor is len(verts) / len(values)). Only ``bar`` -- a histogram's bins do
        # not line up with an input column, and a fill band is one shape, not one-per-row.
        face_colors = None
        if c is not None and spec.key == "bar":
            bar_vals = np.asarray(c, dtype=np.float64).ravel()
            bar_colors = colormap_colors(bar_vals, str(cmap))
            rep = len(verts) // max(len(bar_colors), 1)
            if rep >= 1 and rep * len(bar_colors) == len(verts):
                face_colors = np.repeat(bar_colors, rep, axis=0).astype(np.float32)
                cvalues = bar_vals
        plot.add_patch(
            np.ascontiguousarray(verts, dtype=np.float32),
            None if indices is None else np.ascontiguousarray(indices, dtype=np.uint32),
            mode=mode,
            face_color=rgba,
            edge_color=rgba,
            label=str(label),
            colors=face_colors,
        )

    # Every `add_*` above appends, so the new layer is the last one -- except the family,
    # which the engine keeps as a singleton, inserts at index 0 so it renders first, and
    # *reuses* on a second call. Reading `layers[-1]` for it would tag whatever happened to
    # be plotted last, so the engine's own handle is the only correct answer.
    if spec.layer_type == "line_family":
        layer = plot._primary_line_layer
    else:
        layer = plot.scene.layers[-1]

    # Tag the colour source (the kind's own values, or an explicit ``c`` column) so the
    # Scene panel's colormap picker can re-map it, exactly as pyplot.scatter does. ``cvalues``
    # is only bound in the scatter branch, so the short-circuit keeps non-scatter kinds safe.
    if cvalues is not None:
        # cvalues is set by the scatter branch (explicit c or a colormapped kind's values) or
        # the bar branch (per-bar c). Its cmap is the ``cmap`` param whenever ``c`` drove it.
        tag_cmap = str(cmap) if c is not None else str(opts.get("cmap", "viridis"))
        _tag_colormapped(layer, cvalues, tag_cmap)
    if spec.layer_type == "scatter" and s is not None:
        layer.metadata["svalues"] = np.asarray(s, dtype=np.float32).ravel()
    tag_layer_kind(layer, spec.key, options=opts, source_xy=(xs, ys) if spec.derived else None)
    mark_scene_dirty(plot)
    return layer


def add_image_layer(
    plot: GPULinePlot,
    matrix: Any,
    extent: Tuple[float, float, float, float],
    *,
    label: str,
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> Any:
    """Add a 2-D grid as an image layer and return it.

    **Queued commands only.** The counterpart to :func:`add_xy_layer` for the data shape
    that is not a table: a matrix. Not part of the kind registry on purpose -- a kind
    converts a two-column table, and no table shape means "image"
    (:data:`UNSUPPORTED_KINDS`). Nothing can be converted *to* this and it converts to
    nothing; :func:`layer_kind` returns None for it, so the type menu greys out.

    An imshow layer is a real GL texture, not a grid of points: ``scatter.py:142``
    branches to the textured-quad path on ``artist == "imshow"``, and rebuilds the texture
    from ``metadata["matrix"]`` whenever ``gpu_dirty`` is set. The scatter underneath it
    carries the extent. That is why this mirrors ``pyplot.imshow`` (``pyplot.py:1657-1687``)
    key for key rather than inventing its own metadata: miss one and the layer silently
    degrades into the dot grid the points describe.

    ``origin="lower"`` (rather than imshow's ``"upper"`` default) because the caller here
    is a *field*, not an image: row 0 is the first y sample, so y increases upward like
    every other plot. An image's row 0 is its top row, which is why imshow defaults the
    other way.
    """
    if not label or not str(label).strip():
        raise ValueError("label is required: the engine's index-based default is unstable")

    grid = np.asarray(matrix, dtype=np.float64)
    if grid.ndim != 2:
        raise ValueError(f"matrix must be 2-D, got {grid.ndim}-D")

    xmin, xmax, ymin, ymax = (float(v) for v in extent)
    rows, cols = grid.shape
    xs = np.linspace(xmin, xmax, cols, dtype=np.float32)
    ys = np.linspace(ymin, ymax, rows, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    colors = colormap_colors(grid.ravel(), cmap, vmin, vmax)
    # The same heuristic `pyplot.imshow` uses: the points only have to cover the quad the
    # texture is drawn over, so this is about not leaving gaps, not about readability.
    size = max(1.0, 650.0 / max(rows, cols))

    plot.add_scatter(xx.ravel(), yy.ravel(), colors, float(size), label=str(label))
    layer = plot.scene.layers[-1]
    layer.metadata.update(
        {
            "artist": "imshow",
            "matrix": grid,
            "extent": (xmin, xmax, ymin, ymax),
            "origin": "lower",
            "cmap": str(cmap),
            "vmin": vmin,
            "vmax": vmax,
        }
    )
    mark_scene_dirty(plot)
    return layer


def add_function_layer(
    plot: GPULinePlot,
    func: Any,
    xlim: Tuple[float, float],
    *,
    label: str,
    color: Optional[Sequence[float]] = None,
    width: float = 1.5,
    domain: Optional[Tuple[float, float]] = None,
    samples_per_px: float = 1.0,
) -> Any:
    """Add a **screen-sampled** curve — one that re-evaluates ``func`` when the view moves.

    **Queued commands only.** The live counterpart to :func:`add_xy_layer`: that one bakes
    a table of points, this one keeps the function and samples it against the viewport at
    roughly one sample per pixel column, so zooming computes new detail instead of
    magnifying old samples. See :class:`~glplot.core.layers.FunctionLayer`.

    ``func`` must be vectorised, ``f(ndarray) -> ndarray``, and must not raise for input it
    dislikes — a raising function leaves the previous curve on screen rather than taking
    the frame down, which is what makes it safe to drive from a live-edited expression.
    :func:`glplot.gui.expressions.compile_1d` builds exactly such a callable.

    Appended directly rather than through an ``add_*`` engine method because the engine has
    none for this layer type: there is no data to hand it. The layer is tagged with the
    ``line`` kind so the Scene and Style panels treat it as the polyline it is.
    """
    if not label or not str(label).strip():
        raise ValueError("label is required: the engine's index-based default is unstable")

    layer = FunctionLayer(
        func=func,
        domain=domain,
        samples_per_px=float(samples_per_px),
        color=_normalize_color(color),
        width=float(width),
        label=str(label),
    )
    # Sampled once here so the layer has geometry (and therefore bounds) before its first
    # frame; from then on the engine's `_resample_view_layers` owns it.
    layer.resample(float(xlim[0]), float(xlim[1]), int(getattr(plot, "width", 1280) or 1280))
    plot.scene.layers.append(layer)
    layer.metadata["live_function"] = True
    tag_layer_kind(layer, "line")
    mark_scene_dirty(plot)
    return layer


def _tag_colormapped(layer: Any, values: np.ndarray, cmap: str) -> None:
    """Record what a colormapped layer's colours came from, the way ``pyplot`` does.

    ``cvalues`` is what makes the mapping two-way: the colours are already baked into a
    per-vertex VBO, so without the scalars beside them ``cmap`` is undoable decoration and
    the Scene panel's colormap picker has nothing left to re-map against
    (``pyplot.py:1267-1271`` says the same, and :func:`layer_colormap_kind` keys the whole
    ``values2d`` path on this being present).

    ``marker="s"`` matches what ``pyplot.hist2d`` writes and **is inert today**: nothing
    reads ``metadata["marker"]`` -- not a renderer, not a shader. ``pyplot`` writes it in
    four places and every scatter still draws as a round point, so a heat map's cells are
    dots with gaps rather than tiles. Written anyway so both heat-map paths already agree
    on the day the scatter renderer learns marker shapes, not because it does anything.
    """
    # `pyplot`'s own retainer, not a copy of it: the size cap it enforces is already
    # coupled to this module by design ("past it ... layerops.layer_colormap_kind reports
    # no colormap", pyplot.py:253-255), so a second implementation would be a second cap
    # to keep in step. Imported here because `pyplot` pulls the whole engine.
    from ..pyplot import _retained_cvalues

    layer.metadata.update(
        {
            "marker": "s",
            "cmap": cmap,
            "vmin": None,
            "vmax": None,
            "cvalues": _retained_cvalues(np.asarray(values, dtype=np.float64)),
        }
    )


def _swap_pts(plot: GPULinePlot, layer: Any, pts: np.ndarray) -> None:
    """Swap a scatter/polyline layer's ``pts``, re-fitting colours and the legacy mirror.

    Shared by every kind that compiles to points, so a step layer's staircase reaches
    the GPU through exactly the same path a line's raw points do. Does **not** set the
    dirty flags — the callers do, because they know whether anything else moved too.
    """
    old = getattr(layer, "pts", None)
    layer.pts = pts
    if getattr(layer, "colors", None) is not None:
        layer.colors = _fit_colors(layer.colors, len(pts))
    # Re-point the mirror rather than orphaning it: an orphan both pins the old
    # array and survives remove_layer's identity purge.
    layer_type = getattr(layer, "layer_type", "")
    mirrors = getattr(plot.scene, "scatters" if layer_type == "scatter" else "strips", None)
    if mirrors and old is not None:
        for mirror in mirrors:
            if getattr(mirror, "pts", None) is old:
                mirror.pts = pts
                if getattr(mirror, "colors", None) is not None:
                    mirror.colors = _fit_colors(mirror.colors, len(pts))
                break


def update_layer_kind_data(
    plot: GPULinePlot,
    layer: Any,
    x: Any,
    y: Any,
    *,
    kind: str,
    options: Optional[Dict[str, Any]] = None,
) -> bool:
    """Re-derive ``layer``'s geometry for ``kind`` from ``x``/``y``, in place.

    **Queued commands only.** This is what makes a bar chart or a histogram follow a
    cell edit: the geometry is recomputed by the same :attr:`PlotKind.geom` that built
    it, so the live link cannot diverge from what Plot would draw.

    Returns False — having changed nothing — when the layer cannot absorb the new
    geometry, in which case the caller must rebuild it. The one case that matters is a
    patch's element buffer: ``patch.py:64-70`` creates the EBO **only if
    ``layer.indices`` was set at construction**, so a layer born without indices can
    never gain them, and one born with them must keep them. Within a kind that never
    changes, so this only fires on a mismatched ``kind`` argument.
    """
    spec = kind_spec(kind)
    if str(getattr(layer, "layer_type", "")) != spec.layer_type:
        return False

    xs, ys = _coerce_xy(x, y)
    opts = _resolve_options(spec.key, options)
    geometry = spec.geom(xs, ys, opts)

    if spec.colormapped:
        # The colours are the data here, so they are re-derived with the points: a bin
        # count change moves every cell and rescales every count, and `_swap_pts`'s
        # colour re-fit only ever stretches the old array to the new length -- it would
        # leave the previous grid's colours on the new grid's cells.
        pts, values = geometry
        _swap_pts(plot, layer, np.ascontiguousarray(pts, dtype=np.float32))
        cmap = str(opts.get("cmap", "viridis"))
        layer.colors = colormap_colors(values, cmap)
        _tag_colormapped(layer, values, cmap)
    elif spec.layer_type in ("scatter", "polyline"):
        _swap_pts(plot, layer, np.ascontiguousarray(geometry, dtype=np.float32))
    else:
        verts, indices, mode = geometry
        if (getattr(layer, "indices", None) is None) != (indices is None):
            return False
        layer.vertices = np.ascontiguousarray(verts, dtype=np.float32)
        if indices is not None:
            layer.indices = np.ascontiguousarray(indices, dtype=np.uint32)
        layer.mode = mode

    tag_layer_kind(layer, spec.key, options=opts, source_xy=(xs, ys) if spec.derived else None)
    layer.dirty.gpu_dirty = True
    mark_scene_dirty(plot)
    return True


def update_layer_xy(plot: GPULinePlot, layer: Any, x: Any, y: Any) -> None:
    """Swap a layer's data arrays in place and force a GPU re-upload.

    **Queued commands only.** Handles the tabular layer types that
    ``datasets.from_layer`` exposes: scatter/polyline (``pts``) and line_family
    (``ab``, whose columns are a/b, not x/y).

    Colours are re-fitted to the new length (see ``_fit_colors``) and the legacy mirror
    is re-pointed at the new array so removal's identity-based purge keeps working.

    Raises ValueError on a layer whose kind is *derived* (a step's ``pts`` are its
    staircase, not its columns): writing raw x/y over them would draw a plain line and
    leave the layer still claiming to be a step. Those go through
    :func:`update_layer_kind_data`.
    """
    kind = layer_kind(layer)
    if kind is not None and kind_is_derived(kind):
        raise ValueError(
            f"update_layer_xy cannot update a {kind!r} layer: its geometry is derived "
            "from the columns, not equal to them. Use update_layer_kind_data."
        )

    xs, ys = _coerce_xy(x, y)
    pts = np.ascontiguousarray(np.column_stack([xs, ys]), dtype=np.float32)
    layer_type = getattr(layer, "layer_type", "")

    if layer_type in ("scatter", "polyline"):
        _swap_pts(plot, layer, pts)
    elif layer_type == "line_family":
        layer.ab = pts
        if getattr(layer, "colors", None) is not None:
            layer.colors = _fit_colors(layer.colors, len(pts))
        if getattr(plot, "_primary_line_layer", None) is layer:
            # scene.lines drives _get_adaptive_alpha and the picking readback; leaving
            # it stale makes alpha and picks disagree with what is drawn.
            plot.scene.lines = LineDataset(
                ab=pts, colors=layer.colors, x_range=tuple(layer.x_range)
            )
            plot._cpu_line_copy = pts
    else:
        raise ValueError(f"update_layer_xy does not support layer_type {layer_type!r}")

    layer.dirty.gpu_dirty = True
    mark_scene_dirty(plot)


def set_layer_style(plot: GPULinePlot, layer: Any, **fields: Any) -> None:
    """Set style fields on ``layer``. Unknown fields raise rather than silently no-op.

    ``label`` and ``translation`` live on the layer itself; everything else must be a
    ``LayerStyle`` attribute.

    **The 3D colour trap:** ``geometry3d.py:130`` reads ``style.color`` at *GPU-upload*
    time, not draw time, and ``style_dirty`` has no consumer anywhere — so changing a 3D
    layer's colour without ``gpu_dirty`` leaves the old VBO in place and the picker
    silently does nothing. 2D uniforms re-push every draw and need no flag.

    (Note for panels: 3D ``style.color`` is only the fallback for ``layer.colors is
    None`` — on a layer with per-vertex colours it is inert by design, per §4.4.)
    """
    if not fields:
        return

    changed: List[str] = []
    for name, value in fields.items():
        if name in _LAYER_LEVEL_FIELDS:
            target: Any = layer
        elif hasattr(layer.style, name):
            target = layer.style
        else:
            raise ValueError(f"unknown style field {name!r} for layer_type {layer.layer_type!r}")

        if name == "translation":
            value = (float(value[0]), float(value[1]))

        if _differs(getattr(target, name, None), value):
            setattr(target, name, value)
            changed.append(name)

    if not changed:
        return

    if "color" in changed and _is_3d(layer):
        layer.dirty.gpu_dirty = True

    mark_scene_dirty(plot)


# ----------------------------------------------------------------------------------
# Reorder / duplicate
# ----------------------------------------------------------------------------------


def move_layer(plot: GPULinePlot, src: int, dst: int) -> None:
    """Move a layer within ``scene.layers`` (§1.7).

    **Queued commands only** — the draw callback collects the intent, this applies it.
    List order is only the tie-breaker for the ``style.zorder`` stable sort
    (``renderer_manager.py:96``); this deliberately does not touch ``zorder``.
    """
    layers = plot.scene.layers
    if not layers:
        return
    if not (0 <= src < len(layers)) or not (0 <= dst < len(layers)) or src == dst:
        return
    layers.insert(dst, layers.pop(src))
    mark_scene_dirty(plot)


def duplicate_layer(plot: GPULinePlot, layer: Any, *, label: Optional[str] = None) -> Any:
    """Insert an independent copy of ``layer`` directly after it, and return it.

    **Queued commands only.**

    Not ``copy.deepcopy``: that would clone the ``_gl`` handles as plain ints, so the
    clone and the original would name the same VAO/VBOs and removing either would
    delete the other's buffers. The clone gets no ``_gl`` at all (the renderers create
    it lazily on first draw), a fresh ``layer_id``, and copies of the data arrays so
    editing one does not silently rewrite the other.
    """
    clone = copy.copy(layer)
    clone.__dict__.pop("_gl", None)
    clone.__dict__.pop("_image_gl", None)

    clone.layer_id = uuid.uuid4().int & (1 << 31) - 1
    clone.style = copy.deepcopy(layer.style)
    clone.dirty = LayerDirtyState()
    clone.translation = tuple(getattr(layer, "translation", (0.0, 0.0)))
    clone.label = str(label) if label else f"{getattr(layer, 'label', '') or layer.layer_type} copy"

    try:
        # metadata is load-bearing: "artist" drives draw ordering, "capabilities" gates
        # the render passes, "camera"/"scene_bounds" carry 3D state (§4.4).
        clone.metadata = copy.deepcopy(layer.metadata)
    except Exception:  # pragma: no cover - exotic metadata payload
        clone.metadata = dict(layer.metadata)

    for field in _DATA_ARRAY_FIELDS:
        arr = getattr(layer, field, None)
        if isinstance(arr, np.ndarray):
            setattr(clone, field, arr.copy())

    layers = plot.scene.layers
    found = _index_of(layers, layer)
    layers.insert(len(layers) if found < 0 else found + 1, clone)

    mark_scene_dirty(plot)
    return clone


# ----------------------------------------------------------------------------------
# Kind identity and re-plot — the DataSet.layer_id update path
# ----------------------------------------------------------------------------------

#: ``metadata`` key recording which :func:`add_xy_layer` kind produced a layer.
#: ``layer_type`` alone cannot answer that question: several kinds can legitimately
#: compile to the same ``layer_type`` (a "step" is a polyline too), so re-plotting off
#: ``layer_type`` would silently update a step layer with un-stepped data.
KIND_METADATA_KEY = "gui_kind"

#: Fallback kind lookup for layers created before the tag existed, or by the plain
#: engine API (``pyplot``/``add_line_strip``). Only unambiguous types belong here — a
#: type absent from the table resolves to *unknown*, and an unknown kind never matches,
#: so the caller rebuilds instead of updating. Rebuilding is always correct; a wrong
#: match is not.
_LEGACY_KIND_LAYER_TYPES: dict[str, tuple[str, ...]] = {
    "line": ("polyline",),
    "scatter": ("scatter",),
    # `gplt.plot_lines` tags nothing, and `line_family` is a layer_type only this kind
    # compiles to -- so unlike the two above it needs no artist to disambiguate it, and
    # reading it back is unambiguous. Without this entry a family plotted from code has
    # no kind, the Data panel offers to replot it as any of the other eight, and none of
    # them can put it back: a one-way door out of the only kind that draws it.
    "line_family": ("line_family",),
}

#: The ``pyplot`` artists whose geometry *is* the x/y the caller plotted, and so are the
#: only ones :data:`_LEGACY_KIND_LAYER_TYPES` may speak for. A whitelist, not a blacklist:
#: ``layer_type`` is far too coarse to infer a kind from on its own, because ``imshow``,
#: ``pcolormesh``, ``hist2d`` and ``contour`` all render as a ``scatter`` of one point per
#: cell -- that is how the engine draws a field, not a plot of rows. Reading those as the
#: plain ``scatter`` kind offers the user a type change that throws the picture away and
#: cannot put it back: rebuilt "as a scatter" they return as bare points, the colormap and
#: the grid gone. ``axhline``/``axvline`` are guides that span the view rather than lines
#: through data. An artist that is not named here has no kind, the type menu greys out,
#: and the layer keeps working -- which is why the list must stay a whitelist: a new
#: artist is unknown until someone confirms it belongs, rather than silently misread.
_PLAIN_XY_ARTISTS: frozenset = frozenset({"line", "scatter", "marker", "step", None})

#: Style fields carried across a kind change. ``color``/``line_width``/``point_size``
#: are not here: they are passed to :func:`add_xy_layer` instead, because a scatter
#: bakes its colour into a per-vertex VBO rather than reading ``style.color`` (§4.4),
#: so assigning the style after construction would leave the old VBO colour on screen.
#:
#: The four ``outline_*`` fields are here because a silhouette is a property of *the
#: layer the user is looking at*, not of the primitive underneath it: converting a
#: surface to a wireframe and finding the outline switched itself off would read as the
#: conversion having lost the setting. They are plain draw-time uniforms on every
#: primitive (``geometry3d.py``'s ``_push_outline_uniforms``), so unlike ``color`` they
#: survive being assigned after construction.
_KIND_CHANGE_STYLE_FIELDS = (
    "visible",
    "alpha",
    "zorder",
    "point_outline_enabled",
    "point_outline_color",
    "point_outline_width",
    "outline_enabled",
    "outline_color",
    "outline_width",
    "outline_alpha",
    # Per-layer 3D compositing: a blend/occlusion/auto-alpha choice belongs to the layer,
    # not to the plot type it happens to be, so converting a scatter3d to a volume3d must
    # not silently drop the ADDITIVE the user set on it.
    "blend_mode",
    "depth_write",
    "auto_alpha",
)

#: The data attribute holding each tabular layer type's (x, y) columns.
_XY_ATTR_BY_TYPE = {"scatter": "pts", "polyline": "pts", "line_family": "ab"}

#: ``metadata`` key holding a derived layer's per-kind parameters (bar width, bins...).
KIND_OPTIONS_METADATA_KEY = "gui_kind_options"

#: ``metadata`` key holding the (x, y) a *derived* layer was built from.
#: Mandatory, not an optimisation: a bar layer's ``vertices`` are 4N interleaved corners
#: and a step layer's ``pts`` are the staircase, so neither can be read back as the
#: columns the user plotted. Without this, converting a bar back to a line would plot
#: the corner list — visibly wrong, and silently so.
KIND_SOURCE_XY_METADATA_KEY = "gui_source_xy"


def tag_layer_kind(
    layer: Any,
    kind: str,
    *,
    options: Optional[Dict[str, Any]] = None,
    source_xy: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> None:
    """Record the :func:`add_xy_layer` kind that produced ``layer``, and how.

    Pure bookkeeping on ``metadata`` (which the GUI must preserve anyway, §4.4); no
    renderer reads these keys. Safe to call on an already-tagged layer.

    ``source_xy`` is stored as float32 **copies** — the caller's arrays are usually a
    live ``Column.values``, and retaining a reference would make the tag drift with the
    table until the next re-plot, which is the opposite of a record of what was drawn.
    Omitting it on a *non-derived* kind clears any stored source (a line's ``pts`` are
    its own source); omitting it on a derived kind leaves the existing one, so a bare
    re-tag cannot silently strand a bar layer with no way back to its columns.
    """
    metadata = getattr(layer, "metadata", None)
    if not isinstance(metadata, dict):
        return
    kind_key = str(kind).strip().lower()
    metadata[KIND_METADATA_KEY] = kind_key
    if options is not None:
        metadata[KIND_OPTIONS_METADATA_KEY] = dict(options)
    if source_xy is not None:
        metadata[KIND_SOURCE_XY_METADATA_KEY] = (
            np.array(source_xy[0], dtype=np.float32, copy=True),
            np.array(source_xy[1], dtype=np.float32, copy=True),
        )
    elif not kind_is_derived(kind_key):
        metadata.pop(KIND_SOURCE_XY_METADATA_KEY, None)


def layer_kind(layer: Any) -> Optional[str]:
    """Return ``layer``'s :func:`add_xy_layer` kind, or None when it is not knowable.

    None is a real answer, not a failure: a layer whose type no ``add_xy_layer`` kind
    produces (a patch, a 3D mesh, a line_family) has no kind, and callers must treat
    that as "cannot update in place".
    """
    metadata = getattr(layer, "metadata", None)
    if isinstance(metadata, dict):
        tagged = metadata.get(KIND_METADATA_KEY)
        if isinstance(tagged, str) and tagged:
            return tagged

    # An untagged layer came from pyplot or the raw engine API, and only its artist says
    # whether its geometry is the plotted x/y or the scaffolding of something else --
    # `layer_type` cannot tell an `imshow` from a scatter (see `_PLAIN_XY_ARTISTS`).
    artist = metadata.get("artist") if isinstance(metadata, dict) else None
    if artist not in _PLAIN_XY_ARTISTS:
        return None

    layer_type = str(getattr(layer, "layer_type", ""))
    for kind, types in _LEGACY_KIND_LAYER_TYPES.items():
        if layer_type in types:
            return kind
    return None


def layer_matches_kind(layer: Any, kind: str) -> bool:
    """True when ``layer`` already *is* a ``kind`` layer and can be updated in place."""
    resolved = layer_kind(layer)
    return resolved is not None and resolved == str(kind).strip().lower()


def layer_kind_options(layer: Any) -> Dict[str, Any]:
    """Return a copy of ``layer``'s per-kind parameters, defaulted by its kind.

    A copy: the caller is a panel that will hand it to a widget, and an alias would let
    a slider drag rewrite what the layer was built with without ever re-plotting it.
    """
    kind = layer_kind(layer)
    resolved = default_kind_options(kind) if kind else {}

    # A line family carries its own x range as an attribute, so the honest default for an
    # untagged one (anything `gplt.plot_lines` built) is the range it is actually drawn
    # with -- not the registry's placeholder. Without this a family plotted over (-3, 3)
    # would come back from a type change spanning (-1, 1): the same lines, silently
    # cropped. Stored options still win below, as for every other kind.
    if kind == "line_family":
        x_range = getattr(layer, "x_range", None)
        if isinstance(x_range, (tuple, list)) and len(x_range) == 2:
            resolved["x_lo"], resolved["x_hi"] = float(x_range[0]), float(x_range[1])

    metadata = getattr(layer, "metadata", None)
    if isinstance(metadata, dict):
        stored = metadata.get(KIND_OPTIONS_METADATA_KEY)
        if isinstance(stored, dict):
            resolved.update({k: v for k, v in stored.items() if k in resolved})
    return resolved


def layer_xy(layer: Any) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Return independent copies of a tabular layer's x/y columns, or None.

    **This is the layer's geometry, not necessarily its source data.** For a derived
    kind (bar, step, hist...) the geometry is not the columns that produced it — use
    :func:`layer_source_xy` when you need those.

    Copies, because the caller's use is undo snapshots: handing back a view of
    ``layer.pts`` would let the next in-place edit rewrite the history.
    """
    attr = _XY_ATTR_BY_TYPE.get(str(getattr(layer, "layer_type", "")))
    if attr is None:
        return None
    arr = getattr(layer, attr, None)
    if not isinstance(arr, np.ndarray) or arr.ndim != 2 or arr.shape[1] < 2:
        return None
    return arr[:, 0].copy(), arr[:, 1].copy()


def layer_source_xy(layer: Any) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Return copies of the (x, y) ``layer`` was plotted from, or None if unknowable.

    The data a *type change* must carry across: a bar layer's ``vertices`` are its
    corners and a step layer's ``pts`` are its staircase, so re-plotting either as a
    line off its geometry would draw the scaffolding instead of the data. Derived kinds
    record their source at construction (:func:`tag_layer_kind`); for everything else
    the geometry *is* the source, so this falls back to :func:`layer_xy`.

    None means "this layer cannot be converted" — a patch built by ``pyplot`` or by a
    pre-tag GUI, whose columns nothing recorded. Callers must refuse rather than guess.
    """
    metadata = getattr(layer, "metadata", None)
    if isinstance(metadata, dict):
        stored = metadata.get(KIND_SOURCE_XY_METADATA_KEY)
        if isinstance(stored, tuple) and len(stored) == 2:
            xs, ys = stored
            if isinstance(xs, np.ndarray) and isinstance(ys, np.ndarray) and len(xs) == len(ys):
                return xs.copy(), ys.copy()

    if kind_is_derived(layer_kind(layer)):
        # A derived layer with no recorded source: its geometry is scaffolding, and
        # returning it would look like data. Refusing is the only honest answer.
        return None
    return layer_xy(layer)


def _representative_color(layer: Any) -> Optional[Tuple[float, float, float, float]]:
    """Best single RGBA describing ``layer``, for carrying colour across a kind change.

    ``style.face_color`` on a patch (``patch.py:112`` is the only colour it draws with;
    ``style.color`` on a patch is never read); ``style.color`` when it is set (polyline,
    the 3D fallback); otherwise the first row of the per-vertex colour array, which is
    where a scatter's colour actually lives.
    """
    style = getattr(layer, "style", None)
    if str(getattr(layer, "layer_type", "")) == "patch":
        face = getattr(style, "face_color", None)
        if face is not None:
            try:
                return _normalize_color(face)
            except (TypeError, ValueError):  # pragma: no cover - malformed face colour
                pass

    style_color = getattr(getattr(layer, "style", None), "color", None)
    if style_color is not None:
        try:
            return _normalize_color(style_color)
        except (TypeError, ValueError):  # pragma: no cover - malformed style colour
            pass

    colors = getattr(layer, "colors", None)
    if isinstance(colors, np.ndarray) and colors.size:
        row = colors[0] if colors.ndim == 2 else colors
        try:
            return _normalize_color(np.asarray(row).reshape(-1)[:4])
        except (TypeError, ValueError):  # pragma: no cover - malformed colour array
            return None
    return None


def replot_layer_xy(
    plot: GPULinePlot,
    hud: Any,
    layer: Any,
    x: Any,
    y: Any,
    *,
    kind: str,
    label: Optional[str] = None,
    options: Optional[Dict[str, Any]] = None,
) -> Any:
    """Re-plot ``x``/``y`` **into** ``layer``, and return the layer now holding them.

    **Queued commands only.** This is the "update, do not append" primitive: a dataset
    that is already bound to a live layer must re-plot into it, not stack a second,
    unbound copy of itself on top (which is what plain :func:`add_xy_layer` gives you —
    two identically-labelled layers, only one of which tracks the table).

    Two paths, decided by :func:`layer_matches_kind`:

    * **same kind** — :func:`update_layer_kind_data` re-derives the geometry in place.
      The layer's identity, ``layer_id``, style, list position and selection all
      survive, so every binding pointing at it stays valid. A changed row count is
      fine: the point paths re-fit the colour array, which is mandatory rather than
      cosmetic (``scatter.py`` sizes the draw from ``len(pts)`` but uploads ``colors``
      at its own length).
    * **different kind** — the type is baked in at construction and no conversion API
      exists, so it is delete+recreate. Colour, width, point size, the transferable
      style fields, the translation and the list index are carried over, and the
      selection is re-pointed at the new layer. **``layer_id`` necessarily changes**:
      the caller owns re-pointing any ``DataSet.layer_id`` at the returned layer.

    A same-kind layer that cannot absorb the geometry (see
    :func:`update_layer_kind_data`) falls through to the rebuild path rather than
    failing: rebuilding is always correct, only more expensive.
    """
    kind_key = str(kind).strip().lower()

    if layer_matches_kind(layer, kind_key) and update_layer_kind_data(
        plot, layer, x, y, kind=kind_key, options=options
    ):
        if label and getattr(layer, "label", None) != str(label):
            layer.label = str(label)
        return layer

    layers = plot.scene.layers
    index = _index_of(layers, layer)
    old_style = getattr(layer, "style", None)
    new_label = str(label) if label else (getattr(layer, "label", "") or kind_key)

    # Capture the selection before remove_layer clears it: the user selected this row,
    # and a type change should not deselect it (SPEC_FIX, user decision 2).
    old_id = getattr(layer, "layer_id", None)
    interaction = getattr(plot, "interaction", None)
    was_selected = (
        interaction is not None and getattr(interaction, "selected_layer_id", None) == old_id
    )
    hud_state = getattr(hud, "state", None)
    hud_selected = hud_state is not None and getattr(hud_state, "selected_layer_id", None) == old_id

    new_layer = add_xy_layer(
        plot,
        x,
        y,
        kind=kind_key,
        label=new_label,
        color=_representative_color(layer),
        width=float(getattr(old_style, "line_width", 1.0)),
        size=float(getattr(old_style, "point_size", 6.0)),
        options=options,
    )

    if old_style is not None:
        for field in _KIND_CHANGE_STYLE_FIELDS:
            if hasattr(old_style, field) and hasattr(new_layer.style, field):
                setattr(new_layer.style, field, copy.copy(getattr(old_style, field)))
    new_layer.translation = tuple(getattr(layer, "translation", (0.0, 0.0)))

    remove_layer(plot, hud, layer)

    # index was read before the removal, and the old layer occupied it, so inserting the
    # new one there restores the position exactly.
    new_index = _index_of(layers, new_layer)
    if 0 <= index < len(layers) and new_index >= 0 and new_index != index:
        layers.insert(index, layers.pop(new_index))

    new_id = getattr(new_layer, "layer_id", None)
    if was_selected and interaction is not None:
        interaction.selected_layer_id = new_id
    if hud_selected and hud_state is not None:
        hud_state.selected_layer_id = new_id

    mark_scene_dirty(plot)
    return new_layer


def replot_layer_xyz(
    plot: GPULinePlot,
    hud: Any,
    layer: Any,
    *,
    kind: str,
    options: Optional[Dict[str, Any]] = None,
    label: Optional[str] = None,
) -> Any:
    """Rebuild a 3D ``layer`` as another 3D kind, and return the layer now drawing it.

    **Queued commands only.** The 3D twin of :func:`replot_layer_xy`, and deliberately
    the same shape, because it answers the same question: *this layer, drawn as that
    instead*. It reads the columns back off the layer itself
    (:func:`glplot.gui.layerops3d.layer_source_xyz`) rather than from a dataset, which is
    the whole reason ``add_xyz_layer`` retains them on every derived layer — a surface's
    vertices are its triangulation and a quiver's are its arrows, so re-plotting either
    off its own geometry would draw the scaffolding instead of the data.

    Two paths, as in 2D:

    * **same kind** — :func:`glplot.gui.layerops3d.update_layer_xyz` re-derives the
      geometry in place, so ``layer_id``, style, list position and selection all survive
      and every binding pointing at the layer stays valid. This is the path an options
      edit (stride, baseline, grid shape...) takes.
    * **different kind** — the ``layer_type`` and the primitive are baked in at
      construction (a ``mesh3d`` cannot become a ``wireframe3d`` by assignment; even the
      VAO's index buffer only exists if the layer was born with one), so it is
      delete+recreate. The label, colour, line width, point size, the transferable style
      fields of :data:`_KIND_CHANGE_STYLE_FIELDS` (including the four outline fields),
      the per-layer SSAO/clip metadata, the list index and the selection are all carried
      over. **``layer_id`` necessarily changes**, exactly as in 2D.

    Colour is carried by *re-mapping*, not by copying the old colour array: the vertex
    count almost always changes (a 100-row path is 198 vertices, the same rows as bars
    are 3600), so the old array does not fit the new geometry. A layer that was
    colour-mapped is re-mapped through the same ``metadata["cmap"]``; when the retained
    ``cvalues`` are per **row** they are passed through as the colour column, so an
    explicit "colour by this column" survives verbatim. When they are per *vertex* of the
    old kind — which is what a derived kind stores — they cannot be re-used honestly, and
    the new kind's own scalar (z, or arrow length for a quiver) is mapped instead. That
    is a real, visible limitation and not a rounding error: converting a ``c``-coloured
    ribbon into a surface returns a surface coloured by height.

    Raises ValueError when the layer carries no 3D kind tag or no recoverable source
    columns — the two cases :func:`glplot.gui.layerops3d.layer_source_xyz` reports as
    None. Callers must gate their UI on the same test rather than catching this.
    """
    from . import layerops3d

    spec = layerops3d.kind3d_spec(kind)
    source = layerops3d.layer_source_xyz(layer)
    if source is None:
        raise ValueError(
            f"layer {getattr(layer, 'label', '?')!r} has no recoverable (x, y, z): it is "
            "either untagged (built outside the GUI) or a derived layer whose source "
            "columns were never recorded, so its vertices are geometry rather than data."
        )
    xs, ys, zs = source
    opts = layerops3d.resolve_kind3d_options(spec.key, options)

    if layerops3d.layer_kind3d(layer) == spec.key:
        layerops3d.update_layer_xyz(plot, layer, xs, ys, zs, options=opts)
        if label and getattr(layer, "label", None) != str(label):
            layer.label = str(label)
        return layer

    # The layer that *follows* this one, by identity — not its index, and not a reference
    # to the list it is in. Both of those are unusable here, for the same underlying
    # reason and neither of them applies in 2D:
    #
    #   * ``add_geometry3d`` calls ``set_3d_view``, which rebuilds the whole axis
    #     decoration (floor, grid, ticks, box). Layers appear at index 0 and disappear from
    #     the end, so an index read beforehand points somewhere else afterwards.
    #   * ``engine._system_3d_layer`` **rebinds** ``scene.layers`` to a fresh list in three
    #     of its branches. A ``layers = plot.scene.layers`` held across the add would leave
    #     every later insert/pop happening on an orphan list — the scene would keep the
    #     unmoved layer and nothing would report a failure.
    #
    # So: remember the neighbour, and re-read the list every time it is touched. A
    # neighbour that was itself rebuilt leaves the new layer at the end, which is where it
    # would have been anyway.
    index = _index_of(plot.scene.layers, layer)
    successor = plot.scene.layers[index + 1] if 0 <= index < len(plot.scene.layers) - 1 else None
    old_style = getattr(layer, "style", None)
    metadata = getattr(layer, "metadata", None)
    metadata = metadata if isinstance(metadata, dict) else {}
    new_label = str(label) if label else (getattr(layer, "label", "") or spec.key)

    # Selection, captured before remove_layer clears it: the user selected this row, and
    # changing its type must not deselect it (the rule replot_layer_xy follows).
    old_id = getattr(layer, "layer_id", None)
    interaction = getattr(plot, "interaction", None)
    was_selected = (
        interaction is not None and getattr(interaction, "selected_layer_id", None) == old_id
    )
    hud_state = getattr(hud, "state", None)
    hud_selected = hud_state is not None and getattr(hud_state, "selected_layer_id", None) == old_id

    cmap = metadata.get("cmap")
    stored_values = metadata.get("cvalues")
    column: Optional[np.ndarray] = None
    if stored_values is not None:
        candidate = np.asarray(stored_values, dtype=np.float64).ravel()
        # Per-row is the only length that can be re-used verbatim; anything else is the
        # *old* kind's per-vertex expansion (see the docstring).
        if candidate.size == xs.size:
            column = candidate

    new_layer = layerops3d.add_xyz_layer(
        plot,
        xs,
        ys,
        zs,
        kind=spec.key,
        label=new_label,
        color=_representative_color(layer),
        width=float(getattr(old_style, "line_width", 1.5)),
        size=float(getattr(old_style, "point_size", 3.0)),
        options=opts,
        c=column,
        cmap=str(cmap or "viridis"),
        # Only re-map by the kind's own scalar if the layer *was* mapped. A layer the user
        # gave a flat colour must keep it: changing the plot type is not a restyle.
        colormap_by_z=cmap is not None or stored_values is not None,
    )

    if old_style is not None:
        for field in _KIND_CHANGE_STYLE_FIELDS:
            if hasattr(old_style, field) and hasattr(new_layer.style, field):
                setattr(new_layer.style, field, copy.copy(getattr(old_style, field)))

    # Per-layer 3D state that is the user's, not the kind's. ``camera`` is deliberately
    # absent: ``add_geometry3d`` already synced the new layer onto the panel's camera, and
    # copying the old dict over it would put the layer back under a stale view.
    for key in ("ssao", "ssao_strength", "clip_bounds"):
        if key in metadata:
            new_layer.metadata[key] = copy.copy(metadata[key])

    remove_layer(plot, hud, layer)

    # Put the new layer back where the old one sat: immediately before whatever followed
    # it. The list is re-read here, never held (see the comment where ``successor`` is
    # taken). A successor that no longer exists leaves the layer at the end.
    layers = plot.scene.layers
    new_index = _index_of(layers, new_layer)
    if new_index >= 0 and successor is not None:
        moved = layers.pop(new_index)
        target = _index_of(layers, successor)
        layers.insert(len(layers) if target < 0 else target, moved)

    new_id = getattr(new_layer, "layer_id", None)
    if was_selected and interaction is not None:
        interaction.selected_layer_id = new_id
    if hud_selected and hud_state is not None:
        hud_state.selected_layer_id = new_id

    mark_scene_dirty(plot)
    return new_layer


def bind_dataset(ds: Any, layer: Any, x_name: str, y_name: str) -> None:
    """Point ``ds`` at ``layer``, with a binding that matches the layer's kind.

    The single place that decides *how* a dataset writes to a layer, so the Data panel
    and a Scene/Style type change cannot disagree about it:

    * ``line``/``scatter`` — the geometry **is** the two columns, so the binding is
      live: ``write_back`` reassembles ``pts`` from them on every cell edit.
    * every derived kind — the geometry is computed, so the binding is marked
      ``derived`` and ``write_back`` refuses. The link still exists (the data editor
      re-derives through :func:`update_layer_kind_data`); what it must never do is
      column-stack N table rows over 4N bar corners.
    * ``line_family`` — its geometry lives in ``ab``, not ``pts``. A wrong ``attr``
      does not raise: it grows a phantom array on the layer that no renderer reads,
      while the drawn data never changes.

    Duck-typed on ``ds`` (it only needs ``layer_id``/``binding``) so this module keeps
    its "no dependency on the data model" shape.
    """
    from .datasets import LayerBinding

    if layer is None:
        return
    kind = layer_kind(layer)
    layer_type = str(getattr(layer, "layer_type", ""))
    if layer_type == "line_family":
        attr = "ab"
    elif layer_type == "patch":
        attr = "vertices"
    else:
        attr = "pts"
    ds.layer_id = getattr(layer, "layer_id", None)
    ds.binding = LayerBinding(
        attr=attr,
        columns=[x_name, y_name],
        dtype=np.dtype(np.float32),
        kind=kind,
        derived=kind_is_derived(kind) if kind is not None else False,
    )


def set_layer_kind(
    plot: GPULinePlot,
    hud: Any,
    layer: Any,
    kind: str,
    *,
    options: Optional[Dict[str, Any]] = None,
    store: Any = None,
    label: Optional[str] = None,
) -> Optional[Any]:
    """Change ``layer``'s plot type in place, keeping its data. Returns the new layer.

    **Queued commands only.** The shared implementation behind the type picker in
    *Data*, *Scene* and *Style* — the user's decision (SPEC_FIX #2) is that a type
    change preserves **style, list position and selection**, and that decision has to
    hold identically wherever the gesture is offered, so it lives here rather than in
    any one panel.

    Underneath it is delete+recreate (``layer_type`` is baked in at construction and
    the engine has no conversion API), which makes the sharp edge ``layer_id``: it
    necessarily changes. Everything that names the old id is re-pointed —
    ``interaction``/HUD selection by :func:`replot_layer_xy`, and the ``DataSet``
    bound to it when a ``store`` is given. **Pass the store** if one exists: without
    it, the dataset keeps pointing at a corpse and the next Plot silently creates a
    second layer, which is exactly bug 10c.

    Returns None — having changed nothing — when the layer's source data is unknowable
    (:func:`layer_source_xy`), i.e. a patch or 3D layer this module did not build. The
    caller should disable the picker for those rather than offer a no-op.
    """
    spec = kind_spec(kind)
    if layer_matches_kind(layer, spec.key) and not options:
        return layer

    xy = layer_source_xy(layer)
    if xy is None:
        return None

    old_id = getattr(layer, "layer_id", None)
    new_layer = replot_layer_xy(
        plot,
        hud,
        layer,
        xy[0],
        xy[1],
        kind=spec.key,
        label=label if label is not None else getattr(layer, "label", None),
        options=options,
    )

    if store is not None and old_id is not None:
        ds = None
        getter = getattr(store, "get_by_layer_id", None)
        if callable(getter):
            ds = getter(old_id)
        if ds is not None:
            columns = list(getattr(getattr(ds, "binding", None), "columns", []) or [])
            names = ds.column_names() if hasattr(ds, "column_names") else []
            x_name = columns[0] if columns else (names[0] if names else "x")
            y_name = columns[-1] if len(columns) > 1 else (names[1] if len(names) > 1 else x_name)
            bind_dataset(ds, new_layer, x_name, y_name)

    return new_layer


# ----------------------------------------------------------------------------------
# Metadata — the set_layer_style trap
# ----------------------------------------------------------------------------------

#: Sentinel for "argument not supplied", where ``None`` is itself a meaningful value.
#: ``vmin=None`` means *autoscale this end of the range*, which is a different request
#: from "leave vmin alone", and a plain ``None`` default cannot tell them apart.
_UNSET: Any = object()

#: ``metadata`` keys the *renderers* read. Writing anything else through
#: :func:`set_layer_metadata` is legal (metadata is a free-form dict) but only these
#: change what is on screen. Kept as documentation, not as a guard: ``"artist"``,
#: ``"capabilities"``, ``"camera"`` and friends are equally load-bearing (§4.4) and a
#: whitelist here would have to grow every time the engine learns a key.
LIVE_METADATA_KEYS = frozenset(
    {
        "cmap",
        "vmin",
        "vmax",
        "matrix",
        "origin",
        "extent",
        "ssao",
        "ssao_strength",
        "use_colormap",
        "cmap_index",
    }
)


def set_layer_metadata(
    plot: GPULinePlot, layer: Any, *, gpu_dirty: bool = True, **fields: Any
) -> List[str]:
    """Write ``fields`` into ``layer.metadata`` and make the change take effect.

    **Queued commands only.** Returns the list of keys whose value actually changed.

    **Use this instead of** :func:`set_layer_style` **for anything a renderer reads out
    of metadata.** This is not a stylistic preference — it is the fix for a live trap.
    ``LayerStyle`` declares ``cmap``/``vmin``/``vmax``/``use_colormap``, so
    ``set_layer_style(plot, layer, cmap="hot")`` is *accepted* and writes
    ``layer.style.cmap`` — and then nothing happens, because ``scatter.py:185-189``
    reads ``layer.metadata["cmap"]`` and no renderer anywhere reads ``layer.style.cmap``
    (CONTRACT §4.1 lists all four as DEAD). The call succeeds, the picture does not move,
    and there is no error to follow. Every metadata write goes through here.

    ``gpu_dirty`` defaults to True because the metadata keys worth writing are precisely
    the ones consumed at *upload* time rather than draw time: ``scatter.py:243-247``
    rebuilds the imshow texture only when ``dirty.gpu_dirty`` is set, and re-reads
    ``cmap``/``vmin``/``vmax`` on every rebuild. Without the flag the new colormap sits
    in the dict and the old texture stays on screen — the same silent no-op one layer up.
    Pass ``gpu_dirty=False`` only for bookkeeping keys no upload path reads (e.g.
    :data:`KIND_METADATA_KEY`).
    """
    if not fields:
        return []

    metadata = getattr(layer, "metadata", None)
    if not isinstance(metadata, dict):
        raise ValueError(f"layer {getattr(layer, 'label', '?')!r} has no metadata dict")

    changed: List[str] = []
    for name, value in fields.items():
        if value is _UNSET:
            continue
        if name not in metadata or _differs(metadata.get(name), value):
            metadata[name] = value
            changed.append(name)

    if not changed:
        return []

    if gpu_dirty:
        layer.dirty.gpu_dirty = True
    mark_scene_dirty(plot)
    return changed


# ----------------------------------------------------------------------------------
# Per-layer colormaps
# ----------------------------------------------------------------------------------

#: The GL colormap names, in ``options.density_scheme_index`` order. Re-exported from
#: ``shaders.DENSITY_SCHEMES`` so panels never hardcode the list (CONTRACT §4.5) and so
#: the index a panel shows is the index the shader dispatches on.
GL_COLORMAP_NAMES: Tuple[str, ...] = tuple(DENSITY_SCHEMES)

#: Matplotlib colormap names offered for the CPU colormap paths. Not exhaustive —
#: :func:`set_layer_colormap` accepts any name matplotlib knows; this is only what a
#: picker should show without turning into a 170-entry list.
CPU_COLORMAP_NAMES: Tuple[str, ...] = (
    "viridis",
    "plasma",
    "inferno",
    "magma",
    "cividis",
    "turbo",
    "coolwarm",
    "RdBu",
    "Spectral",
    "jet",
    "hot",
    "gray",
)


def layer_colormap_kind(layer: Any) -> Optional[str]:
    """Which per-layer colormap mechanism ``layer`` supports, or None if none does.

    Colormapping in this engine is not one feature, it is four, and they are reached by
    two different functions. This is the dispatcher a panel gates its widgets on; None
    means *do not draw a colormap control for this layer*, which is the honest answer for
    a patch, a text layer, or a scatter whose colours were handed in as literal RGBA.

    * ``"image"`` — an imshow layer (a scatter carrying ``metadata["matrix"]``). The
      texture is rebuilt from metadata on ``gpu_dirty``, so this is free.
      Use :func:`set_layer_colormap`.
    * ``"values2d"`` — a 2D scatter that retained ``metadata["cvalues"]`` (the scalars
      ``pyplot.scatter(c=...)`` mapped through the colormap). Recoloured on the CPU.
      Use :func:`set_layer_colormap`.
    * ``"values3d"`` — **any** 3D layer that retained the scalars its colours were mapped
      from: ``metadata["cvalues"]`` for one :func:`glplot.gui.layerops3d.add_xyz_layer`
      built, ``metadata["zdata"]`` for a ``scatter3d`` from ``pyplot``. Recoloured on the
      CPU. Use :func:`set_layer_colormap`.

      Every 3D kind, not just ``scatter3d``, on purpose: a surface, a wireframe, a bar
      field and a vector field are all colour-mapped by construction (``add_xyz_layer``
      maps the kind's own scalar through ``cmap`` unless told otherwise), and their
      colours live in the same per-vertex VBO a scatter3d's do. Restricting this to
      ``scatter3d`` is what left the Style panel with no colormap control on exactly the
      layers — surfaces — where the palette carries the most meaning.

      The ``zdata`` half is **only faithful when the layer was coloured by z**:
      ``pyplot.py:1105`` colours by ``c`` when ``c`` is given but only ever retains
      ``z``, so recolouring a ``c``-coloured scatter3d would quietly re-map it against
      the wrong scalars — :func:`_colormap_values_for` refuses that case, which is why
      the retained ``cvalues`` matter more than they look and are preferred here.
    * ``"gl_line"`` — a polyline or line_family, coloured by a *shader* colormap keyed on
      the vertex id, not by data values. Different namespace (:data:`GL_COLORMAP_NAMES`,
      an index) and a different setter: :func:`set_layer_gl_colormap`.
    """
    layer_type = str(getattr(layer, "layer_type", ""))
    metadata = getattr(layer, "metadata", None)
    if not isinstance(metadata, dict):
        metadata = {}

    if layer_type == "scatter":
        if metadata.get("matrix") is not None:
            return "image"
        if metadata.get("cvalues") is not None:
            return "values2d"
        return None
    if _is_3d(layer):
        if metadata.get("cvalues") is not None:
            return "values3d"
        # ``zdata`` is pyplot's own retainer and only ever appears on a scatter3d.
        return "values3d" if metadata.get("zdata") is not None else None
    if layer_type in ("polyline", "line_family"):
        return "gl_line"
    return None


def layer_colormap(layer: Any) -> Tuple[Optional[str], Optional[float], Optional[float]]:
    """Return ``(cmap, vmin, vmax)`` as the renderers see them — from metadata, not style.

    ``None`` for ``vmin``/``vmax`` is the engine's own "autoscale this end" and is
    reported as-is rather than resolved, so a picker can show an empty field instead of
    inventing a number the user never typed.
    """
    metadata = getattr(layer, "metadata", None)
    if not isinstance(metadata, dict):
        return (None, None, None)
    cmap = metadata.get("cmap")
    vmin = metadata.get("vmin")
    vmax = metadata.get("vmax")
    return (
        str(cmap) if isinstance(cmap, str) and cmap else None,
        None if vmin is None else float(vmin),
        None if vmax is None else float(vmax),
    )


def _colormap_values_for(layer: Any) -> Optional[np.ndarray]:
    """The scalars ``layer``'s colours should be re-mapped from, or None if unknowable.

    None is a refusal, not a failure: recolouring against the wrong scalars produces a
    plausible-looking picture that lies about the data, which is worse than declining.
    """
    metadata = getattr(layer, "metadata", None)
    if not isinstance(metadata, dict):
        return None

    kind = layer_colormap_kind(layer)
    if kind == "values2d":
        values = metadata.get("cvalues")
    elif kind == "values3d":
        # ``cvalues`` first: it is what the colours were *actually* mapped from, whereas
        # ``zdata`` is only z and is a lie on a scatter3d coloured by an explicit ``c``.
        values = metadata.get("cvalues")
        if values is None:
            values = metadata.get("zdata")
    else:
        return None

    if values is None:
        return None
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return None

    # The colour array must span the geometry: scatter.py sizes the draw from len(pts)
    # but uploads colors at their own length, so a mismatch reads past the VBO's end.
    for attr in ("pts", "vertices"):
        geometry = getattr(layer, attr, None)
        if isinstance(geometry, np.ndarray) and geometry.ndim == 2:
            if len(geometry) != len(arr):
                return None
            break
    return arr


def _colormap_rgba(
    values: np.ndarray, cmap: Optional[str], vmin: Optional[float], vmax: Optional[float]
) -> np.ndarray:
    """Map scalars to RGBA. Mirrors ``pyplot._colormap_values`` deliberately.

    Reimplemented rather than imported: ``glplot.gui`` importing ``glplot.pyplot`` would
    pull the module-global ``_CURRENT_PLOT`` machinery into every panel, and this module
    is required to import cleanly headless. matplotlib is imported inside the function
    for the same reason — it is a real dependency (``pyplot`` uses it identically), but
    it must not be a cost of importing ``layerops``.
    """
    from matplotlib import colormaps

    arr = np.asarray(values, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        lo, hi = 0.0, 1.0
    else:
        lo = float(finite.min()) if vmin is None else float(vmin)
        hi = float(finite.max()) if vmax is None else float(vmax)
    denom = max(hi - lo, 1e-12)
    normed = np.clip((arr - lo) / denom, 0.0, 1.0)
    normed = np.nan_to_num(normed, nan=0.0, posinf=1.0, neginf=0.0)
    rgba = np.asarray(colormaps.get_cmap(cmap or "viridis")(normed), dtype=np.float32)
    return np.ascontiguousarray(rgba.reshape(len(arr), 4), dtype=np.float32)


def set_layer_colormap(
    plot: GPULinePlot,
    layer: Any,
    *,
    cmap: Any = _UNSET,
    vmin: Any = _UNSET,
    vmax: Any = _UNSET,
) -> bool:
    """Set a layer's **data** colormap and range. Returns True if anything changed.

    **Queued commands only.** For ``"image"``, ``"values2d"`` and ``"values3d"`` layers
    (see :func:`layer_colormap_kind`); raises for anything else, because silently
    ignoring the call is how ``set_layer_style(cmap=...)`` got to be a trap.
    Omitted arguments are left alone; an explicit ``vmin=None`` restores autoscaling.

    ``"image"`` needs nothing but the metadata write — ``scatter.py`` re-reads
    ``cmap``/``vmin``/``vmax`` every time it rebuilds the texture. The two value paths
    additionally recompute ``layer.colors`` here on the CPU, because those colours live
    in a per-vertex VBO that no uniform can reach.
    """
    kind = layer_colormap_kind(layer)
    if kind == "gl_line":
        raise ValueError(
            f"layer_type {getattr(layer, 'layer_type', '?')!r} uses the shader colormap; "
            "call set_layer_gl_colormap instead"
        )
    if kind is None:
        raise ValueError(
            f"layer_type {getattr(layer, 'layer_type', '?')!r} has no per-layer colormap "
            "(see layer_colormap_kind)"
        )

    if cmap is not _UNSET and cmap is not None:
        cmap = str(cmap)
    if vmin is not _UNSET and vmin is not None:
        vmin = float(vmin)
    if vmax is not _UNSET and vmax is not None:
        vmax = float(vmax)

    changed = set_layer_metadata(plot, layer, cmap=cmap, vmin=vmin, vmax=vmax)
    if not changed:
        return False

    if kind in ("values2d", "values3d"):
        values = _colormap_values_for(layer)
        if values is not None:
            resolved_cmap, resolved_vmin, resolved_vmax = layer_colormap(layer)
            colors = _colormap_rgba(values, resolved_cmap, resolved_vmin, resolved_vmax)
            # Preserve an alpha the creation path already baked in (pyplot.scatter
            # multiplies style alpha into column 3), so a recolour does not silently
            # make a translucent scatter opaque.
            old = getattr(layer, "colors", None)
            if isinstance(old, np.ndarray) and old.ndim == 2 and len(old) == len(colors):
                colors[:, 3] = old[:, 3]
            layer.colors = colors
            layer.dirty.gpu_dirty = True

    return True


def set_layer_gl_colormap(
    plot: GPULinePlot,
    layer: Any,
    *,
    enabled: Any = _UNSET,
    scheme_index: Any = _UNSET,
) -> bool:
    """Set a polyline/line_family's **shader** colormap override. Returns True if changed.

    **Queued commands only.** Writes ``metadata["use_colormap"]`` / ``["cmap_index"]``,
    which ``polyline.py`` and ``line_family.py`` read at draw time in preference to the
    scene-global ``options.line_colormap_enabled`` / ``options.density_scheme_index``.

    ``None`` is the third state and the important one: it means *follow the scene*, which
    is the default for every layer and is what makes the global toggle keep working
    unchanged. Only a layer that has been given an explicit override departs from it.

    ``scheme_index`` indexes :data:`GL_COLORMAP_NAMES`, **not** matplotlib. The GLSL
    ``colormap()`` dispatches 1-10 and falls through to Classic(0) for anything else
    (``shaders.py:163-172``), so an out-of-range index would silently render as Classic —
    it is clamped here instead.
    """
    if str(getattr(layer, "layer_type", "")) not in ("polyline", "line_family"):
        raise ValueError(
            f"layer_type {getattr(layer, 'layer_type', '?')!r} has no shader colormap; "
            "call set_layer_colormap instead"
        )

    if enabled is not _UNSET and enabled is not None:
        enabled = bool(enabled)
    if scheme_index is not _UNSET and scheme_index is not None:
        scheme_index = int(np.clip(int(scheme_index), 0, len(GL_COLORMAP_NAMES) - 1))

    # gpu_dirty is not needed: both keys are pushed as uniforms on every draw, and the
    # 2D renderers re-push unconditionally (§1.4). mark_scene_dirty alone re-renders.
    changed = set_layer_metadata(
        plot, layer, gpu_dirty=False, use_colormap=enabled, cmap_index=scheme_index
    )
    return bool(changed)


def layer_gl_colormap(layer: Any) -> Tuple[Optional[bool], Optional[int]]:
    """Return ``(use_colormap, cmap_index)`` overrides, ``None`` meaning follow the scene."""
    metadata = getattr(layer, "metadata", None)
    if not isinstance(metadata, dict):
        return (None, None)
    enabled = metadata.get("use_colormap")
    index = metadata.get("cmap_index")
    return (
        None if enabled is None else bool(enabled),
        None if index is None else int(index),
    )


# ----------------------------------------------------------------------------------
# Per-layer SSAO — live at geometry3d.py:196-198, never exposed
# ----------------------------------------------------------------------------------


def set_layer_ssao(
    plot: GPULinePlot, layer: Any, *, enabled: Any = _UNSET, strength: Any = _UNSET
) -> bool:
    """Set a 3D layer's per-layer SSAO override. Returns True if anything changed.

    **Queued commands only.** ``geometry3d.py:197`` ORs ``metadata["ssao"]`` with the
    scene-global ``visual.ssao.enabled``, so a layer can *opt in* while the scene is off
    but **cannot opt out** while the scene is on — a panel must not draw this as a
    plain checkbox that appears to turn SSAO off. ``metadata["ssao_strength"]``
    (``geometry3d.py:198``) does fully override the global strength.

    Note ``ssao.radius`` is absent on purpose: no ``u_ssao_radius`` uniform exists
    anywhere in the repo (CONTRACT §4.1), so a radius control would be a lie.
    """
    if not _is_3d(layer):
        raise ValueError(f"SSAO is 3D-only; layer_type is {getattr(layer, 'layer_type', '?')!r}")

    if enabled is not _UNSET and enabled is not None:
        enabled = bool(enabled)
    if strength is not _UNSET and strength is not None:
        strength = float(np.clip(float(strength), 0.0, 1.0))

    # Read as uniforms at draw time (geometry3d.py:199-200), so no gpu_dirty.
    return bool(
        set_layer_metadata(plot, layer, gpu_dirty=False, ssao=enabled, ssao_strength=strength)
    )


# ----------------------------------------------------------------------------------
# Draw order — reconciling list order with style.zorder
# ----------------------------------------------------------------------------------


def draw_order(plot: GPULinePlot) -> List[Any]:
    """The layers in the order the renderer actually draws them. Read-only.

    ``renderer_manager.py:96`` is ``sorted(eligible, key=lambda l: l.style.zorder)`` — a
    *stable* sort, so list order survives only as the tie-break within one zorder. This
    reproduces that ranking over the whole list rather than the visible subset: the
    manager filters before it sorts, and filtering preserves relative order, so the
    survivors appear here in the same sequence they appear there.

    This is the ordering a layer panel must show. Showing ``scene.layers`` instead is
    the bug this exists to kill: with any non-default zorder in the scene, the list the
    user reads and the picture they see disagree, and nothing on screen explains why.
    """
    return sorted(plot.scene.layers, key=lambda layer: int(layer.style.zorder))


def zorder_is_authoritative(plot: GPULinePlot) -> bool:
    """True when ``style.zorder`` re-ranks the layers away from their list order.

    In other words: True exactly when a naive list-order panel would be lying, and when
    dragging a row would appear to do nothing. False (the common case — every layer at
    the default zorder 0) means list order already *is* draw order.
    """
    zorders = [int(layer.style.zorder) for layer in plot.scene.layers]
    return zorders != sorted(zorders)


def renumber_zorder(plot: GPULinePlot) -> bool:
    """Rewrite every ``style.zorder`` to its rank in the current *draw* order.

    **Queued commands only.** Returns True if any zorder changed.

    Visually a no-op by construction: the new numbers are derived from the existing draw
    order, so the picture is identical before and after — the sort is merely made
    idempotent. What changes is the *meaning*: afterwards list order and zorder agree,
    so ``scene.layers`` alone answers "what draws on top" and a drag-reorder does what
    it says.

    It does discard the user's chosen zorder *values* (a deliberate 100 becomes its
    rank). That is the price of one source of truth, and it is why nothing calls this
    passively — only an explicit reorder gesture does, where the user has already
    declared that list position is what they mean.
    """
    ordered = draw_order(plot)
    changed = False
    for rank, layer in enumerate(ordered):
        if int(layer.style.zorder) != rank:
            layer.style.zorder = rank
            changed = True

    # Make the list itself match, so the two orderings are equal and not merely
    # consistent. Slice-assign rather than rebind: panels and the engine hold a
    # reference to this list object (hud.py:249), and rebinding would strand them.
    plot.scene.layers[:] = ordered

    if changed:
        mark_scene_dirty(plot)
    return changed


def move_layer_to_index(plot: GPULinePlot, src_id: int, dst_index: int) -> bool:
    """Move the layer with ``src_id`` to ``dst_index`` **in draw order**. Returns True if moved.

    **Queued commands only.** This is :func:`move_layer` with the lie fixed. ``move_layer``
    moves the layer within ``scene.layers`` and deliberately leaves ``style.zorder``
    alone — which is correct as a primitive and useless as a gesture, because the
    renderer sorts by zorder and the picture simply does not move (proven in
    ``diag_2_data_scene.md``).

    Here the reorder happens in draw-order space and is then *committed* by renumbering
    every zorder to the new ranking, so what the user dragged is what the renderer draws.
    """
    ordered = draw_order(plot)
    src = next((i for i, layer in enumerate(ordered) if layer.layer_id == src_id), -1)
    if src < 0:
        return False
    dst = int(np.clip(dst_index, 0, len(ordered) - 1))
    if src == dst:
        # Still commit: the drag is a statement that list order is what is meant, and a
        # scene whose zorders already disagreed must stop disagreeing either way.
        return renumber_zorder(plot)

    ordered.insert(dst, ordered.pop(src))
    plot.scene.layers[:] = ordered
    for rank, layer in enumerate(ordered):
        layer.style.zorder = rank
    mark_scene_dirty(plot)
    return True


# ----------------------------------------------------------------------------------
# Bounds / stats readout
# ----------------------------------------------------------------------------------


def layer_bounds(plot: GPULinePlot, layer: Any) -> Optional[Tuple[float, float, float, float]]:
    """``(xmin, xmax, ymin, ymax)`` in world space, translation applied. Read-only.

    Matches what autoscale frames rather than what the arrays literally hold:
    ``renderer_manager.py:199-203`` adds ``layer.translation`` to the intrinsic bounds,
    so a translated layer's readout would otherwise disagree with the view that
    ``autoscale`` produces for it.

    None is a real answer for a text layer (``TextLayer.get_intrinsic_bounds`` returns
    None by design, §1.9) and for any empty layer — callers must not treat it as an error.
    """
    del plot  # signature symmetry with the other helpers; bounds are computed live
    getter = getattr(layer, "get_intrinsic_bounds", None)
    if getter is None:
        return None
    try:
        bounds = getter()
    except Exception:  # pragma: no cover - malformed layer arrays
        return None
    if bounds is None or len(bounds) < 4:
        return None

    dx, dy = getattr(layer, "translation", (0.0, 0.0))
    xmin, xmax, ymin, ymax = (float(v) for v in bounds[:4])
    if not all(np.isfinite([xmin, xmax, ymin, ymax])):
        return None
    return (xmin + float(dx), xmax + float(dx), ymin + float(dy), ymax + float(dy))


def layer_element_count(layer: Any) -> Optional[int]:
    """How many elements ``layer`` holds (points / vertices / lines), or None."""
    for attr in ("pts", "vertices", "ab"):
        arr = getattr(layer, attr, None)
        if isinstance(arr, np.ndarray) and arr.ndim >= 1:
            return int(len(arr))
    return None


def layer_colormap_data_range(layer: Any) -> Optional[Tuple[float, float]]:
    """The ``(lo, hi)`` a layer's colormap spans when ``vmin``/``vmax`` are left to autoscale.

    Read-only. This is what a picker seeds its range fields with the moment the user
    turns autoscaling off — without it the fields start at 0/1 and the first keystroke
    silently rescales the whole layer.

    None when the layer has no colormap, or when its scalars are all NaN.
    """
    kind = layer_colormap_kind(layer)
    if kind == "image":
        metadata = getattr(layer, "metadata", None)
        values = np.asarray(metadata.get("matrix"), dtype=np.float64).reshape(-1)
    elif kind in ("values2d", "values3d"):
        found = _colormap_values_for(layer)
        if found is None:
            return None
        values = np.asarray(found, dtype=np.float64)
    else:
        return None

    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    return (float(finite.min()), float(finite.max()))


# ----------------------------------------------------------------------------------
# Annotations
# ----------------------------------------------------------------------------------

#: ``metadata`` key marking a layer as a GUI annotation, holding its kind.
ANNOTATION_METADATA_KEY = "gui_annotation"

#: ``metadata`` key holding the annotation group id. Several layers can make up one
#: annotation (an arrow is a polyline *and* a head patch), and the list must show, and
#: delete, one row rather than two — so the parts carry a shared id rather than the
#: panel trying to re-pair them by position afterwards.
ANNOTATION_GROUP_METADATA_KEY = "gui_annotation_group"


@dataclass(frozen=True)
class AnnotationKind:
    """One annotation type: what it compiles to, and what is honest to offer for it.

    ``live_fields``/``export_only_fields`` are not documentation — the panel gates its
    widgets on them, so a field absent from both cannot get a control. That is the
    mechanism enforcing "only wire controls that do something" (CONTRACT §4.1) for a
    surface where three of the obvious knobs are dead in GL.
    """

    key: str
    label: str
    icon: str
    #: Style fields read at *draw* time by a GL renderer. A widget here works on screen.
    live_fields: Tuple[str, ...]
    #: Fields read only by the matplotlib savefig/preview path (``utils/preview.py``).
    #: A widget here must be badged "export only" or it reads as broken on screen.
    export_only_fields: Tuple[str, ...]
    description: str


#: The annotation registry. **Deliberately not a superset of what looks plausible** —
#: every entry here is reachable from a primitive that already renders in GL today.
#: :data:`UNREACHABLE_ANNOTATIONS` records what was left out, and why.
_ANNOTATION_REGISTRY: Dict[str, AnnotationKind] = {
    "text": AnnotationKind(
        key="text",
        label="Text",
        icon="table",
        live_fields=("color", "visible", "zorder"),
        export_only_fields=("text_size_px",),
        description="A text label anchored to data coordinates.",
    ),
    "arrow": AnnotationKind(
        key="arrow",
        label="Arrow",
        icon="chevron_right",
        live_fields=("color", "face_color", "line_width", "alpha", "visible", "zorder"),
        export_only_fields=(),
        description="A shaft polyline plus a filled triangular head patch.",
    ),
    "hline": AnnotationKind(
        key="hline",
        label="Horizontal line",
        icon="minus",
        live_fields=("color", "line_width", "alpha", "visible", "zorder"),
        export_only_fields=(),
        description="A horizontal rule at a y value, spanning an x range.",
    ),
    "vline": AnnotationKind(
        key="vline",
        label="Vertical line",
        icon="minus",
        live_fields=("color", "line_width", "alpha", "visible", "zorder"),
        export_only_fields=(),
        description="A vertical rule at an x value, spanning a y range.",
    ),
    "rect": AnnotationKind(
        key="rect",
        label="Rectangle",
        icon="grid",
        live_fields=("face_color", "alpha", "visible", "zorder"),
        export_only_fields=(),
        description="A filled axis-aligned box (two triangles via add_patch).",
    ),
}

#: Every annotation kind, in picker order.
ANNOTATION_KEYS: Tuple[str, ...] = tuple(_ANNOTATION_REGISTRY)

#: Shapes a user will look for and **not** find, with the reason each is out of reach.
#: Kept as data, and rendered verbatim by the panel, so the honest answer is in the UI
#: rather than only in a commit message. Adding any of these means adding a renderer or
#: a uniform, not a panel widget.
UNREACHABLE_ANNOTATIONS: Dict[str, str] = {
    "Text background box": (
        "renderers/text.py draws the string straight into the background draw list "
        "with no box behind it. A box drawn from a panel would vanish when the panel "
        "is closed, and a patch layer behind the text would be in world space while "
        "the glyphs are fixed-size in screen space -- so it would drift on zoom. "
        "Needs renderers/text.py."
    ),
    "On-screen font size": (
        "style.text_size_px has no GL consumer: renderers/text.py:66 calls "
        "add_text with the default ImGui font and no size. It IS honoured by the "
        "matplotlib export path (utils/preview.py:300), which is why the panel keeps "
        "the control and badges it 'export only' instead of hiding a working knob."
    ),
    "Ellipse / circle": (
        "add_patch has no curve mode; a circle would have to be tessellated to "
        "triangles at a fixed segment count, which stops looking like a circle when "
        "zoomed in. Reachable in principle, dishonest at the resolution GLPlot "
        "invites -- left out rather than faked."
    ),
    "Unfilled (outline-only) rectangle": (
        "style.edge_color is DEAD in GL (CONTRACT 4.4): patch.py reads face_color "
        "only. An outline box is reachable today only as a 5-point polyline, which is "
        "what 'Rectangle (outline)' would have to be; the filled patch is offered "
        "instead because it is the one that honours its own colour picker."
    ),
    "Dashed / dotted lines": (
        "No renderer implements a line stipple; polyline.py has no dash uniform and "
        "no glLineStipple call exists in the repo. pyplot's linestyle= is accepted "
        "and ignored on screen."
    ),
}


def annotation_spec(kind: str) -> AnnotationKind:
    """Return the :class:`AnnotationKind` for ``kind``. Raises ValueError if unknown."""
    spec = _ANNOTATION_REGISTRY.get(str(kind).strip().lower())
    if spec is None:
        raise ValueError(
            f"annotation kind must be one of {', '.join(ANNOTATION_KEYS)}, got {kind!r}"
        )
    return spec


def tag_annotation(layer: Any, kind: str, group: Optional[str] = None) -> str:
    """Mark ``layer`` as part of an annotation of ``kind``, returning the group id.

    Preserves the rest of ``metadata`` (which is load-bearing, §4.4) and never
    overwrites an ``"artist"`` key an ``add_*`` path already set.
    """
    group_id = group or uuid.uuid4().hex
    metadata = getattr(layer, "metadata", None)
    if isinstance(metadata, dict):
        metadata[ANNOTATION_METADATA_KEY] = str(kind).strip().lower()
        metadata[ANNOTATION_GROUP_METADATA_KEY] = group_id
    return group_id


def annotation_kind(layer: Any) -> Optional[str]:
    """``layer``'s annotation kind, or None when it is not a GUI annotation."""
    metadata = getattr(layer, "metadata", None)
    if not isinstance(metadata, dict):
        return None
    kind = metadata.get(ANNOTATION_METADATA_KEY)
    return kind if isinstance(kind, str) and kind in _ANNOTATION_REGISTRY else None


def annotation_group(layer: Any) -> Optional[str]:
    """``layer``'s annotation group id, or None."""
    metadata = getattr(layer, "metadata", None)
    if not isinstance(metadata, dict):
        return None
    group = metadata.get(ANNOTATION_GROUP_METADATA_KEY)
    return group if isinstance(group, str) and group else None


def is_annotation(layer: Any) -> bool:
    """True when ``layer`` was created by the annotate panel."""
    return annotation_kind(layer) is not None


def annotation_groups(plot: GPULinePlot) -> List[Tuple[str, str, List[Any]]]:
    """Every annotation in the scene as ``(group_id, kind, [layers])``, in scene order.

    Read-only; safe from a draw callback. Grouped so the panel lists one row per
    annotation rather than one per layer — an arrow is two layers and must delete,
    restyle and hide as one thing.
    """
    order: List[str] = []
    by_group: Dict[str, Tuple[str, List[Any]]] = {}
    for layer in plot.scene.layers:
        kind = annotation_kind(layer)
        if kind is None:
            continue
        group = annotation_group(layer) or f"_orphan_{id(layer):x}"
        if group not in by_group:
            by_group[group] = (kind, [])
            order.append(group)
        by_group[group][1].append(layer)
    return [(group, by_group[group][0], by_group[group][1]) for group in order]


def _text_mirror_entry(plot: GPULinePlot, layer: Any) -> Optional[Dict[str, Any]]:
    """The ``scene.texts`` mirror row for ``layer``, matched the way the purge matches.

    Kept in lockstep on every text edit for exactly one reason: ``_purge_legacy_mirrors``
    finds the row by comparing x/y/str, so an edited layer whose mirror still holds the
    old scalars would leave that row behind on delete — pinning the string forever and,
    worse, letting the purge match a *different* text layer that happens to sit where
    this one used to.
    """
    texts = getattr(plot.scene, "texts", None)
    if not texts:
        return None
    for entry in texts:
        if (
            entry.get("x") == getattr(layer, "x", None)
            and entry.get("y") == getattr(layer, "y", None)
            and entry.get("str") == getattr(layer, "text", None)
        ):
            return entry
    return None


def add_text_annotation(
    plot: GPULinePlot,
    x: float,
    y: float,
    text: str,
    *,
    label: Optional[str] = None,
    fontsize: int = 12,
    color: Optional[Sequence[float]] = None,
    group: Optional[str] = None,
) -> Any:
    """Add a text annotation at world ``(x, y)`` and return the layer.

    **Queued commands only.** ``engine.add_text`` sets ``frame.dirty_ui`` but *not*
    ``dirty_scene`` (``engine.py:450``) even though text draws in the scene pass — the
    queue's epilogue is what repairs that, so calling this outside the queue produces a
    label that does not appear until an unrelated event wakes the loop (CONTRACT §1.5).
    ``mark_scene_dirty`` below covers the same ground for a caller draining by hand.
    """
    rgba = _normalize_color(color, default=(0.0, 0.0, 0.0, 1.0))
    layer_label = label or f"Text: {str(text)[:24]}"
    plot.add_text(
        float(x), float(y), str(text), fontsize=int(fontsize), color=rgba, label=layer_label
    )
    layer = plot.scene.layers[-1]
    tag_annotation(layer, "text", group)
    mark_scene_dirty(plot)
    return layer


def update_text_annotation(
    plot: GPULinePlot,
    layer: Any,
    *,
    text: Optional[str] = None,
    x: Optional[float] = None,
    y: Optional[float] = None,
) -> None:
    """Edit a text layer's string and/or position in place, mirror included.

    **Queued commands only.** No GPU state is involved — ``renderers/text.py`` reads
    ``layer.x``/``.y``/``.text`` at draw time and projects them per frame, so there is
    nothing to re-upload and ``gpu_dirty`` is deliberately not set (§1.4 sets it only
    for VBO-backed arrays).
    """
    if layer is None or getattr(layer, "layer_type", None) != "text":
        return

    entry = _text_mirror_entry(plot, layer)
    changed = False
    if text is not None and _differs(getattr(layer, "text", None), str(text)):
        layer.text = str(text)
        if entry is not None:
            entry["str"] = layer.text
        changed = True
    if x is not None and _differs(getattr(layer, "x", None), float(x)):
        layer.x = float(x)
        if entry is not None:
            entry["x"] = layer.x
        changed = True
    if y is not None and _differs(getattr(layer, "y", None), float(y)):
        layer.y = float(y)
        if entry is not None:
            entry["y"] = layer.y
        changed = True

    if changed:
        mark_scene_dirty(plot)


def text_annotation_position(layer: Any) -> Tuple[float, float]:
    """A text layer's world position with its translation applied.

    ``renderers/text.py:39`` projects ``(layer.x + tx, layer.y + ty)``, so this is where
    the glyphs actually are — which is what a drag must move and what the list must show.
    """
    tx, ty = getattr(layer, "translation", (0.0, 0.0))
    return (
        float(getattr(layer, "x", 0.0)) + float(tx),
        float(getattr(layer, "y", 0.0)) + float(ty),
    )


def arrow_head_geometry(
    x: float, y: float, dx: float, dy: float, *, head_width: float, head_length: float
) -> Optional[np.ndarray]:
    """The three head vertices for an arrow, or None when the shaft has no length.

    Mirrors ``pyplot.arrow`` (``pyplot.py:2279``) exactly, so a GUI arrow and a scripted
    one are the same object. Split out from :func:`add_arrow_annotation` because it is
    pure geometry and is the part worth testing without a plot.
    """
    length = float(np.hypot(dx, dy))
    if length <= 1e-12:
        return None
    ux, uy = dx / length, dy / length
    px, py = -uy, ux
    tip = np.array([x + dx, y + dy], dtype=np.float32)
    base = tip - float(head_length) * np.array([ux, uy], dtype=np.float32)
    half = 0.5 * float(head_width) * np.array([px, py], dtype=np.float32)
    return np.ascontiguousarray(
        np.array([tip, base + half, base - half], dtype=np.float32), dtype=np.float32
    )


def add_arrow_annotation(
    plot: GPULinePlot,
    x: float,
    y: float,
    dx: float,
    dy: float,
    *,
    label: Optional[str] = None,
    color: Optional[Sequence[float]] = None,
    width: float = 1.0,
    head_width: float = 0.12,
    head_length: float = 0.18,
    group: Optional[str] = None,
) -> List[Any]:
    """Add an arrow (shaft polyline + head patch) and return its layers.

    **Queued commands only.** Deliberately re-implements ``pyplot.arrow``'s geometry
    against ``plot`` rather than calling it: every pyplot function resolves the *global*
    pyplot figure via ``_get_or_create_plot()``, so calling it from the GUI would draw
    the arrow into whatever plot pyplot last touched — not necessarily this one.
    """
    rgba = _normalize_color(color, default=(0.0, 0.0, 0.0, 1.0))
    group_id = group or uuid.uuid4().hex
    base_label = label or "Arrow"
    layers: List[Any] = []

    plot.add_line_strip(
        np.array([x, x + dx], dtype=np.float32),
        np.array([y, y + dy], dtype=np.float32),
        color=rgba,
        width=float(width),
        label=base_label,
    )
    shaft = plot.scene.layers[-1]
    tag_annotation(shaft, "arrow", group_id)
    layers.append(shaft)

    head = arrow_head_geometry(x, y, dx, dy, head_width=head_width, head_length=head_length)
    if head is not None:
        plot.add_patch(
            head,
            np.ascontiguousarray([0, 1, 2], dtype=np.uint32),
            mode="triangles",
            face_color=rgba,
            edge_color=rgba,
            label=f"{base_label} head",
        )
        head_layer = plot.scene.layers[-1]
        tag_annotation(head_layer, "arrow", group_id)
        layers.append(head_layer)

    mark_scene_dirty(plot)
    return layers


def add_line_annotation(
    plot: GPULinePlot,
    *,
    kind: str,
    value: float,
    lo: float,
    hi: float,
    label: Optional[str] = None,
    color: Optional[Sequence[float]] = None,
    width: float = 1.0,
    group: Optional[str] = None,
) -> Any:
    """Add an ``hline``/``vline`` annotation spanning ``[lo, hi]``, and return the layer.

    **Queued commands only.** A plain two-point polyline, matching what
    ``pyplot.hlines``/``vlines`` build (``pyplot.py:2641``): those go through ``plot()``
    and land as polylines too, so this is the same object by a shorter route.

    Note these are *finite* rules with an explicit span, like ``hlines``/``vlines`` and
    unlike ``axhline``/``axvline``, whose "spans the whole axis forever" behaviour has
    no layer that tracks the view — the engine would have to re-emit the geometry on
    every pan. The panel offers "span the current view", which is that, once.
    """
    kind_key = str(kind).strip().lower()
    if kind_key not in ("hline", "vline"):
        raise ValueError(f"kind must be 'hline' or 'vline', got {kind!r}")

    rgba = _normalize_color(color, default=(0.0, 0.0, 0.0, 1.0))
    if kind_key == "hline":
        xs = np.array([lo, hi], dtype=np.float32)
        ys = np.array([value, value], dtype=np.float32)
        default_label = f"y = {float(value):g}"
    else:
        xs = np.array([value, value], dtype=np.float32)
        ys = np.array([lo, hi], dtype=np.float32)
        default_label = f"x = {float(value):g}"

    plot.add_line_strip(xs, ys, color=rgba, width=float(width), label=label or default_label)
    layer = plot.scene.layers[-1]
    tag_annotation(layer, kind_key, group)
    mark_scene_dirty(plot)
    return layer


def rect_annotation_geometry(
    x0: float, y0: float, x1: float, y1: float
) -> Tuple[np.ndarray, np.ndarray]:
    """``(vertices, indices)`` for a filled axis-aligned box, as two triangles.

    Corners are normalised, so a box dragged right-to-left or bottom-to-top is the same
    rectangle rather than a degenerate one with a flipped winding.
    """
    xlo, xhi = (float(x0), float(x1)) if x0 <= x1 else (float(x1), float(x0))
    ylo, yhi = (float(y0), float(y1)) if y0 <= y1 else (float(y1), float(y0))
    verts = np.ascontiguousarray(
        np.array([[xlo, ylo], [xhi, ylo], [xhi, yhi], [xlo, yhi]], dtype=np.float32),
        dtype=np.float32,
    )
    indices = np.ascontiguousarray(np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32), dtype=np.uint32)
    return verts, indices


def add_rect_annotation(
    plot: GPULinePlot,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    *,
    label: Optional[str] = None,
    face_color: Optional[Sequence[float]] = None,
    group: Optional[str] = None,
) -> Any:
    """Add a filled rectangle annotation and return the layer.

    **Queued commands only.** ``add_patch`` does no dtype coercion (§1.5), so the
    float32/uint32 arrays from :func:`rect_annotation_geometry` are mandatory, not tidy.

    Only ``face_color`` is passed: ``edge_color`` is DEAD in GL (``patch.py`` reads
    ``face_color`` only), so setting it would create a picker that does nothing.
    """
    rgba = _normalize_color(face_color, default=(0.3, 0.6, 1.0, 0.35))
    verts, indices = rect_annotation_geometry(x0, y0, x1, y1)
    plot.add_patch(
        verts,
        indices,
        mode="triangles",
        face_color=rgba,
        label=label or "Rectangle",
    )
    layer = plot.scene.layers[-1]
    tag_annotation(layer, "rect", group)
    mark_scene_dirty(plot)
    return layer


def remove_annotation_group(plot: GPULinePlot, hud: Any, layers: Sequence[Any]) -> None:
    """Remove every layer of one annotation, each through the full §1.6 ritual.

    **Queued commands only** (``release_gl`` is GL-thread-bound). Iterates a copy: the
    ritual mutates ``scene.layers``.
    """
    for layer in list(layers):
        remove_layer(plot, hud, layer)


def set_annotation_style(plot: GPULinePlot, layers: Sequence[Any], **fields: Any) -> None:
    """Apply style fields across an annotation's layers, skipping those they lack.

    An arrow is a polyline plus a patch, and the two honour *different* colour fields
    (``polyline.py:173`` reads ``style.color``; ``patch.py:112`` reads
    ``style.face_color``). ``set_layer_style`` raises on an unknown field by design, so
    this filters per layer rather than making the caller fan out by layer_type — the
    panel shows one colour picker and both halves of the arrow follow it.
    """
    for layer in layers:
        if layer is None:
            continue
        applicable = {
            name: value
            for name, value in fields.items()
            if name in _LAYER_LEVEL_FIELDS or hasattr(getattr(layer, "style", None), name)
        }
        if applicable:
            set_layer_style(plot, layer, **applicable)


#: The layer-level (non-``LayerStyle``) field names, as a public name for panels.
#: Same object as the private constant the rest of this module uses, so the two cannot
#: drift; exported because the annotate panel snapshots style fields for undo and must
#: ask the same question ``set_layer_style`` asks.
LAYER_LEVEL_FIELDS = _LAYER_LEVEL_FIELDS


def normalize_color(
    color: Optional[Sequence[float]],
    default: Tuple[float, float, float, float] = DEFAULT_LAYER_COLOR,
) -> Tuple[float, float, float, float]:
    """Public alias of the RGB/RGBA normaliser. ``None`` -> ``default``.

    Panels need it to seed a ``color_edit4`` from a layer whose colour may be a tuple,
    a 3-list or an ndarray, and reaching for the underscore-prefixed original from
    another module is how a private helper quietly becomes API anyway.
    """
    return _normalize_color(color, default)


def annotation_color(layers: Sequence[Any]) -> Tuple[float, float, float, float]:
    """The colour to seed an annotation's picker with: the first layer's real one.

    Prefers ``style.color`` and falls back to ``face_color``, matching the order the
    renderers read them in. A rect has only ``face_color``; a text only ``color``.
    """
    for layer in layers:
        style = getattr(layer, "style", None)
        if style is None:
            continue
        for name in ("color", "face_color"):
            value = getattr(style, name, None)
            if value is not None:
                try:
                    return _normalize_color(value)
                except (TypeError, ValueError):
                    continue
    return DEFAULT_LAYER_COLOR
