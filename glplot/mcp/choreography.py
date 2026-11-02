"""Stage effects: ``create``, ``write``, ``fade_in``, ``indicate``... as in a manim scene.

A step-by-step explainer is a sequence of *plays* (as in manim): draw the axes' curve, write the
equation, fade a dot in, flash the term that matters, fade it all out. ``animate`` expresses
that with ``stages=[{"name", "duration", "effects": {"create": "a1", "write": ["a2"]}}]``.

Nothing new runs at render time. This module **compiles** the effects into what ``animate``
already understands -- a stage variable per artist and channel, moved by the stages, and a
binding from each variable to a property of the artist -- so the timeline, the frame-by-frame
rebuild, the motion review, the limits and ``export_script`` all work on effects unchanged.

Channels, one variable each, named ``fx_<artist>_<channel>``:

* ``show`` (0..1) multiplies the artist's ``alpha``. An artist whose first effect brings it in
  is hidden (0) until that stage; one whose last effect takes it out stays hidden after.
* ``draw`` (0..1) is the artist's ``reveal``: how much of a curve, of a text's letters or of an
  arrow's length is drawn. Kinds that cannot be cut short (a field, a histogram...) fade instead.
* ``pulse`` (0 -> 1 -> 0) is ``indicate``: the colour moves towards the highlight and the size
  grows, then both come back (manim's ``there_and_back``).
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from matplotlib import colors as mcolors

from .state import McpError, suggest

__all__ = ["EFFECTS", "HIGHLIGHT", "compile_effects", "effects_guide"]

#: Effect -> (channel, direction). ``in`` brings the artist in, ``out`` takes it away.
EFFECTS: Dict[str, Tuple[str, str]] = {
    "create": ("draw", "in"),
    "uncreate": ("draw", "out"),
    "write": ("draw", "in"),
    "unwrite": ("draw", "out"),
    "grow": ("draw", "in"),
    "fade_in": ("show", "in"),
    "fade_out": ("show", "out"),
    "indicate": ("pulse", "in"),
    # {"transform": {"a2": "a3"}}: a2 fades out while a3 fades in (one equation becoming the next)
    "transform": ("show", "in"),
}

#: Other spellings a model writing manim-like code reaches for.
ALIASES: Dict[str, str] = {
    "showcreation": "create",
    "show_creation": "create",
    "draw": "create",
    "drawborderthenfill": "write",
    "growarrow": "grow",
    "grow_arrow": "grow",
    "growfromcenter": "grow",
    "fadein": "fade_in",
    "fadeout": "fade_out",
    "flash": "indicate",
    "highlight": "indicate",
    "replace": "transform",
    "replacementtransform": "transform",
    "transformmatchingtex": "transform",
    "transformmatchingshapes": "transform",
    "fadetransform": "transform",
}

#: manim's YELLOW_C, what ``Indicate`` turns a thing while it points at it.
HIGHLIGHT = "#F7D96F"

#: manim's default rate function (``smooth``), which GLPlot calls ``smoother``.
DEFAULT_EASE = "smoother"

#: Kinds of add_plot / add_function whose series can be drawn in part (``reveal``).
_REVEALABLE_PLOTS = frozenset({"line", "scatter", "step", "area"})


def _canonical(name: Any, where: str) -> str:
    raw = str(name).strip()
    key = raw.lower().replace(" ", "_")
    key = ALIASES.get(key.replace("_", ""), ALIASES.get(key, key))
    if key not in EFFECTS:
        known = sorted(EFFECTS)
        raise McpError(
            f"{where}: unknown effect {raw!r}.{suggest(key, known)} Effects: {', '.join(known)}."
        )
    return key


def _targets(value: Any, where: str) -> List[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)) and value and all(isinstance(v, str) for v in value):
        return list(value)
    raise McpError(f"{where}: give an artist id ('a2') or a list of them (['a2', 'a3']).")


def _stage_effects(spec: Mapping[str, Any], where: str) -> List[Tuple[str, str]]:
    """``(effect, artist id)`` pairs of one stage, in the order given."""
    raw = spec.get("effects")
    if raw is None:
        return []
    items: List[Mapping[str, Any]]
    if isinstance(raw, Mapping):
        items = [raw]
    elif isinstance(raw, (list, tuple)) and all(isinstance(i, Mapping) for i in raw):
        items = list(raw)
    else:
        raise McpError(
            f"{where}: effects is an object like {{'create': 'a1', 'write': ['a2', 'a3']}}."
        )
    pairs: List[Tuple[str, str]] = []
    for item in items:
        for name, value in item.items():
            effect = _canonical(name, where)
            if effect == "transform":
                if not isinstance(value, Mapping) or not value:
                    raise McpError(
                        f"{where}: transform is {{'a2': 'a3'}}: a2 fades out as a3 fades in."
                    )
                for old, new in value.items():
                    pairs.append(("fade_out", str(old)))
                    pairs.append(("fade_in", str(new)))
                continue
            for target in _targets(value, f"{where}: {name}"):
                pairs.append((effect, target))
    return pairs


class _Subject:
    """What the compiler needs to know about one artist: how it was drawn, what it looks like."""

    def __init__(self, box: Any, entry: Any, artist_id: str) -> None:
        op = next((o for o in entry.log if o.artist_id == artist_id), None)
        if op is None:
            known = ", ".join(a.artist_id for a in entry.artists) or "none"
            raise McpError(f"effects: {entry.figure_id} has no artist {artist_id!r} ({known}).")
        self.id = artist_id
        self.op = op
        self.tool = op.tool
        self.args = op.args
        self.kind = str(op.args.get("kind") or "")
        self.layers = self._layers(entry, artist_id)
        self.arguments = set(_signature(op.tool))

    @staticmethod
    def _layers(entry: Any, artist_id: str) -> List[Any]:
        artist = next((a for a in entry.artists if a.artist_id == artist_id), None)
        if artist is None:
            return []
        ids = set(artist.layer_ids)
        return [layer for layer in entry.plot.scene.layers if layer.layer_id in ids]

    # -- what it can do ---------------------------------------------------------------

    def can_reveal(self) -> bool:
        if self.tool in ("add_plot", "add_function"):
            return (self.kind or "line") in _REVEALABLE_PLOTS
        if self.tool == "add_annotation":
            return self.kind in ("text", "arrow")
        return False

    def can_fade(self) -> bool:
        return "alpha" in self.arguments

    # -- what it looks like now ---------------------------------------------------------

    def alpha(self) -> float:
        value = self.args.get("alpha", self.op.resolved.get("alpha"))
        return 1.0 if value is None else float(value)

    def color(self) -> Optional[str]:
        value = self.args.get("color") or self.op.resolved.get("color")
        if value is None:
            return None
        try:
            return mcolors.to_hex(value)
        except ValueError:
            return None

    def style_number(self, attribute: str, default: float) -> float:
        for layer in self.layers:
            value = getattr(getattr(layer, "style", None), attribute, None)
            if isinstance(value, (int, float)) and value > 0:
                return float(value)
        return default


def _signature(tool: str) -> List[str]:
    import inspect

    from .toolbox import Toolbox

    return [
        name
        for name in inspect.signature(getattr(Toolbox, tool)).parameters
        if name not in ("self", "figure_id")
    ]


def _number(value: float) -> str:
    return repr(round(float(value), 6))


def compile_effects(
    box: Any,
    entry: Any,
    stages: Optional[Sequence[Mapping[str, Any]]],
    variables: Optional[Mapping[str, Any]],
    bindings: Optional[Sequence[Mapping[str, Any]]],
) -> Tuple[Optional[List[Dict[str, Any]]], Optional[Dict[str, Any]], Optional[List[Any]], Any]:
    """``stages`` with ``effects`` turned into stage variables and ``bindings``.

    Returns ``(stages, variables, bindings, plan)``: the first three are what ``animate``
    runs (no stage has ``effects`` any more), ``plan`` says what each stage does, for the reply
    (``None`` when no stage had effects, and then the inputs come back unchanged).
    """
    has_effects = isinstance(stages, (list, tuple)) and any(
        isinstance(s, Mapping) and s.get("effects") for s in stages
    )
    if not has_effects:
        # untouched, so animate's own checks see (and word their refusals about) what was sent
        return stages, variables, bindings, None  # type: ignore[return-value]
    if bindings is not None and not isinstance(bindings, (list, tuple)):
        raise McpError("bindings must be a list of {target, property, value} objects.")

    per_stage: List[List[Tuple[str, str]]] = []
    for index, spec in enumerate(stages):
        if not isinstance(spec, Mapping):
            raise McpError(f"stage {index + 1} must be an object.")
        where = f"stage {spec.get('name', index + 1)!r}"
        per_stage.append(_stage_effects(spec, where))

    subjects: Dict[str, _Subject] = {}
    for pairs in per_stage:
        for _effect, target in pairs:
            if target not in subjects:
                subjects[target] = _Subject(box, entry, target)

    out_vars: Dict[str, Any] = dict(variables or {})
    out_stages: List[Dict[str, Any]] = [
        {k: copy.deepcopy(v) for k, v in s.items() if k != "effects"} for s in stages
    ]
    plan: List[Dict[str, Any]] = []
    notes: List[str] = []
    channels: Dict[Tuple[str, str], str] = {}  # (artist, channel) -> variable

    def variable(artist: str, channel: str, start: float) -> str:
        key = (artist, channel)
        if key not in channels:
            name = f"fx_{artist}_{channel}"
            if name in out_vars:
                raise McpError(f"effects need the variable name {name!r}; rename yours.")
            out_vars[name] = start
            channels[key] = name
        return channels[key]

    # first effect on each channel decides where the channel starts (hidden or shown)
    first: Dict[Tuple[str, str], str] = {}
    for pairs in per_stage:
        for effect, target in pairs:
            channel, direction = EFFECTS[effect]
            subject = subjects[target]
            channel = _resolve_channel(subject, effect, channel, notes)
            first.setdefault((target, channel), direction)
            if channel == "draw":
                first.setdefault((target, "show"), direction)

    for index, (spec, pairs) in enumerate(zip(stages, per_stage)):
        moves = out_stages[index].setdefault("animate", {})
        if not isinstance(moves, dict):
            raise McpError(f"stage {index + 1}: animate must be an object.")
        ease = spec.get("ease") or DEFAULT_EASE
        summary: Dict[str, List[str]] = {}
        for effect, target in pairs:
            subject = subjects[target]
            channel, direction = EFFECTS[effect]
            channel = _resolve_channel(subject, effect, channel, None)
            summary.setdefault(effect, []).append(target)
            if channel == "pulse":
                name = variable(target, "pulse", 0.0)
                moves[name] = {"from": 0, "to": 1, "ease": "there_and_back"}
                continue
            start = 0.0 if first[(target, channel)] == "in" else 1.0
            name = variable(target, channel, start)
            moves[name] = {"to": 1 if direction == "in" else 0, "ease": ease}
            if channel == "draw":
                shown = variable(target, "show", 0.0 if first[(target, "show")] == "in" else 1.0)
                if shown not in moves:
                    # drawn in: visible from the stage's first frame; drawn out: gone at its end
                    moves[shown] = "1" if direction == "in" else "1 - floor(s)"
        plan.append(
            {"stage": spec.get("name", f"stage {index + 1}"), "effects": summary}
            if summary
            else {"stage": spec.get("name", f"stage {index + 1}")}
        )

    made = _bindings(subjects, channels)
    taken = {(b["target"], b["property"]) for b in made}
    for source in list(bindings or []) + [b.describe() for b in entry.bindings]:
        if not isinstance(source, Mapping):
            continue
        clash = (source.get("target"), source.get("property"))
        if clash in taken:
            raise McpError(
                f"{clash[0]}.{clash[1]} is driven by an effect and by a binding; drop one of "
                "them (effects drive alpha, reveal, color and the size of the artists they name)."
            )
    out_bindings = list(bindings or []) + made
    return out_stages, out_vars, out_bindings, {"stages": plan, "notes": notes}


def _resolve_channel(
    subject: _Subject, effect: str, channel: str, notes: Optional[List[str]]
) -> str:
    """The channel an effect really uses on this artist (a field cannot be drawn: it fades)."""
    if channel == "draw" and not subject.can_reveal():
        if not subject.can_fade():
            raise McpError(
                f"{effect} on {subject.id}: {subject.tool} draws something that can neither be "
                "drawn in part nor faded. Use effects on add_plot, add_function, add_field and "
                "add_annotation artists."
            )
        if notes is not None:
            what = subject.kind or subject.tool
            note = f"{subject.id} ({what}) cannot be drawn in part: {effect} fades it."
            if note not in notes:
                notes.append(note)
        return "show"
    if channel == "show" and not subject.can_fade():
        raise McpError(f"{effect} on {subject.id}: {subject.tool} artists have no alpha to fade.")
    if channel == "pulse" and subject.color() is None:
        raise McpError(
            f"indicate on {subject.id}: it has no single colour to highlight (a colormapped "
            "field or recipe); fade it, or give it a color."
        )
    return channel


def _bindings(subjects: Mapping[str, _Subject], channels: Mapping[Tuple[str, str], str]):
    made: List[Dict[str, Any]] = []
    for (artist, channel), name in channels.items():
        subject = subjects[artist]
        if channel == "show":
            base = subject.alpha()
            value = name if base == 1.0 else f"{_number(base)} * {name}"
            made.append({"target": artist, "property": "alpha", "value": value})
        elif channel == "draw":
            made.append({"target": artist, "property": "reveal", "value": name})
        else:
            color = subject.color()
            made.append(
                {
                    "target": artist,
                    "property": "color",
                    "value": {"colormap": [color, HIGHLIGHT], "at": name},
                }
            )
            prop, base, grow = _size_of(subject)
            if prop:
                made.append(
                    {
                        "target": artist,
                        "property": prop,
                        "value": f"{_number(base)} * (1 + {grow} * {name})",
                    }
                )
    return made


def _size_of(subject: _Subject) -> Tuple[Optional[str], float, float]:
    """The property ``indicate`` enlarges, its current value and how much it grows."""
    if subject.tool == "add_annotation" and subject.kind == "text":
        base = subject.args.get("fontsize") or subject.style_number("text_size_px", 12.0)
        return "fontsize", float(base), 0.2
    if subject.tool == "add_annotation" and subject.kind == "arrow":
        return "linewidth", float(subject.args.get("linewidth") or 3.0), 0.8
    if subject.tool in ("add_plot", "add_function"):
        if (subject.kind or "line") == "scatter":
            base = subject.args.get("size") or subject.style_number("point_size", 6.0)
            return "size", float(base), 0.5
        if "linewidth" in subject.arguments:
            base = subject.args.get("linewidth") or subject.style_number("line_width", 1.5)
            return "linewidth", float(base), 0.8
    return None, 0.0, 0.0


def effects_guide() -> str:
    """The effect names, for the guide."""
    return ", ".join(f"`{name}`" for name in EFFECTS)
