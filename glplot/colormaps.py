"""GLPlot's own colormaps, registered with matplotlib on import.

A style preset has a palette for *categories* (which line is which) and, since presets own a
colormap too, a ramp for *values* (a heat map, a surface, a colour-by-column scatter). The
matplotlib built-ins cover the scientific cases well -- viridis, cividis and gray are what
Clean, Colour-safe and Print use -- but they cannot give a look its own voice: a synthwave
page whose heat map is viridis is a synthwave page with a stock chart pasted into it.

These seven are written the way the built-ins are, as a short list of anchor colours
interpolated in RGB, and are held to the same rule that makes a ramp readable: **lightness
only ever moves one way**. Dark-page ramps climb from near the page colour to near-white
(so "low" dissolves into the background and "high" glows); the one light-page ramp descends
from paper to ink. Nothing here is a rainbow, which is the usual way a decorative ramp
stops being able to answer "which of these two is bigger".

Registration is idempotent and never overrides somebody else's colormap of the same name:
if another library already registered ``aurora`` the matplotlib one stays and ours is
reachable only as ``glplot_aurora``. :func:`resolve` hides that from callers.
"""

from __future__ import annotations

from typing import Dict, Tuple

__all__ = ["CUSTOM_COLORMAPS", "ensure_registered", "resolve"]

RGB = Tuple[float, float, float]


def _hex(value: str) -> RGB:
    value = value.lstrip("#")
    return tuple(int(value[i : i + 2], 16) / 255.0 for i in (0, 2, 4))  # type: ignore[return-value]


#: name -> anchor colours, low to high.
CUSTOM_COLORMAPS: Dict[str, Tuple[RGB, ...]] = {
    # Polar night to the green curtain: deep navy, teal, aurora green, then pale lime.
    "aurora": tuple(
        _hex(c) for c in ("#071427", "#0b3a52", "#0e7c86", "#27d3a2", "#a6f56b", "#f3ffd2")
    ),
    # Outrun: ultraviolet, magenta, hot pink, orange, then a near-white yellow.
    "synthwave": tuple(
        _hex(c) for c in ("#12062a", "#4a1485", "#b81fb0", "#ff4f84", "#ffa24a", "#fff4b0")
    ),
    # Dusk to horizon: plum, rose, orange, gold, cream.
    "sunset": tuple(
        _hex(c) for c in ("#220c2e", "#6b1d5c", "#c93f5e", "#f27b3a", "#ffc65e", "#fff5d6")
    ),
    # Deep water to shallows: abyss, cobalt, teal, seafoam, foam.
    "lagoon": tuple(
        _hex(c) for c in ("#03112a", "#0a3566", "#0f78a6", "#22bfc0", "#8bedcf", "#effff7")
    ),
    # Paper to ink, the only descending ramp: for a light page, where "low" is the page.
    "parchment": tuple(
        _hex(c) for c in ("#f7efd9", "#e6c985", "#cf8a45", "#a04a31", "#5a2638", "#1f1226")
    ),
    # manim's blues climbing to its yellow, for the manim look: night, BLUE_E, BLUE_D,
    # BLUE_C, TEAL_B, YELLOW_C, YELLOW_A. Lightness rises the whole way.
    "manim": tuple(
        _hex(c)
        for c in ("#0b1a26", "#236B8E", "#29ABCA", "#58C4DD", "#76DDC0", "#F7D96F", "#FFF1B6")
    ),
    # A light, saturated pop ramp: white-pink to raspberry to plum.
    "candy": tuple(
        _hex(c) for c in ("#fff1f7", "#ffb3d1", "#ff6fa6", "#d6308a", "#7d1f8e", "#2c1763")
    ),
    # Deep space to a sunlit planet, for the Kurzgesagt look: night blue, indigo, violet,
    # hot pink, orange, pale gold.
    "kurzgesagt": tuple(
        _hex(c) for c in ("#11163A", "#33287A", "#7B3FA8", "#E0457B", "#FF9F45", "#FFE29A")
    ),
    # Salmon paper down to claret ink, for the Financial Times look (a light-page ramp).
    "claret": tuple(
        _hex(c) for c in ("#FFF1E5", "#F2C7AE", "#E0857A", "#B5345A", "#7A0F35", "#3D0A1E")
    ),
    # ggplot2's default continuous scale (#132B43 to #56B1F7), carried on to a pale sky so
    # it spans enough lightness to read.
    "ggblue": tuple(_hex(c) for c in ("#132B43", "#22496A", "#3A7DB0", "#56B1F7", "#BEE3FF")),
    # Nord: polar night through the frost blues to snow storm.
    "frost": tuple(
        _hex(c) for c in ("#2E3440", "#3B4A63", "#5E81AC", "#81A1C1", "#88C0D0", "#ECEFF4")
    ),
    # Catppuccin Mocha: base, surface, mauve, pink, a cream yellow.
    "mocha": tuple(
        _hex(c) for c in ("#1E1E2E", "#45475A", "#9C7BD0", "#CBA6F7", "#F5C2E7", "#F9E2AF")
    ),
    # Tokyo Night: the night, its indigo and blues, cyan, then near-white.
    "tokyo": tuple(
        _hex(c) for c in ("#1A1B26", "#2F3B73", "#4C6BC8", "#7AA2F7", "#7DCFFF", "#E6EEFF")
    ),
    # A terminal's amber phosphor: black to amber to a pale glow.
    "amber": tuple(
        _hex(c) for c in ("#050505", "#3D1F00", "#8A4B00", "#FB8B1E", "#FFC46B", "#FFF1D6")
    ),
}

_registered = False


def ensure_registered() -> None:
    """Register every colormap in :data:`CUSTOM_COLORMAPS` with matplotlib, once."""
    global _registered
    if _registered:
        return
    import matplotlib
    from matplotlib.colors import LinearSegmentedColormap

    registry = matplotlib.colormaps
    for name, anchors in CUSTOM_COLORMAPS.items():
        colormap = LinearSegmentedColormap.from_list(name, list(anchors), N=256)
        existing = name in registry
        if existing and getattr(registry[name], "_glplot_owned", False):
            continue
        target = f"glplot_{name}" if existing else name
        if target in registry:
            continue
        colormap._glplot_owned = True  # type: ignore[attr-defined]
        colormap.name = target
        registry.register(colormap, name=target)
    _registered = True


def resolve(name: str) -> str:
    """The registered name to ask matplotlib for when you mean GLPlot's ``name``.

    Plain ``name`` unless another library got there first, then ``glplot_<name>``. Any name
    that is not one of ours is returned unchanged, so this is safe to call on every colormap
    a style carries.
    """
    ensure_registered()
    if name not in CUSTOM_COLORMAPS:
        return name
    import matplotlib

    colormap = matplotlib.colormaps[name] if name in matplotlib.colormaps else None
    if colormap is not None and getattr(colormap, "_glplot_owned", False):
        return name
    return f"glplot_{name}"


ensure_registered()
