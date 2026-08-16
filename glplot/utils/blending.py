"""The blend function for a :class:`~glplot.options.BlendMode`, in one place.

Two callers set blending and they have to agree, because the second one runs *inside* the
first: :meth:`GPULinePlot._apply_blending_policy` sets the figure's mode once before the
scene pass, and :class:`~glplot.renderers.geometry3d.Geometry3DRenderer` overrides it for
the duration of a layer that carries a mode of its own and puts the figure's back
afterwards. A second implementation of "what does ADDITIVE mean" would be a bug waiting for
the day the two drift.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from OpenGL.GL import *

if TYPE_CHECKING:
    from ..options import BlendMode

#: Modes whose result does not depend on the order the fragments arrive in.
#:
#: Addition and reverse subtraction are commutative in the destination, and screen
#: (``1 - (1-s)(1-d)``) is symmetric in the two operands, so a cloud drawn in any of them
#: composites to the same image however its points are ordered. Alpha blending is the odd
#: one out -- ``s*a + d*(1-a)`` weights whatever arrived first differently from what arrives
#: last, which is why a translucent point cloud has to be depth-sorted and why one of these
#: modes lets it skip that (:func:`glplot.renderers.geometry3d.depth_order`).
ORDER_INDEPENDENT_MODES = frozenset({"ADDITIVE", "SUBTRACTIVE", "SCREEN"})


def is_order_independent(mode: "BlendMode | None") -> bool:
    """Whether ``mode`` composites to the same image regardless of draw order."""
    return mode is not None and mode.name in ORDER_INDEPENDENT_MODES


def hides_nothing(mode: "BlendMode | None") -> bool:
    """Whether ``mode`` always lets what is already in the framebuffer show through.

    True for the accumulating modes, whose destination factor is ``GL_ONE`` (or, for screen,
    ``1 - src``): the background is never replaced, only added to, so such a layer is
    see-through at *any* alpha -- including 1.0. That is the difference from alpha blending,
    where opacity really is opacity, and it is why translucency cannot be read off the alpha
    channel alone once a layer can choose its own mode.
    """
    return is_order_independent(mode)


def apply_blend_mode(mode: "BlendMode", *, premultiplied_target: bool = False) -> None:
    """Set ``glBlendFunc``/``glBlendEquation`` for ``mode``. Assumes ``GL_BLEND`` is on.

    ``premultiplied_target`` is for the interaction cache, which is cleared to transparent
    black and composited with ``GL_ONE, GL_ONE_MINUS_SRC_ALPHA`` -- i.e. it stores
    *premultiplied* RGB. The colour channels come out premultiplied for free (``src*a +
    dst*(1-a)`` over a zero background is exactly ``a*src``), but the alpha channel does not:
    the same factors give ``a*a + dst*(1-a)``, so the cache records alpha *squared*. At the
    ``alpha=0.1`` an overplotted cloud is drawn with, the cache ended up with a hundredth of
    the coverage it painted, the composite pass then barely dimmed the page underneath, and
    the scatter washed out to white the instant the camera moved. ``GL_ONE`` for the alpha
    channel accumulates coverage the way the colour channels already accumulate
    premultiplied colour.

    ``OFF`` (and any mode a future release adds) leaves the state untouched rather than
    guessing at it.
    """
    from ..options import BlendMode

    glBlendEquation(GL_FUNC_ADD)  # Default reset

    if mode == BlendMode.ADDITIVE:
        src, dst = GL_SRC_ALPHA, GL_ONE
    elif mode == BlendMode.SUBTRACTIVE:
        glBlendEquation(GL_FUNC_REVERSE_SUBTRACT)
        src, dst = GL_SRC_ALPHA, GL_ONE
    elif mode == BlendMode.SCREEN:
        src, dst = GL_ONE, GL_ONE_MINUS_SRC_COLOR
    elif mode == BlendMode.ALPHA or mode == BlendMode.AUTO:
        src, dst = GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA
    else:
        return  # OFF, and anything a future mode adds: leave the state alone

    if premultiplied_target:
        glBlendFuncSeparate(src, dst, GL_ONE, GL_ONE_MINUS_SRC_ALPHA)
    else:
        glBlendFunc(src, dst)
