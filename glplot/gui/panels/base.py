"""The :class:`Panel` base class — the contract every workspace panel implements.

A panel renders a *body* only. The :class:`~glplot.gui.workspace.Workspace` owns the
window: it calls ``imgui.begin``/``imgui.end`` around :meth:`Panel.draw`, so a panel
must never open or close its own top-level window (child regions are fine).

Panels run inside an imgui draw callback, which makes CONTRACT §1.1 binding on every
line of :meth:`draw`: **no direct mutation of the live scene**. Anything that touches
``plot.scene``, ``plot.frame`` or ``plot.cache`` goes through :meth:`Panel.submit` (or
:meth:`Panel.push_command` when it should also be undoable), which defers the work to
the queue drain at the top of ``_main_loop``. Mutating from ``draw`` does not raise —
it silently loses the dirty-flag latch (``engine.py:1238`` clears it later in the same
frame) and the edit does not appear until an unrelated event wakes the loop. That is
why the helpers exist: they are the short path, so the correct thing is also the
convenient one.

This module deliberately does not import imgui: it holds no drawing code, and staying
import-clean lets panel subclasses be constructed in a headless test.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    from ..commands import CommandQueue
    from ..datasets import DataStore
    from ..history import Command, UndoStack
    from ..workspace import Workspace


class Panel:
    """Base class for every workspace panel.

    Subclasses override the three class attributes and :meth:`draw`::

        class MyPanel(Panel):
            title = "My Panel"
            icon = "gear"
            default_open = True

            def draw(self) -> None:
                imgui.text("hello")

    Subclasses that need per-panel state override ``__init__`` and **must** call
    ``super().__init__(ws)`` first.
    """

    #: Window title. Also the panel's key in ``Workspace.panels`` unless the workspace
    #: says otherwise, so keep it unique and stable — imgui keys window position and
    #: size off the title, and renaming a panel resets the user's layout.
    title: str = "Panel"

    #: An icon name from :data:`glplot.gui.icons.ICON_SHAPES`. Drawn on the icon rail
    #: and beside the title. An unknown name draws a placeholder rather than raising.
    icon: str = "layers"

    #: Whether the panel is open the first time the workspace is built.
    default_open: bool = False

    def __init__(self, ws: Workspace) -> None:
        self._ws = ws

    # -- convenience proxies ---------------------------------------------------
    # Read-only by design: a panel that rebinds ws.store or ws.queue would silently
    # desynchronise every other panel, all of which hold the workspace's originals.

    @property
    def ws(self) -> Workspace:
        """The owning workspace."""
        return self._ws

    @property
    def plot(self) -> Any:
        """The live ``GPULinePlot``. Read freely; mutate only from :meth:`submit`."""
        return self._ws.plot

    @property
    def store(self) -> DataStore:
        """The shared :class:`~glplot.gui.datasets.DataStore`."""
        return self._ws.store

    @property
    def undo(self) -> UndoStack:
        """The shared :class:`~glplot.gui.history.UndoStack`."""
        return self._ws.undo

    @property
    def queue(self) -> CommandQueue:
        """The shared :class:`~glplot.gui.commands.CommandQueue`."""
        return self._ws.queue

    @property
    def hud(self) -> Optional[Any]:
        """The ``HudManager``, or None when the plot has no HUD.

        Needed by :func:`glplot.gui.layerops.remove_layer`, which clears the HUD's
        dangling selection. Prefers the workspace's own reference and falls back to
        the engine's, so it resolves whether or not the workspace holds one.
        """
        hud = getattr(self._ws, "hud", None)
        if hud is not None:
            return hud
        return getattr(self._ws.plot, "hud", None)

    # -- deferred mutation -----------------------------------------------------

    def submit(self, fn: Callable[[], None]) -> None:
        """Defer ``fn`` to the queue drain (CONTRACT §1.3). **The only way to mutate.**

        ``fn`` runs at the top of the next ``_main_loop`` iteration, on the GL thread,
        with the dirty-flag epilogue applied afterwards. Use for anything that is not
        worth an undo entry — autoscale, reset view, a visibility toggle.
        """
        self._ws.queue.submit(fn)

    def push_command(self, cmd: Command) -> None:
        """Defer ``cmd`` and record it on the undo stack — the undoable :meth:`submit`.

        ``UndoStack.push`` executes ``cmd.do`` itself, so pushing from inside the queued
        closure is what keeps the execution on the GL thread: the stack entry and the
        mutation land in the same drain, and neither can be interleaved with a frame.
        Pushing directly from ``draw`` would run ``do`` in the draw callback — exactly
        the §1.1 violation this class exists to prevent.

        A command built with ``undoable=False`` still runs; the stack refuses to record
        it and clears the history instead (see :class:`glplot.gui.history.Command`).
        """
        undo_stack = self._ws.undo
        self._ws.queue.submit(lambda: undo_stack.push(cmd))

    # -- lifecycle -------------------------------------------------------------

    def draw(self) -> None:
        """Render the panel body. The workspace owns the surrounding begin/end.

        The base implementation is a no-op so a subclass can be stood up incrementally
        without breaking the frame.
        """

    def __repr__(self) -> str:
        return f"<{type(self).__name__} title={self.title!r}>"
