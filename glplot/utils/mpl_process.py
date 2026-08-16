"""Hand a GLPlot snapshot to a matplotlib viewer running in a separate process.

Why a subprocess and not an in-process figure?

On macOS the GLFW event pump (``glfw.poll_events``) drains and recreates the
shared ``NSAutoreleasePool``.  A matplotlib window created from inside the GL
loop is autoreleased into that pool, so it is deallocated on the next poll: the
figure opens, then vanishes roughly a second later.  Running matplotlib on a
second thread is not an option either -- the MacOSX backend aborts the process
when driven off the main thread.  A child process owns its own main thread and
its own event loop, so the figure is a real interactive window that is fully
decoupled from the GL loop and outlives the parent.

The module is pure (numpy + stdlib only) and imports headless.  It is also a
runnable entry point::

    python -m glplot.utils.mpl_process <payload.npz>

which is exactly how :func:`launch_snapshot_viewer` starts the child.
"""

from __future__ import annotations

import atexit
import importlib.util
import os
import subprocess
import sys
import tempfile
import warnings
from typing import Any, List, Optional, Tuple

import numpy as np

__all__ = [
    "MATPLOTLIB_MISSING_MESSAGE",
    "active_viewers",
    "launch_snapshot_viewer",
    "load_payload",
    "main",
    "matplotlib_available",
    "write_payload",
]

MATPLOTLIB_MISSING_MESSAGE = (
    "matplotlib is not installed, so the snapshot cannot be opened in a "
    "matplotlib window. Install it with 'pip install matplotlib'."
)

# (process, payload_path) for every viewer this process has spawned.
_VIEWERS: List[Tuple[Any, str]] = []


def matplotlib_available() -> bool:
    """Return True if matplotlib can be imported in this interpreter."""
    try:
        return importlib.util.find_spec("matplotlib") is not None
    except (ImportError, ValueError):
        return False


def _text_or_none(value: Any) -> Optional[str]:
    """Normalise an optional label to a non-empty string or None."""
    if value is None:
        return None
    text = str(value)
    return text if text else None


def write_payload(
    snapshot: Any,
    path: str,
    *,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
) -> str:
    """Serialise a :class:`GLPlotSnapshot` plus axis labels to ``path`` (npz).

    Only plain arrays and strings are written, so the child can load the file
    with ``allow_pickle=False``.
    """
    rgba = np.ascontiguousarray(snapshot.rgba)
    with open(path, "wb") as handle:
        np.savez_compressed(
            handle,
            rgba=rgba,
            extent=np.asarray(snapshot.extent, dtype=np.float64),
            xlim=np.asarray(snapshot.xlim, dtype=np.float64),
            ylim=np.asarray(snapshot.ylim, dtype=np.float64),
            width_px=np.asarray(int(snapshot.width_px), dtype=np.int64),
            height_px=np.asarray(int(snapshot.height_px), dtype=np.int64),
            transparent=np.asarray(bool(snapshot.transparent), dtype=np.bool_),
            projected_3d=np.asarray(bool(getattr(snapshot, "projected_3d", False)), dtype=np.bool_),
            xlabel=np.asarray(_text_or_none(xlabel) or ""),
            ylabel=np.asarray(_text_or_none(ylabel) or ""),
            title=np.asarray(_text_or_none(title) or ""),
        )
    return path


def load_payload(path: str) -> Tuple[Any, dict]:
    """Load a payload written by :func:`write_payload`.

    Returns ``(snapshot, labels)`` where ``labels`` has ``xlabel``/``ylabel``/
    ``title`` keys, each either a string or None.
    """
    from .mpl_bridge import GLPlotSnapshot

    with np.load(path, allow_pickle=False) as data:
        snapshot = GLPlotSnapshot(
            rgba=np.array(data["rgba"]),
            extent=tuple(float(v) for v in data["extent"]),
            xlim=tuple(float(v) for v in data["xlim"]),
            ylim=tuple(float(v) for v in data["ylim"]),
            width_px=int(data["width_px"]),
            height_px=int(data["height_px"]),
            transparent=bool(data["transparent"]),
            # Payloads written before this field existed simply do not carry it.
            projected_3d=bool(data["projected_3d"]) if "projected_3d" in data else False,
        )
        labels = {
            "xlabel": _text_or_none(str(data["xlabel"])),
            "ylabel": _text_or_none(str(data["ylabel"])),
            "title": _text_or_none(str(data["title"])),
        }
    return snapshot, labels


def _unlink(path: str) -> None:
    """Delete ``path``, ignoring a file that is already gone."""
    try:
        os.unlink(path)
    except OSError:
        pass


def _reap() -> None:
    """Reap exited viewers so they never linger as zombies, and drop payloads."""
    alive: List[Tuple[Any, str]] = []
    for proc, path in _VIEWERS:
        if proc.poll() is None:
            alive.append((proc, path))
        else:
            # The child unlinks the payload once loaded; clean up if it died first.
            _unlink(path)
    _VIEWERS[:] = alive


def active_viewers() -> int:
    """Return the number of viewer processes this process spawned that are alive."""
    _reap()
    return len(_VIEWERS)


atexit.register(_reap)


def _child_env() -> dict:
    """Environment for the child, with this glplot importable on PYTHONPATH."""
    env = dict(os.environ)
    # .../<root>/glplot/utils/mpl_process.py -> <root>
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    existing = env.get("PYTHONPATH", "")
    parts = [root] + [p for p in existing.split(os.pathsep) if p]
    env["PYTHONPATH"] = os.pathsep.join(parts)
    return env


def _spawn_kwargs() -> dict:
    """Platform flags that detach the viewer from the parent's signals/session."""
    if os.name == "nt":
        flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        return {"creationflags": flags} if flags else {}
    # POSIX: a new session means Ctrl-C in the parent's terminal does not also
    # kill the figure, and the child keeps running after the parent exits.
    return {"start_new_session": True}


def launch_snapshot_viewer(
    snapshot: Any,
    *,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
) -> Optional[Any]:
    """Open ``snapshot`` in a matplotlib window owned by a separate process.

    Returns the :class:`subprocess.Popen` handle, or None if the viewer could
    not be started (matplotlib missing, or the spawn failed).  Never raises and
    never blocks the caller -- it is safe to call from the GL loop.
    """
    _reap()

    if not matplotlib_available():
        warnings.warn(MATPLOTLIB_MISSING_MESSAGE, RuntimeWarning, stacklevel=2)
        return None

    fd, path = tempfile.mkstemp(prefix="glplot_snapshot_", suffix=".npz")
    os.close(fd)
    try:
        write_payload(snapshot, path, xlabel=xlabel, ylabel=ylabel, title=title)
    except Exception as exc:  # pragma: no cover - defensive, e.g. disk full
        _unlink(path)
        warnings.warn(f"Could not serialize snapshot for matplotlib: {exc}", RuntimeWarning)
        return None

    # Always a fresh interpreter (spawn, never fork): forking a process with a
    # live GL context is unsafe on macOS.
    cmd = [sys.executable, "-m", "glplot.utils.mpl_process", path]
    try:
        proc = subprocess.Popen(cmd, env=_child_env(), cwd=os.getcwd(), **_spawn_kwargs())
    except Exception as exc:
        _unlink(path)
        warnings.warn(f"Could not start the matplotlib viewer process: {exc}", RuntimeWarning)
        return None

    _VIEWERS.append((proc, path))
    return proc


def main(argv: Optional[List[str]] = None) -> int:
    """Child entry point: load a payload, show it, block until the user closes it."""
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        print("usage: python -m glplot.utils.mpl_process <payload.npz>", file=sys.stderr)
        return 2

    path = args[0]
    try:
        snapshot, labels = load_payload(path)
    except Exception as exc:
        print(f"glplot: could not read snapshot payload: {exc}", file=sys.stderr)
        return 1
    finally:
        # The payload has served its purpose either way; never leave it behind.
        _unlink(path)

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"glplot: {MATPLOTLIB_MISSING_MESSAGE} ({exc})", file=sys.stderr)
        return 1

    from .mpl_bridge import snapshot_to_matplotlib

    fig, ax, _artist = snapshot_to_matplotlib(snapshot)
    if labels["xlabel"]:
        ax.set_xlabel(labels["xlabel"])
    if labels["ylabel"]:
        ax.set_ylabel(labels["ylabel"])
    if labels["title"]:
        ax.set_title(labels["title"])

    manager = getattr(fig.canvas, "manager", None)
    if manager is not None and hasattr(manager, "set_window_title"):
        try:
            manager.set_window_title(labels["title"] or "GLPlot -> Matplotlib")
        except Exception:
            pass

    # This process owns its main thread and its own event loop, so blocking here
    # is exactly right: it keeps the figure interactive until the user closes it.
    plt.show()
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via subprocess
    sys.exit(main())
