"""What this GLPlot install can do, and whether it works -- for people and for models.

:func:`capabilities` is cheap and read-only (versions, counts, optional pieces present).
:func:`doctor` additionally *runs* things: it renders a tiny figure headless, which is the one check
that tells you the install can really make a plot.
"""

from __future__ import annotations

import importlib
import importlib.metadata as metadata
import platform
import shutil
import sys
from typing import Any, Dict, List, Optional

__all__ = ["capabilities", "doctor", "format_doctor"]


def _version(distribution: str) -> Optional[str]:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def _mcp_sdk() -> Optional[str]:
    """The installed ``mcp`` SDK version, or None (the built-in server is used then)."""
    version = _version("mcp")
    if version is None:
        return None
    try:
        importlib.import_module("mcp")
    except ImportError:
        return None
    return version


def capabilities() -> Dict[str, Any]:
    """Versions, optional dependencies and the size of each part of the API."""
    import glplot

    from .. import pyplot
    from ..gui import recipes
    from . import apiindex, example_catalog, guides

    return {
        "glplot": glplot.__version__,
        "python": platform.python_version(),
        "platform": f"{platform.system()} {platform.release()}",
        "dependencies": {
            "numpy": _version("numpy"),
            "scipy": _version("scipy"),
            "matplotlib": _version("matplotlib"),
            "glfw": _version("glfw"),
            "PyOpenGL": _version("PyOpenGL"),
            "imgui-bundle": _version("imgui-bundle"),
        },
        "optional": {
            "mcp_sdk": _mcp_sdk(),
            "ffmpeg": shutil.which("ffmpeg"),
        },
        "api": {
            "functions": len([e for e in apiindex.entries() if e.module == pyplot.__name__]),
            "categories": len(apiindex.categories()),
            "style_presets": len(pyplot.plot_styles()),
            "recipes": len(recipes.RECIPES),
            "guide_topics": len(guides.list_topics()),
            "examples": len(example_catalog.list_examples()),
        },
        "outputs": {
            "headless_export": "png, pdf, svg, jpg via matplotlib; no display needed",
            "live_window": "plt.show() needs a display and OpenGL 3.3+",
            "animation": "glplot.animation: gif (Pillow); mp4 needs ffmpeg"
            + ("" if shutil.which("ffmpeg") else " (not found on PATH)"),
        },
    }


def _check(name: str, fn: Any) -> Dict[str, Any]:
    try:
        detail = fn()
        return {"check": name, "ok": True, "detail": detail}
    except Exception as exc:  # noqa: BLE001 - a diagnostic reports, it does not raise
        return {"check": name, "ok": False, "detail": f"{type(exc).__name__}: {exc}"}


def doctor() -> List[Dict[str, Any]]:
    """Run the checks that decide whether a plot can be made here."""
    results: List[Dict[str, Any]] = []

    results.append(
        _check("python >= 3.11", lambda: _need(sys.version_info >= (3, 11), sys.version.split()[0]))
    )
    for module in ("numpy", "scipy", "matplotlib"):
        results.append(
            _check(f"import {module}", lambda m=module: importlib.import_module(m).__version__)
        )
    results.append(
        _check("import glplot.pyplot", lambda: importlib.import_module("glplot.pyplot") and "ok")
    )

    def _render() -> str:
        import numpy as np

        import glplot.pyplot as plt

        from ..utils.preview import render_preview_bytes

        plt.figure("doctor", figsize=(3, 2))
        x = np.linspace(0, 1, 20)
        plt.plot(x, x * x)
        plt.title("doctor")
        try:
            data = render_preview_bytes(plt.get_engine())
        finally:
            plt.close("all")
        if not data.startswith(b"\x89PNG"):
            raise RuntimeError("the render did not produce a PNG")
        return f"headless render ok ({len(data)} bytes)"

    results.append(_check("headless render", _render))
    results.append(
        _check(
            "mcp",
            lambda: (
                f"official SDK {_mcp_sdk()}"
                if _mcp_sdk()
                else "SDK not installed: the built-in stdio server is used "
                "(pip install 'glplot[mcp]' for HTTP)"
            ),
        )
    )
    results.append(
        _check(
            "ffmpeg (mp4 animations)",
            lambda: shutil.which("ffmpeg") or "not on PATH: gif works, mp4 does not",
        )
    )
    return results


def _need(condition: bool, detail: str) -> str:
    if not condition:
        raise RuntimeError(f"{detail} is too old; GLPlot needs Python 3.11+")
    return detail


def format_doctor(results: List[Dict[str, Any]]) -> str:
    """The doctor report as plain text."""
    lines = []
    for item in results:
        mark = "ok  " if item["ok"] else "FAIL"
        lines.append(f"[{mark}] {item['check']}: {item['detail']}")
    failed = sum(1 for item in results if not item["ok"])
    lines.append("")
    lines.append("All checks passed." if not failed else f"{failed} check(s) failed.")
    return "\n".join(lines)
