"""Regenerate the gallery thumbnails the workstation shows. Development tool.

Every card in the Style panel and the Gallery panel shows a *real render*: the figure is built
with the same functions the GUI calls, styled with the same ``apply_style`` the card's click
runs, drawn by the real OpenGL window and read back from its framebuffer (see
``tools/gl_capture.py``). Nothing is mocked up, so a preset that looks bad in a thumbnail looks
bad in the app -- which is the point of looking at them.

Needs a display and a GPU, so it is run by hand and its output (``glplot/gui/assets/gallery``)
is committed. The test-suite never runs it: it checks that the committed files exist and decode.

    python tools/build_gallery_assets.py                    # everything
    python tools/build_gallery_assets.py --styles neon,chalk --scenes lines,surface
    python tools/build_gallery_assets.py --recipes spiral_scatter
    python tools/build_gallery_assets.py --sheet sheet.png --styles clean,neon   # audit sheet
    python tools/build_gallery_assets.py --post-only        # re-encode raw captures, no GL

Two stages, so the cheap one can be repeated without a GPU: captures go to a raw directory as
lossless PNG (one child process each -- a second window in one process fails on imgui's stale
font texture), then are cropped to the card shape and encoded to WebP into the assets folder.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ASSETS = ROOT / "glplot" / "gui" / "assets" / "gallery"
DEFAULT_RAW = Path(tempfile.gettempdir()) / "glplot_gallery_raw"

#: Logical size of the window being captured. The framebuffer is 2x that on a retina display.
WINDOW = (480, 300)
#: Size of the stored thumbnail. 8:5, the window's own shape, so nothing is cropped.
THUMB = (400, 250)
WEBP_QUALITY = 84


# ----------------------------------------------------------------------------------
# Style scenes: five figures that between them exercise everything a preset sets
# ----------------------------------------------------------------------------------


def _new_plot():
    from glplot.engine import GPULinePlot

    return GPULinePlot(width=WINDOW[0], height=WINDOW[1], title="GLPlot")


def _labels(plot, title: str, xlabel: str = "", ylabel: str = "") -> None:
    plot.set_title(title)
    plot.xlabel = xlabel
    plot.ylabel = ylabel


def scene_lines(plot) -> None:
    """Five damped, phase-shifted modes: line width, palette order, grid, frame."""
    from glplot.gui import layerops

    x = np.linspace(0.0, 14.0, 700)
    for k in range(5):
        y = np.sin(x * (1.0 + 0.17 * k) + 0.62 * k) * np.exp(-x / (11.0 + 2.5 * k))
        layerops.add_xy_layer(plot, x, y, kind="line", label=f"mode {k + 1}", width=1.6)
    _labels(plot, "Damped modes", "time (s)", "amplitude")


def scene_scatter(plot) -> None:
    """Four clusters: point size and outline, additive blending, the palette as categories."""
    from glplot.gui import layerops

    rng = np.random.default_rng(11)
    centres = ((-1.6, 0.9), (1.4, 1.2), (-0.2, -1.3), (2.0, -0.9))
    for index, (cx, cy) in enumerate(centres):
        n = 900
        theta = rng.uniform(0, 2 * np.pi, n)
        radius = np.abs(rng.normal(0.0, 0.55, n))
        layerops.add_xy_layer(
            plot,
            cx + radius * np.cos(theta),
            cy + radius * np.sin(theta) * 0.8,
            kind="scatter",
            label=f"group {index + 1}",
            size=4.0,
        )
    _labels(plot, "Four populations", "PC 1", "PC 2")


def scene_bars(plot) -> None:
    """Three grouped series: filled patches, the palette cycle, a baseline."""
    from glplot.gui import layerops

    rng = np.random.default_rng(5)
    base = np.arange(7, dtype=float)
    for series in range(3):
        heights = 3.0 + 2.2 * np.sin(base * 0.9 + series) + rng.uniform(0.0, 1.2, base.size)
        layerops.add_xy_layer(
            plot,
            base + (series - 1) * 0.27,
            heights,
            kind="bar",
            label=f"series {series + 1}",
            options={"bar_width": 0.25},
        )
    _labels(plot, "Quarterly output", "quarter", "units")


def scene_field(plot) -> None:
    """A 2-D field as an image: the preset's value colormap, and nothing else to hide it."""
    from glplot.gui import layerops

    n = 180
    xs = np.linspace(-3.0, 3.0, n)
    xx, yy = np.meshgrid(xs, xs)
    field = (
        np.exp(-((xx - 1.0) ** 2 + (yy - 0.6) ** 2) * 1.1)
        + 0.8 * np.exp(-((xx + 1.2) ** 2 + (yy + 0.8) ** 2) * 0.8)
        + 0.18 * np.sin(2.2 * xx) * np.cos(2.2 * yy)
    )
    layerops.add_image_layer(plot, field, (-3.0, 3.0, -3.0, 3.0), label="field", cmap="viridis")
    _labels(plot, "Intensity map", "x", "y")


def scene_surface(plot) -> None:
    """A 3-D surface: the 3D box, depth shading, and the colormap on a mesh."""
    from glplot.gui import layerops3d

    n = 72
    xs = np.linspace(-4.0, 4.0, n)
    xx, yy = np.meshgrid(xs, xs)
    radius = np.hypot(xx, yy)
    # Rings falling off with distance, tilted by a slow saddle: full colormap range across the
    # whole sheet (a "peaks" function is flat over most of its domain, which hides the ramp).
    zz = 1.5 * np.sin(1.6 * radius) * np.exp(-0.05 * radius**2) + 0.35 * np.cos(
        0.9 * xx
    ) * np.sin(0.9 * yy)
    plot.set_ndim(3)
    layerops3d.add_xyz_layer(
        plot, xx.ravel(), yy.ravel(), zz.ravel(), kind="surface3d", label="peaks", cmap="viridis"
    )
    plot.set_3d_view(elev=30.0, azim=-58.0)
    plot.frame_3d_view()


#: Scene name -> builder. The order is the order the Style panel offers them in.
SCENES: Dict[str, Callable] = {
    "lines": scene_lines,
    "scatter": scene_scatter,
    "bars": scene_bars,
    "field": scene_field,
    "surface": scene_surface,
}


# ----------------------------------------------------------------------------------
# Jobs
# ----------------------------------------------------------------------------------


def _style_job(key: str, scene: str):
    from glplot.gui import styles

    plot = _new_plot()
    SCENES[scene](plot)
    styles.apply_style(plot, styles.get_style(key), layers=True, colormaps=True)
    return plot


def _recipe_job(key: str):
    from glplot.gui import recipes

    return recipes.build_preview_plot(
        recipes.get_recipe(key), width=WINDOW[0], height=WINDOW[1], new_plot=_new_plot
    )


def run_child(spec: str, out: Path) -> int:
    """Build ``spec`` (``style:KEY:SCENE`` or ``recipe:KEY``), capture it, write ``out``."""
    from PIL import Image

    from tools.gl_capture import capture

    kind, _, rest = spec.partition(":")
    if kind == "style":
        key, _, scene = rest.partition(":")
        plot = _style_job(key, scene)
    elif kind == "recipe":
        plot = _recipe_job(rest)
    else:
        print(f"unknown job {spec!r}", file=sys.stderr)
        return 2
    image = capture(plot)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(out)
    return 0


def _raw_path(raw_dir: Path, spec: str) -> Path:
    return raw_dir / (spec.replace(":", "__") + ".png")


def _run_parallel(specs: Sequence[str], raw_dir: Path, workers: int) -> List[str]:
    """Capture every spec in its own process, ``workers`` at a time; return the failures."""
    failures: List[str] = []

    def one(spec: str) -> Optional[str]:
        out = _raw_path(raw_dir, spec)
        started = time.time()
        done = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--child", spec, "--out", str(out)],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=240,
        )
        if done.returncode != 0 or not out.exists():
            tail = "\n".join(done.stderr.strip().splitlines()[-6:])
            print(f"  FAILED {spec}\n{tail}", flush=True)
            return spec
        print(f"  {spec}  ({time.time() - started:.1f}s)", flush=True)
        return None

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for result in pool.map(one, specs):
            if result is not None:
                failures.append(result)
    return failures


# ----------------------------------------------------------------------------------
# Post-processing
# ----------------------------------------------------------------------------------


def thumbnail(raw: Path):
    """The raw capture, scaled to :data:`THUMB` (a no-op crop: both are 8:5)."""
    from PIL import Image

    with Image.open(raw) as handle:
        return handle.convert("RGB").resize(THUMB, Image.LANCZOS)


def asset_path(spec: str) -> Path:
    kind, _, rest = spec.partition(":")
    if kind == "style":
        key, _, scene = rest.partition(":")
        return ASSETS / "styles" / f"{key}__{scene}.webp"
    return ASSETS / "recipes" / f"{rest}.webp"


def encode(specs: Sequence[str], raw_dir: Path) -> int:
    """Encode each spec's raw capture into the assets folder. Returns how many were written."""
    written = 0
    total = 0
    for spec in specs:
        raw = _raw_path(raw_dir, spec)
        if not raw.exists():
            print(f"  missing raw capture for {spec}")
            continue
        target = asset_path(spec)
        target.parent.mkdir(parents=True, exist_ok=True)
        thumbnail(raw).save(target, "WEBP", quality=WEBP_QUALITY, method=6)
        written += 1
        total += target.stat().st_size
    print(f"encoded {written} thumbnails, {total / 1024:.0f} KiB")
    return written


def contact_sheet(specs: Sequence[str], raw_dir: Path, out: Path, columns: int) -> None:
    """Tile the captures with their names, for judging a batch of looks at a glance."""
    from PIL import Image, ImageDraw

    cell_w, cell_h, label_h = 360, 225, 16
    rows = (len(specs) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * cell_w, rows * (cell_h + label_h)), (32, 32, 36))
    draw = ImageDraw.Draw(sheet)
    for index, spec in enumerate(specs):
        x = (index % columns) * cell_w
        y = (index // columns) * (cell_h + label_h)
        raw = _raw_path(raw_dir, spec)
        if raw.exists():
            with Image.open(raw) as handle:
                sheet.paste(
                    handle.convert("RGB").resize((cell_w, cell_h), Image.LANCZOS), (x, y + label_h)
                )
        draw.text(
            (x + 4, y + 2), spec.replace("style:", "").replace("recipe:", ""), fill=(235, 235, 235)
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    print(f"wrote {out}")


# ----------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------


def _csv(value: Optional[str]) -> Optional[List[str]]:
    return [part.strip() for part in value.split(",") if part.strip()] if value else None


def collect_specs(
    styles: Optional[List[str]],
    scenes: Optional[List[str]],
    recipes: Optional[List[str]],
    do_styles: bool,
    do_recipes: bool,
) -> List[str]:
    from glplot.gui import styles as style_module

    specs: List[str] = []
    if do_styles:
        keys = styles or list(style_module.STYLE_KEYS)
        for key in keys:
            style_module.get_style(key)  # a typo fails here, not after a minute of capturing
            for scene in scenes or list(SCENES):
                if scene not in SCENES:
                    raise SystemExit(f"unknown scene {scene!r}; choose from {', '.join(SCENES)}")
                specs.append(f"style:{key}:{scene}")
    if do_recipes:
        from glplot.gui import recipes as recipe_module

        for key in recipes or [r.key for r in recipe_module.RECIPES]:
            recipe_module.get_recipe(key)
            specs.append(f"recipe:{key}")
    return specs


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--child", help="(internal) capture one job and exit")
    parser.add_argument("--out", type=Path, help="(internal) where the child writes its capture")
    parser.add_argument("--styles", help="comma-separated style keys (default: all)")
    parser.add_argument("--scenes", help=f"comma-separated scenes (default: {','.join(SCENES)})")
    parser.add_argument("--recipes", help="comma-separated recipe keys (default: all)")
    parser.add_argument("--only", choices=("styles", "recipes"), help="build just one family")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--jobs", type=int, default=3, help="parallel GL windows (default: 3)")
    parser.add_argument("--post-only", action="store_true", help="re-encode existing raw captures")
    parser.add_argument("--sheet", type=Path, help="also write a contact sheet of the captures")
    parser.add_argument("--columns", type=int, default=5, help="contact sheet columns")
    parser.add_argument(
        "--no-encode", action="store_true", help="capture only; leave the assets alone"
    )
    args = parser.parse_args(argv)

    if args.child:
        return run_child(args.child, args.out)

    os.environ.setdefault("MPLBACKEND", "Agg")
    do_styles = args.only != "recipes" and not args.recipes
    do_recipes = args.only != "styles" and not (args.styles or args.scenes)
    specs = collect_specs(
        _csv(args.styles), _csv(args.scenes), _csv(args.recipes), do_styles, do_recipes
    )
    args.raw_dir.mkdir(parents=True, exist_ok=True)

    failures: List[str] = []
    if not args.post_only:
        print(f"capturing {len(specs)} views, {args.jobs} at a time -> {args.raw_dir}")
        failures = _run_parallel(specs, args.raw_dir, args.jobs)
    if args.sheet:
        contact_sheet(specs, args.raw_dir, args.sheet, args.columns)
    if not args.no_encode:
        encode([s for s in specs if s not in failures], args.raw_dir)
    if failures:
        print(f"{len(failures)} failed: {', '.join(failures)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
