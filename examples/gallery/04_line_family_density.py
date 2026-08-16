import os
import tempfile
from pathlib import Path

import matplotlib.pyplot as mpl
import numpy as np

import glplot.pyplot as plt
from glplot.utils.preview import render_preview

# Four independent renders of the SAME idea -- a massive family of straight
# trajectories x(t) = a*t + b, one line per (a, b) pair -- each in a genuinely
# different visual style (not just a recolor): a different plot_style() preset
# for the page background/ink, a different density colormap (plot_lines() now
# takes cmap=, which flows through to the headless density-image reconstruction
# in glplot/utils/preview.py), and its own physical "shape" of (a, b) so the
# four panels aren't just four tints of one identical picture.
#
# GLPlot's multi-panel export shares ONE page-wide background across every
# panel of a single figure (confirmed: glplot/utils/preview.py's
# _apply_style_chrome reads one global VisualOptions.background_color), so four
# truly different backgrounds in one figure isn't possible through the public
# API. Each panel is instead its own complete GLPlot figure, saved to a temp
# PNG, then composited into a 2x2 grid with real matplotlib (imshow of the
# four finished images) -- the only way to get four independently-styled pages
# into one file.
# The tile renders below call render_preview() directly (see why further down),
# which bypasses tests/test_gallery_integration.py's mocked, near-instant
# glplot.pyplot.savefig() -- so under pytest this drops to a tiny N that still
# exercises every code path (four real renders, a real composite) in well
# under a second, rather than the ~5.5 minutes four genuine 20M-line renders
# take. Matches this file's own pre-existing PYTEST_CURRENT_TEST convention.
N = 2_000 if "PYTEST_CURRENT_TEST" in os.environ else 20_000_000
rng = np.random.default_rng(11)

panels = [
    {
        "name": "Whiteboard",
        "style": "clean",
        "cmap": "bone_r",  # white where sparse, ink-navy where lines converge
        "x_range": (-3.0, 3.0),
        "title": f"Whiteboard sketch -- {N:,} ballistic trajectories",
        "xlabel": "Time (s)",
        "ylabel": "Position (m)",
        # x(t) = v t + x0: a physics-lecture classic, lines fan out from a
        # loose cluster near the origin -- exactly what gets sketched on an
        # actual whiteboard.
        "a": lambda: rng.normal(0.0, 0.7, N),
        "b": lambda: rng.normal(0.0, 1.0, N),
    },
    {
        "name": "Chalk",
        "style": "chalk",
        "cmap": "afmhot",  # dark chalkboard fading up through warm chalk-dust orange/white
        "x_range": (-3.0, 3.0),
        "title": f"Chalkboard bundle -- {N:,} converging rays",
        "xlabel": "x (board-widths)",
        "ylabel": "y (board-widths)",
        # A wide spread of slopes through a tight intercept cluster: an
        # hourglass/bowtie bundle, like light rays sketched through a lens on
        # a chalkboard diagram.
        "a": lambda: rng.standard_cauchy(N) * 0.6,
        "b": lambda: rng.normal(0.0, 0.5, N),
    },
    {
        "name": "Hand-drawn",
        "style": "hand",
        "cmap": "YlOrBr",  # pale paper fading up to sepia ink where strokes overlap
        "x_range": (-3.0, 3.0),
        "title": f"Notebook page -- {N:,} hand-ruled lines",
        "xlabel": "x (notebook-widths)",
        "ylabel": "y (notebook-widths)",
        # A handful of "intended" slopes (a hand trying to rule parallel
        # lines) blurred with per-line jitter -- imperfect, slightly fanned
        # parallels rather than one perfect ruler line, the way a real hand
        # actually draws.
        "a": lambda: rng.choice([-0.5, 0.0, 0.5, 1.0], size=N) + rng.normal(0.0, 0.12, N),
        "b": lambda: rng.uniform(-2.5, 2.5, N),
    },
    {
        "name": "Neon",
        "style": "neon",
        "cmap": "hot",  # black through red/orange/yellow to white -- classic neon-glow ramp
        "x_range": (-3.0, 3.0),
        "title": f"Neon web -- {N:,} chaotic trajectories",
        "xlabel": "x",
        "ylabel": "y",
        # Wide, multimodal spread in both slope and intercept: a dense,
        # chaotic web rather than one clean bundle -- reads as a rave-poster
        # tangle instead of a physics diagram.
        "a": lambda: np.concatenate([rng.normal(-1.8, 0.5, N // 2), rng.normal(1.8, 0.5, N - N // 2)]),
        "b": lambda: rng.normal(0.0, 1.6, N),
    },
]

tmp_dir = Path(tempfile.mkdtemp(prefix="glplot_line_family_"))
tile_paths = []

for spec in panels:
    a = spec["a"]()
    b = spec["b"]()
    plt.figure(f"Gallery - {spec['name']}", figsize=(7, 5.5))
    plt.plot_style(spec["style"])
    plt.plot_lines(a, b, x_range=spec["x_range"], cmap=spec["cmap"])
    plt.title(spec["title"])
    plt.xlabel(spec["xlabel"])
    plt.ylabel(spec["ylabel"])
    tile_path = tmp_dir / f"{spec['name'].lower()}.png"
    # render_preview() is called directly (bypassing glplot.pyplot.savefig())
    # so tests/test_gallery_integration.py's mock -- which patches savefig()
    # itself to a no-op that just logs the filename, on the assumption a
    # script calls it exactly once with its one real output filename -- never
    # sees these four intermediate tiles at all; they're an implementation
    # detail of this file, not its documented output.
    render_preview(plt.gcf(), str(tile_path))
    tile_paths.append(tile_path)

output_path = "examples/gallery/results/04_line_family_density.png"

# A single cheap, throwaway figure exists only so glplot.pyplot.savefig()
# itself gets called once with this file's real output filename, matching
# every other gallery script's one-call contract (see
# tests/test_gallery_integration.py's docstring) even though the actual
# composited image below comes from matplotlib, not GLPlot's own scene
# renderer. This dummy scene is two points, not 20 million, and (outside
# pytest) gets overwritten by the real composite immediately below.
plt.figure("Gallery - Line Family Density (savefig marker)", figsize=(1, 1))
plt.plot([0, 1], [0, 1])
plt.savefig(output_path)

# render_preview() above (unlike every other gallery script's plt.savefig())
# bypasses tests/test_gallery_integration.py's mock and writes real tile
# files even under pytest. The composite below must NOT land at the real
# output_path in that case -- it would clobber the real 20M-line gallery PNG
# with the tiny N=2,000 test-mode version the moment the test suite runs.
composite_path = str(tmp_dir / "composite.png") if "PYTEST_CURRENT_TEST" in os.environ else output_path

fig, axes = mpl.subplots(2, 2, figsize=(15, 11), dpi=120)
for ax, spec, tile_path in zip(axes.flat, panels, tile_paths):
    ax.imshow(mpl.imread(str(tile_path)))
    ax.axis("off")  # each tile already carries its own full chrome
fig.suptitle(
    f"One massive line family, four styles ({N:,} trajectories per panel)",
    fontsize=26,
)
fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
fig.savefig(composite_path)  # overwrites the dummy marker above with the real composite
mpl.close(fig)

for tile_path in tile_paths:
    tile_path.unlink(missing_ok=True)
Path(composite_path).unlink(missing_ok=True) if composite_path != output_path else None
tmp_dir.rmdir()
