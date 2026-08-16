"""Panel-grid compositing shared by mpl_version.py and glplot_version.py.

Both scripts render each panel (and inset) as its own single-panel image and
hand the file paths here -- using the exact same code to assemble the final
grid and draw the title is what guarantees the two composites are laid out
and captioned identically, rather than relying on two independent
implementations to happen to agree.

Why per-panel images at all, instead of one `plt.subplots(3, 2)` figure for
mpl_version.py too: GLPlot's headless `savefig()` can only ever render one
panel per figure (see glplot_version.py's module docstring), so matching
mpl_version.py to that -- one figure per panel, composited afterward -- is
what makes the two pipelines comparable at every stage, not just in the
final pixels.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

#: Both scripts build every panel at this exact physical size and DPI. GLPlot's
#: headless preview (glplot/utils/preview.py) hardcodes dpi=120 for the
#: matplotlib figure it reconstructs -- not configurable from the calling
#: script -- so mpl_version.py matches that number rather than the other way
#: around. With the same figsize *and* the same DPI, a "9pt" font or a
#: "1.5pt" line ends up the same pixel size in both outputs; matching only one
#: of the two would still leave a scale mismatch.
PANEL_FIGSIZE = (6.4, 5.6)
PANEL_DPI = 120
#: render_preview also floors the *inset* figure at 4.0x3.0in
#: (``max(engine.width/100, 4.0)``, ``max(engine.height/100, 3.0)`` in
#: glplot/utils/preview.py) regardless of the figsize passed to
#: ``gplt.figure()`` -- requesting anything smaller (as this used to, at
#: 2.6x2.3) silently got the floor size instead, so GLPlot's inset PNG came
#: back at a different pixel size *and* a different aspect ratio (4:3 vs
#: 2.6:2.3) than mpl_version.py's, which does honor a smaller figsize exactly.
#: Matching that floor here, instead of fighting it, is what makes the two
#: inset images the same shape before ``composite_inset`` resizes either of
#: them into the parent panel's inset box.
INSET_FIGSIZE = (4.0, 3.0)

TITLE_FONT_PT = 15
TITLE_BAR_PX = int(round(TITLE_FONT_PT * PANEL_DPI / 72.0 * 2.2))
GAP_PX = 18


def load_font(size_px):
    for candidate in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    ):
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size_px)
    return ImageFont.load_default()


def composite_inset(parent_path, inset_path, frac_box):
    """Paste ``inset_path`` onto ``parent_path`` at a fraction-of-panel box, with a border."""
    parent = Image.open(parent_path).convert("RGB")
    inset = Image.open(inset_path).convert("RGB")
    pw, ph = parent.size
    x0, y0, w, h = frac_box
    box_w, box_h = int(pw * w), int(ph * h)
    inset = inset.resize((box_w - 6, box_h - 6))
    bordered = Image.new("RGB", (box_w, box_h), "black")
    bordered.paste(inset, (3, 3))
    parent.paste(bordered, (int(pw * x0), int(ph * y0)))
    parent.save(parent_path)


def assemble_grid(paths_grid, out_path, title):
    """Lay ``paths_grid`` (a list of rows of image paths) into one canvas with a centered title."""
    images = [[Image.open(p).convert("RGB") for p in row] for row in paths_grid]
    cell_w = max(im.width for row in images for im in row)
    cell_h = max(im.height for row in images for im in row)
    n_rows, n_cols = len(images), len(images[0])
    canvas_w = n_cols * cell_w + (n_cols + 1) * GAP_PX
    canvas_h = TITLE_BAR_PX + n_rows * cell_h + (n_rows + 1) * GAP_PX
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)

    font = load_font(int(round(TITLE_FONT_PT * PANEL_DPI / 72.0)))
    bbox = draw.textbbox((0, 0), title, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    draw.text(
        ((canvas_w - text_w) / 2, (TITLE_BAR_PX - text_h) / 2 - bbox[1]),
        title,
        fill="black",
        font=font,
    )

    for r, row in enumerate(images):
        for c, im in enumerate(row):
            x = GAP_PX + c * (cell_w + GAP_PX)
            y = TITLE_BAR_PX + GAP_PX + r * (cell_h + GAP_PX)
            canvas.paste(im, (x, y))
    canvas.save(out_path)
    print(f"wrote {out_path}")
