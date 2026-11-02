"""One chart, twelve famous looks: the same data exported under each style preset.

Two products' monthly sales and the gap between them, drawn once per preset with no colour named
anywhere: the preset decides the page, the palette, the faces and the furniture of the saved
picture -- a newsroom chart for ``economist``, ``ft`` and ``fivethirtyeight``, a grey panel for
``ggplot``, a range frame for ``tufte``, a monospace screen for ``tokyonight``, a hand-drawn page
for ``chalk``. Each look is written to ``results/31_looks/<key>.png`` and the twelve are tiled into
``results/31_borrowed_looks.png``.

The looks are unaffiliated tributes to famous designs, built from their public colours and habits.
"""

import time
from pathlib import Path

import numpy as np
from PIL import Image

import glplot.pyplot as plt

LOOKS = (
    "economist",
    "ft",
    "fivethirtyeight",
    "ggplot",
    "tufte",
    "material",
    "kurzgesagt",
    "manim",
    "tokyonight",
    "catppuccin",
    "terminal",
    "chalk",
)

RESULTS = Path(__file__).resolve().parent / "results"
OUT = RESULTS / "31_looks"
OUT.mkdir(parents=True, exist_ok=True)

months = np.arange(1, 25)
rng = np.random.default_rng(31)
season = 6 * np.sin(2 * np.pi * (months - 3) / 12)
first = 40 + 1.2 * months + season + rng.normal(0, 1.5, months.size)
second = 30 + 2.1 * months + 0.6 * season + rng.normal(0, 1.5, months.size)
gap = second - first

started = time.time()
for key in LOOKS:
    plt.close("all")
    plt.figure(f"Gallery - {key}", figsize=(8, 5))
    plt.plot_style(key)
    plt.plot(months, first, label="Classic")
    plt.plot(months, second, label="Pro")
    plt.bar(months, gap, width=0.6, label="Pro minus Classic")
    plt.axhline(0.0, linewidth=1.0)
    plt.title("Pro overtook Classic in month 10")
    plt.xlabel("month")
    plt.ylabel("units sold (thousands)")
    plt.legend(loc="upper left")
    plt.savefig(str(OUT / f"{key}.png"))

# the contact sheet: four looks per row, each scaled to the same width -- from the files this
# run wrote, so a dry run that skips the saving leaves the sheet alone
written = [OUT / f"{key}.png" for key in LOOKS]
written = [p for p in written if p.exists() and p.stat().st_mtime >= started - 1.0]
if written:
    width = 640
    tiles = [Image.open(path).convert("RGB") for path in written]
    tiles = [t.resize((width, round(t.height * width / t.width))) for t in tiles]
    height = max(t.height for t in tiles)
    rows = (len(tiles) + 3) // 4
    sheet = Image.new("RGB", (width * 4, height * rows), "white")
    for i, tile in enumerate(tiles):
        sheet.paste(tile, ((i % 4) * width, (i // 4) * height))
    sheet.save(RESULTS / "31_borrowed_looks.png")
