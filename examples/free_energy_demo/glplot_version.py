"""GLPlot rendition of the same 6-panel free-energy-methods figure as
mpl_version.py, using GLPlot's own API for every mark. It documents (in code,
not by silently faking it) the GLPlot limitations discovered while building
this and works around each on purpose:

  * No hatch renderer -> shaded regions get a texture faked from real GLPlot
    scatter/plot primitives (stipple dots, or batched diagonal "hatch" lines).
    Which of the three fakeable kinds (dots/diagonal/cross) applies to which
    series comes from ``data.kind_cycle`` -- the same cycle mpl_version.py
    uses to pick its real hatch characters, so a given series index gets the
    same *kind* of texture in both, even though the pixels can't match.
  * No functional log-scale axis -> panel f's y-data is pre-transformed with
    log10 and the axis is labelled accordingly ("log10(error)"). The y-range
    used is ``data.panel_f_limits()['log_ylim']``, the log10 of the exact same
    range mpl_version.py hands its real ``set_yscale('log')`` axis.
  * No twin-axis (twinx/twiny are both no-op stubs) -> the secondary series in
    panels a and c is scaled into the primary axis's numeric range by
    ``twin_rescale`` (from ``data.panel_a_limits``/``panel_c_limits``), chosen
    so it covers the same *fraction* of panel height as it does in
    matplotlib's real right-hand axis.
  * No inset_axes -> each inset is rendered as its own small GLPlot figure and
    composited onto its parent panel afterwards with Pillow (compose.py,
    shared with mpl_version.py so the two composites are assembled and
    captioned by identical code).
  * Headless savefig()'s matplotlib-preview fallback (render_preview) has
    several fixed behaviors that aren't configurable from the calling script,
    so mpl_version.py is the one that adapts to match GLPlot rather than the
    other way around:
      - Every panel is rendered at a hardcoded ``dpi=120``. mpl_version.py
        therefore also builds every panel at ``dpi=120`` (``compose.PANEL_DPI``)
        -- a mismatched DPI is what made GLPlot's text/lines look smaller
        before, since a "9pt" font is a different pixel size at a different DPI
        even with identical rcParams.
      - The legend is drawn with a hardcoded style (``fontsize=8``,
        ``loc="upper right"``, fixed framealpha/borderpad/labelspacing/
        handlelength, and a 5-item cap before "+N more"). mpl_version.py copies
        these exact values into its own ``ax.legend(**LEGEND_KW)`` calls, and
        the 5-item cap is raised process-wide below (``preview.MAX_LEGEND_ITEMS``)
        since several panels here have more than 5 legend entries and
        mpl_version.py does not truncate its own.
      - ``xlim()``/``ylim()`` have **no effect** here: render_preview ends with
        an unconditional ``ax.autoscale(enable=True)`` that always re-fits the
        view to whatever was actually plotted, discarding any prior limits --
        confirmed by capturing the axes' limits at save time. Worked around by
        ``set_axis_bounds()`` below: two fully transparent, near-zero-size
        points at the corners we want visible, placed slightly inside the
        target range by the inverse of matplotlib's own autoscale margin so
        the final crop lands exactly there.
      - ``errorbar(capsize=...)`` is in *data units*, not points like
        matplotlib's. With both scripts now using the same panel size and DPI,
        ``data.capsize_data_units()`` converts a target point size into the
        data-unit width that covers the same number of pixels on a given
        panel's x-axis.
      - ``errorbar(fmt='o')`` draws a solid connector through the points --
        matplotlib treats a marker-only fmt as "no line", GLPlot does not --
        so every errorbar call here passes ``linestyle="none"`` explicitly.
      - ``violinplot()`` bodies are triangle *strips* (vertices interleave
        left/right sides); the generic patch fallback reconnects them as a
        plain closed polygon in strip order, which draws a zigzag comb instead
        of a smooth violin. Worked around locally by re-sorting each body's
        vertices into proper polygon-boundary order before export (preview-path
        only -- this would be the wrong order for GLPlot's live GPU strip
        render).

GLPlot's headless savefig() (no window ever created) renders only the
*active* panel of a multi-panel figure -- verified empirically, not just
asserted -- so each panel here is built and exported as its own single-panel
GLPlot figure, then stitched into the final 3x2 grid with Pillow via
compose.py. Every visual mark in the output image was drawn by GLPlot;
Pillow only composites finished PNGs.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as mpl
import numpy as np
from compose import INSET_FIGSIZE, PANEL_DPI, PANEL_FIGSIZE, assemble_grid, composite_inset
from data import (
    ALPHA_BAND,
    ALPHA_ERROR_BAND,
    ALPHA_HILL,
    ALPHA_HIST_FILL,
    ALPHA_TWIN_FILL,
    ALPHA_VIOLIN,
    RC_PARAMS,
    capsize_data_units,
    kind_cycle,
    panel_a_data,
    panel_a_limits,
    panel_b_data,
    panel_b_limits,
    panel_c_data,
    panel_c_limits,
    panel_d_data,
    panel_d_limits,
    panel_e_data,
    panel_e_limits,
    panel_f_data,
    panel_f_limits,
)

import glplot.pyplot as gplt
import glplot.utils.preview as preview_mod

RESULTS = Path(__file__).resolve().parent / "results"
PANELS_DIR = RESULTS / "glplot_panels"
PANELS_DIR.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update(RC_PARAMS)
# render_preview hardcodes a 5-entry cap ("+N more") that mpl_version.py's own
# legends do not apply; raised here so panels with >5 entries (b, c, e) show
# every entry in both renderings instead of only in one of them.
preview_mod.MAX_LEGEND_ITEMS = 12

CAPSIZE_PT = 2.0
MARKERSIZE_PT = 3.5


def labeled(result, text=""):
    """Pin every layer in ``result`` to exactly the label wanted for the legend.

    Needed because an unlabeled GLPlot layer is not "no label" -- the engine
    stamps it with a fallback name for its own layer-list bookkeeping, and the
    headless preview's legend builder cannot tell that apart from a real one.
    """
    layers = result if isinstance(result, (list, tuple)) else [result]
    for layer in layers:
        if layer is not None and hasattr(layer, "label"):
            layer.label = text
    return result


def no_edge(layer):
    """Strip the visible boundary stroke matplotlib would draw around a filled patch.

    ``fill_between()`` gives its returned layer the same ``edge_color`` as
    ``face_color`` (both at the requested alpha), which the headless preview
    then hands straight to ``matplotlib.patches.Polygon(facecolor=fc,
    edgecolor=ec)``. mpl_version.py's own fill_between/hist calls all pass
    ``linewidth=0.0`` (a real matplotlib no-edge), except the histogram's
    ``histtype='stepfilled'`` -- which had its own edge only because it was
    told to, and turned out not to want one, one at a time each shaded region
    stacks its own edge on top of five or six others sharing bin boundaries,
    compounding into a far more prominent grid than any single soft fill
    should show. This sets alpha to 0 on the edge alone so the Polygon draws
    with a real edgecolor="none" equivalent, whatever line_width it has.
    """
    fc = layer.style.face_color
    layer.style.edge_color = (fc[0], fc[1], fc[2], 0.0)
    return layer


def set_axis_bounds(xlim, ylim):
    """Force the exported axis range, since ``gplt.xlim()``/``gplt.ylim()`` do nothing here.

    See the module docstring for why -- the short version: render_preview's
    trailing ``ax.autoscale(enable=True)`` always wins. Two fully transparent,
    near-zero-size points at the two corners we want visible still count
    toward the data bounding box (alpha doesn't affect data limits), placed
    slightly *inside* the requested range by the exact amount matplotlib's own
    autoscale margin will restore.
    """
    mx = mpl.rcParams.get("axes.xmargin", 0.05)
    my = mpl.rcParams.get("axes.ymargin", 0.05)
    fx = mx / (1.0 + 2.0 * mx)
    fy = my / (1.0 + 2.0 * my)
    x0, x1 = xlim
    y0, y1 = ylim
    xs = [x0 + fx * (x1 - x0), x1 - fx * (x1 - x0)]
    ys = [y0 + fy * (y1 - y0), y1 - fy * (y1 - y0)]
    labeled(gplt.scatter(xs, ys, color=(0.0, 0.0, 0.0, 0.0), s=0.001, alpha=0.0))


def fix_violin_body_order(body):
    """Re-sort a violinplot body's strip-ordered vertices into polygon order.

    ``verts[0::2]`` is the left side and ``verts[1::2]`` the right side, both
    already in the same (bottom-to-top) order. A proper closed outline is the
    left side forward then the right side reversed; the headless preview
    instead draws the raw strip order as a polygon boundary, which zigzags
    between the two sides.
    """
    verts = body.vertices
    left, right = verts[0::2], verts[1::2]
    body.vertices = np.vstack([left, right[::-1]])


def clip_y(arr, ylim):
    """Clip data to the panel's intended y-range before plotting.

    render_preview's headless path has no real ``set_ylim`` -- it always
    autoscales to whatever was actually plotted (see ``set_axis_bounds``) --
    so any series that dips outside the intended range (a fill_between band
    with a negative excursion, say) drags the autoscaled view along with it,
    unlike matplotlib's ``ax.set_ylim`` + ``clip_on=True`` default, which hides
    the part outside the view instead of re-fitting to include it. Clipping
    the data first reproduces that visual clip rather than fighting autoscale.
    """
    return np.clip(arr, ylim[0], ylim[1])


def mask_to_xlim(x, y, xlim):
    """Drop points outside the panel's intended x-range before plotting.

    Same problem as ``clip_y``, on the other axis: panel a's hills are
    deliberately sampled on a grid wider than the panel's x-range
    (``x_grid`` runs -2.2..3.2 against an xlim of -2..3) so a hill centered
    near the edge doesn't get an abrupt flat cutoff -- matplotlib's real
    ``set_xlim`` + ``clip_on=True`` hides the extra margin for free. Without
    an equivalent clip here, that extra margin drags ``ax.autoscale`` wider
    than the intended range, which is what "GLPlot overextends past where
    matplotlib stops" turned out to be. Masking out-of-range points (rather
    than clamping them, which would stack them at the boundary) reproduces
    the same visual cutoff.
    """
    x = np.asarray(x)
    mask = (x >= xlim[0]) & (x <= xlim[1])
    return x[mask], np.asarray(y)[mask]


def texture_overlay(x, y_lo, y_hi, kind, color, xlim, ylim, alpha=0.5):
    """Fake a hatch texture inside a [y_lo, y_hi] band using real GLPlot marks.

    ``diag``/``cross`` build their lines in coordinates normalized by the
    *panel's* xlim/ylim (not just this band's local extent) before computing
    a 45-degree diagonal, then map back to data coordinates. Skipping this and
    using raw data units directly (as an earlier version did) treats one unit
    of x and one unit of y as the same visual distance, which is only true by
    coincidence -- it broke badly on panel f, where x spans ~1 (a fraction of
    a run) and y spans ~4 (log10 of an error), turning a "diagonal hatch" into
    a dense, wrong-angle mess.
    """
    x = np.asarray(x, dtype=float)
    y_lo = np.broadcast_to(np.asarray(y_lo, dtype=float), x.shape)
    y_hi = np.broadcast_to(np.asarray(y_hi, dtype=float), x.shape)

    x_span = xlim[1] - xlim[0]
    y_span = ylim[1] - ylim[0]
    nx = (x - xlim[0]) / x_span
    ny_lo = (y_lo - ylim[0]) / y_span
    ny_hi = (y_hi - ylim[0]) / y_span
    nxmin, nxmax = float(nx.min()), float(nx.max())
    nymin, nymax = float(min(ny_lo.min(), 0.0)), float(ny_hi.max())

    if kind == "dots":
        # A regular hex-packed grid, not a random scatter: matplotlib's own
        # dot hatch ('.') is an evenly-spaced lattice, and random points
        # instead clump unevenly (some patches empty, some overlapping),
        # which reads as noisy and darker overall than a uniform lattice at
        # the same nominal density.
        spacing = 1.0 / 40.0
        row_h = spacing * np.sqrt(3.0) / 2.0
        gx_all, gy_all = [], []
        rows = np.arange(nymin - row_h, nymax + row_h, row_h)
        for ridx, gy in enumerate(rows):
            offset = spacing / 2.0 if ridx % 2 else 0.0
            gx = np.arange(nxmin - spacing, nxmax + spacing, spacing) + offset
            gx_all.append(gx)
            gy_all.append(np.full_like(gx, gy))
        gx_all = np.concatenate(gx_all)
        gy_all = np.concatenate(gy_all)
        lo_i = np.interp(gx_all, nx, ny_lo, left=np.nan, right=np.nan)
        hi_i = np.interp(gx_all, nx, ny_hi, left=np.nan, right=np.nan)
        mask = np.isfinite(lo_i) & (gy_all >= lo_i) & (gy_all <= hi_i)
        labeled(
            gplt.scatter(
                xlim[0] + gx_all[mask] * x_span,
                ylim[0] + gy_all[mask] * y_span,
                color=color,
                s=2.0,
                alpha=alpha,
            )
        )
        return

    directions = {"diag": (1,), "cross": (1, -1)}[kind]
    span = (nxmax - nxmin) + (nymax - nymin)
    spacing = max(span / 26.0, 1e-3)
    for direction in directions:
        n_lines = int(span / spacing) + 2
        xs_all, ys_all = [], []
        for k in range(n_lines):
            c = nxmin - (nymax - nymin) + k * spacing
            t = np.linspace(0.0, 1.0, 40)
            yy = nymin + t * (nymax - nymin)
            xx = c + direction * yy
            lo_i = np.interp(xx, nx, ny_lo, left=np.nan, right=np.nan)
            hi_i = np.interp(xx, nx, ny_hi, left=np.nan, right=np.nan)
            mask = np.isfinite(lo_i) & (yy >= lo_i) & (yy <= hi_i)
            xs_all.append(np.where(mask, xx, np.nan))
            xs_all.append([np.nan])
            ys_all.append(np.where(mask, yy, np.nan))
            ys_all.append([np.nan])
        xn = np.concatenate(xs_all)
        yn = np.concatenate(ys_all)
        labeled(
            gplt.plot(
                xlim[0] + xn * x_span,
                ylim[0] + yn * y_span,
                color=color,
                linewidth=0.8,
                alpha=alpha,
            )
        )


def save(name):
    path = PANELS_DIR / f"{name}.png"
    # scale=1.0: the default (2.0) would double the physical figure size while
    # render_preview's dpi stays fixed at 120, breaking the pixel-for-pixel
    # match with mpl_version.py's panels (see compose.py).
    gplt.savefig(str(path), scale=1.0)
    return path


def build_panel_a():
    d = panel_a_data()
    lims = panel_a_limits(d)
    cmap = mpl.get_cmap("turbo")
    n = len(d["hills"])
    kinds = kind_cycle(n)
    cap = capsize_data_units(CAPSIZE_PT, lims["xlim"], PANEL_FIGSIZE[0], PANEL_DPI)

    gplt.figure(figsize=PANEL_FIGSIZE)
    for i, hill in enumerate(d["hills"]):
        color = cmap(i / max(n - 1, 1))
        hill_scaled = hill * lims["twin_rescale"]
        x_grid, hill_scaled = mask_to_xlim(d["x_grid"], hill_scaled, lims["xlim"])
        labeled(no_edge(gplt.fill_between(x_grid, hill_scaled, 0, color=color, alpha=ALPHA_HILL)))
        texture_overlay(
            x_grid, 0.0, hill_scaled, kinds[i], color, lims["xlim"], lims["y_main"], alpha=0.45
        )
    labeled(
        gplt.plot(
            d["x_exact"], clip_y(d["f_exact"], lims["y_main"]), color="crimson", linewidth=2.0
        ),
        "exact F(x)",
    )
    labeled(
        gplt.errorbar(
            d["mc_x"],
            clip_y(d["mc_y"], lims["y_main"]),
            yerr=d["mc_sigma"],
            fmt="o",
            ms=MARKERSIZE_PT,
            linestyle="none",
            color="tab:blue",
            ecolor="tab:blue",
            capsize=cap,
        ),
        "MC mean ± sigma",
    )
    set_axis_bounds(lims["xlim"], lims["y_main"])
    gplt.xlabel("position x")
    gplt.ylabel("free energy F(x)  |  biased P_i(x), rescaled (no native twin-axis)")
    gplt.title("a   Umbrella sampling windows")
    return save("a_main")


def build_inset_a():
    d = panel_a_data()
    xlim = (-0.3, 0.3)
    cap = capsize_data_units(2.0, xlim, INSET_FIGSIZE[0], PANEL_DPI)
    gplt.figure(figsize=INSET_FIGSIZE, title="")
    mask = (d["x_exact"] > -0.35) & (d["x_exact"] < 0.35)
    labeled(gplt.plot(d["x_exact"][mask], d["f_exact"][mask], color="crimson", linewidth=1.8))
    mmask = (d["mc_x"] > -0.35) & (d["mc_x"] < 0.35)
    labeled(
        gplt.errorbar(
            d["mc_x"][mmask],
            d["mc_y"][mmask],
            yerr=d["mc_sigma"][mmask],
            fmt="o",
            ms=3.0,
            linestyle="none",
            color="tab:blue",
            ecolor="tab:blue",
            capsize=cap,
        )
    )
    pad = 0.1 * (d["f_exact"][mask].max() - d["f_exact"][mask].min() + 1e-6)
    set_axis_bounds(xlim, (d["f_exact"][mask].min() - pad, d["f_exact"][mask].max() + pad))
    return save("a_inset")


def build_panel_b():
    d = panel_b_data()
    lims = panel_b_limits(d)
    colors = mpl.get_cmap("plasma")(np.linspace(0.05, 0.9, len(d["fractions"])))

    gplt.figure(figsize=PANEL_FIGSIZE)
    f_on_x = np.interp(d["x"], d["x_exact"], d["f_exact"])
    band = 0.35 * np.exp(-(((d["x"] - 0.6) / 1.4) ** 2)) + 0.05
    band_lo = clip_y(f_on_x - band, lims["ylim"])
    band_hi = clip_y(f_on_x + band, lims["ylim"])
    labeled(no_edge(gplt.fill_between(d["x"], band_lo, band_hi, color="crimson", alpha=ALPHA_BAND)))
    texture_overlay(
        d["x"], band_lo, band_hi, "dots", "crimson", lims["xlim"], lims["ylim"], alpha=0.35
    )

    labeled(
        gplt.plot(d["x_exact"], clip_y(d["f_exact"], lims["ylim"]), color="crimson", linewidth=2.2),
        "exact F(x)",
    )
    for (frac, curve), color in zip(d["curves"].items(), colors):
        labeled(
            gplt.plot(d["x"], clip_y(curve, lims["ylim"]), color=color, linewidth=1.6),
            f"{frac * 100:g}%",
        )
    set_axis_bounds(lims["xlim"], lims["ylim"])
    gplt.xlabel("position x")
    gplt.ylabel("F(x) estimate")
    gplt.title("b   Fraction of hills deposited")
    return save("b_main")


def build_inset_b():
    d = panel_b_data()
    colors = mpl.get_cmap("plasma")(np.linspace(0.05, 0.9, len(d["fractions"])))
    gplt.figure(figsize=INSET_FIGSIZE, title="")
    mask = (d["x"] > -0.35) & (d["x"] < 0.35)
    emask = (d["x_exact"] > -0.35) & (d["x_exact"] < 0.35)
    all_y = [d["f_exact"][emask]]
    for (frac, curve), color in zip(d["curves"].items(), colors):
        labeled(gplt.plot(d["x"][mask], curve[mask], color=color, linewidth=1.4))
        all_y.append(curve[mask])
    labeled(gplt.plot(d["x_exact"][emask], d["f_exact"][emask], color="crimson", linewidth=1.8))
    all_y = np.concatenate(all_y)
    pad = 0.1 * (all_y.max() - all_y.min() + 1e-6)
    set_axis_bounds((-0.3, 0.3), (all_y.min() - pad, all_y.max() + pad))
    return save("b_inset")


def build_panel_c():
    d = panel_c_data()
    lims = panel_c_limits(d)
    colors = mpl.get_cmap("viridis")(np.linspace(0.05, 0.9, len(d["temperatures"])))
    kinds = kind_cycle(len(d["temperatures"]))

    gplt.figure(figsize=PANEL_FIGSIZE)
    labeled(
        gplt.plot(
            d["x_exact"], clip_y(d["f_exact"], lims["y_main"]), color="crimson", linewidth=2.2
        ),
        "exact F(x)",
    )
    for i, (T, color) in enumerate(zip(d["temperatures"], colors)):
        p_scaled = d["p_curves"][T] * lims["twin_rescale"]
        labeled(no_edge(gplt.fill_between(d["x"], p_scaled, 0, color=color, alpha=ALPHA_TWIN_FILL)))
        texture_overlay(
            d["x"], 0.0, p_scaled, kinds[i], color, lims["xlim"], lims["y_main"], alpha=0.4
        )
        labeled(
            gplt.plot(d["x"], clip_y(d["f_curves"][T], lims["y_main"]), color=color, linewidth=1.5),
            f"T={T:g}",
        )
    set_axis_bounds(lims["xlim"], lims["y_main"])
    gplt.xlabel("position x")
    gplt.ylabel("free energy F(x)  |  P(x|T), rescaled (no native twin-axis)")
    gplt.title("c   Temperature sweep, double well")
    return save("c_main")


def build_inset_c():
    d = panel_c_data()
    colors = mpl.get_cmap("viridis")(np.linspace(0.05, 0.9, len(d["temperatures"])))
    gplt.figure(figsize=INSET_FIGSIZE, title="")
    mask = (d["x_exact"] > -0.35) & (d["x_exact"] < 0.35)
    xmask = (d["x"] > -0.35) & (d["x"] < 0.35)
    all_y = [d["f_exact"][mask]]
    labeled(gplt.plot(d["x_exact"][mask], d["f_exact"][mask], color="crimson", linewidth=1.8))
    for T, color in zip(d["temperatures"], colors):
        labeled(gplt.plot(d["x"][xmask], d["f_curves"][T][xmask], color=color, linewidth=1.3))
        all_y.append(d["f_curves"][T][xmask])
    all_y = np.concatenate(all_y)
    pad = 0.1 * (all_y.max() - all_y.min() + 1e-6)
    set_axis_bounds((-0.3, 0.3), (all_y.min() - pad, all_y.max() + pad))
    return save("c_inset")


def build_panel_d():
    d = panel_d_data()
    lims = panel_d_limits(d)
    colors = mpl.get_cmap("cividis")(np.linspace(0.05, 0.9, len(d["lambdas"])))
    cap = capsize_data_units(CAPSIZE_PT, lims["xlim"], PANEL_FIGSIZE[0], PANEL_DPI)

    gplt.figure(figsize=PANEL_FIGSIZE)
    parts = gplt.violinplot(d["samples"], positions=d["lambdas"], widths=0.07, showextrema=False)
    for i, body in enumerate(parts["bodies"]):
        # One alpha for both face and edge, matching mpl's ``body.set_alpha(ALPHA_VIOLIN)``,
        # which applies uniformly to the whole patch rather than treating the two separately.
        body.style.face_color = tuple(colors[i][:3]) + (ALPHA_VIOLIN,)
        body.style.edge_color = tuple(colors[i][:3]) + (ALPHA_VIOLIN,)
        fix_violin_body_order(body)
    labeled(parts["bodies"])

    sigmas = np.array([s.std() / np.sqrt(len(s)) for s in d["samples"]])
    labeled(
        gplt.errorbar(
            d["lambdas"],
            d["means"],
            yerr=sigmas,
            fmt="o",
            ms=4.0,
            linestyle="none",
            color="tab:blue",
            ecolor="tab:blue",
            capsize=cap,
        ),
        "MC mean ± sigma",
    )
    fit = np.polyfit(d["lambdas"], d["means"], 1)
    labeled(
        gplt.plot(d["lambdas"], np.polyval(fit, d["lambdas"]), color="crimson", linewidth=1.8),
        "analytic <dV/dlambda>",
    )
    set_axis_bounds(lims["xlim"], lims["ylim"])
    gplt.xlabel("coupling lambda")
    gplt.ylabel("<dV/dlambda>_lambda")
    gplt.title("d   Thermodynamic integration")
    tx = lims["xlim"][0] + 0.03 * (lims["xlim"][1] - lims["xlim"][0])
    ty = lims["ylim"][0] + 0.06 * (lims["ylim"][1] - lims["ylim"][0])
    labeled(gplt.text(tx, ty, f"dF = {d['df_estimate']:.3f} +/- {d['df_sigma']:.3f}", fontsize=8.5))
    return save("d_main")


def build_panel_e():
    d = panel_e_data()
    lims = panel_e_limits(d)
    colors = mpl.get_cmap("cool")(np.linspace(0.05, 0.9, len(d["n_switch_values"])))
    kinds = kind_cycle(len(d["n_switch_values"]))

    gplt.figure(figsize=PANEL_FIGSIZE)
    # axvline() spans get_ylim() at call time (an in-memory engine value, separate
    # from the headless export crop) -- gplt.ylim() must run first so the vline
    # covers the right range; set_axis_bounds() at the end still does the real
    # work of cropping the exported PNG, since gplt.ylim() alone has no effect
    # on that. Calling axvline before the histogram loop also matches
    # mpl_version.py's legend order (its exact-dF entry is added first there too).
    gplt.xlim(*lims["xlim"])
    gplt.ylim(*lims["ylim"])
    labeled(
        gplt.axvline(d["df_exact"], color="crimson", linewidth=1.6, linestyle="--"),
        f"exact dF = {d['df_exact']:.3f}",
    )
    bins = np.linspace(-1.4, 1.0, 41)
    for i, (n, color) in enumerate(zip(d["n_switch_values"], colors)):
        w = d["work_samples"][n]
        counts, edges = np.histogram(w, bins=bins, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        # A single fill_between carries both the shape and the legend label --
        # matplotlib's own stepfilled histtype is one patch too. An earlier version
        # drew a *second*, separate step() line on top for the legend, which
        # doubled up the alpha at every bin edge (two semi-transparent strokes
        # stacked exactly on top of each other) into a much bolder border than
        # mpl's single-patch edge: the "marked edge" the histogram didn't need.
        labeled(
            no_edge(
                gplt.fill_between(
                    centers, counts, 0, color=color, alpha=ALPHA_HIST_FILL, step="mid"
                )
            ),
            f"n_switch={n}",
        )
        texture_overlay(
            centers, 0.0, counts, kinds[i], color, lims["xlim"], lims["ylim"], alpha=0.4
        )
    set_axis_bounds(lims["xlim"], lims["ylim"])
    gplt.xlabel("work W")
    gplt.ylabel("P_F(W)")
    gplt.title("e   Jarzynski work distributions")
    return save("e_main")


def build_panel_f():
    d = panel_f_data()
    lims = panel_f_limits(d)
    colors = mpl.get_cmap("tab10")(np.linspace(0, 0.9, len(d["curves"])))
    kinds = kind_cycle(len(d["curves"]))

    gplt.figure(figsize=PANEL_FIGSIZE)
    for i, ((name, c), color) in enumerate(zip(d["curves"].items(), colors)):
        log_lo = np.log10(c["lo"])
        log_hi = np.log10(c["hi"])
        log_mean = np.log10(c["mean"])
        labeled(
            no_edge(gplt.fill_between(d["t"], log_lo, log_hi, color=color, alpha=ALPHA_ERROR_BAND))
        )
        texture_overlay(
            d["t"], log_lo, log_hi, kinds[i], color, lims["xlim"], lims["log_ylim"], alpha=0.35
        )
        labeled(gplt.plot(d["t"], log_mean, color=color, linewidth=1.6), name)
    set_axis_bounds(lims["xlim"], lims["log_ylim"])
    gplt.xlabel("fraction of run")
    gplt.ylabel("log10(error)  --  no native log-scale axis, data pre-transformed")
    gplt.title("f   Error convergence (log10 y-axis workaround)")
    return save("f_main")


def main():
    a = build_panel_a()
    composite_inset(a, build_inset_a(), frac_box=(0.5, 0.42, 0.42, 0.4))
    b = build_panel_b()
    composite_inset(b, build_inset_b(), frac_box=(0.5, 0.42, 0.42, 0.4))
    c = build_panel_c()
    composite_inset(c, build_inset_c(), frac_box=(0.06, 0.1, 0.4, 0.32))
    d = build_panel_d()
    e = build_panel_e()
    f = build_panel_f()

    out = RESULTS / "glplot_free_energy.png"
    assemble_grid(
        [[a, b], [c, d], [e, f]],
        out,
        "Free-energy methods -- synthetic demo (GLPlot, textured shaded areas)",
    )


if __name__ == "__main__":
    main()
