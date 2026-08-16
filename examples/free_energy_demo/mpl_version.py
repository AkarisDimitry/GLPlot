"""Matplotlib reference figure: a 6-panel free-energy-methods style figure with
shaded regions rendered in a variety of hatch textures (in addition to the plain
alpha shading the original reference image uses).

Each panel is its own independent figure, built at the exact same physical
size and DPI as glplot_version.py's panels (``compose.PANEL_FIGSIZE`` /
``compose.PANEL_DPI``) and composited afterward with the same code
(``compose.assemble_grid``/``composite_inset``) -- not because matplotlib
needs that (a single ``plt.subplots(3, 2)`` figure would be simpler), but
because GLPlot's headless export can only ever produce one panel per figure,
and matching that pipeline stage-for-stage is what makes point sizes, line
widths and the title end up pixel-comparable rather than only "close" at the
end. See compose.py's module docstring for the mechanics.

Axis ranges, texture-kind assignment, legend style, alphas and rcParams all
come from data.py so this script and glplot_version.py draw the exact same
numeric boxes and as-close-as-possible chrome.

Panels:
  a. Umbrella sampling windows (biased P_i(x)) + exact/MC free energy, twin axis, inset
  b. Metadynamics-style F(x) convergence vs. fraction of hills deposited, inset
  c. Temperature sweep of a double-well F(x) and P(x|T), twin axis, inset
  d. Thermodynamic-integration violins of <dV/dlambda> vs coupling, with a linear fit
  e. Jarzynski work-distribution step histograms for several n_switch values
  f. Log-scale error convergence for five different free-energy methods
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from compose import INSET_FIGSIZE, PANEL_DPI, PANEL_FIGSIZE, assemble_grid, composite_inset
from data import (
    ALPHA_BAND,
    ALPHA_ERROR_BAND,
    ALPHA_HILL,
    ALPHA_HIST_FILL,
    ALPHA_TWIN_FILL,
    ALPHA_VIOLIN,
    LEGEND_KW,
    MPL_HATCH_FOR_KIND,
    PANEL_E_BINS,
    RC_PARAMS,
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

RESULTS = Path(__file__).resolve().parent / "results"
PANELS_DIR = RESULTS / "mpl_panels"
PANELS_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(RC_PARAMS)


def new_panel():
    fig = plt.figure(figsize=PANEL_FIGSIZE, dpi=PANEL_DPI)
    return fig, fig.add_subplot(111)


def new_inset_fig():
    fig = plt.figure(figsize=INSET_FIGSIZE, dpi=PANEL_DPI)
    return fig, fig.add_subplot(111)


def save(fig, name):
    path = PANELS_DIR / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def panel_a():
    d = panel_a_data()
    lims = panel_a_limits(d)
    cmap = plt.get_cmap("turbo")
    n = len(d["hills"])
    kinds = kind_cycle(n)
    fig, ax = new_panel()
    ax2 = ax.twinx()
    for i, hill in enumerate(d["hills"]):
        color = cmap(i / max(n - 1, 1))
        ax2.fill_between(
            d["x_grid"],
            0,
            hill,
            facecolor=color,
            edgecolor=color,
            alpha=ALPHA_HILL,
            hatch=MPL_HATCH_FOR_KIND[kinds[i]],
            linewidth=0.0,
        )
    ax2.set_ylim(*lims["y_twin"])
    ax2.set_yticks([])

    ax.plot(d["x_exact"], d["f_exact"], color="crimson", lw=2.0, label="exact F(x)", zorder=5)
    ax.errorbar(
        d["mc_x"],
        d["mc_y"],
        yerr=d["mc_sigma"],
        fmt="o",
        ms=3.5,
        color="tab:blue",
        ecolor="tab:blue",
        elinewidth=1.0,
        capsize=2.0,
        label="MC mean ± sigma",
        zorder=6,
    )
    ax.set_xlim(*lims["xlim"])
    ax.set_ylim(*lims["y_main"])
    ax.set_xlabel("position x")
    ax.set_ylabel("free energy F(x)  |  biased P_i(x), rescaled (no native twin-axis)")
    ax.legend(**LEGEND_KW)
    ax.set_title("a   Umbrella sampling windows", fontsize=11)
    return save(fig, "a_main")


def inset_a():
    d = panel_a_data()
    fig, axin = new_inset_fig()
    mask = (d["x_exact"] > -0.35) & (d["x_exact"] < 0.35)
    axin.plot(d["x_exact"][mask], d["f_exact"][mask], color="crimson", lw=1.6)
    mmask = (d["mc_x"] > -0.35) & (d["mc_x"] < 0.35)
    axin.errorbar(
        d["mc_x"][mmask],
        d["mc_y"][mmask],
        yerr=d["mc_sigma"][mmask],
        fmt="o",
        ms=3,
        color="tab:blue",
        ecolor="tab:blue",
        capsize=2,
    )
    axin.set_xlim(-0.3, 0.3)
    pad = 0.1 * (d["f_exact"][mask].max() - d["f_exact"][mask].min() + 1e-6)
    axin.set_ylim(d["f_exact"][mask].min() - pad, d["f_exact"][mask].max() + pad)
    axin.tick_params(labelsize=6.5)
    for spine in axin.spines.values():
        spine.set_linewidth(0.6)
    return save(fig, "a_inset")


def panel_b():
    d = panel_b_data()
    lims = panel_b_limits(d)
    colors = plt.get_cmap("plasma")(np.linspace(0.05, 0.9, len(d["fractions"])))
    fig, ax = new_panel()
    ax.plot(d["x_exact"], d["f_exact"], color="crimson", lw=2.2, label="exact F(x)", zorder=5)

    band = 0.35 * np.exp(-((d["x"] - 0.6) / 1.4) ** 2) + 0.05
    f_on_x = np.interp(d["x"], d["x_exact"], d["f_exact"])
    ax.fill_between(
        d["x"],
        f_on_x - band,
        f_on_x + band,
        facecolor="crimson",
        edgecolor="crimson",
        alpha=ALPHA_BAND,
        hatch=MPL_HATCH_FOR_KIND["dots"],
        linewidth=0.0,
        zorder=1,
    )

    for (frac, curve), color in zip(d["curves"].items(), colors):
        pct = f"{frac * 100:g}%"
        ax.plot(d["x"], curve, color=color, lw=1.6, label=pct)
    ax.set_xlim(*lims["xlim"])
    ax.set_ylim(*lims["ylim"])
    ax.set_xlabel("position x")
    ax.set_ylabel("F(x) estimate")
    ax.legend(**LEGEND_KW)
    ax.set_title("b   Fraction of hills deposited", fontsize=11)
    return save(fig, "b_main")


def inset_b():
    d = panel_b_data()
    colors = plt.get_cmap("plasma")(np.linspace(0.05, 0.9, len(d["fractions"])))
    fig, axin = new_inset_fig()
    mask = (d["x"] > -0.35) & (d["x"] < 0.35)
    emask = (d["x_exact"] > -0.35) & (d["x_exact"] < 0.35)
    all_y = [d["f_exact"][emask]]
    for (frac, curve), color in zip(d["curves"].items(), colors):
        axin.plot(d["x"][mask], curve[mask], color=color, lw=1.4)
        all_y.append(curve[mask])
    axin.plot(d["x_exact"][emask], d["f_exact"][emask], color="crimson", lw=1.6)
    axin.set_xlim(-0.3, 0.3)
    all_y = np.concatenate(all_y)
    pad = 0.1 * (all_y.max() - all_y.min() + 1e-6)
    axin.set_ylim(all_y.min() - pad, all_y.max() + pad)
    axin.tick_params(labelsize=6.5)
    for spine in axin.spines.values():
        spine.set_linewidth(0.6)
    return save(fig, "b_inset")


def panel_c():
    d = panel_c_data()
    lims = panel_c_limits(d)
    colors = plt.get_cmap("viridis")(np.linspace(0.05, 0.9, len(d["temperatures"])))
    kinds = kind_cycle(len(d["temperatures"]))
    fig, ax = new_panel()
    ax2 = ax.twinx()

    ax.plot(d["x_exact"], d["f_exact"], color="crimson", lw=2.2, label="exact F(x)", zorder=6)
    for i, (T, color) in enumerate(zip(d["temperatures"], colors)):
        ax.plot(d["x"], d["f_curves"][T], color=color, lw=1.5, zorder=5, label=f"T={T:g}")
        ax2.fill_between(
            d["x"],
            0,
            d["p_curves"][T],
            facecolor=color,
            edgecolor=color,
            alpha=ALPHA_TWIN_FILL,
            hatch=MPL_HATCH_FOR_KIND[kinds[i]],
            linewidth=0.0,
        )
    ax.set_xlim(*lims["xlim"])
    ax.set_ylim(*lims["y_main"])
    ax2.set_ylim(*lims["y_twin"])
    ax2.set_yticks([])
    ax.set_xlabel("position x")
    ax.set_ylabel("free energy F(x)  |  P(x|T), rescaled (no native twin-axis)")
    ax.legend(**LEGEND_KW)
    ax.set_title("c   Temperature sweep, double well", fontsize=11)
    return save(fig, "c_main")


def inset_c():
    d = panel_c_data()
    colors = plt.get_cmap("viridis")(np.linspace(0.05, 0.9, len(d["temperatures"])))
    fig, axin = new_inset_fig()
    mask = (d["x_exact"] > -0.35) & (d["x_exact"] < 0.35)
    all_y = [d["f_exact"][mask]]
    axin.plot(d["x_exact"][mask], d["f_exact"][mask], color="crimson", lw=1.6)
    xmask = (d["x"] > -0.35) & (d["x"] < 0.35)
    for T, color in zip(d["temperatures"], colors):
        axin.plot(d["x"][xmask], d["f_curves"][T][xmask], color=color, lw=1.3)
        all_y.append(d["f_curves"][T][xmask])
    axin.set_xlim(-0.3, 0.3)
    all_y = np.concatenate(all_y)
    pad = 0.1 * (all_y.max() - all_y.min() + 1e-6)
    axin.set_ylim(all_y.min() - pad, all_y.max() + pad)
    axin.tick_params(labelsize=6.5)
    for spine in axin.spines.values():
        spine.set_linewidth(0.6)
    return save(fig, "c_inset")


def panel_d():
    d = panel_d_data()
    lims = panel_d_limits(d)
    positions = d["lambdas"]
    fig, ax = new_panel()
    parts = ax.violinplot(d["samples"], positions=positions, widths=0.07, showextrema=False)
    colors = plt.get_cmap("cividis")(np.linspace(0.05, 0.9, len(positions)))
    for i, body in enumerate(parts["bodies"]):
        body.set_facecolor(colors[i])
        body.set_edgecolor(colors[i])
        body.set_alpha(ALPHA_VIOLIN)

    sigmas = np.array([s.std() / np.sqrt(len(s)) for s in d["samples"]])
    ax.errorbar(
        positions,
        d["means"],
        yerr=sigmas,
        fmt="o",
        ms=4,
        color="tab:blue",
        ecolor="tab:blue",
        capsize=2,
        label="MC mean ± sigma",
        zorder=5,
    )
    fit = np.polyfit(positions, d["means"], 1)
    ax.plot(
        positions,
        np.polyval(fit, positions),
        color="crimson",
        lw=1.8,
        label="analytic <dV/dlambda>",
        zorder=4,
    )
    ax.set_xlim(*lims["xlim"])
    ax.set_ylim(*lims["ylim"])
    ax.set_xlabel("coupling lambda")
    ax.set_ylabel("<dV/dlambda>_lambda")
    ax.legend(**LEGEND_KW)
    tx = lims["xlim"][0] + 0.03 * (lims["xlim"][1] - lims["xlim"][0])
    ty = lims["ylim"][0] + 0.06 * (lims["ylim"][1] - lims["ylim"][0])
    ax.text(
        tx,
        ty,
        f"dF = {d['df_estimate']:.3f} +/- {d['df_sigma']:.3f}",
        fontsize=8.5,
        color="black",
    )
    ax.set_title("d   Thermodynamic integration", fontsize=11)
    return save(fig, "d_main")


def panel_e():
    d = panel_e_data()
    lims = panel_e_limits(d)
    colors = plt.get_cmap("cool")(np.linspace(0.05, 0.9, len(d["n_switch_values"])))
    kinds = kind_cycle(len(d["n_switch_values"]))
    fig, ax = new_panel()
    ax.axvline(
        d["df_exact"], color="crimson", ls="--", lw=1.6, label=f"exact dF = {d['df_exact']:.3f}"
    )

    for i, (n, color) in enumerate(zip(d["n_switch_values"], colors)):
        w = d["work_samples"][n]
        ax.hist(
            w,
            bins=PANEL_E_BINS,
            density=True,
            histtype="stepfilled",
            facecolor=color,
            edgecolor="none",
            alpha=ALPHA_HIST_FILL,
            hatch=MPL_HATCH_FOR_KIND[kinds[i]],
            linewidth=0.0,
            label=f"n_switch={n}",
        )
    ax.set_xlim(*lims["xlim"])
    ax.set_ylim(*lims["ylim"])
    ax.set_xlabel("work W")
    ax.set_ylabel("P_F(W)")
    ax.legend(**LEGEND_KW)
    ax.set_title("e   Jarzynski work distributions", fontsize=11)
    return save(fig, "e_main")


def panel_f():
    d = panel_f_data()
    lims = panel_f_limits(d)
    colors = plt.get_cmap("tab10")(np.linspace(0, 0.9, len(d["curves"])))
    kinds = kind_cycle(len(d["curves"]))
    fig, ax = new_panel()
    for i, ((name, c), color) in enumerate(zip(d["curves"].items(), colors)):
        ax.fill_between(
            d["t"],
            c["lo"],
            c["hi"],
            facecolor=color,
            edgecolor=color,
            alpha=ALPHA_ERROR_BAND,
            hatch=MPL_HATCH_FOR_KIND[kinds[i]],
            linewidth=0.0,
        )
        ax.plot(d["t"], c["mean"], color=color, lw=1.6, label=name)
    ax.set_xlim(*lims["xlim"])
    ax.set_ylim(*lims["ylim"])
    ax.set_yscale("log")
    ax.set_xlabel("fraction of run")
    ax.set_ylabel("error (F(x) or dF)")
    ax.legend(**LEGEND_KW)
    ax.set_title("f   Error convergence (log10 y-axis workaround)", fontsize=11)
    return save(fig, "f_main")


def main():
    a = panel_a()
    composite_inset(a, inset_a(), frac_box=(0.5, 0.42, 0.42, 0.4))
    b = panel_b()
    composite_inset(b, inset_b(), frac_box=(0.5, 0.42, 0.42, 0.4))
    c = panel_c()
    composite_inset(c, inset_c(), frac_box=(0.06, 0.1, 0.4, 0.32))
    d = panel_d()
    e = panel_e()
    f = panel_f()

    out = RESULTS / "matplotlib_free_energy.png"
    assemble_grid(
        [[a, b], [c, d], [e, f]],
        out,
        "Free-energy methods -- synthetic demo (matplotlib, textured shaded areas)",
    )


if __name__ == "__main__":
    main()
