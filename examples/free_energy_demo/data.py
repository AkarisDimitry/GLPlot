"""Synthetic "free energy methods" dataset shared by the matplotlib and GLPlot
demo scripts, so both draw exactly the same numbers.

None of this is real physics -- it is a plausible-looking stand-in for a
free-energy-methods figure (umbrella sampling, metadynamics, TMMC, thermodynamic
integration, Jarzynski) built purely to exercise plotting, shading and texture
code identically in both libraries.
"""

from __future__ import annotations

import numpy as np

RNG = np.random.default_rng(20260804)

DF_EXACT = -0.108  # "exact" free energy difference used by panels d, e, f


def exact_F1(x):
    """Single steep well used by panels a, b, e, f."""
    x = np.asarray(x, dtype=float)
    base = 0.55 * (x - 0.6) ** 2
    wiggle = 0.16 * np.sin(3.0 * (x - 0.6)) * np.exp(-((x - 0.6) / 1.2) ** 2)
    return base + wiggle


def exact_F2(x):
    """Double well used by panel c."""
    x = np.asarray(x, dtype=float)
    return 1.5 * (x**2 - 1.0) ** 2


def panel_a_data():
    x_exact = np.linspace(-2.0, 3.0, 400)
    f_exact = exact_F1(x_exact)

    centers = np.linspace(-1.8, 2.8, 16)
    sigma_win = 0.28
    x_grid = np.linspace(-2.2, 3.2, 300)
    hills = []
    for c in centers:
        amp = 0.9 + 0.4 * np.cos(0.7 * c)
        hills.append(amp * np.exp(-0.5 * ((x_grid - c) / sigma_win) ** 2))

    mc_x = centers
    mc_y = exact_F1(mc_x) + RNG.normal(scale=0.03, size=len(mc_x))
    mc_sigma = 0.06 + 0.05 * np.abs(np.sin(mc_x))
    return dict(
        x_exact=x_exact,
        f_exact=f_exact,
        x_grid=x_grid,
        hills=hills,
        centers=centers,
        mc_x=mc_x,
        mc_y=mc_y,
        mc_sigma=mc_sigma,
    )


def panel_b_data():
    x_exact = np.linspace(-2.0, 3.0, 400)
    f_exact = exact_F1(x_exact)

    fractions = [0.004, 0.01, 0.03, 0.12, 1.0]
    x = np.linspace(-2.0, 3.0, 200)
    curves = {}
    for frac in fractions:
        weight = min(1.0, frac * 6.0)
        residual_bump = 1.3 * np.exp(-((x - 0.6) / 0.9) ** 2) * (1.0 - weight)
        edge_sag = 0.9 * np.exp(-((x + 1.6) / 0.5) ** 2) * (1.0 - weight)
        noise = RNG.normal(scale=0.03 * (1.0 - weight) + 0.01, size=x.shape)
        curves[frac] = weight * exact_F1(x) + residual_bump + edge_sag + noise
    return dict(x_exact=x_exact, f_exact=f_exact, x=x, fractions=fractions, curves=curves)


def panel_c_data():
    x_exact = np.linspace(-1.9, 1.9, 400)
    f_exact = exact_F2(x_exact)

    temperatures = [0.2, 0.35, 0.5, 0.8, 1.2]
    x = np.linspace(-1.9, 1.9, 250)
    f_curves = {}
    p_curves = {}
    for T in temperatures:
        scale = (0.3 / T) ** 0.6
        f_t = exact_F2(x) * scale + 0.05 * RNG.normal(size=x.shape)
        f_curves[T] = f_t
        p = np.exp(-(f_t - f_t.min()) / T)
        p_curves[T] = p / p.max()

    mc_x = np.linspace(-1.8, 1.8, 22)
    mc_y = exact_F2(mc_x) + RNG.normal(scale=0.05, size=mc_x.shape)
    mc_sigma = 0.08 + 0.04 * np.abs(mc_x)
    return dict(
        x_exact=x_exact,
        f_exact=f_exact,
        x=x,
        temperatures=temperatures,
        f_curves=f_curves,
        p_curves=p_curves,
        mc_x=mc_x,
        mc_y=mc_y,
        mc_sigma=mc_sigma,
    )


def panel_d_data():
    lambdas = np.linspace(0.0, 1.0, 11)
    samples = [RNG.normal(loc=0.25 - 0.7 * lam, scale=0.35, size=180) for lam in lambdas]
    means = np.array([s.mean() for s in samples])
    return dict(lambdas=lambdas, samples=samples, means=means, df_estimate=-0.107, df_sigma=0.004)


def panel_e_data():
    n_switch_values = [10, 25, 50, 100, 200, 400]
    work_samples = {}
    for n in n_switch_values:
        dissipation = 0.9 / np.sqrt(n)
        spread = 0.25 + 1.1 / np.sqrt(n)
        skew_component = RNG.exponential(scale=spread * 0.6, size=4000) * (20.0 / n)
        base = RNG.normal(loc=DF_EXACT + dissipation, scale=spread, size=4000)
        work_samples[n] = base + skew_component
    return dict(n_switch_values=n_switch_values, work_samples=work_samples, df_exact=DF_EXACT)


def panel_a_limits(d):
    """Shared axis ranges for panel a, so both scripts plot the same numeric box.

    ``twin_rescale`` is the number GLPlot multiplies the (unscaled) hill
    amplitudes by so that, drawn directly on the main F(x) axis, they occupy
    the same *fraction* of panel height as they do in matplotlib's real right
    -hand twin axis -- the two renderers hit the same target range by
    different means (a second scale vs. a rescale-then-relabel), but the
    fraction of the panel they cover should match.
    """
    xlim = (-2.0, 3.0)
    y_main_max = max(float(d["f_exact"].max()), float((d["mc_y"] + d["mc_sigma"]).max())) * 1.05
    y_main = (0.0, y_main_max)
    hill_max = max(float(h.max()) for h in d["hills"])
    y_twin = (0.0, hill_max * 1.25)
    return dict(xlim=xlim, y_main=y_main, y_twin=y_twin, twin_rescale=y_main[1] / y_twin[1])


def panel_b_limits(d):
    xlim = (-2.0, 3.0)
    ymax = max(float(d["f_exact"].max()), max(float(c.max()) for c in d["curves"].values()))
    return dict(xlim=xlim, ylim=(0.0, ymax * 1.05))


def panel_c_limits(d):
    xlim = (-1.9, 1.9)
    ymax = max(float(d["f_exact"].max()), max(float(c.max()) for c in d["f_curves"].values()))
    y_main = (0.0, ymax * 1.05)
    p_max = max(float(p.max()) for p in d["p_curves"].values())
    y_twin = (0.0, p_max * 1.45)
    return dict(xlim=xlim, y_main=y_main, y_twin=y_twin, twin_rescale=y_main[1] / y_twin[1])


def panel_d_limits(d):
    lam = d["lambdas"]
    width = 0.07
    xlim = (float(lam.min()) - 6 * width, float(lam.max()) + 6 * width)
    all_samples = np.concatenate(d["samples"])
    ylim = (float(all_samples.min()) * 1.08, float(all_samples.max()) * 1.08)
    return dict(xlim=xlim, ylim=ylim)


PANEL_E_BINS = np.linspace(-1.4, 1.0, 41)


def panel_e_limits(d):
    ymax = 0.0
    for w in d["work_samples"].values():
        counts, _ = np.histogram(w, bins=PANEL_E_BINS, density=True)
        ymax = max(ymax, float(counts.max()))
    return dict(xlim=(-1.4, 1.0), ylim=(0.0, ymax * 1.08))


def panel_f_limits(d):
    lo_all = np.concatenate([c["lo"] for c in d["curves"].values()])
    hi_all = np.concatenate([c["hi"] for c in d["curves"].values()])
    y_min = float(lo_all.min()) * 0.8
    y_max = float(hi_all.max()) * 1.2
    return dict(
        xlim=(float(d["t"].min()), float(d["t"].max())),
        ylim=(y_min, y_max),
        log_ylim=(np.log10(y_min), np.log10(y_max)),
    )


#: The only three "hatch families" GLPlot can fake without a real hatch renderer
#: (stippled dots, single-direction diagonal lines, crossed diagonal lines).
#: Both scripts cycle through this same list in the same order so that a given
#: series index gets the *same kind* of texture in both renderings, even though
#: the pixels themselves can't match (matplotlib draws a real vector hatch;
#: GLPlot fakes one from scatter/plot primitives).
TEXTURE_KINDS = ["diag", "cross", "dots"]
MPL_HATCH_FOR_KIND = {"diag": "/", "cross": "x", "dots": "."}


def kind_cycle(n):
    return [TEXTURE_KINDS[i % len(TEXTURE_KINDS)] for i in range(n)]


#: GLPlot's headless-preview legend is drawn by library code with these values
#: hardcoded (see ``glplot.utils.preview.render_preview``), not read from
#: matplotlib rcParams -- so the matplotlib script matches itself to these
#: numbers, rather than the other way around, since GLPlot's can't be
#: customized short of monkeypatching the library function.
LEGEND_KW = dict(
    loc="upper right",
    fontsize=8,
    framealpha=0.78,
    borderpad=0.35,
    labelspacing=0.3,
    handlelength=1.4,
)

#: Applied as ``plt.rcParams.update(RC_PARAMS)`` in both scripts before any
#: figure is built, so tick/label/hatch sizing matches between the real
#: matplotlib figure and GLPlot's matplotlib-based headless preview.
RC_PARAMS = {
    "font.size": 9,
    "axes.labelsize": 9.5,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "hatch.linewidth": 0.6,
}

#: One alpha per kind of shaded region, used by both scripts instead of each
#: guessing its own number -- the two scripts independently picking close-but-
#: different alphas (0.32 vs 0.22, 0.12 vs 0.08, ...) was as much a source of
#: "the shading doesn't match" as the texture-kind mismatch was.
ALPHA_HILL = 0.28
ALPHA_BAND = 0.10
ALPHA_TWIN_FILL = 0.20
ALPHA_HIST_FILL = 0.20
ALPHA_ERROR_BAND = 0.16
ALPHA_VIOLIN = 0.30

#: matplotlib's ``errorbar(capsize=...)`` is in *points*; GLPlot's is in *data
#: units* (see its docstring). With both scripts now rendering every panel at
#: the same physical size and DPI (``compose.PANEL_DPI``/``PANEL_FIGSIZE``),
#: a points value can be converted to the data-unit width that covers the same
#: number of pixels on a given panel's x-axis, given that axis's span and the
#: panel's pixel width.
def capsize_data_units(capsize_pt, xlim, panel_width_in, dpi):
    capsize_px = capsize_pt * dpi / 72.0
    px_per_data_unit = (panel_width_in * dpi) / (xlim[1] - xlim[0])
    return capsize_px / px_per_data_unit


def panel_f_data():
    t = np.linspace(0.02, 1.0, 60)
    methods = {
        "Umbrella F(x) WHAM self-consistency": dict(plateau=1.1e-4, start=0.5, tau=0.06, band=0.25),
        "Metadynamics F(x) RMSE": dict(plateau=9e-3, start=0.55, tau=0.12, band=0.3),
        "TMMC F(x) RMSE": dict(plateau=7e-3, start=0.45, tau=0.15, band=0.28),
        "TI |ΔF(t) - ΔF_exact|": dict(plateau=4e-2, start=0.4, tau=0.35, band=0.35),
        "Jarzynski |ΔF - ΔF_exact| (n_switch=400)": dict(
            plateau=0.16, start=0.5, tau=0.5, band=0.4
        ),
    }
    curves = {}
    for name, p in methods.items():
        mean = p["plateau"] + (p["start"] - p["plateau"]) * np.exp(-t / p["tau"])
        mean *= 1.0 + 0.04 * RNG.normal(size=t.shape)
        mean = np.abs(mean)
        band_lo = mean * (1.0 - p["band"])
        band_hi = mean * (1.0 + p["band"])
        curves[name] = dict(mean=np.clip(mean, 1e-5, None), lo=np.clip(band_lo, 1e-5, None), hi=band_hi)
    return dict(t=t, curves=curves)
