"""
02_gpu_comparison.py
====================
Comparative benchmark: GLPlot vs VisPy vs fastplotlib vs Datashader vs hvPlot
- Line counts : 10^3 → 10^7
- Metrics     : wall-clock time  +  peak RSS memory (psutil) or tracemalloc
- Repetitions : N_REPEATS runs per (library, N); first run is warm-up (discarded)
- Statistics  : mean, std, min, max over the remaining runs
"""

import time
import os
import sys
import gc
import tracemalloc
import dataclasses
from typing import List, Optional
import numpy as np
import warnings
from PIL import Image

# Force local glplot import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import glplot.pyplot as gplt

# Optional: psutil for RSS
try:
    import psutil

    _proc = psutil.Process(os.getpid())
    psutil_available = True
except ImportError:
    psutil_available = False

# VisPy
try:
    import glfw
    from vispy import app, scene

    app.use_app("glfw")
    vispy_available = True
except Exception as e:
    print(f"Warning: VisPy failed: {e}")
    vispy_available = False

# fastplotlib
try:
    import fastplotlib as fpl

    fastplotlib_available = True
except Exception as e:
    print(f"Warning: fastplotlib failed: {e}")
    fastplotlib_available = False

# datashader
try:
    import datashader as ds
    import datashader.transfer_functions as tf
    import pandas as pd

    datashader_available = True
except Exception as e:
    print(f"Warning: datashader failed: {e}")
    datashader_available = False

# hvplot
try:
    import hvplot.pandas
    import holoviews as hv

    hv.extension("matplotlib")
    hvplot_available = True
except Exception as e:
    print(f"Warning: hvplot failed: {e}")
    hvplot_available = False

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
INSET_W, INSET_H = 720, 540
FULL_W, FULL_H = 800, 600
MARGIN_LEFT, MARGIN_TOP = 60, 20
X_RANGE = (-5.0, 5.0)
Y_RANGE = (-10.0, 20.0)
CMAP = ["#00001a", "#003399", "#00ccff", "#ffffff"]

# ── Repetitions ─────────────────────────────────────────────────────────────
N_REPEATS = 5  # total runs per cell
N_WARMUP = 1  # first N_WARMUP runs are discarded (JIT / GPU warm-up)
# ---------------------------------------------------------------------------


def _adaptive_alpha(N):
    return float(np.clip(5.0 / np.sqrt(N), 0.0005, 0.1))


def _make_colors(N, alpha):
    c = np.zeros((N, 4), dtype=np.float32)
    c[:, 1] = 0.7
    c[:, 2] = 1.0
    c[:, 3] = alpha
    return c


def _paste_onto_canvas(pil_img, output_path):
    full = Image.new("RGBA", (FULL_W, FULL_H), (0, 0, 0, 255))
    if pil_img.size != (INSET_W, INSET_H):
        pil_img = pil_img.resize((INSET_W, INSET_H), Image.Resampling.LANCZOS)
    full.paste(pil_img, (MARGIN_LEFT, MARGIN_TOP))
    full.save(output_path)


def _rss_mb():
    return _proc.memory_info().rss / (1024**2) if psutil_available else None


def _fmt_mem(mb):
    if mb is None:
        return "N/A"
    if abs(mb) >= 1024:
        return f"{mb/1024:.1f} GB"
    return f"{mb:.0f} MB"


# ---------------------------------------------------------------------------
# RunStats – statistics over N_REPEATS - N_WARMUP measurements
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class RunStats:
    """Aggregated statistics for one (library, N) cell."""

    # Time
    t_mean: float
    t_std: float
    t_min: float
    t_max: float
    # Memory (RSS delta or tracemalloc peak, in MiB)
    m_mean: float
    m_std: float
    m_min: float
    m_max: float

    def time_str(self):
        return f"{self.t_mean:.3f}±{self.t_std:.3f}s"

    def mem_str(self):
        return f"{_fmt_mem(self.m_mean)}±{self.m_std:.0f}"


def _measure_once(fn):
    """Single timed + memory-measured run of fn().  Returns (elapsed_s, mem_MiB)."""
    gc.collect()
    rss_before = _rss_mb()
    tracemalloc.start()
    t0 = time.perf_counter()
    try:
        fn()
    finally:
        elapsed = time.perf_counter() - t0
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        gc.collect()
    rss_after = _rss_mb()

    peak_tm = peak_bytes / (1024**2)
    if rss_before is not None and rss_after is not None:
        mem = rss_after - rss_before
    else:
        mem = peak_tm
    return elapsed, mem


def _measure_repeated(fn_factory, label, n_repeats=N_REPEATS, n_warmup=N_WARMUP):
    """
    Run fn_factory() n_repeats times.
    fn_factory is called each repetition to get a fresh closure (avoids
    closures capturing stale state).  First n_warmup runs are discarded.
    Returns RunStats or None on error.
    """
    times = []
    mems = []
    for rep in range(n_repeats):
        tag = "(warm-up)" if rep < n_warmup else f"rep {rep - n_warmup + 1}/{n_repeats - n_warmup}"
        try:
            fn = fn_factory()
            t, m = _measure_once(fn)
            if rep >= n_warmup:
                times.append(t)
                mems.append(m)
            print(f"      [{tag}] {t:.4f}s  mem={_fmt_mem(m)}")
        except Exception as e:
            print(f"      [{tag}] FAILED: {e}")
            if rep >= n_warmup:
                # keep None-like sentinel by skipping
                pass

    if not times:
        return None

    ta, tm_arr = np.array(times), np.array(mems)
    return RunStats(
        t_mean=float(np.mean(ta)),
        t_std=float(np.std(ta, ddof=min(1, len(ta) - 1))),
        t_min=float(np.min(ta)),
        t_max=float(np.max(ta)),
        m_mean=float(np.mean(tm_arr)),
        m_std=float(np.std(tm_arr, ddof=min(1, len(tm_arr) - 1))),
        m_min=float(np.min(tm_arr)),
        m_max=float(np.max(tm_arr)),
    )


# ---------------------------------------------------------------------------
# Renderer functions  (unchanged logic, just extracted for reuse)
# ---------------------------------------------------------------------------
def run_vispy(N, a, b, colors, output_path):
    glfw.init()
    glfw.default_window_hints()
    canvas = scene.SceneCanvas(
        show=False, size=(INSET_W, INSET_H), keys="interactive", bgcolor="black"
    )
    view = canvas.central_widget.add_view()
    pos = np.empty((2 * N, 3), dtype=np.float32)
    pos[0::2, 0] = X_RANGE[0]
    pos[0::2, 1] = a * X_RANGE[0] + b
    pos[0::2, 2] = 0.0
    pos[1::2, 0] = X_RANGE[1]
    pos[1::2, 1] = a * X_RANGE[1] + b
    pos[1::2, 2] = 0.0
    color_arr = np.repeat(colors, 2, axis=0)
    line = scene.visuals.Line(pos, color=color_arr, connect="segments", parent=view.scene)
    line.set_gl_state(depth_test=False, blend=True, blend_func=("src_alpha", "one_minus_src_alpha"))
    view.camera = "panzoom"
    view.camera.set_range(x=X_RANGE, y=Y_RANGE)
    canvas.app.process_events()
    img = canvas.render()
    _paste_onto_canvas(Image.fromarray(img), output_path)
    canvas.close()


def run_fastplotlib(N, a, b, colors, output_path):
    fig = fpl.Figure(canvas="offscreen", size=(INSET_W, INSET_H))
    subplot = fig[0, 0]
    pos = np.empty((3 * N, 3), dtype=np.float32)
    pos[0::3, 0] = X_RANGE[0]
    pos[0::3, 1] = a * X_RANGE[0] + b
    pos[0::3, 2] = 0.0
    pos[1::3, 0] = X_RANGE[1]
    pos[1::3, 1] = a * X_RANGE[1] + b
    pos[1::3, 2] = 0.0
    pos[2::3] = np.nan
    color_arr = np.repeat(colors, 3, axis=0)
    subplot.add_line(pos, colors=color_arr)
    subplot.camera.width = X_RANGE[1] - X_RANGE[0]
    subplot.camera.height = Y_RANGE[1] - Y_RANGE[0]
    subplot.camera.position = (
        (X_RANGE[0] + X_RANGE[1]) / 2.0,
        (Y_RANGE[0] + Y_RANGE[1]) / 2.0,
        0.0,
    )
    temp = output_path + ".temp.png"
    fig._render()
    fig.export(temp)
    if fig._output is not None:
        fig.close()
    if os.path.exists(temp):
        _paste_onto_canvas(Image.open(temp), output_path)
        os.remove(temp)


def run_datashader(N, a, b, alpha, output_path):
    pos = np.empty((3 * N, 2), dtype=np.float32)
    pos[0::3, 0] = X_RANGE[0]
    pos[0::3, 1] = a * X_RANGE[0] + b
    pos[1::3, 0] = X_RANGE[1]
    pos[1::3, 1] = a * X_RANGE[1] + b
    pos[2::3] = np.nan
    alphas = np.full(3 * N, alpha, dtype=np.float32)
    alphas[2::3] = np.nan
    df = pd.DataFrame({"x": pos[:, 0], "y": pos[:, 1], "alpha": alphas})
    cvs = ds.Canvas(plot_width=INSET_W, plot_height=INSET_H, x_range=X_RANGE, y_range=Y_RANGE)
    agg = cvs.line(df, "x", "y", agg=ds.sum("alpha"))
    img = tf.shade(agg, cmap=CMAP, how="log")
    _paste_onto_canvas(img.to_pil(), output_path)


def run_hvplot(N, a, b, alpha, output_path):
    pos = np.empty((3 * N, 2), dtype=np.float32)
    pos[0::3, 0] = X_RANGE[0]
    pos[0::3, 1] = a * X_RANGE[0] + b
    pos[1::3, 0] = X_RANGE[1]
    pos[1::3, 1] = a * X_RANGE[1] + b
    pos[2::3] = np.nan
    alphas = np.full(3 * N, alpha, dtype=np.float32)
    alphas[2::3] = np.nan
    df = pd.DataFrame({"x": pos[:, 0], "y": pos[:, 1], "alpha": alphas})
    plot = df.hvplot.line(
        x="x",
        y="y",
        rasterize=True,
        aggregator=ds.sum("alpha"),
        dynamic=False,
        cmap=CMAP,
        colorbar=False,
        xlim=X_RANGE,
        ylim=Y_RANGE,
    )
    plot = plot.opts(bgcolor="black", xaxis=None, yaxis=None, show_frame=False)
    temp = output_path + ".temp.png"
    hv.save(plot, temp)
    if os.path.exists(temp):
        _paste_onto_canvas(Image.open(temp).convert("RGBA"), output_path)
        os.remove(temp)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("   GLPLOT VS VISPY VS FASTPLOTLIB VS DATASHADER VS HVPLOT")
    print(
        f"   Repeats: {N_REPEATS}  |  Warm-up: {N_WARMUP}  |  "
        f"Effective reps: {N_REPEATS - N_WARMUP}"
    )
    print("   Metrics: wall-clock time  +  RSS delta (psutil) / tracemalloc")
    print("=" * 70)

    for name, avail in [
        ("VisPy", vispy_available),
        ("fastplotlib", fastplotlib_available),
        ("Datashader", datashader_available),
        ("hvPlot", hvplot_available),
    ]:
        if not avail:
            print(f"  ⚠  {name} not available")
    print(
        f"  psutil: {'available (RSS)' if psutil_available else 'NOT available – using tracemalloc'}"
    )

    output_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(output_dir, exist_ok=True)

    line_counts = [1_000, 10_000, 100_000, 1_000_000, 10_000_000]

    VISPY_CAP = 10_000_000
    FASTPLOTLIB_CAP = 10_000_000
    DATASHADER_CAP = 10_000_000
    HVPLOT_CAP = 10_000_000

    # Results: list of RunStats (or None) per library, one entry per N
    glplot_density_results: List[Optional[RunStats]] = []
    glplot_exact_results: List[Optional[RunStats]] = []
    vispy_results: List[Optional[RunStats]] = []
    fastplotlib_results: List[Optional[RunStats]] = []
    datashader_results: List[Optional[RunStats]] = []
    hvplot_results: List[Optional[RunStats]] = []

    for N in line_counts:
        print(f"\n{'─'*70}")
        print(f"  N = {N:,} lines")
        print(f"{'─'*70}")

        a = np.random.uniform(-8.0, 8.0, N).astype(np.float32)
        b = -(a**2) / 4.0
        alpha = _adaptive_alpha(N)
        colors = _make_colors(N, alpha)

        # ── GLPlot GPU Density ────────────────────────────────────────────
        print(f"\n  → GLPlot GPU Density  ({N_REPEATS} runs, {N_WARMUP} warm-up)")
        out_d = os.path.join(output_dir, f"glplot_gpu_density_{N}.png")

        def _glplot_density_factory(N=N, a=a, b=b, colors=colors, out=out_d):
            def _fn():
                gplt._cleanup_pyplot_state()
                gplt.figure(f"GLPlot Density {N}", width=FULL_W, height=FULL_H, density=True)
                gplt.options(
                    density_resolution_scale=1.0,
                    density_gain=2.0,
                    density_scheme_index=8,
                    axis_show_grid=False,
                    axis_show_labels=False,
                    axis_show_frame=False,
                    light_bg_mode=False,
                )
                gplt.plot_lines(a, b, x_range=X_RANGE, colors=colors)
                gplt.xlim(*X_RANGE)
                gplt.ylim(*Y_RANGE)
                gplt.show(test_mode=True)
                gplt.savefig(out, density=True)

            return _fn

        stats = _measure_repeated(_glplot_density_factory, "GLPlot GPU Density")
        glplot_density_results.append(stats)
        if stats:
            print(f"    ✓ {stats.time_str()}  mem={stats.mem_str()}")

        # ── GLPlot GPU Exact ──────────────────────────────────────────────
        print(f"\n  → GLPlot GPU Exact  ({N_REPEATS} runs, {N_WARMUP} warm-up)")
        out_e = os.path.join(output_dir, f"glplot_gpu_exact_{N}.png")

        def _glplot_exact_factory(N=N, a=a, b=b, colors=colors, out=out_e):
            def _fn():
                gplt._cleanup_pyplot_state()
                gplt.figure(f"GLPlot Exact {N}", width=FULL_W, height=FULL_H, density=False)
                gplt.options(
                    axis_show_grid=False,
                    axis_show_labels=False,
                    axis_show_frame=False,
                    light_bg_mode=False,
                )
                gplt.plot_lines(a, b, x_range=X_RANGE, colors=colors)
                gplt.xlim(*X_RANGE)
                gplt.ylim(*Y_RANGE)
                gplt.show(test_mode=True)
                gplt.savefig(out, density=False)

            return _fn

        stats = _measure_repeated(_glplot_exact_factory, "GLPlot GPU Exact")
        glplot_exact_results.append(stats)
        if stats:
            print(f"    ✓ {stats.time_str()}  mem={stats.mem_str()}")

        # ── VisPy ─────────────────────────────────────────────────────────
        if vispy_available and N <= VISPY_CAP:
            print(f"\n  → VisPy GPU Lines  ({N_REPEATS} runs, {N_WARMUP} warm-up)")
            out_v = os.path.join(output_dir, f"vispy_gpu_lines_{N}.png")

            def _vispy_factory(N=N, a=a, b=b, colors=colors, out=out_v):
                return lambda: run_vispy(N, a, b, colors, out)

            stats = _measure_repeated(_vispy_factory, "VisPy GPU Lines")
            vispy_results.append(stats)
            if stats:
                print(f"    ✓ {stats.time_str()}  mem={stats.mem_str()}")
        else:
            if vispy_available and N > VISPY_CAP:
                print(f"  Skipping VisPy (N>{VISPY_CAP:,})")
            vispy_results.append(None)

        # ── fastplotlib ───────────────────────────────────────────────────
        if fastplotlib_available and N <= FASTPLOTLIB_CAP:
            print(f"\n  → fastplotlib GPU Lines  ({N_REPEATS} runs, {N_WARMUP} warm-up)")
            out_f = os.path.join(output_dir, f"fastplotlib_gpu_lines_{N}.png")

            def _fpl_factory(N=N, a=a, b=b, colors=colors, out=out_f):
                return lambda: run_fastplotlib(N, a, b, colors, out)

            stats = _measure_repeated(_fpl_factory, "fastplotlib GPU Lines")
            fastplotlib_results.append(stats)
            if stats:
                print(f"    ✓ {stats.time_str()}  mem={stats.mem_str()}")
        else:
            if fastplotlib_available and N > FASTPLOTLIB_CAP:
                print(f"  Skipping fastplotlib (N>{FASTPLOTLIB_CAP:,})")
            fastplotlib_results.append(None)

        # ── Datashader ────────────────────────────────────────────────────
        if datashader_available and N <= DATASHADER_CAP:
            print(f"\n  → Datashader CPU  ({N_REPEATS} runs, {N_WARMUP} warm-up)")
            out_ds = os.path.join(output_dir, f"datashader_density_{N}.png")

            def _ds_factory(N=N, a=a, b=b, alpha=alpha, out=out_ds):
                return lambda: run_datashader(N, a, b, alpha, out)

            stats = _measure_repeated(_ds_factory, "Datashader")
            datashader_results.append(stats)
            if stats:
                print(f"    ✓ {stats.time_str()}  mem={stats.mem_str()}")
        else:
            if datashader_available and N > DATASHADER_CAP:
                print(f"  Skipping Datashader (N>{DATASHADER_CAP:,})")
            datashader_results.append(None)

        # ── hvPlot ────────────────────────────────────────────────────────
        if hvplot_available and N <= HVPLOT_CAP:
            print(f"\n  → hvPlot  ({N_REPEATS} runs, {N_WARMUP} warm-up)")
            out_hv = os.path.join(output_dir, f"hvplot_density_{N}.png")

            def _hv_factory(N=N, a=a, b=b, alpha=alpha, out=out_hv):
                return lambda: run_hvplot(N, a, b, alpha, out)

            stats = _measure_repeated(_hv_factory, "hvPlot")
            hvplot_results.append(stats)
            if stats:
                print(f"    ✓ {stats.time_str()}  mem={stats.mem_str()}")
        else:
            if hvplot_available and N > HVPLOT_CAP:
                print(f"  Skipping hvPlot (N>{HVPLOT_CAP:,})")
            hvplot_results.append(None)

    # ──────────────────────────────────────────────────────────────────────
    # Summary table
    # ──────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print(f"  RESULTS SUMMARY  (mean ± std  over {N_REPEATS - N_WARMUP} runs, format: time / mem)")
    print("=" * 90)
    col_w = 26
    print(f"{'Library':<{col_w}}", end="")
    for N in line_counts:
        label = f"10^{int(np.log10(N))}"
        print(f"  {label:>18}", end="")
    print()
    print("-" * (col_w + 20 * len(line_counts)))

    all_series = [
        ("GLPlot GPU Density", glplot_density_results),
        ("GLPlot GPU Exact", glplot_exact_results),
        ("VisPy GPU Lines", vispy_results),
        ("fastplotlib GPU Lines", fastplotlib_results),
        ("Datashader (CPU)", datashader_results),
        ("hvPlot", hvplot_results),
    ]
    for label, results in all_series:
        print(f"{label:<{col_w}}", end="")
        for r in results:
            if r is None:
                print(f"  {'—':>18}", end="")
            else:
                cell = f"{r.t_mean:.2f}±{r.t_std:.2f}s/{_fmt_mem(r.m_mean)}"
                print(f"  {cell:>18}", end="")
        print()
    print()

    # ──────────────────────────────────────────────────────────────────────
    # Charts: 2 subplots (time + memory) with error bands
    # ──────────────────────────────────────────────────────────────────────
    print("Generating benchmark plot with error bars...")
    try:
        fig, (ax_t, ax_m) = plt.subplots(1, 2, figsize=(18, 7), dpi=120)

        series_meta = [
            ("GLPlot GPU Density", glplot_density_results, "o-", "#1f77b4", 2.5),
            ("GLPlot GPU Exact", glplot_exact_results, "s--", "#2ca02c", 2.0),
            ("VisPy GPU Lines", vispy_results, "d--", "#d62728", 2.0),
            ("fastplotlib GPU Lines (WGPU)", fastplotlib_results, "^--", "#ff7f0e", 2.0),
            ("Datashader (CPU α-density)", datashader_results, "x--", "#9467bd", 2.0),
            ("hvPlot (Datashader + matplotlib)", hvplot_results, "P:", "#e377c2", 2.0),
        ]

        for label, results, fmt, color, lw in series_meta:
            idxs = [i for i, r in enumerate(results) if r is not None]
            if not idxs:
                continue
            xs = np.array([line_counts[i] for i in idxs])
            t_mu = np.array([results[i].t_mean for i in idxs])
            t_sig = np.array([results[i].t_std for i in idxs])
            m_mu = np.array([results[i].m_mean for i in idxs])
            m_sig = np.array([results[i].m_std for i in idxs])

            # Time subplot
            ax_t.plot(xs, t_mu, fmt, color=color, linewidth=lw, label=label)
            ax_t.fill_between(
                xs, np.maximum(t_mu - t_sig, 1e-6), t_mu + t_sig, alpha=0.18, color=color
            )
            for x, mu, sig in zip(xs, t_mu, t_sig):
                if x >= 100_000:
                    ax_t.annotate(
                        f"{mu:.2f}±{sig:.2f}s",
                        (x, mu),
                        textcoords="offset points",
                        xytext=(0, 10),
                        ha="center",
                        fontsize=6.5,
                    )

            # Memory subplot
            ax_m.plot(xs, m_mu, fmt, color=color, linewidth=lw, label=label)
            ax_m.fill_between(xs, m_mu - m_sig, m_mu + m_sig, alpha=0.18, color=color)
            for x, mu, sig in zip(xs, m_mu, m_sig):
                if x >= 100_000:
                    ax_m.annotate(
                        f"{_fmt_mem(mu)}±{sig:.0f}",
                        (x, mu),
                        textcoords="offset points",
                        xytext=(0, 10),
                        ha="center",
                        fontsize=6.5,
                    )

        for ax in (ax_t, ax_m):
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("Number of Lines", fontsize=12)
            ax.grid(True, which="both", linestyle=":", alpha=0.6)
            ax.legend(fontsize=9, loc="upper left")

        ax_t.set_ylabel("Wall-clock Time (s)", fontsize=12)
        ax_t.set_title(
            f"Rendering Time  (10³ → 10⁷)\nmean ± σ over {N_REPEATS-N_WARMUP} reps",
            fontsize=13,
            fontweight="bold",
        )

        mem_label = "Peak RSS Δ (MiB)" if psutil_available else "Peak heap via tracemalloc (MiB)"
        ax_m.set_ylabel(mem_label, fontsize=12)
        ax_m.set_title(
            f"Memory Usage  (10³ → 10⁷)\nmean ± σ over {N_REPEATS-N_WARMUP} reps",
            fontsize=13,
            fontweight="bold",
        )

        fig.suptitle(
            "GLPlot vs VisPy vs fastplotlib vs Datashader vs hvPlot\n"
            "Line Rendering Benchmark — Time & Memory with Variance",
            fontsize=15,
            fontweight="bold",
            y=1.02,
        )
        out_chart = os.path.join(output_dir, "gpu_comparison_benchmark_results.png")
        fig.tight_layout()
        fig.savefig(out_chart, bbox_inches="tight")
        plt.close(fig)
        print(f"Chart saved → {out_chart}")

    except Exception as e:
        print(f"Failed to generate plot: {e}")

    print("\nBenchmark Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
