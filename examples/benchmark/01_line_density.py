import os
import sys
import time
import warnings

import numpy as np

# Force local glplot import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import matplotlib

matplotlib.use("Agg")  # Force non-GUI backend
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

import glplot.pyplot as gplt

# Suppress warnings for clean output
warnings.filterwarnings("ignore")


def main():
    print("=" * 60)
    print("   GLPLOT VS MATPLOTLIB LINE DENSITY BENCHMARK")
    print("=" * 60)
    print("Running in headless mode...")

    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(output_dir, exist_ok=True)

    # Line counts to benchmark
    line_counts = [10, 100, 1000, 10000, 100000, 1000000]

    # Data structures to store benchmark times
    glplot_times = []
    mpl_lines_times = []
    mpl_density_times = []

    # Common parameters
    x_range = (-5.0, 5.0)
    width, height = 800, 600

    for N in line_counts:
        print(f"\n--- Benchmarking N = {N:,} lines ---")

        # 1. Generate line parameters: y = a*x + b
        a = np.random.randn(N).astype(np.float32) * 2.0
        b = np.random.randn(N).astype(np.float32) * 250.0

        # Colors with transparency
        colors = np.random.rand(N, 4).astype(np.float32)
        colors[:, 3] = 0.05  # Alpha

        # ----------------- GLPLOT GPU DENSITY RENDER -----------------
        print("  Running GLPlot GPU Density Rendering...")
        try:
            t0 = time.perf_counter()
            # Reset glplot state
            gplt._cleanup_pyplot_state()
            gplt.figure(f"GLPlot Density {N}", width=width, height=height, density=True)

            # Use GLPlot options for high performance density
            gplt.options(density_resolution_scale=1.0, density_gain=2.0)
            gplt.plot_lines(a, b, x_range=x_range, colors=colors)
            gplt.xlim(x_range[0], x_range[1])
            gplt.ylim(-1000, 1000)

            # Headless show & savefig
            gplt.show(test_mode=True)
            glplot_out = os.path.join(output_dir, f"glplot_density_{N}.png")
            gplt.savefig(glplot_out, density=True)

            t_gl = time.perf_counter() - t0
            glplot_times.append(t_gl)
            print(f"    GLPlot GPU: {t_gl:.4f} seconds")
        except Exception as e:
            print(f"    GLPlot failed: {e}")
            glplot_times.append(None)

        # ----------- MATPLOTLIB STANDARD LINES (LineCollection) -----------
        # We cap standard Matplotlib line plotting to 100,000 lines to avoid long hangs
        if N <= 100000:
            print("  Running Matplotlib CPU LineCollection...")
            try:
                t0 = time.perf_counter()
                fig, ax = plt.subplots(figsize=(width / 100.0, height / 100.0), dpi=100)

                # Construct segments optimized as a single array of shape (N, 2, 2)
                segments = np.empty((N, 2, 2))
                segments[:, 0, 0] = x_range[0]
                segments[:, 0, 1] = a * x_range[0] + b
                segments[:, 1, 0] = x_range[1]
                segments[:, 1, 1] = a * x_range[1] + b

                lc = LineCollection(segments, colors=colors, linewidths=1.0)
                ax.add_collection(lc)
                ax.set_xlim(x_range)
                ax.set_ylim(-1000, 1000)
                ax.set_title(f"Matplotlib Lines {N}")

                mpl_lines_out = os.path.join(output_dir, f"matplotlib_lines_{N}.png")
                fig.savefig(mpl_lines_out, dpi=100)
                plt.close(fig)

                t_mpl_l = time.perf_counter() - t0
                mpl_lines_times.append(t_mpl_l)
                print(f"    Matplotlib Lines: {t_mpl_l:.4f} seconds")
            except Exception as e:
                print(f"    Matplotlib Lines failed: {e}")
                mpl_lines_times.append(None)
        else:
            print("  Running Matplotlib CPU LineCollection... Capped (skipped for N >= 1,000,000)")
            mpl_lines_times.append(None)

        # ----------- MATPLOTLIB CPU 2D HISTOGRAM DENSITY -----------
        print("  Running Matplotlib CPU 2D Histogram Density...")
        try:
            t0 = time.perf_counter()
            fig, ax = plt.subplots(figsize=(width / 100.0, height / 100.0), dpi=100)

            # CPU evaluation of lines and density calculation
            xs = np.linspace(x_range[0], x_range[1], width, dtype=np.float32)
            y0 = a * x_range[0] + b
            y1 = a * x_range[1] + b

            ymin_v, ymax_v = -1000.0, 1000.0
            density = np.zeros((height, len(xs)), dtype=np.float32)
            y_edges = np.linspace(ymin_v, ymax_v, density.shape[0] + 1, dtype=np.float32)

            # Process in chunks to prevent large CPU memory footprint
            chunk_size = 4096
            ab = np.column_stack([a, b])
            for start in range(0, N, chunk_size):
                chunk = ab[start : start + chunk_size]
                # Evaluates y = ax + b for all x
                ys = chunk[:, 0, None] * xs[None, :] + chunk[:, 1, None]
                for col in range(len(xs)):
                    density[:, col] += np.histogram(ys[:, col], bins=y_edges)[0]

            ax.imshow(
                np.log1p(density),
                extent=(x_range[0], x_range[1], ymin_v, ymax_v),
                origin="lower",
                aspect="auto",
                cmap="magma",
                interpolation="bilinear",
            )
            ax.set_title(f"Matplotlib Density {N}")

            mpl_density_out = os.path.join(output_dir, f"matplotlib_density_{N}.png")
            fig.savefig(mpl_density_out, dpi=100)
            plt.close(fig)

            t_mpl_d = time.perf_counter() - t0
            mpl_density_times.append(t_mpl_d)
            print(f"    Matplotlib CPU Density: {t_mpl_d:.4f} seconds")
        except Exception as e:
            print(f"    Matplotlib Density failed: {e}")
            mpl_density_times.append(None)

    # ---------------- GENERATE BENCHMARK RESULTS PLOT ----------------
    print("\nGenerating final comparison plot...")
    try:
        fig, ax = plt.subplots(figsize=(10, 6), dpi=120)

        # Filter None values for plotting
        valid_lc_idx = [
            i
            for i, t in enumerate(line_counts)
            if i < len(glplot_times) and glplot_times[i] is not None
        ]
        valid_mpl_l_idx = [
            i
            for i, t in enumerate(line_counts)
            if i < len(mpl_lines_times) and mpl_lines_times[i] is not None
        ]
        valid_mpl_d_idx = [
            i
            for i, t in enumerate(line_counts)
            if i < len(mpl_density_times) and mpl_density_times[i] is not None
        ]

        ax.plot(
            [line_counts[i] for i in valid_lc_idx],
            [glplot_times[i] for i in valid_lc_idx],
            "o-",
            color="#1f77b4",
            linewidth=2.5,
            label="GLPlot GPU Density",
        )

        ax.plot(
            [line_counts[i] for i in valid_mpl_l_idx],
            [mpl_lines_times[i] for i in valid_mpl_l_idx],
            "s--",
            color="#d62728",
            linewidth=2,
            label="Matplotlib CPU Lines (LineCollection)",
        )

        ax.plot(
            [line_counts[i] for i in valid_mpl_d_idx],
            [mpl_density_times[i] for i in valid_mpl_d_idx],
            "^--",
            color="#ff7f0e",
            linewidth=2,
            label="Matplotlib CPU 2D Density (Histogram)",
        )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Number of Lines", fontsize=12)
        ax.set_ylabel("Execution + Save Time (seconds)", fontsize=12)
        ax.set_title("GLPlot vs Matplotlib: Line Density Benchmark", fontsize=14, fontweight="bold")
        ax.grid(True, which="both", linestyle=":", alpha=0.6)
        ax.legend(fontsize=10, loc="upper left")

        # Add values labels for the highest line counts
        for i in valid_lc_idx:
            count = line_counts[i]
            t = glplot_times[i]
            if count >= 10000:
                ax.annotate(
                    f"{t:.2f}s",
                    (count, t),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha="center",
                    fontsize=9,
                    color="#1f77b4",
                )

        for i in valid_mpl_d_idx:
            count = line_counts[i]
            t = mpl_density_times[i]
            if count >= 10000:
                ax.annotate(
                    f"{t:.2f}s",
                    (count, t),
                    textcoords="offset points",
                    xytext=(0, -15),
                    ha="center",
                    fontsize=9,
                    color="#ff7f0e",
                )

        benchmark_img = os.path.join(output_dir, "line_density_benchmark_results.png")
        fig.tight_layout()
        fig.savefig(benchmark_img)
        plt.close(fig)
        print(f"Benchmark results comparison figure saved: {benchmark_img}")

    except Exception as e:
        print(f"Failed to generate benchmark results plot: {e}")

    print("\nBenchmark Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
