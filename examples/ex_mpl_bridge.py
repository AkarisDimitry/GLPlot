import matplotlib.pyplot as plt
import numpy as np

import glplot.pyplot as gplt


def run_bridge_example():
    # 1. Generate some interesting data (Sine wave with noise)
    n_lines = 100
    x = np.linspace(-10, 10, 1000)

    # Create multiple lines with varying amplitude and phase
    for i in range(n_lines):
        phase = i * 0.1
        amp = 1.0 + np.sin(i * 0.2)
        y = amp * np.sin(x + phase) + np.random.normal(0, 0.05, len(x))
        gplt.plot(x, y, label=f"Line {i}")

    print("\n--- GLPlot with Matplotlib Bridge ---")
    print("Instructions:")
    print("1. An interactive GLPlot window will open.")
    print("2. Adjust the view (zoom/pan) to your liking.")
    print("3. Press 'M' on your keyboard to 'Teleport' the view to Matplotlib.")
    print("4. Or, see the code below for programmatic transfer.")

    # 2. Get the engine instance
    plot = gplt.get_engine()

    # You can configure where 'M' sends the data if you want:
    # plot.set_matplotlib_transfer_target(ax=some_existing_ax)

    # 3. Show the interactive plot
    gplt.show()

    # --- Programmatic Transfer Example ---
    # After you close the GLPlot window, or if running in a script:
    print("\nProgrammatically transferring current view to Matplotlib...")

    # We create a Matplotlib figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Transfer view to the first subplot
    plot.to_matplotlib(
        ax=ax1,
        scale=2.0,  # High-res capture
        transparent=True,
        include_axes=False,  # We want Matplotlib to handle the axes
    )
    ax1.set_title("Transferred Raster (High Res)")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")
    ax1.grid(True, alpha=0.3)

    # We can even add Matplotlib geometry ON TOP of the GLPlot raster
    x_mark = np.linspace(-10, 10, 10)
    ax2.plot(x_mark, np.cos(x_mark), "ro--", label="MPL Overlay")

    # Transfer the same view to the second subplot as a background
    plot.to_matplotlib(
        ax=ax2,
        scale=1.0,
    )  # alpha=0.5) # Semi-transparent background
    ax2.set_title("With Matplotlib Overlays")
    ax2.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_bridge_example()
