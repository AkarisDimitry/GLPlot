import time

import glfw
import numpy as np

import glplot.pyplot as gplt


def diagnostic_view_persistence():
    """
    Diagnostic script to check if camera state is being reset.
    """
    x = np.linspace(-10, 10, 1000)
    y = np.sin(x)
    gplt.plot(x, y, color=(0.2, 0.4, 0.8, 1.0))

    plot = gplt._get_or_create_plot()

    # 1. Set initial view
    print("Setting initial view manually: X[-5, 5]")
    gplt.xlim(-5, 5)

    # Track state
    initial_cx = plot.camera.cx
    initial_zoom = plot.camera.zoom
    print(f"Initial State: cx={initial_cx}, zoom={initial_zoom}")

    # We'll run a few frames and check if it maintains state
    # Since we can't easily interact via script, we'll simulate a 'pan'
    print("Simulating a pan of +1.0 in world units...")
    plot.camera.cx += 1.0

    # Now check if it's maintained after a fake draw
    # This involves calling the internal draw logic
    # In practice, I'll just check if GPULinePlot logic resets it anywhere.

    print("Current State: cx={}".format(plot.camera.cx))

    # Let's see if reset_view/autoscale/set_view have bad defaults
    print("\nChecking for bad defaults in set_view(xlim=None):")
    plot.set_view(xlim=None, ylim=None)
    print(
        "State after empty set_view: cx={}, cy={}, zoom={}".format(
            plot.camera.cx, plot.camera.cy, plot.camera.zoom
        )
    )

    # Check if a resize would reset it
    print("\nChecking if height/width update in set_view affects zoom:")
    plot.width = 1000
    plot.height = 1000
    plot.set_view(xlim=(-5, 5))
    print("State after set_view at 1000x1000: zoom={}".format(plot.camera.zoom))

    plot.width = 2000
    plot.height = 1000
    plot.set_view(xlim=(-5, 5))
    print("State after set_view at 2000x1000: zoom={}".format(plot.camera.zoom))


if __name__ == "__main__":
    diagnostic_view_persistence()
