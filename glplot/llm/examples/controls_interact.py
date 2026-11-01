"""Sliders that drive a plot (live window), and what happens without one.

plt.slider / plt.interact add controls to a Controls panel beside the plot. The function runs
once immediately, so a headless script still gets the figure for the initial values; with
plt.show() every change re-runs it. Use @plt.interact for the common 'function of parameters'.
"""

import numpy as np

import glplot.pyplot as plt

x = np.linspace(0, 10, 1000)

plt.figure("Tunable wave", figsize=(8, 5))
plt.xlabel("x")
plt.ylabel("y")
# The curve arrives later (from the function below); limits set now still hold, and an
# axis you do not set keeps fitting the data.
plt.ylim(-5.5, 5.5)


@plt.interact(amp=(0, 5), freq=(0.1, 10), wave=["sin", "cos"])
def signal(amp=1.0, freq=2.0, wave="sin"):
    return x, amp * getattr(np, wave)(freq * x)


plt.savefig("controls_interact.png")  # plt.show() opens the window with the sliders
