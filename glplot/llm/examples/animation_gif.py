"""An animation saved as a GIF, with no window.

glplot.animation mirrors matplotlib.animation (it is NOT on plt). The update function receives
the frame number and must be pure with respect to it: compute everything from the frame, do not
rely on state from the previous call, so a saved file and live playback agree.
"""

import numpy as np

import glplot.animation as animation
import glplot.pyplot as plt

x = np.linspace(0, 2 * np.pi, 200)
FRAMES = 30

fig = plt.figure("Traveling wave", figsize=(7, 4))
plt.xlim(0, 2 * np.pi)
plt.ylim(-1.2, 1.2)
(line,) = plt.plot(x, np.sin(x), lw=2)
plt.xlabel("x")
plt.ylabel("amplitude")


def update(frame):
    phase = 2 * np.pi * frame / FRAMES
    line.set_ydata(np.sin(x - phase))
    plt.title(f"phase = {phase:.2f} rad")
    return [line]


ani = animation.FuncAnimation(fig, update, frames=FRAMES, interval=50)
ani.save("animation_gif.gif", fps=15)  # plt.show() plays it live instead
