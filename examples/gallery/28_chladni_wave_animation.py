import numpy as np

import glplot.animation as animation
import glplot.pyplot as plt

N = 360
x = np.linspace(-1.0, 1.0, N)
y = np.linspace(-1.0, 1.0, N)
X, Y = np.meshgrid(x, y)

FRAMES = 54


def chladni_field(m: float, n: float) -> np.ndarray:
    """Superposed vibrational eigenmodes of a square plate (a Chladni pattern).

    sin(m*pi*x)*cos(n*pi*y) - sin(n*pi*x)*cos(m*pi*y) is the classic closed form for the
    standing-wave displacement of a square membrane driven at mode numbers (m, n); its
    zero-crossings are exactly the nodal lines sand collects on in a real Chladni-plate
    demonstration.
    """
    return np.sin(m * np.pi * X) * np.cos(n * np.pi * Y) - np.sin(n * np.pi * X) * np.cos(m * np.pi * Y)


def mode_pair(frame: int) -> tuple:
    t = frame / FRAMES
    m = 2.2 + 4.3 * (0.5 - 0.5 * np.cos(2 * np.pi * t))
    n = 3.1 + 3.6 * (0.5 - 0.5 * np.cos(2 * np.pi * t * 1.5 + 1.1))
    return m, n


def zoom_span(frame: int) -> float:
    t = frame / FRAMES
    return 0.75 - 0.35 * np.cos(2 * np.pi * t)


fig = plt.figure("Gallery - Animated Standing-Wave Interference", figsize=(7.8, 6.2))
plt.plot_style("blueprint")


def update(frame: int):
    m, n = mode_pair(frame)
    span = zoom_span(frame)
    Z = chladni_field(m, n)
    plt.cla()
    plt.imshow(Z, extent=(-1, 1, -1, 1), origin="lower", cmap="RdBu_r", vmin=-1.0, vmax=1.0, aspect="equal")
    plt.xlim(-span, span)
    plt.ylim(-span, span)
    plt.title(f"Standing-wave interference (m={m:.1f}, n={n:.1f})")
    plt.xlabel("x (plate width, normalized)")
    plt.ylabel("y (plate width, normalized)")
    return []


ani = animation.FuncAnimation(fig, update, frames=FRAMES, interval=42)
ani.save("examples/gallery/results/28_chladni_wave_animation.gif", fps=20)
