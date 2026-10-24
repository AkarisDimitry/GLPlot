import numpy as np

import glplot.pyplot as plt

# Low-poly potential energy surface (PES) built from nothing but sine and
# cosine: a sum of many plane waves (sin(kx*x + ky*y + phase) * cos(...)) at
# random frequencies, directions and phases, amplitude falling off with each
# wave so the terrain reads as one coherent rocky landscape rather than pure
# noise -- no Gaussian wells, no macro shape, just trigonometric terms.
# Rendered "low poly": the surface is sampled on a deliberately coarse base
# grid, so plot_surface's default (no further decimation) facets are already
# large, flat panels, with a same-resolution black wireframe overlay tracing
# every facet edge -- the same technique example 12 uses for its floor
# projection scaffold.
rng = np.random.default_rng(7)
n_waves = 24
wave_freq = rng.uniform(3.0, 11.0, n_waves)
wave_angle = rng.uniform(0.0, 2.0 * np.pi, n_waves)
wave_phase = rng.uniform(0.0, 2.0 * np.pi, n_waves)
wave_amp = 1.0 / np.arange(1, n_waves + 1)  # higher frequencies contribute less, like octave noise


def wave_pes(x, y):
    v = np.zeros_like(x)
    for amp, freq, angle, phase in zip(wave_amp, wave_freq, wave_angle, wave_phase):
        kx = freq * np.cos(angle)
        ky = freq * np.sin(angle)
        v += amp * np.sin(kx * x + ky * y + phase) * np.cos(0.5 * kx * x - 0.5 * ky * y - phase)
    return v


xs = np.linspace(-8.8, 8.8, 900)
ys = np.linspace(-8.4, 8.4, 900)
x, y = np.meshgrid(xs, ys)
v = wave_pes(x, y)

plt.figure("Gallery - Low-Poly Sine/Cosine Potential Energy Surface", figsize=(9, 6))
plt.plot_surface(
    x,
    y,
    v,
    cmap="coolwarm",
    elev=48,
    azim=-58,
    alpha=1.0,
)
plt.plot_wireframe(
    x,
    y,
    v + 0.02,
    elev=48,
    azim=-58,
    rstride=1,
    cstride=1,
    color=(0.05, 0.05, 0.05, 0.35),
    linewidth=0.5,
)
plt.title("Low-poly PES -- built from sine and cosine plane waves only")
plt.set_axis3d_off()  # drop the box, floor, grid, ticks and labels -- just the terrain
plt.show()
plt.savefig("examples/gallery/results/30_low_poly_potential_energy_surface.png")
