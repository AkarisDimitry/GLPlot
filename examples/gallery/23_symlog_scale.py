import numpy as np

import glplot.pyplot as plt

rng = np.random.default_rng(23)
N = 2_000_000
t = np.linspace(0, 60, N)
# A damped LC-circuit ringdown recorded at high sample rate: it starts with a swing of
# +-500 mA and decays into small ripples around zero riding on real sensor noise. A plain
# log axis can't show this at all (it masks every negative half-cycle), and a linear axis
# would flatten the late, small structure -- and the noise that lives with it -- into an
# invisible flat line next to the early peak. symlog is built for exactly this -- linear
# inside `linthresh` so the zero crossings and the noise floor both stay visible,
# logarithmic outside it so the early peak doesn't dominate the whole axis. At 2,000,000
# raw samples the near-zero region isn't a clean analytic curve but an actual dense, noisy
# trace -- which is the whole point of resolving it at all.
envelope = 500.0 * np.exp(-t / 8.0)
ringdown = envelope * np.cos(2.0 * t)
# A much smaller, higher-frequency parasitic resonance that a coarser sampling would alias
# away, plus per-sample sensor noise -- both only become visible once the main ringdown
# decays into symlog's linear zone.
ripple = 1.1 * np.cos(2.0 * np.pi * 55.0 * t) * np.exp(-t / 22.0)
noise = 0.3 * rng.standard_normal(N)
signal_clean = ringdown + ripple  # the deterministic physics, no sensor noise
signal = signal_clean + noise  # what the instrument actually measured, 2,000,000 times

plt.figure("Gallery - Damped Ringdown (Symlog Scale)", figsize=(9, 5))
plt.plot_style("neon")  # true black background so the glow has somewhere to bloom into

# Two-color palette: acid-green for the trace itself (the classic CRT-phosphor color this
# kind of ringdown is displayed in), hot magenta for everything that explains it -- the raw
# sample cloud underneath and the decay envelope on top -- so a single glance separates
# "the signal" from "the millions of measurements and the physics that bounds them".
scope_green = "#39ff14"
accent_magenta = "#ff2fd0"
base_lw = 0.6

# Raw-sample texture: every 5th one of the 2,000,000 individual noisy measurements (the real
# instrument reading, ringdown + parasitic ripple + sensor noise), plotted as its own scatter
# in magenta at very low alpha -- not the same values as the green line below, so the grain
# genuinely spreads around the core instead of hiding underneath it. The dense early lobes
# pile the dots into a solid magenta wash while the sparse late ripple resolves into visible
# individual grains, making the huge N a texture you can see rather than a number in a title.
scatter_stride = 5
plt.scatter(
    t[::scatter_stride],
    signal[::scatter_stride],
    color=accent_magenta,
    s=1.6,
    alpha=0.05,
    edgecolors="none",
)

# Oscilloscope trace: the noise-free ringdown + parasitic ripple, crisp and thin at full
# 2,000,000-sample resolution, so the fine high-frequency ripple shape stays sharp under the
# glow -- the clean core the noisy magenta measurements above scatter around.
plt.plot(t, signal_clean, color=scope_green, lw=base_lw, label="coil current")

# Underglow fill: a soft wash from the trace down to the zero line. Downsampled -- at full
# resolution the ripple crosses zero often enough that a filled band turns into fine
# zebra-striping instead of a smooth glow; every 4th sample keeps the big early lobes solid
# and the fill fast without changing what the eye reads.
fill_stride = 4
plt.fill_between(
    t[::fill_stride], signal_clean[::fill_stride], 0.0, color=scope_green, alpha=0.1
)

# Glow: the same trace redrawn progressively thicker and more transparent so the stacked
# translucent copies blend into a soft halo around the crisp core above -- the real
# mplcyberpunk recipe (10 layers, alpha budget 0.3 split evenly, linewidth +1.05/layer).
# The glow copies are downsampled (every 6th sample) -- at 2,000,000 points x 10 extra full
# plot() calls the headless Agg export would crawl, and a glow this blurred never needed
# full resolution to look soft in the first place.
glow_stride = 6
t_glow = t[::glow_stride]
signal_glow = signal_clean[::glow_stride]
n_layers = 10
for i in range(1, n_layers + 1):
    plt.plot(
        t_glow,
        signal_glow,
        color=scope_green,
        linewidth=base_lw + 1.05 * i,
        alpha=0.3 / n_layers,
        solid_capstyle="round",
    )

# Decay envelope: the +-500*exp(-t/8) bound the ringdown actually rides inside, drawn as its
# own thin glowing magenta curve on a smooth 400-point grid (an analytic bound, so it never
# needed the raw sample rate). It ties back to the same scatter color, cuts straight through
# both the logarithmic outer zone and symlog's linear zone around zero, and gives the plot a
# second story on top of "here is the signal": the physical envelope that shape is bounded by.
t_env = np.linspace(0, 60, 400)
env_curve = 500.0 * np.exp(-t_env / 8.0)
env_lw = 1.2
n_env_glow = 5
for sign, lbl in ((1.0, "decay envelope"), (-1.0, None)):
    plt.plot(t_env, sign * env_curve, color=accent_magenta, lw=env_lw, alpha=0.9, label=lbl)
    for i in range(1, n_env_glow + 1):
        plt.plot(
            t_env,
            sign * env_curve,
            color=accent_magenta,
            linewidth=env_lw + 1.3 * i,
            alpha=0.25 / n_env_glow,
            solid_capstyle="round",
        )

plt.yscale("symlog", linthresh=5)
plt.grid(True)
plt.xlabel("Time (s)")
plt.ylabel("Current (mA)")
plt.title(f"Damped ringdown + decay envelope (symlog, linthresh=5, N={N // 1_000_000}M)")
plt.legend(loc="upper left")
# plt.show()
plt.savefig("examples/gallery/results/23_symlog_scale.png")
