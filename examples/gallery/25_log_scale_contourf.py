import numpy as np

import glplot.pyplot as plt

# A device's response surface swept across three decades of drive frequency, with a
# simulated particle detector sweeping the same bench: millions of individual hits, landing
# more often wherever the response is strong, plotted one dot per detection. On a linear
# axis the interesting structure near the low end of x would be crushed into a sliver a few
# pixels wide. contourf() draws through imshow() internally, and imshow()'s image quad now
# respects a real axis scale, so both the field and the detector hits stay legible on a
# genuine log x-axis.
rng = np.random.default_rng(25)
x = np.logspace(0, 3, 2000)  # drive frequency, Hz (1 Hz to 1 kHz)
y = np.linspace(1.0, 50.0, 1500)  # drive amplitude, mV
X, Y = np.meshgrid(x, y)
Z = np.sin(np.log10(X) * 2.0) * np.cos(Y / 10.0)  # response, arbitrary units

plt.figure("Gallery - Real Log Scale contourf", figsize=(9, 5))
# Charcoal page instead of the default light chrome -- a neutral dark ground makes the
# warm contourf cmap (inferno) read as a genuine heat field and gives the cyan hot-spot
# glow below something dark to bloom against. Unlike "blueprint", "dark" doesn't tint the
# page with a strong hue of its own, so it stays out of the way of a colormap that means
# something (the contourf field), which is exactly the trade its own docstring flags.
plt.plot_style("dark")
plt.contourf(X, Y, Z, levels=12, cmap="inferno")

# Millions of simulated detector hits, softly weighted toward the strong-response lobes of
# the same field -- drawn directly from Z's own distribution as a single weighted sample
# rather than rejection-sampled, so a huge point count costs one draw instead of millions
# of retries. The bias is gentle on purpose: a hard cutoff would flood the bright lobes
# into one solid blob and hide the contour bands under it, so the density instead grades
# with the field, sparse over the negative-response lobes and denser -- but still
# translucent -- over the positive ones. Jitter is applied in log10(x) so a hit lands
# uniformly within its own logspace cell rather than being biased toward one edge of it.
n_hits = 1_500_000
weights = np.exp(1.5 * Z).ravel()
weights /= weights.sum()
flat_idx = rng.choice(weights.size, size=n_hits, p=weights)
iy, ix = np.divmod(flat_idx, X.shape[1])
log10_x = np.log10(x)
half_dlogx = (log10_x[1] - log10_x[0]) / 2.0
half_dy = (y[1] - y[0]) / 2.0
hit_x = 10.0 ** (log10_x[ix] + rng.uniform(-half_dlogx, half_dlogx, n_hits))
hit_y = y[iy] + rng.uniform(-half_dy, half_dy, n_hits)
# Pale, low-alpha stipple over the whole field -- the "millions of elements" backbone --
# in a near-white ink so it reads as fine detector grain against the dark inferno field
# rather than fighting the colormap for attention.
plt.scatter(hit_x, hit_y, color="#eaf3ff", s=0.35, alpha=0.045, edgecolors="none")

# Hot-spot glow: re-weight the *existing* hit cloud far more sharply toward its own
# strongest-response hits (weight ~ hit's own field value^4, versus the gentle ^1.5 the
# whole cloud was drawn with) and draw a few thousand of them without replacement -- a
# genuinely field-strength-ranked subset of real hits, not a fresh sample. The weighting
# stays continuous rather than a hard top-N% cutoff, so the selected points thin out
# smoothly away from each peak instead of stopping at a sharp geographic edge; that
# smooth thinning is what lets the stacked translucent copies below actually blend into a
# soft halo rather than tiling a small area solid. Redrawing only this small subset --
# not the full 1.5M-point cloud -- is also what keeps the ten extra glow draws cheap
# through the headless Agg export.
hit_z = Z.ravel()[flat_idx]
hot_weight = np.clip(hit_z, 0.0, None) ** 4 + 1e-12
hot_weight /= hot_weight.sum()
n_glow = 6000
hot_sel = rng.choice(hit_x.size, size=n_glow, replace=False, p=hot_weight)
hot_x, hot_y = hit_x[hot_sel], hit_y[hot_sel]
glow_color = "#00e5ff"
base_size = 4.0
plt.scatter(hot_x, hot_y, color=glow_color, s=base_size, alpha=0.55, edgecolors="none")
n_layers = 10
for i in range(1, n_layers + 1):
    plt.scatter(
        hot_x,
        hot_y,
        s=base_size * (1.35**i),
        color=glow_color,
        alpha=0.3 / n_layers,
        edgecolors="none",
    )

plt.xscale("log")
plt.title(f"Device response across three decades ({n_glow:,} hot spots glowing)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude (mV)")
# plt.show()
plt.savefig("examples/gallery/results/25_log_scale_contourf.png")
