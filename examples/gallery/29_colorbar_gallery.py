import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm, PowerNorm, SymLogNorm

import glplot.pyplot as plt

# Four independent demonstrations of gplt.colorbar(), each a small simulation rather
# than a formula sampled on a grid: the four artist types a colorbar is useful on
# (scatter, a hand-colored line family, contour, imshow), the four bar positions
# matplotlib supports (right/left/top/bottom -- panel C takes matplotlib's own default,
# right; A/B/D each take one of the other three explicitly), and four distinct
# Normalize subclasses (linear, LogNorm, SymLogNorm, PowerNorm) so the gradient shape
# itself, not just its position, differs panel to panel.
#
# Every panel names its own axes (gplt now stores xlabel/ylabel/title per panel, not
# once per figure) and each external colorbar shrinks its own panel's rect in place --
# a real geometry change, not an overlay -- so data and bar never overlap even in this
# tight 2x2 grid. Positions are assigned so every external bar's ticks land in the gap
# between two panels rather than off the window's own edge: GLPlot's panel grid
# reserves no outer figure margin (each panel draws its chrome *inside* its own rect).
# Panel D sidesteps that a different way, with `inset=True`.
rng = np.random.default_rng(29)

fig, axs = plt.subplots(2, 2, figsize=(13, 10), wspace=0.11, hspace=0.13)

# Short titles on purpose: matplotlib centres a title over its own panel, and on a grid
# this tight two long ones meet in the middle. The physics lives in the axis names.
TITLE_SIZE = 26

# Bars are drawn far thinner than matplotlib's chunky 0.15 default, which is sized for
# one full-width figure and reads oversized against a panel that is a quarter of the
# canvas.
BAR_FRACTION = 0.035

# ---- Panel A: N-body-ish stellar field, linear scale, bar on the bottom ----
# A galactic disc sampled as three real populations rather than one radial blob: a
# two-armed spiral disc, a dense central bulge, and a diffuse halo. Colour is surface
# temperature, and it is deliberately *not* a function of radius -- a star is hot
# because it is young, and young stars sit in the spiral arms where the gas is being
# compressed. So the colorbar reads out star formation, the arms show up in colour as
# well as in density, and nothing here is confined to a circle.
n_disc, n_bulge, n_halo = 70_000, 16_000, 5_000

# Stars scatter *perpendicular* to their arm's ridge line by a fixed width in kpc, not
# by a fixed angle: an angular spread covers `r * angle` kpc, so it fans out with radius
# and washes the arms into concentric rings by the disc's edge. `arm_offset` places the
# star and sets its temperature below, which is what ties the colour to the structure.
arm = rng.integers(0, 2, n_disc)
t_disc = rng.gamma(shape=2.3, scale=1.7, size=n_disc)  # distance along the arm
theta_ridge = 0.55 * t_disc + arm * np.pi  # ~1.3 turns across the disc: arms, not rings
# Arms widen outward (a constant width draws a hairline at large radius, where the arm
# is long and sparse), and the temperature ridge is defined relative to that local
# width so the arms stay soft-edged populations rather than drawn curves.
arm_width = 0.55 + 0.13 * t_disc
arm_offset = rng.normal(0.0, 1.0, n_disc) * arm_width
perp_x, perp_y = -np.sin(theta_ridge), np.cos(theta_ridge)
disc_x = (t_disc * np.cos(theta_ridge) + arm_offset * perp_x) * 1.35  # inclined on sky
disc_y = (t_disc * np.sin(theta_ridge) + arm_offset * perp_y) * 0.78
disc_temp = (
    4_300.0
    + 6_400.0 * np.exp(-((arm_offset / arm_width) ** 2))  # young + hot on the arm ridge
    + rng.normal(0.0, 380.0, n_disc)
)

bulge_x = rng.normal(0.0, 1.15, n_bulge) * 1.35
bulge_y = rng.normal(0.0, 0.95, n_bulge) * 0.78
bulge_temp = 3_900.0 + rng.normal(0.0, 300.0, n_bulge)  # old, red, metal-rich

# Exponentially-thinning halo rather than a uniform annulus, which read as a hard ring
# sitting on top of the disc instead of a population behind it.
# Clipped, or the exponential's tail throws a handful of stars far enough out to set
# the autoscale by itself and shrink the galaxy to a smudge in the middle of the frame.
halo_r = np.clip(4.0 + rng.exponential(4.0, n_halo), 0.0, 19.0)
halo_th = rng.uniform(0.0, 2 * np.pi, n_halo)
halo_x = halo_r * np.cos(halo_th)
halo_y = halo_r * np.sin(halo_th) * 0.85
halo_temp = 11_600.0 + rng.normal(0.0, 700.0, n_halo)  # hot, metal-poor

temp_lo = float(min(disc_temp.min(), bulge_temp.min(), halo_temp.min()))
temp_hi = float(max(disc_temp.max(), bulge_temp.max(), halo_temp.max()))
# Halo first, disc over it, bulge last: painter's order, faintest population behind.
# Each gets its own marker so the three are separable even where they overlap, and one
# shared vmin/vmax so a colour means the same temperature in all three (and on the bar).
axs[0, 0].scatter(
    halo_x,
    halo_y,
    c=halo_temp,
    cmap="viridis",
    s=6.0,
    alpha=0.40,
    marker="^",
    vmin=temp_lo,
    vmax=temp_hi,
)
axs[0, 0].scatter(
    disc_x, disc_y, c=disc_temp, cmap="viridis", s=2.4, alpha=0.70, vmin=temp_lo, vmax=temp_hi
)
axs[0, 0].scatter(
    bulge_x,
    bulge_y,
    c=bulge_temp,
    cmap="viridis",
    s=3.2,
    alpha=0.60,
    marker="s",
    vmin=temp_lo,
    vmax=temp_hi,
)
axs[0, 0].set_xlabel("Galactic X (kpc)")
axs[0, 0].set_ylabel("Galactic Y (kpc)")
axs[0, 0].set_title("Stellar field | linear, bottom", fontsize=TITLE_SIZE)
plt.colorbar(
    ax=axs[0, 0], location="bottom", fraction=BAR_FRACTION, pad=0.17, shrink=0.85
).set_label("Surface temperature (K)")

# ---- Panel B: radioactive decay bench, log scale, bar on the left ----------
# Thirteen isotopes whose half-lives span two decades, each simulated as a real
# counting experiment: the smooth curve is the theory, the points are Poisson-sampled
# detector counts at a shared, sparse readout schedule (so the scatter is heteroscedastic
# -- noisier where the count rate is low -- exactly as a real decay bench is). A LogNorm
# is what keeps the short-lived isotopes from bunching into one indistinguishable colour.
t = np.linspace(0.0, 10.0, 400)
t_meas = np.sort(rng.uniform(0.2, 10.0, 60))
half_lives = np.logspace(-0.5, 1.4, 13)
decay_sm = ScalarMappable(
    norm=LogNorm(vmin=float(half_lives.min()), vmax=float(half_lives.max())), cmap="plasma"
)
N0 = 4_000  # initial nuclei per isotope -> Poisson counting noise scales as sqrt(N)
for hl in half_lives:
    color = decay_sm.to_rgba(hl)
    expected = N0 * np.exp(-t_meas * np.log(2.0) / hl)
    counts = rng.poisson(np.clip(expected, 0.0, None)) / N0
    axs[0, 1].scatter(t_meas, counts, color=color, s=11.0, alpha=0.55, marker="o")
    axs[0, 1].plot(t, np.exp(-t * np.log(2.0) / hl), color=color, linewidth=2.2)
axs[0, 1].set_xlabel("Elapsed time (s)")
axs[0, 1].set_ylabel("Surviving fraction")
axs[0, 1].set_title("Decay bench | log, left", fontsize=TITLE_SIZE)
# A generous `pad` here is not cosmetic: it is the gap this panel's own y-name has to
# fit into, since the bar took the side the name is drawn on.
plt.colorbar(
    decay_sm, ax=axs[0, 1], location="left", fraction=BAR_FRACTION, pad=0.20, shrink=0.85
).set_label("Half-life (s)")

# ---- Panel C: electrostatic dipole, symlog scale, default (right) bar ------
# A dipole's potential is strongly positive at one charge, strongly negative at the
# other, and crosses zero on the perpendicular bisector -- the shape a plain LogNorm
# cannot represent (no negative domain) and SymLogNorm exists for. Levels are explicit
# and geometrically spaced (ratio ~1.5 per step) because the field is a 1/r singularity:
# equal-value spacing crams every line into the last pixel around each charge. A
# scattered probe grid -- as if a sensor had sampled the same field with real
# measurement error -- sits under the lines on the exact same colour scale, and
# gplt.clabel() seats each level's own value inline on its line (real matplotlib
# geometry, breaking the line to fit the number; headless export only, see its docstring).
gx = np.linspace(-3.0, 3.0, 400)
GX, GY = np.meshgrid(gx, gx)


def _dipole_potential(x, y):
    v = np.zeros_like(x)
    for cx, cy, q in ((-1.2, 0.0, 1.0), (1.2, 0.0, -1.0)):
        r = np.clip(np.sqrt((x - cx) ** 2 + (y - cy) ** 2), 0.12, None)
        v = v + q / r
    return v


potential = _dipole_potential(GX, GY)
pot_lo, pot_hi = float(potential.min()), float(potential.max())
dipole_norm = SymLogNorm(linthresh=0.5, base=10, vmin=pot_lo, vmax=pot_hi)

n_probes = 2_600
probe_x = rng.uniform(-3.0, 3.0, n_probes)
probe_y = rng.uniform(-3.0, 3.0, n_probes)
probe_v = _dipole_potential(probe_x, probe_y) + rng.normal(0.0, 0.06, n_probes)
axs[1, 0].scatter(
    probe_x, probe_y, c=probe_v, cmap="RdBu_r", norm=dipole_norm, s=14.0, alpha=0.65, marker="x"
)

level_magnitudes = np.array([0.15, 0.3, 0.5, 0.8, 1.2, 1.8, 2.6, 3.8, 5.5, 8.0])
cs = axs[1, 0].contour(
    GX,
    GY,
    potential,
    levels=np.concatenate([-level_magnitudes[::-1], level_magnitudes]),
    cmap="RdBu_r",
    norm=dipole_norm,
    linewidths=1.8,
)
axs[1, 0].clabel(cs, inline=True, fontsize=11, fmt=lambda v: f"{v:.2g}")
axs[1, 0].set_xlabel("x (m)")
axs[1, 0].set_ylabel("y (m)")
axs[1, 0].set_title("Dipole field | symlog, right", fontsize=TITLE_SIZE)
plt.colorbar(cs, ax=axs[1, 0], fraction=BAR_FRACTION, pad=0.03, shrink=0.9).set_label(
    "Potential (a.u.)"
)

# ---- Panel D: speckled diffraction frame, power norm, inset bar on top -----
# A CCD frame from a diffraction experiment: an Airy-like pattern, multiplied by
# speckle (a fully-developed exponential speckle field, as a coherent source really
# produces), plus read noise and a scatter of cosmic-ray hits. Intensity is dominated
# by one narrow, extremely bright core, so a linear colour ramp spends its whole range
# on a handful of pixels and renders every fainter ring as one flat colour --
# PowerNorm(gamma<1) is the standard fix, compressing the peak until the outer rings
# become visible without the core blowing out.
#
# `inset=True`: this panel's own title sits above it, in the same strip an external
# "top" bar would use, so the bar is drawn inside the panel instead -- clipped to the
# panel's own bounds, with its numbers haloed rather than sitting on an opaque box that
# would hide the very image the bar describes. `shrink` keeps it from spanning the
# panel edge to edge.
gx2 = np.linspace(-6.0, 6.0, 320)
GX2, GY2 = np.meshgrid(gx2, gx2)
r2 = np.sqrt(GX2**2 + GY2**2)
airy = np.sinc(r2) ** 2
speckle = rng.exponential(1.0, airy.shape)  # fully-developed speckle: I ~ Exp(mean)
intensity = airy * (0.45 + 0.55 * speckle) + 0.012 * rng.random(airy.shape)
for _ in range(28):  # cosmic-ray hits: a few saturated pixels, as any real CCD frame has
    cy, cx = rng.integers(0, airy.shape[0]), rng.integers(0, airy.shape[1])
    intensity[cy, cx] = intensity.max() * rng.uniform(0.6, 1.0)

im = axs[1, 1].imshow(
    intensity,
    extent=[-6.0, 6.0, -6.0, 6.0],
    cmap="inferno",
    norm=PowerNorm(gamma=0.4, vmin=float(intensity.min()), vmax=float(intensity.max())),
)
axs[1, 1].set_xlabel("Detector x (mm)")
axs[1, 1].set_ylabel("Detector y (mm)")
axs[1, 1].set_title("Speckle CCD | power, inset", fontsize=TITLE_SIZE)
# Explicit ticks: a PowerNorm bar auto-locates evenly in *data* space, which the gamma
# then bunches together toward the bright end -- unreadable on a bar this short.
plt.colorbar(
    im,
    ax=axs[1, 1],
    location="top",
    inset=True,
    fraction=0.035,
    shrink=0.62,
    ticks=[0.0, 0.5, 1.5, 3.0],
).set_label("Intensity (a.u.)")

# plt.show()
plt.savefig("examples/gallery/results/29_colorbar_gallery.png")
