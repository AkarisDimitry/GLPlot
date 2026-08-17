"""Two-wavetrain interference, drawn as a family of characteristic rays.

The classic way to *see* the interference of two wave trains without ever plotting a
sinusoid is the ray (characteristic) picture used in dispersion/optics teaching: each
plane-wave component of a wave train traces a straight "phase ray" x = v*t + x0 in a
position-vs-time diagram, where v is its phase velocity and x0 its starting offset. Bundle
together many rays whose velocities cluster tightly around a phase velocity, and the
*density* of where those rays crowd together approximates where the corresponding wave
packet's energy actually sits -- crowded rays mean constructive overlap, thin rays mean
destructive interference. Two such wave trains, superposed, cross each other in an "X" and
the shape of that crossing carries exactly the physics of the two-wave beat:

  * the opening angle of the "X" is set by the *velocity difference* Delta v between the
    two wave trains -- a wide separation makes a wide-open X (a big walk-off), a narrow one
    makes a near-parallel bundle (nearly co-propagating waves, a slow beat).
  * the crossover point's position tracks the *relative phase* between the two trains --
    as the phase slides, the X's waist slides sideways in position, which is the moving
    interference-fringe pattern you would actually see or hear as a wave train's beat note.

This script renders that ray family with GLPlot's density-mode line-family renderer
(`plot_lines(a, b, x_range=..., cmap=...)`, ~24,000 rays per frame -- cheap because the
cost is a fixed-resolution 2D histogram, not per-line drawing) and animates both knobs at
once: the velocity spread breathes open and shut while the relative phase slides back and
forth, so the interference lattice visibly evolves rather than sitting still.
"""

import numpy as np

import glplot.animation as animation
import glplot.pyplot as plt

FRAMES = 72
T = 5.0  # duration of the ray diagram's time axis, seconds
N_PER_TRAIN = 12_000  # rays per wave train, 24,000 total per frame

V0 = 1.0  # baseline phase velocity shared by both wave trains, m/s
SLOPE_JITTER = 0.04  # per-ray velocity scatter within a train, m/s (keeps each "X" arm a soft band, not a razor line)
PHASE_JITTER = 0.35  # per-ray starting-position scatter within a train, m -- tight enough that a
# real phase separation reads as two distinct bands rather than one smeared blur

rng = np.random.default_rng(7)


def wave_params(frame: int):
    """Velocity separation and relative phase between the two wave trains, this frame.

    Kept modest on purpose (max ray position stays under ~9 m across the whole run): the
    headless density preview auto-fits the y-axis to the data every frame, and a range that
    pushes tick labels into double digits (e.g. "10.0") crowds the rotated y-axis label
    against the figure edge. Staying single-digit keeps "Position (m)" comfortably clear of
    the frame regardless of which frame lands in the GIF.
    """
    t = frame / FRAMES
    # Carrier velocity itself wobbles slowly, so the whole X-diagram drifts as it breathes.
    v_center = V0 + 0.15 * np.sin(2 * np.pi * t * 0.5)
    # Velocity spread breathes through one full open/close cycle over the run: 0.20..0.70 m/s
    # -- wide enough that the "X" visibly opens into two separate arms and closes back down.
    delta_v = 0.20 + 0.50 * (0.5 - 0.5 * np.cos(2 * np.pi * t * 1.0))
    # Relative phase slides back and forth twice over the run: at phase=0 the two wave
    # trains land exactly on top of each other (constructive, one bright band); at the
    # extremes they pull apart into two clearly separate strands (destructive walk-off) --
    # the classic beat between two nearby frequencies, animated as the beat itself.
    delta_phase = 1.4 * np.sin(2 * np.pi * t * 2.0)
    return v_center, delta_v, delta_phase


fig = plt.figure("Gallery - Traveling Wave Family", figsize=(8.6, 6.2))
plt.plot_style("neon")
# Pin the view: plot_lines()'s density image otherwise auto-fits its y-range to each
# frame's own ray spread (percentile of that frame's (a, b) values), so the diagram
# visibly rescaled -- rays looked like they sped up or slowed down -- every frame purely
# from the axis rescaling, not from the actual wave physics. 9.5 m comfortably covers
# the "~9 m" max ray position noted above across the whole velocity/phase sweep.
plt.xlim(0.0, T)
plt.ylim(-9.5, 9.5)


def update(frame: int):
    v_center, delta_v, delta_phase = wave_params(frame)

    a1 = rng.normal(v_center - delta_v / 2.0, SLOPE_JITTER, N_PER_TRAIN)
    a2 = rng.normal(v_center + delta_v / 2.0, SLOPE_JITTER, N_PER_TRAIN)
    b1 = rng.normal(-delta_phase / 2.0, PHASE_JITTER, N_PER_TRAIN)
    b2 = rng.normal(delta_phase / 2.0, PHASE_JITTER, N_PER_TRAIN)

    a = np.concatenate([a1, a2])
    b = np.concatenate([b1, b2])

    # No plt.cla() here: this figure only ever holds the one plot_lines() layer, which
    # set_lines_ab() already refreshes in place every call, and title/xlabel/ylabel are
    # plain overwrites -- so nothing needs clearing frame to frame. That turns out to
    # matter more than it looks: clf()/cla() replaces the whole scene (`self.scene =
    # SceneData()`) but does not forget the engine's cached "reuse this layer" handle for
    # the line family, so a *second* plot_lines() call after a cla() updates a layer
    # object that is no longer anywhere in the (new, empty) scene -- the next frame then
    # finds no layers at all. Skipping the never-needed cla() sidesteps that entirely.
    plt.plot_lines(a, b, x_range=(0.0, T), cmap="plasma")
    plt.title(f"Two-wavetrain interference (Δv={delta_v:.2f} m/s, Δφ={delta_phase:+.2f} m)")
    plt.xlabel("Time (s)")
    plt.ylabel("Position (m)")
    return []


ani = animation.FuncAnimation(fig, update, frames=FRAMES, interval=42)
ani.save("examples/gallery/animations/results/02_traveling_wave_family.gif", fps=20)
# plt.show()
