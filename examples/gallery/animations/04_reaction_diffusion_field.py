"""Gray-Scott reaction-diffusion -- a Turing pattern growing in a shallow reactor.

Two chemical species diffuse across a thin dish: an activator ``U`` (fed in from the
edges at a constant rate) and an autocatalytic product ``V`` (which converts U into more
of itself, then decays). The classic Gray-Scott system is

    dU/dt = Du * Laplacian(U) - U*V^2 + F*(1 - U)
    dV/dt = Dv * Laplacian(V) + U*V^2 - (F + k)*V

Depending only on the feed rate ``F`` and kill rate ``k``, this pair of equations is
famous for spontaneously organising a *uniform* starting mixture into spots, stripes,
maze-like walls, or (the case animated here, F=0.0545, k=0.062) spots that grow and then
split in two -- the "mitosis" regime, a favourite of Turing-pattern demos because the
splitting reads instantly as *alive* rather than merely decorative. This is the same
mechanism Alan Turing proposed in 1952 for how a chemically uniform embryo grows spots and
stripes -- animal coat patterns, coral growth, and (down to slightly different constants)
this exact simulation are all the same two-reactant instability.

Every frame advances the finite-difference simulation by many small Euler sub-steps (the
diffusion term needs a small time step to stay numerically stable) and only then draws the
current V-field as a dense heatmap -- so the per-frame COST is one small array update times
a handful of sub-steps, not a re-render of "millions of points": a 170x170 grid is a
resolution choice, not a point count, and it stays comfortably fast every frame while still
reading as a rich, continuous concentration field.
"""

import numpy as np

import glplot.animation as animation
import glplot.pyplot as plt

# --- Reactor grid -----------------------------------------------------------------
N = 170  # grid cells per side; the "resolution" lever, not a per-point cost
SIZE_MM = 2.5  # half-width of the reactor dish, mm -- purely for axis units
DU, DV = 0.16, 0.08  # diffusion rates of activator U and product V
FEED, KILL = 0.0545, 0.0620  # feed/kill pair tuned for the "mitosis" spot-splitting regime
DT = 1.0  # Euler step (the classic discrete Gray-Scott stencil is stable at dt=1)
SUBSTEPS_PER_FRAME = 110  # sim steps advanced between each drawn frame
FRAMES = 72


def laplacian(field: np.ndarray) -> np.ndarray:
    """5-point discrete Laplacian with periodic (wrap-around) boundaries.

    ``np.roll`` shifts the whole grid instead of looping over cells -- the same
    data-parallel trick as the Mandelbrot example, just for a stencil instead of an
    escape-time recursion.
    """
    return (
        np.roll(field, 1, axis=0)
        + np.roll(field, -1, axis=0)
        + np.roll(field, 1, axis=1)
        + np.roll(field, -1, axis=1)
        - 4.0 * field
    )


def seed_reactor() -> tuple:
    """Start near-uniform (U=1, V=0) with a few small perturbed patches to break symmetry.

    A perfectly uniform field is an unstable equilibrium that never moves; real reactors
    (and this simulation) need a nudge, so a handful of square droplets of V are dosed in
    off-centre, plus a whisper of noise so the patches do not all evolve in lockstep.
    """
    rng = np.random.default_rng(7)
    U = np.ones((N, N))
    V = np.zeros((N, N))
    for cx, cy in [(0.5, 0.5), (0.32, 0.62), (0.68, 0.38), (0.4, 0.28), (0.62, 0.7)]:
        r = 5
        ix, iy = int(cx * N), int(cy * N)
        U[ix - r : ix + r, iy - r : iy + r] = 0.50
        V[ix - r : ix + r, iy - r : iy + r] = 0.25
    U += 0.02 * rng.standard_normal((N, N))
    V += 0.02 * rng.standard_normal((N, N))
    return np.clip(U, 0.0, 1.0), np.clip(V, 0.0, 1.0)


U, V = seed_reactor()

fig = plt.figure("Gallery - Reaction-Diffusion Field", figsize=(7.6, 6.4))
plt.plot_style("dark")

extent = (-SIZE_MM, SIZE_MM, -SIZE_MM, SIZE_MM)


def step_reactor() -> None:
    """Advance U, V by one Euler step of the Gray-Scott PDE, in place."""
    global U, V
    Lu, Lv = laplacian(U), laplacian(V)
    uvv = U * V * V
    U = U + DT * (DU * Lu - uvv + FEED * (1.0 - U))
    V = V + DT * (DV * Lv + uvv - (FEED + KILL) * V)
    np.clip(U, 0.0, 1.0, out=U)
    np.clip(V, 0.0, 1.0, out=V)


def update(frame: int):
    for _ in range(SUBSTEPS_PER_FRAME):
        step_reactor()

    sim_time = (frame + 1) * SUBSTEPS_PER_FRAME * DT
    plt.cla()
    plt.imshow(
        V,
        extent=extent,
        origin="lower",
        cmap="inferno",
        vmin=0.0,
        vmax=0.42,
        aspect="equal",
    )
    plt.title(f"Gray-Scott reaction-diffusion -- Turing pattern (t={sim_time:.0f})")
    plt.xlabel("x (mm)")
    plt.ylabel("y (mm)")
    return []


ani = animation.FuncAnimation(fig, update, frames=FRAMES, interval=45)
# plt.show()
ani.save("examples/gallery/animations/results/04_reaction_diffusion_field.gif", fps=20)
