"""High-temperature Langevin dynamics on a rotating 3D potential-energy surface.

This example deliberately separates data generation from plotting:

1. :func:`simulate_langevin` knows only about NumPy and returns a ``DynamicsData`` object.
2. :func:`build_animation` receives that finished dataset and only creates visual layers.
3. :func:`main` is the small orchestration layer that connects those two independent parts.

The particles evolve on a tilted double-well potential in reduced units, where particle
mass and Boltzmann's constant are both one. A Langevin thermostat at ``T = 0.60`` adds
strong random thermal kicks while friction removes energy. The resulting trajectories
cross the central barrier in both directions instead of simply relaxing into one minimum.
"""

import argparse
from dataclasses import dataclass

import numpy as np

import glplot.animation as animation
import glplot.pyplot as plt

# -----------------------------------------------------------------------------
# Simulation controls. These values belong to data generation, not rendering.
# -----------------------------------------------------------------------------
DT = 0.015
STEPS = 400
N_PARTICLES = 24
FRICTION = 0.80
TEMPERATURE = 0.60  # Reduced temperature: k_B = 1 and particle mass = 1.
RANDOM_SEED = 12

# -----------------------------------------------------------------------------
# Rendering controls. Changing these does not require changing the dynamics.
# -----------------------------------------------------------------------------
FRAMES = 48
FPS = 20
TRAIL_SAMPLES = 14
TRAIL_STRIDE = 2
PARTICLE_DIAMETER_PX = 6.0
TRAIL_DIAMETER_PX = 3.0
PARTICLE_SURFACE_OFFSET = 0.10
# Below 1.0 the PES is see-through, so particles and trails behind a ridge stay visible
# (dimmed by the surface) instead of being hidden by it.
SURFACE_ALPHA = 0.55


@dataclass(frozen=True)
class DynamicsData:
    """Arrays produced by the simulator and consumed by the plotting code.

    Shapes:
        times: ``(steps,)``
        positions: ``(steps, particles, 2)``
        velocities: ``(steps, particles, 2)``
        potential_energy: ``(steps, particles)``
        total_energy: ``(steps, particles)``
        surface_normals: ``(steps, particles, 3)``
    """

    times: np.ndarray
    positions: np.ndarray
    velocities: np.ndarray
    potential_energy: np.ndarray
    total_energy: np.ndarray
    surface_normals: np.ndarray
    temperature: float


# =============================================================================
# PART 1: PHYSICAL MODEL AND DATA GENERATION
# Everything in this section is independent of GLPlot.
# =============================================================================


def potential(x, y):
    """Evaluate the tilted double-well potential V(x, y)."""
    return 0.25 * (x * x - 1.0) ** 2 + 0.5 * y * y + 0.15 * x * y


def potential_gradient(x, y):
    """Return the analytic gradient dV/dx, dV/dy used to compute force = -grad(V)."""
    return x * (x * x - 1.0) + 0.15 * y, y + 0.15 * x


def simulate_langevin(
    *,
    temperature=TEMPERATURE,
    steps=STEPS,
    n_particles=N_PARTICLES,
    dt=DT,
    friction=FRICTION,
    seed=RANDOM_SEED,
):
    """Generate an ensemble with an Euler-Maruyama Langevin integrator.

    In reduced units (mass = k_B = 1), each velocity obeys

        dv = (-grad(V) - gamma*v) dt + sqrt(2*gamma*T) dW.

    The random increment ``dW`` has variance ``dt``. Therefore the discrete noise
    amplitude is ``sqrt(2*gamma*T*dt)``. This fluctuation-dissipation pairing is what
    makes ``temperature`` a physical thermostat parameter rather than arbitrary jitter.
    """
    rng = np.random.default_rng(seed)

    # Step 1: place the ensemble in the left well with a small positional spread.
    x = -1.05 + rng.normal(0.0, 0.07, n_particles)
    y = np.linspace(-0.55, 0.55, n_particles) + rng.normal(0.0, 0.04, n_particles)

    # Step 2: sample thermal velocities. A small positive x drift helps the first barrier
    # crossings happen early enough to be visible, while the stochastic bath dominates
    # the later motion and can drive particles in either direction.
    thermal_speed = np.sqrt(temperature)
    vx = rng.normal(0.35, thermal_speed, n_particles)
    vy = rng.normal(0.00, thermal_speed, n_particles)

    positions = np.empty((steps, n_particles, 2), dtype=np.float64)
    velocities = np.empty_like(positions)
    noise_scale = np.sqrt(2.0 * friction * temperature * dt)

    for step in range(steps):
        # Step 3a: conservative acceleration points downhill on the PES.
        grad_x, grad_y = potential_gradient(x, y)

        # Step 3b: Euler-Maruyama adds deterministic force/friction and independent
        # Gaussian thermal kicks to every particle and Cartesian component.
        vx += dt * (-grad_x - friction * vx) + noise_scale * rng.normal(size=n_particles)
        vy += dt * (-grad_y - friction * vy) + noise_scale * rng.normal(size=n_particles)

        # Step 3c: semi-implicit position update uses the newly updated velocities.
        x += dt * vx
        y += dt * vy

        # Step 3d: store a complete snapshot. No plotting object is created here.
        positions[step, :, 0] = x
        positions[step, :, 1] = y
        velocities[step, :, 0] = vx
        velocities[step, :, 1] = vy

    # Step 4: derive observables once after integration. The plotting function receives
    # ready-to-use arrays and never needs to know how the integrator works.
    potential_energy = potential(positions[:, :, 0], positions[:, :, 1])
    grad_x, grad_y = potential_gradient(positions[:, :, 0], positions[:, :, 1])
    surface_normals = np.stack((-grad_x, -grad_y, np.ones_like(grad_x)), axis=2)
    surface_normals /= np.linalg.norm(surface_normals, axis=2, keepdims=True)
    kinetic_energy = 0.5 * np.sum(velocities * velocities, axis=2)
    total_energy = potential_energy + kinetic_energy
    times = dt * np.arange(steps)

    return DynamicsData(
        times=times,
        positions=positions,
        velocities=velocities,
        potential_energy=potential_energy,
        total_energy=total_energy,
        surface_normals=surface_normals,
        temperature=float(temperature),
    )


# =============================================================================
# PART 2: VISUALIZATION
# This section consumes DynamicsData and does not run or modify the simulation.
# =============================================================================


def _mark_geometry_changed(layer):
    """Tell the live GPU renderer that an in-place array update must be uploaded."""
    layer.dirty.data_dirty = True
    layer.dirty.gpu_dirty = True
    layer.dirty.bounds_dirty = True


def _reveal_2d(layer, times, values, stop):
    """Reveal a time series without allocating a new line on every frame."""
    layer.pts[:stop, 0] = times[:stop]
    layer.pts[:stop, 1] = values[:stop]

    # Collapse the not-yet-visible tail onto the current endpoint. This avoids NaNs and
    # makes all unused segments zero-length, so only the history appears on screen.
    layer.pts[stop:, 0] = times[stop - 1]
    layer.pts[stop:, 1] = values[stop - 1]
    _mark_geometry_changed(layer)


def build_animation(data: DynamicsData):
    """Create a GLPlot animation from an already generated ``DynamicsData`` dataset."""
    times = data.times
    x = data.positions[:, :, 0]
    y = data.positions[:, :, 1]

    # Step 1: start each marker exactly on the PES, then move it a short distance along
    # the local surface normal. Unlike a simple vertical z offset, this gives the same
    # physical clearance on flat and steep parts of the surface. The regular depth buffer
    # can therefore keep nearby particles visible while still hiding particles behind a
    # ridge as the camera rotates.
    surface_points = np.stack((x, y, data.potential_energy), axis=2)
    particle_points = surface_points + PARTICLE_SURFACE_OFFSET * data.surface_normals

    # Step 2: sample the analytic PES on a modest grid. The grid is static, so it is built
    # once and remains untouched throughout the animation.
    axis_x = np.linspace(-2.2, 2.2, 52)
    axis_y = np.linspace(-2.2, 2.2, 52)
    grid_x, grid_y = np.meshgrid(axis_x, axis_y)
    grid_v = potential(grid_x, grid_y)

    # Step 3: create independent 3D and time-series panels.
    fig, axes = plt.subplots(
        1,
        2,
        title="Gallery - High-temperature PES dynamics",
        width=1500,
        height=760,
        wspace=0.06,
        width_ratios=[1.15, 1.0],
    )
    plt.plot_style("dark")
    ax_pes, ax_time = axes
    ax_pes.set_projection("3d")
    # Orthographic projection keeps a 6 px particle at 6 px throughout the camera orbit.
    # In perspective mode the live shader intentionally enlarges near particles, which is
    # useful for depth perception but makes marker size disagree with a fixed-size GIF.
    ax_pes.set_proj_type("ortho")

    # Step 4: create one scatter layer for all particles and one for all trail samples.
    # Later frames mutate these fixed-size vertex arrays instead of creating new layers.
    #
    # Declaration order matters once the PES is translucent. GLPlot draws opaque 3D layers
    # first and translucent ones afterwards, in the order they were created, and each draw
    # is depth-tested against what is already there. The opaque particles therefore always
    # go down before the surface. The translucent trails must be created *before* the PES
    # too: created after it, every trail sample behind the surface would fail the depth
    # test against the PES and never be drawn at all.
    initial_particles = particle_points[0]
    particles = ax_pes.scatter(
        initial_particles[:, 0],
        initial_particles[:, 1],
        initial_particles[:, 2],
        color=(1.0, 0.28, 0.08, 1.0),
        s=PARTICLE_DIAMETER_PX,
        alpha=1.0,
        label=f"{data.positions.shape[1]} particles",
        depth_write=True,
    )

    initial_trails = np.repeat(initial_particles[None, :, :], TRAIL_SAMPLES, axis=0)
    trails = ax_pes.scatter(
        initial_trails[:, :, 0].ravel(),
        initial_trails[:, :, 1].ravel(),
        initial_trails[:, :, 2].ravel(),
        color=(1.0, 0.78, 0.20, 1.0),
        s=TRAIL_DIAMETER_PX,
        alpha=0.36,
        # Translucent points skip depth writes by default. Writing depth here lets the PES,
        # drawn next, skip the trail samples in front of it so they stay crisp, while it
        # still blends over the ones behind it so they show through.
        depth_write=True,
    )

    # Step 5: the PES goes last so it blends over everything behind it.
    ax_pes.plot_surface(
        grid_x,
        grid_y,
        grid_v,
        cmap="viridis",
        alpha=SURFACE_ALPHA,
        # Keep the surface's own depth writes: a translucent mesh without them would blend
        # its far side over its near side in whatever order the triangles happen to be in.
        depth_write=True,
        elev=32,
        azim=-60,
        rstride=1,
        cstride=1,
    )
    ax_pes.set_xlim(-2.2, 2.2)
    ax_pes.set_ylim(-2.2, 2.2)
    ax_pes.set_zlim(-0.15, 6.8)
    ax_pes.set_axis3d_off()
    ax_pes.set_title(f"Langevin dynamics — T = {data.temperature:.2f}", fontsize=16)

    # Step 6: reduce the particle ensemble to three compact observables for the right
    # panel. At finite temperature, mean energy fluctuates instead of decaying monotonically.
    mean_x = x.mean(axis=1)
    mean_y = y.mean(axis=1)
    mean_energy = data.total_energy.mean(axis=1)
    collapsed_time = np.full(len(times), times[0])
    x_line = ax_time.plot(collapsed_time, np.full(len(times), mean_x[0]), label="mean x", lw=2.2)[0]
    y_line = ax_time.plot(collapsed_time, np.full(len(times), mean_y[0]), label="mean y", lw=2.2)[0]
    energy_line = ax_time.plot(
        collapsed_time,
        np.full(len(times), mean_energy[0]),
        label="mean energy",
        lw=2.2,
    )[0]

    all_series = np.concatenate((mean_x, mean_y, mean_energy))
    series_pad = 0.08 * np.ptp(all_series)
    ax_time.set_xlim(times[0], times[-1])
    ax_time.set_ylim(float(all_series.min() - series_pad), float(all_series.max() + series_pad))
    ax_time.set_xlabel("time", fontsize=12)
    ax_time.set_ylabel("ensemble mean", fontsize=12)
    ax_time.tick_params(labelsize=11)
    ax_time.set_title("Thermal evolution", fontsize=16)
    ax_time.legend()

    # Step 7: map a compact set of rendered frames onto every stored integration step.
    frame_stops = np.linspace(2, len(times), FRAMES, dtype=int)
    n_particles = data.positions.shape[1]

    def update(frame):
        """Update existing layers for one animation frame; never rerun the dynamics."""
        stop = int(frame_stops[frame])
        current = stop - 1

        # 7a: move the particles to the current precomputed coordinates.
        particles.vertices[:] = particle_points[current]
        _mark_geometry_changed(particles)

        # 7b: gather earlier snapshots into short dotted trails. Old indices are clipped
        # to zero at the beginning, so the trail grows naturally during the first frames.
        trail_indices = np.maximum(
            current - TRAIL_STRIDE * np.arange(TRAIL_SAMPLES),
            0,
        )
        trails.vertices[:] = particle_points[trail_indices].reshape(-1, 3)
        _mark_geometry_changed(trails)

        # 7c: orbit the camera once while gently changing elevation. This modifies only
        # the view; particle positions and the PES remain in physical coordinates.
        phase = frame / max(1, FRAMES - 1)
        azim = -60.0 + 360.0 * phase
        elev = 30.0 + 8.0 * np.sin(2.0 * np.pi * phase)
        ax_pes.view_init(elev=elev, azim=azim)
        in_right_well = int(np.count_nonzero(x[current] > 0.0))
        ax_pes.set_title(
            f"T = {data.temperature:.2f} — {in_right_well}/{n_particles} in right well",
            fontsize=16,
        )

        # 7d: reveal the matching portion of each precomputed time series.
        _reveal_2d(x_line, times, mean_x, stop)
        _reveal_2d(y_line, times, mean_y, stop)
        _reveal_2d(energy_line, times, mean_energy, stop)
        ax_time.set_title(f"Thermal evolution — t = {times[current]:.2f}", fontsize=16)
        return particles, trails, x_line, y_line, energy_line

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=FRAMES,
        interval=1000.0 / FPS,
        repeat=True,
    )
    return fig, ani


# =============================================================================
# PART 3: ORCHESTRATION
# Generate data first, then pass it into the completely separate plotting stage.
# =============================================================================


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--show",
        action="store_true",
        help="open the live OpenGL animation instead of writing the gallery GIF",
    )
    parser.add_argument(
        "--renderer",
        choices=("auto", "opengl", "matplotlib"),
        default="auto",
        help=(
            "renderer used for the saved GIF: auto tries the GUI's OpenGL renderer and "
            "falls back to matplotlib (default)"
        ),
    )
    args = parser.parse_args(argv)
    data = simulate_langevin()
    _fig, ani = build_animation(data)

    if args.show:
        print(
            f"Live particles: {PARTICLE_DIAMETER_PX:.0f} px diameter, "
            f"V(x,y) + {PARTICLE_SURFACE_OFFSET:.2f}, orthographic camera"
        )
        plt.show()
    else:
        ani.save(
            "examples/gallery/animations/results/16_pes_particle_dynamics.gif",
            fps=FPS,
            savefig_kwargs={"renderer": args.renderer},
        )


if __name__ == "__main__":
    main()
