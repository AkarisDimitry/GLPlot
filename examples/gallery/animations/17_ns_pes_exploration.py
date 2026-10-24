"""Nested sampling on the multi-well PES from PES.py, with 15 live points.

Adapted from scripts/Nested_sampling_2D.py's NestedSampler2D, generalised to
evaluate the rugged multi-well landscape (Gaussian wells in a shallow
confining bowl) defined in presentation/PES.py instead of the double well.
"""

import argparse
import math
import os

import contourpy
import numpy as np
import PES
from numba import njit
from tqdm import tqdm

import glplot.animation as animation
import glplot.pyplot as gplt

#: This script's own directory (presentation/), regardless of the caller's cwd.
_HERE = os.path.dirname(os.path.abspath(__file__))


def _in_presentation(out_dir):
    """Resolve a relative movie out_dir against presentation/, not the caller's cwd."""
    return out_dir if os.path.isabs(out_dir) else os.path.join(_HERE, out_dir)


# ---------------------------------------------------------------------------
# PES.pes(x, y) re-normalises by the min of whatever array is passed to it,
# which is right for plotting a full grid but meaningless for one walker at a
# time. So the raw (unshifted) well sum is rebuilt here from PES's own well
# parameters, and the additive shift is fixed once from the same reference
# grid PES.main() plots, then reused for every energy evaluation below --
# this stays in sync with PES.py without duplicating its formula logic
# beyond the well sum itself.
# ---------------------------------------------------------------------------
def _raw_v(x, y):
    v = PES._CONFINE * (x**2 + y**2)
    for depth, x0, y0, width in PES._WELLS:
        v -= depth * np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2.0 * width**2))
    return v


_WELLS_ARR = np.array(PES._WELLS, dtype=np.float64)  # columns: depth, x0, y0, width
_CONFINE = float(PES._CONFINE)

_ref_X, _ref_Y = np.meshgrid(np.linspace(-3.2, 3.2, 220), np.linspace(-3.2, 3.2, 220))
_V_SHIFT = float(_raw_v(_ref_X, _ref_Y).min())


def pes(P):
    """Vectorised PES, evaluated on (..., 2) points; matches PES.pes's field exactly."""
    x = P[..., 0]
    y = P[..., 1]
    return _raw_v(x, y) - _V_SHIFT


def pes_normals(x, y):
    """Unit upward normals of the PES surface z = V(x, y), from the analytic gradient.

    d/dx of the confining bowl is 2*c*x; d/dx of a Gaussian well
    -depth * exp(-r^2 / 2w^2) is +depth * exp(-r^2 / 2w^2) * (x - x0) / w^2.
    """
    gx = 2.0 * _CONFINE * x
    gy = 2.0 * _CONFINE * y
    for depth, x0, y0, width in PES._WELLS:
        g = depth * np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2.0 * width**2)) / width**2
        gx = gx + g * (x - x0)
        gy = gy + g * (y - y0)
    n = np.stack((-gx, -gy, np.ones_like(gx)), axis=-1)
    return n / np.linalg.norm(n, axis=-1, keepdims=True)


@njit(cache=True, inline="always")
def _pes_scalar(x, y, wells, confine, v_shift):
    v = confine * (x * x + y * y)
    for i in range(wells.shape[0]):
        depth = wells[i, 0]
        x0 = wells[i, 1]
        y0 = wells[i, 2]
        width = wells[i, 3]
        dx = x - x0
        dy = y - y0
        v -= depth * math.exp(-(dx * dx + dy * dy) / (2.0 * width * width))
    return v - v_shift


@njit(cache=True)
def _seed_rng(seed):
    np.random.seed(seed)


def _resolve_seed(seed):
    if seed is None:
        return int(np.random.default_rng().integers(0, 2**31 - 1))
    return seed


@njit(cache=True, nogil=True)
def _explore(x0, y0, sigma, e_worst, lo_x, hi_x, lo_y, hi_y, mc_steps, wells, confine, v_shift):
    """Clone a survivor and diffuse it under the constraint {V < e_worst}."""
    xc, yc = x0, y0
    ec = _pes_scalar(xc, yc, wells, confine, v_shift)
    n_acc = 0
    for _ in range(mc_steps):
        xt = xc + np.random.normal(0.0, sigma)
        yt = yc + np.random.normal(0.0, sigma)
        if lo_x <= xt <= hi_x and lo_y <= yt <= hi_y:
            et = _pes_scalar(xt, yt, wells, confine, v_shift)
            if et < e_worst:
                xc, yc, ec = xt, yt, et
                n_acc += 1
    return xc, yc, ec, n_acc


@njit(cache=True, nogil=True)
def _advance_all(
    pts, e_pts, sigma, e_worst, lo_x, hi_x, lo_y, hi_y, mc_steps, wells, confine, v_shift
):
    """Advance every walker one MC sweep under the ceiling {V < e_worst}."""
    n_acc = 0
    for i in range(pts.shape[0]):
        x, y, e, acc = _explore(
            pts[i, 0],
            pts[i, 1],
            sigma,
            e_worst,
            lo_x,
            hi_x,
            lo_y,
            hi_y,
            mc_steps,
            wells,
            confine,
            v_shift,
        )
        pts[i, 0] = x
        pts[i, 1] = y
        e_pts[i] = e
        n_acc += acc
    return n_acc


class NestedSampler2D:
    """Ensemble contraction over the multi-well PES defined in PES.py."""

    def __init__(
        self,
        n_live,
        bounds=((-3.2, 3.2), (-3.2, 3.2)),
        sigma=0.3,
        target_accept=0.5,
        seed=None,
    ):
        self.n_live = n_live
        self.bounds = bounds
        self.sigma = sigma
        self.target_accept = target_accept

        self.seed = _resolve_seed(seed)
        _seed_rng(self.seed)
        rng = np.random.default_rng(self.seed)

        (lo_x, hi_x), (lo_y, hi_y) = bounds
        self.live = np.column_stack(
            (rng.uniform(lo_x, hi_x, n_live), rng.uniform(lo_y, hi_y, n_live))
        )
        self.e_live = pes(self.live)

    def step(self, mc_steps=40):
        """Advance every live point by one MC sweep under the current worst-energy
        ceiling. Returns that ceiling (the e_worst this step enforced)."""
        (lo_x, hi_x), (lo_y, hi_y) = self.bounds
        e_worst = float(self.e_live.max())

        n_acc = _advance_all(
            self.live,
            self.e_live,
            self.sigma,
            e_worst,
            lo_x,
            hi_x,
            lo_y,
            hi_y,
            mc_steps,
            _WELLS_ARR,
            _CONFINE,
            _V_SHIFT,
        )

        accept_rate = n_acc / (self.n_live * mc_steps)
        self.sigma *= 1.1 if accept_rate > self.target_accept else 1 / 1.1
        self.sigma = min(max(self.sigma, 1e-4), min(hi_x - lo_x, hi_y - lo_y))
        return e_worst


#: Below 1.0 the walkers behind a ridge stay visible (dimmed) in the live GUI; see the
#: plot_surface() call in build_animation().
SURFACE_ALPHA = 0.8
#: Total width of the e_worst boundary line, in pixels (GUI) / points (export).
CONTOUR_WIDTH = 3.5
#: How far outside the true e_worst boundary the drawn ring sits, in x/y units. The ring
#: stays at height e_worst, so this is what lifts it off the surface: it wraps each well
#: from the outside like a collar instead of lying on (and z-fighting with) the mesh.
CONTOUR_XY_EXPANSION = 0.15


def _mark_geometry_changed(layer):
    """Tell the live GPU renderer that an in-place array update must be uploaded.

    Copied from example.py's helper of the same name.
    """
    layer.dirty.data_dirty = True
    layer.dirty.gpu_dirty = True
    layer.dirty.bounds_dirty = True


def run_nested_sampling(sampler, n_iter, mc_steps, stride=1):
    """Run the whole contraction once and return one snapshot per rendered frame.

    Frame 0 is the untouched starting configuration (no ceiling yet, e_worst = inf);
    frame k is the state after k * `stride` MC sweeps. `stride` still advances the
    sampler by every iteration (the physics is unaffected) but keeps only one snapshot
    per `stride` iterations -- the knob for a 400-iteration run that would otherwise
    mean 400 rendered frames.

    Precomputing is what lets the live GUI play the animation correctly. The GUI's
    timeline calls update(frame) on its own schedule: it loops back to frame 0, lets
    you scrub, and may skip frames when rendering falls behind. An update() that
    stepped a stateful sampler would keep contracting past the end on every loop and
    show a different run for every scrub; indexing a finished run cannot.

    Returns (live, e_live, e_worst) with shapes (n_frames, n_live, 2),
    (n_frames, n_live) and (n_frames,).
    """
    n_frames = n_iter // stride + 1
    live = np.empty((n_frames, sampler.n_live, 2))
    e_live = np.empty((n_frames, sampler.n_live))
    e_worst = np.full(n_frames, np.inf)
    live[0], e_live[0] = sampler.live, sampler.e_live
    for frame in tqdm(range(1, n_frames), desc="Running nested sampling"):
        for _ in range(stride):
            e_worst[frame] = sampler.step(mc_steps=mc_steps)
        live[frame], e_live[frame] = sampler.live, sampler.e_live
    return live, e_live, e_worst


def build_animation(sampler, n_iter, mc_steps, stride=1):
    """Build the glplot FuncAnimation for the nested-sampling run.

    Mirrors new_example.py's build_animation(): this only ever constructs (fig, ani)
    and hands them back -- it never calls gplt.show() or ani.save() itself, so main()
    decides afterwards whether to play it live in the GUI or ani.save() it.

    The sampler is run to completion first (run_nested_sampling()), then the PES
    surface and the walker scatter layer are each created once; every frame just
    looks up its precomputed snapshot and mutates the scatter layer's vertices in
    place, rather than clearing and re-adding layers per frame.

    Returns (fig, ani, n_frames, update); n_frames is how many frames ani.save() will
    write, and update is the same per-frame callback ani was built with.
    """
    live, e_live, e_worst_per_frame = run_nested_sampling(sampler, n_iter, mc_steps, stride)
    (lo_x, hi_x), (lo_y, hi_y) = sampler.bounds
    x = np.linspace(lo_x, hi_x, 220)
    y = np.linspace(lo_y, hi_y, 220)
    X, Y = np.meshgrid(x, y)
    Z = pes(np.stack((X, Y), axis=-1))
    # At the low, near-side-on elev=10 view below, a point sitting only slightly above
    # its local surface height is still easily grazed by a nearby ridge in the camera's
    # line of sight -- the on-screen clearance from a fixed world-space lift shrinks a
    # lot at a grazing angle. 0.15 (vs. the ~0.05 that was fine at the old elev=35) gives
    # real clearance instead of a razor-thin one.
    z_lift = 0.15
    # Marching-squares generator for the {PES(x, y) = e_worst} boundary -- the same
    # curve the 2D contourf/contour pair in scripts/Nested_sampling_2D.py's
    # plot_snapshots() draws, just lifted onto the 3D surface here.
    contour_gen = contourpy.contour_generator(x=x, y=y, z=Z)

    # facecolor is (visually) white but not the literal (1.0, 1.0, 1.0) tuple: the
    # headless preview renderer builds a real matplotlib Axes3D, which shades its x/y/z
    # walls light grey by default. glplot's own _apply_style_chrome() (utils/preview.py)
    # *can* recolour those panes to match the figure background, via
    # axis3d.set_pane_color() -- but only runs that step when the background isn't
    # exactly (1.0, 1.0, 1.0), which it special-cases as "the historical white export,
    # leave it untouched". This is how the grey panes are removed without reaching past
    # glplot into matplotlib internals.
    fig = gplt.figure(figsize=(9, 7), facecolor=(0.999, 0.999, 0.999))
    # depth_write=True on both layers keeps normal depth testing. The surface is slightly
    # translucent (SURFACE_ALPHA) because at elev=10 the ridges in front genuinely hide
    # most well bottoms: with a real depth buffer (the live GUI) an opaque PES would hide
    # the walkers exactly where they collect. GLPlot draws opaque 3D layers (walkers,
    # contour) first and translucent ones last, so the PES blends over whatever is
    # behind it and the walkers show through, dimmed -- the same look the matplotlib
    # export gives, where the surface is drawn over the points at alpha <= 0.92.
    gplt.plot_surface(X, Y, Z, cmap="coolwarm", alpha=SURFACE_ALPHA, depth_write=True, zorder=20)
    points = gplt.scatter3d(
        live[0, :, 0],
        live[0, :, 1],
        e_live[0] + z_lift,
        color="k",
        s=10,
        alpha=1.0,
        depth_write=True,
    )
    gplt.set_box_aspect((1, 1, 0.6))
    gplt.view_init(elev=10, azim=-60)
    gplt.set_axis3d_off()  # hides the box, grid, ticks and axis labels in one call

    # savefig()/ani.save() choose their renderer by whether a GL window already
    # exists (see glplot/animation.py's own module docstring): none yet -> every
    # frame goes through the matplotlib/Agg headless preview (CPU), which is what
    # was happening here before this call. test_mode=True builds a real (hidden,
    # since VISIBLE is hinted false in that mode) OpenGL context and renders exactly
    # one frame before returning -- engine.py's run() takes a dedicated early-return
    # branch for it instead of entering the blocking interactive loop -- so every
    # later save (or the --show window) runs through the actual GPU pipeline.
    # gplt.show()

    # The constraint boundary has a variable number of disconnected loops -- one per
    # well once e_worst drops below its rim -- so its layers can't be a fixed-size
    # array mutated in place like `points` above. Instead they're tracked here and
    # rebuilt every frame: removed from the scene and re-added via plot3d(), which is
    # animation.py's other supported idiom ("calling glplot.pyplot functions from
    # inside the callback") alongside in-place vertex mutation.
    contour_layers = []

    def _redraw_contour(e_worst):
        for layer in contour_layers:
            fig.scene.layers.remove(layer)
        contour_layers.clear()
        # At step 0 there is no ceiling yet (e_worst = inf, as in
        # Nested_sampling_2D.py's own first snapshot) -- nothing to draw a boundary
        # for, exactly as that script's plot_snapshots() skips the contour then.
        if not np.isfinite(e_worst):
            return
        for loop in contour_gen.lines(e_worst):
            # The boundary is a real 3D curve, depth-tested like everything else: in
            # front of the mesh where it is nearer the camera, behind it (seen through
            # the translucent PES, dimmed) where the mesh is in the way. Drawn exactly on
            # the level set it would lie on the surface and z-fight with it, so it is
            # widened in XY by CONTOUR_XY_EXPANSION, straight uphill (outward, the
            # direction V rises), and kept at height e_worst: a horizontal collar just
            # outside each well's wall, at exactly the energy ceiling it represents.
            #
            # Laying the widened ring back onto the surface instead was tried and
            # rejected: on the narrow, steep wells 0.15 outward climbs the wall unevenly
            # and the ring comes out tilted and far taller than the boundary it marks.
            uphill = -pes_normals(loop[:, 0], loop[:, 1])[:, :2]
            slope = np.linalg.norm(uphill, axis=1, keepdims=True)
            # A flat spot has no uphill; leave those points where they are.
            uphill = np.divide(uphill, slope, out=np.zeros_like(uphill), where=slope > 1e-9)
            ring = loop + CONTOUR_XY_EXPANSION * uphill
            curve = np.column_stack((ring, np.full(len(ring), e_worst)))
            for layer in gplt.plot3d(
                curve[:, 0], curve[:, 1], curve[:, 2], color="black", linewidth=1.0
            ):
                # Width: core-profile OpenGL (macOS included) draws 3D lines 1 px wide
                # whatever linewidth says, so the extra width comes from an outline in
                # the line's own colour -- 1 px + 2 * outline_width in the GUI, and
                # linewidth + 2 * outline_width in the matplotlib export.
                layer.style.outline_enabled = True
                layer.style.outline_color = (0.0, 0.0, 0.0, 1.0)
                layer.style.outline_width = (CONTOUR_WIDTH - 1.0) / 2.0
                contour_layers.append(layer)

    # Every frame is a pure lookup into the precomputed run, so update() can be called
    # in any order and any number of times -- FuncAnimation's warm-up-then-rewind, the
    # GUI looping back to frame 0, or the user scrubbing the timeline -- and always
    # draws the same picture for the same frame. No custom init_func is needed.
    n_frames = len(e_worst_per_frame)

    def update(frame):
        points.vertices[:] = np.column_stack(
            (live[frame, :, 0], live[frame, :, 1], e_live[frame] + z_lift)
        )
        _mark_geometry_changed(points)
        _redraw_contour(e_worst_per_frame[frame])
        return (points, *contour_layers)

    ani = animation.FuncAnimation(fig, update, frames=n_frames)
    return fig, ani, n_frames, update


def save_movie(ani, n_frames, out_dir, fps=20, gif=False):
    """ani.save() with a tqdm bar wired to Animation.save()'s own progress_callback
    hook (matplotlib's signature).

    Animation.save() picks its writer from the output filename's extension -- .png
    writes a frame sequence (out_dir/frame_0000.png, frame_0001.png, ...), .gif
    writes one animated GIF via Pillow (out_dir/movie.gif), matching how
    new_example.py's own ani.save() targets a .gif path. A relative out_dir is
    resolved against this script's own directory (presentation/), not the caller's
    cwd, so every movie lands there regardless of where the script is launched from.
    """
    out_dir = _in_presentation(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "movie.gif" if gif else "frame.png")
    with tqdm(total=n_frames, desc="Rendering nested-sampling movie") as pbar:

        def report_progress(frame_index, total):
            pbar.n = frame_index + 1
            pbar.refresh()

        ani.save(path, fps=fps, progress_callback=report_progress)
    if gif:
        print(f"saved animated GIF ({n_frames} frames) to {path}")
    else:
        print(f"saved {n_frames} frames to {out_dir}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--show",
        action="store_true",
        help="open the live OpenGL viewer instead of saving movie frames",
    )
    parser.add_argument(
        "--gif",
        action="store_true",
        help="save an animated GIF instead of a PNG frame sequence (ignored with --show)",
    )
    args = parser.parse_args(argv)

    sampler = NestedSampler2D(n_live=15, sigma=0.3, seed=2)
    _fig, ani, n_frames, update = build_animation(sampler, n_iter=100, mc_steps=40, stride=10)

    if args.show:
        # The GUI plays the FuncAnimation live on the panel's timeline, starting from
        # frame 0 and looping; Space pauses and the Timeline panel scrubs. Do not run
        # update() here first: the run is already precomputed, and the window would
        # only open on the last frame before replaying from the first anyway.
        gplt.show()
    else:
        save_movie(ani, n_frames, out_dir="NS_movie", fps=20, gif=args.gif)


if __name__ == "__main__":
    main()
