from __future__ import annotations

import time
from contextlib import contextmanager
from typing import TYPE_CHECKING, Iterator, Optional

import numpy as np
from OpenGL.GL import *

from ..core.context import RenderContext
from ..options import DEFAULT_AXIS_MARGINS, RenderMode, override_axis_margins

if TYPE_CHECKING:
    from ..engine import GPULinePlot


@contextmanager
def _default_axis_margins(options: object) -> Iterator[None]:
    """Render with the stock axis gutters, whatever the live window is using.

    An exported PNG is a figure, not a screenshot: it has no icon rail, no panels, no
    chrome of any kind. But the GUI widens ``axis_margin_l`` so its rail stops covering
    the Y tick labels on screen, and the export builds its projection from the very same
    ``camera_controller.mvp()``. Without this, every exported image silently inherits the
    rail-sized gutter and grows a mystery empty band down its left edge that nothing in
    the picture explains.

    Restores the caller's values on the way out, including on exception -- the live view
    must not be left mis-projected because a save failed.
    """
    with override_axis_margins(options, DEFAULT_AXIS_MARGINS):
        yield


class ExportManager:
    """
    Handles offscreen rendering for high-resolution exports.
    """

    def __init__(self, engine: GPULinePlot) -> None:
        self.engine = engine

    def savefig(
        self,
        filename: str,
        scale: float = 1.0,
        mode: Optional[RenderMode] = None,
        exact_budget: Optional[int] = None,
    ) -> None:
        """
        Renders the current scene to an offscreen buffer at high resolution.
        """
        # 1. Prepare target resolution
        width = int(self.engine.fb_width * scale)
        height = int(self.engine.fb_height * scale)

        # 2. Create offscreen resources
        fbo = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, fbo)

        tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tex)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, None)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0)

        rbo = glGenRenderbuffers(1)
        glBindRenderbuffer(GL_RENDERBUFFER, rbo)
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, width, height)
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rbo)
        glBindRenderbuffer(GL_RENDERBUFFER, 0)

        if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
            glBindFramebuffer(GL_FRAMEBUFFER, 0)
            glDeleteFramebuffers(1, [fbo])
            glDeleteTextures(1, [tex])
            glDeleteRenderbuffers(1, [rbo])
            raise RuntimeError("Failed to create export framebuffer")

        # 3. Save engine state and configure for export
        old_viewport = glGetIntegerv(GL_VIEWPORT)
        glViewport(0, 0, width, height)
        glClearColor(1.0, 1.0, 1.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # 4. Render
        # Auto-fit the view if no explicit limits have been set yet (mirrors the
        # behaviour of matplotlib's savefig, which always renders the autoscaled bounds).
        if (
            getattr(self.engine, "_needs_initial_autoscale", False)
            and not self.engine._is_pure_3d_scene()
        ):
            self.engine._autoscale_all_panels()

        # Screen-sampled layers (``FunctionLayer``) take the view as an input, and the view
        # here may be one the main loop never saw -- a script that sets limits and saves
        # without ever opening a window. Resample before drawing so the exported curve is
        # the one the limits ask for rather than whatever interval it was born with, and at
        # the *image's* pixel width rather than the window's, or a 4x export would just
        # magnify the screen's sampling.
        resample = getattr(self.engine, "_resample_view_layers", None)
        if callable(resample):
            resample(width / max(int(getattr(self.engine, "width", width) or width), 1))

        # The whole projection-and-draw span runs on the stock gutters: `mvp()` reads them,
        # and so does the density resolve pass's UV clip box (`renderers/density.py`). Both
        # must see the same numbers or the density image would clip against a box the
        # geometry was never projected into.
        with _default_axis_margins(self.engine.options):
            self.engine._apply_blending_policy()
            # One figure, potentially several panels: each is rendered into its own
            # sub-rectangle of the export image (scissored so a panel cannot bleed into a
            # neighbour), with a projection built from that panel's own pixel size and camera.
            # A single full-window panel is just the degenerate case and matches the old path.
            saved_idx = self.engine.active_panel_index
            # Scissor only matters when panels tile the image; a single full-image panel
            # needs none, and skipping it keeps the fast path free of extra GL state.
            multi = len(self.engine.panels) > 1
            try:
                for i, panel in enumerate(self.engine.panels):
                    self.engine.active_panel_index = i
                    vx, vy, vw, vh = panel.pixel_rect(width, height)
                    glViewport(vx, vy, vw, vh)
                    if multi:
                        glEnable(GL_SCISSOR_TEST)
                        glScissor(vx, vy, vw, vh)

                    mvp = self.engine.camera_controller.mvp(vw, vh)
                    window = self.engine.camera_controller.world_window(vw, vh)
                    ndc_scale, ndc_offset = self.engine._get_ndc_transform(window)

                    prob = 1.0
                    if exact_budget is not None and self.engine.scene.lines.count > 0:
                        prob = min(1.0, (exact_budget * vw) / self.engine.scene.lines.count)

                    alpha = self.engine.options.default_global_alpha
                    if prob < 1.0:
                        alpha = min(1.0, alpha / (prob**0.5))

                    ctx = RenderContext(
                        mvp=mvp,
                        window_world=window,
                        ndc_scale=ndc_scale,
                        ndc_offset=ndc_offset,
                        width_px=vw,
                        height_px=vh,
                        fb_width=vw,
                        fb_height=vh,
                        dpr=scale * (self.engine.fb_width / max(self.engine.width, 1)),
                        mode=mode or self.engine.policy.runtime.current_mode,
                        global_alpha=alpha,
                        lod_keep_prob=prob,
                        is_density=self.engine.display_density,
                        time=time.perf_counter(),
                    )

                    layers = self.engine._get_all_layers()
                    # ``_density_active`` rather than the raw flag: density is a 2D
                    # accumulator and the density pass drops 3D geometry, so exporting a 3D
                    # scene with density left on would write a blank frame. See
                    # ``engine._density_active``.
                    if self.engine._density_active():
                        self.engine.renderer_manager.draw_density(
                            layers, ctx, target_fbo=fbo, target_size=(width, height)
                        )
                    else:
                        self.engine.renderer_manager.draw_exact(layers, ctx)

                    # The axis pass — grid, spines and tick marks.
                    #
                    # This export used to draw the data layers and *nothing else*, so every
                    # PNG produced through GL came out with no frame, no grid and no ticks.
                    # It was easy to miss because ``pyplot.savefig`` only reaches this code
                    # when a GL context is already live; with no window it falls back to the
                    # matplotlib bridge, which draws full axes. The same call therefore
                    # produced two very different pictures depending on whether a window
                    # happened to be open — and the gallery PNGs, all generated headless,
                    # went down the fallback and looked correct.
                    #
                    # A 3D scene is skipped for the same reason the live overlay skips it:
                    # its decoration is real geometry in ``scene.layers`` and was already
                    # drawn above, so running the 2D axis pass over it would stamp a flat
                    # frame across the box.
                    #
                    # **Numeric tick labels are still absent from a GL export.** They are
                    # drawn into an imgui draw list, which needs an active imgui frame, and
                    # there is none here. Only the geometry is recoverable at this point;
                    # the matplotlib fallback remains the path that produces labelled axes.
                    if not self.engine._is_pure_3d_scene():
                        self.engine.axis_manager.update(ctx)
                        self.engine.renderer_manager.draw_axes(self.engine.axis_manager, ctx)
            finally:
                if multi:
                    glDisable(GL_SCISSOR_TEST)
                self.engine.active_panel_index = saved_idx
                glViewport(0, 0, width, height)

        # 5. Read back and save
        glFinish()
        pixels = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)
        image = np.frombuffer(pixels, dtype=np.uint8).reshape((height, width, 3))
        image = np.flipud(image)

        import matplotlib.pyplot as plt

        plt.imsave(filename, image)

        # 6. Cleanup
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glViewport(*old_viewport)
        glDeleteFramebuffers(1, [fbo])
        glDeleteTextures(1, [tex])
        glDeleteRenderbuffers(1, [rbo])

        print(f"Exported high-res image to {filename} ({width}x{height})")
