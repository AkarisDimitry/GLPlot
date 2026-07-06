from __future__ import annotations

import time
from typing import TYPE_CHECKING, Optional

import numpy as np
from OpenGL.GL import *

from ..core.context import RenderContext
from ..options import RenderMode

if TYPE_CHECKING:
    from ..engine import GPULinePlot


class ExportManager:
    """
    Handles offscreen rendering for high-resolution exports.
    """

    def __init__(self, engine: GPULinePlot):
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
            self.engine.autoscale()

        # Create a proper RenderContext for high-resolution export
        mvp = self.engine.camera_controller.mvp(width, height)
        window = self.engine.camera_controller.world_window(width, height)

        # Calculate NDC transform for high-res
        ndc_scale, ndc_offset = self.engine._get_ndc_transform(window)

        # Quality policy for exports
        prob = 1.0
        if exact_budget is not None and self.engine.scene.lines.count > 0:
            prob = min(1.0, (exact_budget * width) / self.engine.scene.lines.count)

        alpha = self.engine.options.default_global_alpha
        if prob < 1.0:
            alpha = min(1.0, alpha / (prob**0.5))

        ctx = RenderContext(
            mvp=mvp,
            window_world=window,
            ndc_scale=ndc_scale,
            ndc_offset=ndc_offset,
            width_px=width,
            height_px=height,
            fb_width=width,
            fb_height=height,
            dpr=scale * (self.engine.fb_width / max(self.engine.width, 1)),
            mode=mode or self.engine.policy.runtime.current_mode,
            global_alpha=alpha,
            lod_keep_prob=prob,
            is_density=self.engine.display_density,
            time=time.perf_counter(),
        )

        # Apply engine blending policy
        self.engine._apply_blending_policy()

        # Render all layers via the modular manager
        layers = self.engine._get_all_layers()
        if self.engine.display_density:
            self.engine.renderer_manager.draw_density(
                layers, ctx, target_fbo=fbo, target_size=(width, height)
            )
        else:
            self.engine.renderer_manager.draw_exact(layers, ctx)

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
