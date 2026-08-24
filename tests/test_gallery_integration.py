import runpy
from pathlib import Path

import pytest

import glplot.animation as _animation
import glplot.pyplot as _gplt
from glplot.engine import GPULinePlot

EXPECTED_FILES = [
    "01_line_plot.png",
    "02_scatter_fill.png",
    "03_bar_hist.png",
    "04_line_family_density.png",
    "05_guides_and_colormap.png",
    "06_signal_tools.png",
    "07_projected_3d_cloud.png",
    "08_vector_field_quiver.png",
    "09_large_matrix_heatmap.png",
    "10_massive_hist2d_density.png",
    "11_contour_pcolormesh_field.png",
    "12_surface_wireframe_bar3d.png",
    "13_volumetric_nebula.png",
    "14_bar3d_hex_box_city.png",
    "15_vector_field_3d.png",
    "16_ssao_comparison.png",
    "17_square_bars3d.png",
    "19_turbulent_vector_field_3d.png",
    "20_categorical_bar_and_spines.png",
    "21_date_axis_and_text_bbox.png",
    "22_log_scale_plots.png",
    "23_symlog_scale.png",
    "24_log_scale_bar_chart.png",
    "25_log_scale_contourf.png",
    "26_logit_scale.png",
    "27_inset_axes_image.png",
    "29_colorbar_gallery.png",
]


@pytest.mark.timeout(180)
def test_gallery_scripts_build_without_rendering(monkeypatch):
    """All 21 gallery scripts must run to completion without crashing and each
    must call savefig() exactly once with the expected filename.

    savefig() now falls back to render_preview() when no GL window exists
    (the no-window path added to fix the macOS headless crash), so we patch
    at the pyplot module level rather than patching GPULinePlot.savefig.

    28_chladni_wave_animation.py is the one script that calls Animation.save()
    (not savefig()) to write a real animated GIF -- 54 frames each re-rendered
    through the full matplotlib preview pipeline and then Pillow-encoded.
    Animation.save() itself already has dedicated coverage in
    test_animation_integration.py, so faking it here is the same call as
    faking savefig(): this test exists to prove every gallery script runs to
    completion without crashing, not to reproduce every real render or
    re-encode a GIF that already lives in examples/gallery/results/.

    Running 26 scripts through full line coverage genuinely takes ~90s
    (verified), well past the suite's default 60s hang ceiling -- this is real
    work, not a hang, so it gets its own generous timeout rather than raising
    the ceiling for every other test.
    """
    saved = []

    def fake_savefig(filename, density=None, scale=2.0):
        saved.append(Path(filename).name)

    monkeypatch.setattr(_gplt, "savefig", fake_savefig)
    monkeypatch.setattr(GPULinePlot, "run", lambda self: None)

    def fake_animation_save(self, *args, **kwargs):
        # Mirror Animation.save()'s own first line so its "deleted without
        # rendering anything" __del__ warning doesn't fire on a save() that was
        # deliberately skipped, not forgotten.
        self._draw_was_started = True

    monkeypatch.setattr(_animation.Animation, "save", fake_animation_save)

    root = Path(__file__).resolve().parents[1] / "examples" / "gallery"
    for script in sorted(root.glob("[0-9][0-9]_*.py")):
        runpy.run_path(str(script), run_name="__main__")

    assert saved == EXPECTED_FILES
