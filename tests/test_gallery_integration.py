import runpy
from pathlib import Path

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
    "18_hex_bars3d.png",
    "19_turbulent_vector_field_3d.png",
]


def test_gallery_scripts_build_without_rendering(monkeypatch):
    """All 19 gallery scripts must run to completion without crashing and each
    must call savefig() exactly once with the expected filename.

    savefig() now falls back to render_preview() when no GL window exists
    (the no-window path added to fix the macOS headless crash), so we patch
    at the pyplot module level rather than patching GPULinePlot.savefig.
    """
    saved = []

    def fake_savefig(filename, density=None, scale=2.0):
        saved.append(Path(filename).name)

    monkeypatch.setattr(_gplt, "savefig", fake_savefig)
    monkeypatch.setattr(GPULinePlot, "run", lambda self: None)

    root = Path(__file__).resolve().parents[1] / "examples" / "gallery"
    for script in sorted(root.glob("[0-9][0-9]_*.py")):
        runpy.run_path(str(script), run_name="__main__")

    assert saved == EXPECTED_FILES
