"""Test the 2-D field path: ``f(x, y)`` from the Functions panel to an image layer.

Three pieces, and each can fail quietly rather than loudly:

* :func:`expressions.evaluate_2d` — a transposed grid still renders, just rotated, so the
  row/column convention is pinned against ``imshow``'s here rather than eyeballed.
* :func:`layerops.add_image_layer` — an image is a real GL texture only while
  ``metadata["artist"] == "imshow"`` (``scatter.py:142``); drop one key and the layer
  silently degrades into the grid of points that lives underneath the texture.
* The panel's mode switch — ``y`` has to become an *axis*, not the parameter slider every
  other free name becomes.

No OpenGL and no GPU: nothing here calls ``show()``.
"""

from __future__ import annotations

import numpy as np
import pytest

import glplot.pyplot as gplt
from glplot.engine import GPULinePlot
from glplot.gui import expressions, layerops
from glplot.gui.expressions import ExpressionError


@pytest.fixture(autouse=True)
def clean_state():
    gplt._cleanup_pyplot_state()
    yield
    gplt._cleanup_pyplot_state()


@pytest.fixture
def axes():
    return np.linspace(-3.0, 3.0, 5), np.linspace(-2.0, 2.0, 4)


class TestEvaluate2D:
    def test_grid_is_rows_of_y_by_columns_of_x(self, axes):
        """The layout `imshow` expects. Transposed, the picture comes out rotated."""
        x, y = axes
        assert expressions.evaluate_2d("x*0 + y*0", x, y).shape == (len(y), len(x))

    def test_matches_the_same_meshgrid_numpy_would_build(self, axes):
        x, y = axes
        xx, yy = np.meshgrid(x, y)
        got = expressions.evaluate_2d("sin(x)*cos(y)", x, y)
        assert np.allclose(got, np.sin(xx) * np.cos(yy))

    def test_y_is_an_axis_not_a_binding(self, axes):
        """`y` must win over a same-named variable, or the axis would be a constant."""
        x, y = axes
        got = expressions.evaluate_2d("y", x, y, variables={"y": 999.0})
        assert np.allclose(got[:, 0], y)

    def test_an_x_only_expression_broadcasts(self, axes):
        """Not an error: the field is simply constant along the axis it ignores."""
        x, y = axes
        got = expressions.evaluate_2d("x", x, y)
        assert got.shape == (len(y), len(x))
        assert np.allclose(got[0], got[-1])

    def test_a_constant_broadcasts(self, axes):
        x, y = axes
        assert np.allclose(expressions.evaluate_2d("7", x, y), 7.0)

    def test_parameters_still_bind(self, axes):
        x, y = axes
        got = expressions.evaluate_2d("a*y", x, y, variables={"a": 3.0})
        assert np.allclose(got[:, 0], 3.0 * y)

    def test_the_result_is_writable_and_unaliased(self, axes):
        """`broadcast_to` hands back a read-only view; a layer needs a real array."""
        x, y = axes
        got = expressions.evaluate_2d("1", x, y)
        got[0, 0] = 5.0  # must not raise
        assert got[0, 0] == 5.0

    def test_complex_is_refused(self, axes):
        x, y = axes
        with pytest.raises(ExpressionError, match="complex"):
            expressions.evaluate_2d("sqrt(x + 0j)", x, y)

    def test_hostile_input_is_still_refused(self, axes):
        """It goes through the same validated AST as `evaluate_1d`, and must stay there."""
        x, y = axes
        with pytest.raises(ExpressionError):
            expressions.evaluate_2d("__import__('os').system('true')", x, y)

    def test_a_2d_domain_is_refused(self, axes):
        x, y = axes
        with pytest.raises(ExpressionError, match="1-D"):
            expressions.evaluate_2d("x", np.zeros((2, 2)), y)


class TestImageLayer:
    @pytest.fixture
    def matrix(self):
        x, y = np.linspace(-3.0, 3.0, 16), np.linspace(-2.0, 2.0, 12)
        return expressions.evaluate_2d("sin(x)*cos(y)", x, y)

    def test_takes_the_textured_path(self, matrix):
        """`scatter.py:142` branches on this exact string; without it, dots."""
        plot = GPULinePlot()
        layer = layerops.add_image_layer(plot, matrix, (-3.0, 3.0, -2.0, 2.0), label="f")
        assert layer.metadata["artist"] == "imshow"
        assert layer.metadata["matrix"].shape == matrix.shape

    def test_carries_everything_the_texture_rebuild_reads(self, matrix):
        """The rebuild re-reads these on every gpu_dirty; a missing one is a silent no-op."""
        plot = GPULinePlot()
        layer = layerops.add_image_layer(
            plot, matrix, (-3.0, 3.0, -2.0, 2.0), label="f", cmap="turbo"
        )
        for key in ("matrix", "extent", "origin", "cmap", "vmin", "vmax"):
            assert key in layer.metadata, f"{key} missing: the texture cannot be rebuilt"
        assert layer.metadata["cmap"] == "turbo"

    def test_origin_is_lower_so_y_increases_upward(self, matrix):
        """A field's row 0 is its first y sample; an image's row 0 is its top."""
        plot = GPULinePlot()
        layer = layerops.add_image_layer(plot, matrix, (-3.0, 3.0, -2.0, 2.0), label="f")
        assert layer.metadata["origin"] == "lower"

    def test_the_colormap_stays_editable(self, matrix):
        """The Scene inspector's picker gates on this."""
        plot = GPULinePlot()
        layer = layerops.add_image_layer(plot, matrix, (-3.0, 3.0, -2.0, 2.0), label="f")
        assert layerops.layer_colormap_kind(layer) == "image"

    def test_has_no_plot_kind(self, matrix):
        """An image converts to nothing: the type menu must stay greyed out."""
        plot = GPULinePlot()
        layer = layerops.add_image_layer(plot, matrix, (-3.0, 3.0, -2.0, 2.0), label="f")
        assert layerops.layer_kind(layer) is None

    def test_points_cover_every_cell(self, matrix):
        plot = GPULinePlot()
        layer = layerops.add_image_layer(plot, matrix, (-3.0, 3.0, -2.0, 2.0), label="f")
        assert len(layer.pts) == matrix.size

    def test_a_1d_matrix_is_refused(self):
        plot = GPULinePlot()
        with pytest.raises(ValueError, match="2-D"):
            layerops.add_image_layer(plot, np.zeros(4), (0.0, 1.0, 0.0, 1.0), label="f")

    def test_an_empty_label_is_refused(self, matrix):
        """§1.5: the engine's index-based default renames itself as the scene changes."""
        plot = GPULinePlot()
        with pytest.raises(ValueError, match="label"):
            layerops.add_image_layer(plot, matrix, (-3.0, 3.0, -2.0, 2.0), label="")


class TestPanelFieldMode:
    """`y` in the text is what switches modes -- there is no toggle to get wrong."""

    def _panel(self):
        from glplot.gui.panels.functions import FunctionsPanel
        from glplot.gui.workspace import Workspace

        plot = GPULinePlot()
        ws = Workspace(plot)
        return plot, ws, FunctionsPanel(ws)

    def _sync(self, panel, expr):
        panel.expr = expr
        panel._synced_expr = None
        panel._sync_params()
        return panel

    def test_a_curve_stays_a_curve(self):
        _plot, _ws, panel = self._panel()
        self._sync(panel, "a*sin(b*x)")
        assert panel.is_field is False
        assert panel._param_order == ["a", "b"]

    def test_naming_y_makes_a_field(self):
        _plot, _ws, panel = self._panel()
        self._sync(panel, "sin(x)*cos(y)")
        assert panel.is_field is True

    def test_y_does_not_become_a_slider(self):
        """A name cannot be an axis and a knob at once."""
        _plot, _ws, panel = self._panel()
        self._sync(panel, "sin(x)*cos(y)")
        assert "y" not in panel._param_order

    def test_other_free_names_are_still_sliders(self):
        _plot, _ws, panel = self._panel()
        self._sync(panel, "a*sin(x)*cos(k*y)")
        assert panel.is_field is True
        assert panel._param_order == ["a", "k"]

    def test_the_label_reports_both_axes(self):
        _plot, _ws, panel = self._panel()
        self._sync(panel, "sin(x)*cos(y)")
        assert panel._label().startswith("f(x, y)")
        self._sync(panel, "sin(x)")
        assert panel._label().startswith("f(x) ")

    def test_the_preview_grid_is_capped_by_cells_not_samples(self):
        """nx*ny, not nx: 20k samples is a cheap curve and a vast grid."""
        from glplot.gui.panels.functions import PREVIEW_MAX_CELLS

        _plot, _ws, panel = self._panel()
        self._sync(panel, "sin(x)*cos(y)")
        panel.samples, panel.y_samples = 4000, 4000
        nx, ny = panel._field_shape(PREVIEW_MAX_CELLS)
        assert nx * ny <= PREVIEW_MAX_CELLS
        assert panel._field_shape() == (4000, 4000)  # uncapped is what Plot uses

    def test_the_cap_keeps_the_aspect(self):
        """Scaling one axis would preview a shape the field does not have."""
        from glplot.gui.panels.functions import PREVIEW_MAX_CELLS

        _plot, _ws, panel = self._panel()
        self._sync(panel, "sin(x)*cos(y)")
        panel.samples, panel.y_samples = 2000, 1000
        nx, ny = panel._field_shape(PREVIEW_MAX_CELLS)
        assert nx / ny == pytest.approx(2.0, rel=0.05)

    def test_plot_uses_the_full_grid_not_the_preview(self):
        """Plotting what the preview sampled would hand over a coarser field."""
        _plot, ws, panel = self._panel()
        self._sync(panel, "sin(x)*cos(y)")
        panel.x_min, panel.x_max, panel.samples = -6.0, 6.0, 200
        panel.y_min, panel.y_max, panel.y_samples = -4.0, 4.0, 150
        panel._refresh_preview()
        panel._action_plot_field()
        ws.queue.drain(panel.plot)

        layer = panel.plot.scene.layers[-1]
        assert layer.metadata["matrix"].shape == (150, 200)
        assert layer.metadata["extent"] == (-6.0, 6.0, -4.0, 4.0)

    def test_plotting_a_field_is_undoable(self):
        _plot, ws, panel = self._panel()
        self._sync(panel, "sin(x)*cos(y)")
        panel._refresh_preview()
        panel._action_plot_field()
        ws.queue.drain(panel.plot)
        assert len(panel.plot.scene.layers) == 1

        ws.undo.undo()
        ws.queue.drain(panel.plot)
        assert len(panel.plot.scene.layers) == 0

    def test_a_broken_expression_plots_nothing(self):
        _plot, ws, panel = self._panel()
        self._sync(panel, "sin(x)*cos(y)")
        panel._refresh_preview()
        panel.expr = "sin(x) +"  # invalid, and never synced
        panel._action_plot_field()
        ws.queue.drain(panel.plot)
        assert len(panel.plot.scene.layers) == 0
        assert panel._status_ok is False
