"""Test the pure signal-transform pipeline in glplot.gui.pipeline.

The chain logic and every step's numerics are pure numpy (delegating to mathops), so these
run fully headless -- no imgui, no GL. The panel that wraps them is tested separately in
:mod:`test_gui_pipeline_panel` conditions below, guarded by an imgui import.
"""

from __future__ import annotations

import numpy as np
import pytest

from glplot.gui import pipeline


def _signal(n=1000):
    """Two peaks on a sloped, noisy background -- a realistic chain input."""
    x = np.linspace(0.0, 20.0, n)

    def g(a, mu, s):
        return a * np.exp(-((x - mu) ** 2) / (2.0 * s * s))

    y = g(5.0, 6.0, 0.4) + g(4.0, 13.0, 0.4) + 0.3 * x + 1.0
    return x, y


class TestRegistry:
    """Every registered step must be internally consistent."""

    def test_every_step_has_defaults_for_all_its_params(self):
        """A missing default would KeyError the first time apply reads that parameter."""
        for key in pipeline.STEP_KEYS:
            step_type = pipeline.STEP_TYPES[key]
            for spec in step_type.params:
                assert spec.name in step_type.defaults, f"{key} has no default for {spec.name}"

    def test_step_keys_match_the_registry(self):
        assert set(pipeline.STEP_KEYS) == set(pipeline.STEP_TYPES)
        assert len(pipeline.STEP_KEYS) == len(pipeline.STEP_TYPES)

    def test_new_step_copies_defaults(self):
        """Two steps of one kind must hold independent parameter dicts."""
        a = pipeline.new_step("smooth")
        b = pipeline.new_step("smooth")
        a.params["window"] = 99
        assert b.params["window"] != 99
        assert a.params is not pipeline.STEP_TYPES["smooth"].defaults


class TestRunPipeline:
    """Test that a chain runs its steps in order, each feeding the next."""

    def test_empty_chain_returns_the_input(self):
        x, y = _signal()
        result = pipeline.run_pipeline(x, y, [])
        assert result.ok
        assert np.array_equal(result.y, y)

    def test_chain_applies_in_order(self):
        """baseline -> normalize(minmax) yields a signal spanning exactly [0, 1]."""
        x, y = _signal()
        steps = [pipeline.new_step("baseline"), pipeline.new_step("normalize")]
        steps[0].params["kind"] = "linear"
        result = pipeline.run_pipeline(x, y, steps)
        assert result.ok
        assert float(np.min(result.y)) == pytest.approx(0.0, abs=1e-9)
        assert float(np.max(result.y)) == pytest.approx(1.0, abs=1e-9)

    def test_order_matters(self):
        """Swapping two steps changes the result (normalize-then-derivative differs)."""
        x, y = _signal()
        a = [pipeline.new_step("normalize"), pipeline.new_step("derivative")]
        b = [pipeline.new_step("derivative"), pipeline.new_step("normalize")]
        ra = pipeline.run_pipeline(x, y, a)
        rb = pipeline.run_pipeline(x, y, b)
        assert not np.allclose(ra.y, rb.y)

    def test_a_failing_step_reports_its_index_and_message(self):
        """A bad band cutoff stops the chain at that step, not with a traceback."""
        x, y = _signal()
        steps = [pipeline.new_step("smooth"), pipeline.new_step("filter")]
        steps[1].params.update(btype="bandpass", cutoff=0.8, cutoff_hi=0.2)
        result = pipeline.run_pipeline(x, y, steps)
        assert not result.ok
        assert result.failed_index == 1
        assert "below" in (result.error or "")

    def test_partial_result_survives_a_failure(self):
        """On failure the result holds the signal produced by the steps before it."""
        x, y = _signal()
        steps = [pipeline.new_step("normalize"), pipeline.new_step("filter")]
        steps[1].params.update(btype="bandpass", cutoff=0.8, cutoff_hi=0.2)
        result = pipeline.run_pipeline(x, y, steps)
        # The first step (normalize minmax) still ran: its output spans [0, 1].
        assert float(np.max(result.y)) == pytest.approx(1.0, abs=1e-9)

    def test_unknown_step_is_reported_not_raised(self):
        x, y = _signal()
        result = pipeline.run_pipeline(x, y, [pipeline.PipelineStep("nope", {})])
        assert not result.ok
        assert "unknown step" in (result.error or "")

    @pytest.mark.parametrize("key", list(pipeline.STEP_KEYS))
    def test_every_step_runs_on_a_real_signal(self, key):
        """Each step type applies with its defaults and returns finite, same-or-valid data."""
        x, y = _signal()
        result = pipeline.run_pipeline(x, y, [pipeline.new_step(key)])
        assert result.ok, f"{key} failed: {result.error}"
        assert result.x.size == result.y.size
        assert bool(np.all(np.isfinite(result.y)))

    def test_derivative_then_integral_recovers_the_shape(self):
        """d/dx then cumulative-integral returns a curve parallel to the original."""
        x = np.linspace(0.0, 10.0, 2000)
        y = np.sin(x)
        steps = [pipeline.new_step("derivative"), pipeline.new_step("integral")]
        result = pipeline.run_pipeline(x, y, steps)
        assert result.ok
        # Integral of the derivative equals y minus its value at x[0] (a constant offset).
        recovered = result.y - result.y[0]
        assert np.allclose(recovered, y - y[0], atol=1e-2)


class TestMoveStep:
    """Test the reorder a drag performs."""

    def test_move_forward_and_back(self):
        steps = [pipeline.new_step(k) for k in ("baseline", "smooth", "normalize")]
        pipeline.move_step(steps, 0, 2)
        assert [s.kind for s in steps] == ["smooth", "normalize", "baseline"]
        pipeline.move_step(steps, 2, 0)
        assert [s.kind for s in steps] == ["baseline", "smooth", "normalize"]

    def test_out_of_range_is_a_no_op(self):
        steps = [pipeline.new_step("smooth")]
        pipeline.move_step(steps, 0, 5)
        pipeline.move_step(steps, -1, 0)
        pipeline.move_step(steps, 3, 4)
        assert [s.kind for s in steps] == ["smooth"]

    def test_same_index_is_a_no_op(self):
        steps = [pipeline.new_step("smooth"), pipeline.new_step("baseline")]
        pipeline.move_step(steps, 1, 1)
        assert [s.kind for s in steps] == ["smooth", "baseline"]


# ----------------------------------------------------------------------------------
# The panel, driven through the headless imgui harness (CONTRACT 2.10).
# ----------------------------------------------------------------------------------

imgui = pytest.importorskip("imgui_bundle").imgui

from typing import Any, List  # noqa: E402

from glplot.gui.commands import CommandQueue  # noqa: E402
from glplot.gui.datasets import Column, DataSet, DataStore  # noqa: E402
from glplot.gui.history import UndoStack  # noqa: E402
from glplot.gui.panels.pipeline import PipelinePanel  # noqa: E402


class _FakeScene:
    def __init__(self) -> None:
        self.layers: List[Any] = []


class _FakePlot:
    def __init__(self) -> None:
        self.scene = _FakeScene()
        self.hud = None


class _FakeWorkspace:
    def __init__(self) -> None:
        self.plot = _FakePlot()
        self.store = DataStore()
        self.queue = CommandQueue()
        self.undo = UndoStack()
        self.hud = None


@pytest.fixture
def imgui_context():
    ctx = imgui.create_context()
    io = imgui.get_io()
    io.display_size = 1280, 900
    io.delta_time = 1 / 60.0
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
    yield io
    imgui.destroy_context(ctx)


@pytest.fixture
def workspace():
    ws = _FakeWorkspace()
    t = np.linspace(0.0, 20.0, 500)
    ws.store.add(DataSet("d", [Column("x", t), Column("y", np.sin(t) + 0.2 * t)]))
    return ws


def _draw_frame(panel, io, pos=(0.0, 0.0), down=False):
    io.mouse_pos = pos
    io.mouse_down[0] = down
    imgui.new_frame()
    imgui.set_next_window_pos((100, 100))
    imgui.set_next_window_size((470, 460))
    imgui.begin("Pipeline")
    panel.draw()
    imgui.end()
    imgui.render()


class TestPipelinePanel:
    """The panel must draw, add/remove/reorder steps, and commit a result."""

    def test_draws_empty_without_crashing(self, imgui_context, workspace):
        panel = PipelinePanel(workspace)
        for _ in range(3):
            _draw_frame(panel, imgui_context)
        assert panel._steps == []

    def test_click_to_add_appends_a_step(self, imgui_context, workspace, monkeypatch):
        """Clicking a palette button appends that step type to the chain."""
        real_button = imgui.button
        monkeypatch.setattr(
            "glplot.gui.panels.pipeline.imgui.button",
            lambda label, *a, **k: True if label == "Smooth" else real_button(label, *a, **k),
        )
        panel = PipelinePanel(workspace)
        _draw_frame(panel, imgui_context)
        assert [s.kind for s in panel._steps] == ["smooth"]

    def test_pending_move_reorders_after_the_loop(self, imgui_context, workspace):
        """A queued reorder is applied by draw, not while the row loop is running."""
        panel = PipelinePanel(workspace)
        panel._steps = [pipeline.new_step(k) for k in ("baseline", "smooth", "normalize")]
        panel._pending = ("move", 0, 2)
        _draw_frame(panel, imgui_context)
        assert [s.kind for s in panel._steps] == ["smooth", "normalize", "baseline"]

    def test_pending_remove_drops_a_step(self, imgui_context, workspace):
        panel = PipelinePanel(workspace)
        panel._steps = [pipeline.new_step("smooth"), pipeline.new_step("baseline")]
        panel._pending = ("remove", 0)
        _draw_frame(panel, imgui_context)
        assert [s.kind for s in panel._steps] == ["baseline"]

    def test_pending_insert_adds_at_index(self, imgui_context, workspace):
        panel = PipelinePanel(workspace)
        panel._steps = [pipeline.new_step("smooth")]
        panel._pending = ("insert", "baseline", 0)
        _draw_frame(panel, imgui_context)
        assert [s.kind for s in panel._steps] == ["baseline", "smooth"]

    def test_apply_new_dataset_commits_the_chain_output(self, imgui_context, workspace):
        """Applying the chain as a new dataset registers the transformed signal."""
        panel = PipelinePanel(workspace)
        panel._steps = [pipeline.new_step("normalize")]
        panel._apply_mode = "new_dataset"
        for _ in range(2):
            _draw_frame(panel, imgui_context)
        x = np.asarray(workspace.store.get("d").get("x"))
        y = np.asarray(workspace.store.get("d").get("y"))
        result = pipeline.run_pipeline(x, y, panel._steps)
        cmd = panel._command_new_dataset(result.x, result.y, "chain out")
        cmd.do()
        assert "chain out" in workspace.store.names()
        ds = workspace.store.get("chain out")
        assert float(np.max(ds.get("y"))) == pytest.approx(1.0, abs=1e-9)
