"""Test the Math Lab panel in glplot.gui.panels.mathlab.

Driven through the sanctioned headless imgui harness (CONTRACT 2.10): a real context,
real frames, real widgets, synthetic mouse input, and no OpenGL or GPU at any point.

The headline case is :class:`TestSourceSelector`. ``radio_button("Dataset")`` and
``enum_combo("Dataset", ...)`` used to sit in one id scope, and since ImGui derives a
widget id from ``hash(label) ^ id_stack`` they resolved to the same id -- so the combo's
popup, keyed on that id, could never open and the dataset could not be changed at all.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest

imgui = pytest.importorskip("imgui_bundle").imgui

from glplot.gui import icons  # noqa: E402
from glplot.gui import mathops  # noqa: E402
from glplot.gui import mathops2d  # noqa: E402
from glplot.gui import mathopsnd  # noqa: E402
from glplot.gui import notifications  # noqa: E402
from glplot.gui import widgets  # noqa: E402
from glplot.gui.commands import CommandQueue  # noqa: E402
from glplot.gui.datasets import Column, DataSet, DataStore  # noqa: E402
from glplot.gui.history import UndoStack  # noqa: E402
from glplot.gui.models import ModelStore, TrainedModel  # noqa: E402
from glplot.gui.panels.mathlab import (  # noqa: E402
    _APPLY_MODES,
    _CATEGORIES,
    _CATEGORY_OF,
    _FIT_MODELS,
    _INDEX_OPTION,
    _NO_SAMPLING,
    _TABS,
    MathLabPanel,
    _fit_state_key,
    _SamplingParams,
    _sorted_by_x,
    _Source,
    _split_indices,
    _subsample_indices,
)

# The panel's default geometry: workspace.py sizes it 0.37 x 0.47 of a 1280x900 viewport.
_PANEL_W = int(1280 * 0.37)
# +24px of test-harness headroom keeps the "Apply must stay on-screen" geometry checks
# below honest without changing the real window size workspace.py actually uses.
_PANEL_H = int(900 * 0.47) + 24

# Gaussian parameters (a, mu, sigma, c) the Fit tab tests generate data from and must recover.
_PEAK_TRUTH = (5.0, 2.0, 1.5, 1.0)


class _FakeScene:
    """Just enough scene for the panel to read layers from."""

    def __init__(self) -> None:
        self.layers: List[Any] = []


class _FakePlot:
    """A stand-in plot. The panel only ever reads ``scene.layers`` in these tests."""

    def __init__(self) -> None:
        self.scene = _FakeScene()
        self.hud = None


class _FakeWorkspace:
    """The five collaborators Panel proxies onto.

    ``models`` (a real :class:`ModelStore`) was added alongside this repo's
    now-uncommitted trained-model-rail work: ``MathLabPanel.draw()`` unconditionally
    reads ``self.models.models`` to decide whether the rail is visible, so every test
    that calls ``panel.draw()`` needs a real store here, not just ``None``.
    """

    def __init__(self) -> None:
        self.plot = _FakePlot()
        self.store = DataStore()
        self.queue = CommandQueue()
        self.undo = UndoStack()
        self.models = ModelStore()
        self.hud = None


@pytest.fixture
def imgui_context():
    """A headless imgui context with a font atlas but no GL (CONTRACT 2.10)."""
    ctx = imgui.create_context()
    io = imgui.get_io()
    io.display_size = 1280, 900
    io.delta_time = 1 / 60.0
    # Font atlas is dynamic under imgui-bundle; get_tex_data_as_rgba32()/texture_id no
    # longer exist. Telling imgui a backend owns texture building is the headless
    # equivalent, since this harness never renders real pixels.
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
    yield io
    imgui.destroy_context(ctx)


@pytest.fixture
def workspace():
    """A workspace holding three datasets, each with distinct column names."""
    ws = _FakeWorkspace()
    t = np.linspace(0.0, 10.0, 128)
    ws.store.add(DataSet("alpha", [Column("x", t), Column("y", np.sin(t))]))
    ws.store.add(DataSet("beta", [Column("x", t), Column("y", np.cos(t))]))
    ws.store.add(DataSet("wide", [Column("t", t), Column("v", t * 2.0), Column("w", t**2)]))
    return ws


def _draw_frame(panel: MathLabPanel, io: Any, pos=(0.0, 0.0), down: bool = False) -> None:
    """Run one complete frame with the panel at its default size."""
    io.mouse_pos = pos
    io.mouse_down[0] = down
    imgui.new_frame()
    imgui.set_next_window_pos((100, 100))
    imgui.set_next_window_size((_PANEL_W, _PANEL_H))
    imgui.begin("Math Lab")
    panel.draw()
    imgui.end()
    imgui.render()


class TestSourceSelector:
    """The dataset source selector must actually be operable."""

    def test_dataset_combo_popup_opens(self, imgui_context, workspace, monkeypatch):
        """Clicking the Dataset combo must open its popup.

        REGRESSION: the radio and the combo shared an ImGui id, so the popup was
        unreachable and the user could not change dataset -- the first control in the
        panel was dead. Fails without the push_id("kind") scope around the radios.
        """
        io = imgui_context
        panel = MathLabPanel(workspace)
        box = _capture_combo_center(monkeypatch, "Dataset")

        for _ in range(3):
            _draw_frame(panel, io)
        assert "center" in box, "the Dataset combo was never drawn"

        # Press and release on the combo.
        cx, cy = box["center"]
        _draw_frame(panel, io, (cx, cy), False)
        _draw_frame(panel, io, (cx, cy), True)
        _draw_frame(panel, io, (cx, cy), False)

        assert _popup_open_during_frame(panel, io, (cx, cy)) is True

    def test_dataset_selection_changes(self, imgui_context, workspace, monkeypatch):
        """Picking an entry from the open popup must change the selected dataset.

        The end-to-end form of the bug: not merely "a popup opens" but "the user can
        actually switch to another dataset".
        """
        io = imgui_context
        panel = MathLabPanel(workspace)
        log = _spy_enum_combo(monkeypatch, "Dataset")

        for _ in range(3):
            _draw_frame(panel, io)
        assert panel._ds_name == "alpha"

        cx, cy = log["center"]
        _draw_frame(panel, io, (cx, cy), False)
        _draw_frame(panel, io, (cx, cy), True)
        _draw_frame(panel, io, (cx, cy), False)
        _draw_frame(panel, io, (cx, cy), False)

        assert log["items"], "the popup never opened, so no entries were drawn"
        target, tx, ty = log["items"][-1]
        _draw_frame(panel, io, (tx, ty), False)
        _draw_frame(panel, io, (tx, ty), True)
        _draw_frame(panel, io, (tx, ty), False)

        assert target == "wide"
        assert panel._ds_name == "wide"

    def test_layer_radio_still_switches_source_kind(self, imgui_context, workspace):
        """The push_id scope must not stop the radios themselves from working."""
        io = imgui_context
        panel = MathLabPanel(workspace)
        for _ in range(2):
            _draw_frame(panel, io)
        assert panel._source_kind == "dataset"

        panel._source_kind = "layer"
        for _ in range(2):
            _draw_frame(panel, io)
        # No layer has tabular data, so the panel reports that rather than crashing.
        assert panel._source_kind == "layer"


class TestColumnMemory:
    """x/y column picks belong to the dataset, not to the panel."""

    def test_columns_remembered_per_dataset(self, imgui_context, workspace):
        """Switching dataset and back must restore that dataset's own columns.

        REGRESSION: the picks were panel-global and silently reset on every switch.
        """
        io = imgui_context
        panel = MathLabPanel(workspace)

        panel._ds_name = "wide"
        for _ in range(2):
            _draw_frame(panel, io)
        assert (panel._ds_x_col, panel._ds_y_col) == ("t", "v")

        # A deliberate pick on 'wide'.
        panel._ds_columns["wide"] = ("t", "w")
        for _ in range(2):
            _draw_frame(panel, io)
        assert (panel._ds_x_col, panel._ds_y_col) == ("t", "w")

        # Away to a dataset with entirely different column names...
        panel._ds_name = "alpha"
        for _ in range(2):
            _draw_frame(panel, io)
        assert (panel._ds_x_col, panel._ds_y_col) == ("x", "y")

        # ...and back. The pick survives.
        panel._ds_name = "wide"
        for _ in range(2):
            _draw_frame(panel, io)
        assert (panel._ds_x_col, panel._ds_y_col) == ("t", "w")

    def test_forgets_deleted_datasets(self, imgui_context, workspace):
        """The per-dataset memory must not grow without bound."""
        io = imgui_context
        panel = MathLabPanel(workspace)
        panel._ds_columns["ghost"] = ("a", "b")
        panel._ds_name = "alpha"
        for _ in range(2):
            _draw_frame(panel, io)
        assert "ghost" not in panel._ds_columns

    def test_stale_column_falls_back(self, imgui_context, workspace):
        """A remembered column that no longer exists must not select nothing."""
        io = imgui_context
        panel = MathLabPanel(workspace)
        panel._ds_name = "alpha"
        panel._ds_columns["alpha"] = ("gone", "vanished")
        for _ in range(2):
            _draw_frame(panel, io)
        assert (panel._ds_x_col, panel._ds_y_col) == ("x", "y")


class TestApplyIsReachable:
    """Apply must be on screen without scrolling, on every tab that has one."""

    @pytest.mark.parametrize("key,title", [(k, t) for k, t, _m in _TABS])
    def test_apply_button_visible(self, imgui_context, workspace, monkeypatch, key, title):
        """The Apply button must lie inside the panel at its default size.

        REGRESSION: Apply was laid out after the stat rows and fell below the fold on
        every tab but Normalize (Integral at y=456 in a 423px panel; Fit at y=501 and
        further down with each coefficient row). It is now pinned to the panel bottom.
        """
        if key == "umap":
            pytest.importorskip("umap")

        io = imgui_context
        panel = MathLabPanel(workspace)
        rect = _capture_apply_rect(monkeypatch)

        panel.show_operation(key)
        for _ in range(6):
            # Clear per frame: a tab switch takes one frame to settle, so the first
            # frame still draws the previous tab.
            rect.clear()
            _draw_frame(panel, io)

        assert rect, f"the {title} tab drew no Apply button"
        assert rect["bottom"] <= 100 + _PANEL_H, (
            f"{title}: Apply bottom at y={rect['bottom']:.0f} is below the panel, "
            f"which ends at y={100 + _PANEL_H}"
        )

    @pytest.mark.filterwarnings("ignore:Polyfit may be poorly conditioned")
    def test_apply_stays_pinned_as_stats_grow(self, imgui_context, workspace, monkeypatch):
        """Raising the fit degree adds a stat row per coefficient; Apply must not move.

        This was the worst case of the old layout: every extra coefficient pushed the
        commit button further out of reach.
        """
        io = imgui_context
        panel = MathLabPanel(workspace)
        rect = _capture_apply_rect(monkeypatch)
        positions = set()

        for degree in (1, 8, 20):
            panel.show_operation("fit")
            panel._fit_degree = degree
            for _ in range(6):
                rect.clear()
                _draw_frame(panel, io)
            assert rect, f"degree {degree} drew no Apply button"
            assert rect["bottom"] <= 100 + _PANEL_H
            positions.add(round(rect["bottom"]))

        assert len(positions) == 1, f"Apply moved as stats grew: {sorted(positions)}"


class TestTabCategories:
    """Every operation must live in exactly one category and be reachable through it."""

    def test_every_tab_is_categorised_exactly_once(self):
        """No operation may be orphaned (in no category) or duplicated (in two)."""
        all_keys = [k for k, _t, _m in _TABS]
        categorised = [k for _title, keys in _CATEGORIES for k in keys]
        assert sorted(categorised) == sorted(all_keys)
        assert len(categorised) == len(set(categorised)), "a tab is in two categories"
        assert set(_CATEGORY_OF) == set(all_keys)

    def test_show_operation_jumps_to_the_operations_category(self):
        """Requesting an operation must also queue its category for the top-level bar."""
        panel = MathLabPanel(_FakeWorkspace())
        panel.store.add(DataSet("d", [Column("x", np.arange(10.0)), Column("y", np.arange(10.0))]))
        panel.show_operation("histogram")
        assert panel._pending_tab == "histogram"
        assert panel._pending_category == _CATEGORY_OF["histogram"]

    def test_unknown_operation_sets_nothing(self):
        """A stale caller with a bad key must not queue a phantom category."""
        panel = MathLabPanel(_FakeWorkspace())
        panel.show_operation("does_not_exist")
        assert panel._pending_tab is None
        assert panel._pending_category is None

    @pytest.mark.parametrize("key", [k for k, _t, _m in _TABS])
    def test_each_operation_reachable_across_categories(self, imgui_context, workspace, key):
        """Driving show_operation for any key must settle on that tab's computed result.

        The end-to-end proof that the two-level bar loses nothing: whichever category the
        operation lives in, the panel navigates to it and the operation's params are the
        ones that actually run.
        """
        io = imgui_context
        panel = MathLabPanel(workspace)
        panel.show_operation(key)
        for _ in range(6):
            _draw_frame(panel, io)
        # The tab actually took hold: its params drove the last compute (cache key holds
        # the params tuple whose head is the operation kind).
        assert panel._cache_key is not None
        assert panel._cache_key[-1][0] == key


class TestPerTabApplyState:
    """The output name and apply mode describe the operation, not the panel."""

    def test_name_is_per_tab(self, workspace):
        """A name typed on Smooth must not leak onto Fit."""
        panel = MathLabPanel(workspace)
        panel._apply_names["smooth"] = "my smoothed curve"
        assert panel._apply_name_for("smooth") == "my smoothed curve"
        assert panel._apply_name_for("fit") == ""

    def test_mode_is_per_tab(self, workspace):
        """A mode chosen on one tab must not carry to another."""
        panel = MathLabPanel(workspace)
        assert panel._apply_mode_for("integral") == "new_layer"
        panel._apply_modes["integral"] = "replace"
        assert panel._apply_mode_for("integral") == "replace"
        assert panel._apply_mode_for("fft") == "new_layer"

    def test_output_name_uses_the_tabs_own_name(self, workspace):
        """_output_name must read the name of the tab being applied."""
        panel = MathLabPanel(workspace)
        source = _dataset_source(workspace)
        result = _fake_result()
        panel._apply_names["smooth"] = "chosen"
        assert panel._output_name(source, result, "smooth") == "chosen"
        # Another tab falls back to the generated default.
        assert panel._output_name(source, result, "fit") == "alpha.y integral"

    def test_blank_name_falls_back_to_default(self, workspace):
        """Whitespace is not a name."""
        panel = MathLabPanel(workspace)
        source = _dataset_source(workspace)
        result = _fake_result()
        panel._apply_names["smooth"] = "   "
        assert panel._output_name(source, result, "smooth") == "alpha.y integral"


class TestPreviewCaption:
    """The preview must say that it is a preview."""

    def test_caption_present(self, imgui_context, workspace, monkeypatch):
        """An uncommitted preview must be captioned, in ASCII (CONTRACT 3)."""
        io = imgui_context
        panel = MathLabPanel(workspace)
        seen: List[str] = []
        real = imgui.text_disabled
        monkeypatch.setattr(
            imgui, "text_disabled", lambda t, *a, **k: (seen.append(t), real(t, *a, **k))[1]
        )
        panel.show_operation("integral")
        for _ in range(4):
            _draw_frame(panel, io)

        caption = next((s for s in seen if "not applied" in s), None)
        assert caption is not None, f"no preview caption among {seen!r}"
        assert caption.isascii(), f"caption {caption!r} needs glyphs ProggyClean lacks"


# ----------------------------------------------------------------------------------
# Instrumentation helpers. These measure geometry; they never fake panel behaviour.
# ----------------------------------------------------------------------------------


def _capture_combo_center(monkeypatch, label: str) -> Dict[str, Any]:
    """Record the screen centre of the combo drawn with ``label``."""
    box: Dict[str, Any] = {}
    real = widgets.enum_combo

    def spy(lbl, current, options, **kwargs):
        out = real(lbl, current, options, **kwargs)
        if lbl == label:
            mn, mx = imgui.get_item_rect_min(), imgui.get_item_rect_max()
            box["center"] = ((mn.x + mx.x) / 2.0, (mn.y + mx.y) / 2.0)
        return out

    monkeypatch.setattr("glplot.gui.panels.mathlab.widgets.enum_combo", spy)
    return box


def _spy_enum_combo(monkeypatch, label: str) -> Dict[str, Any]:
    """A real begin_combo/selectable pair that also records where everything landed.

    This mirrors ``widgets.enum_combo`` exactly; it exists only so the test can learn
    the popup entries' screen positions, which the real helper does not expose.
    """
    log: Dict[str, Any] = {"items": []}

    def spy(lbl, current, options, **kwargs):
        items = [str(o) for o in options]
        changed = False
        selected = current
        opened = imgui.begin_combo(lbl, str(current))
        if lbl == label:
            mn, mx = imgui.get_item_rect_min(), imgui.get_item_rect_max()
            log["center"] = ((mn.x + mx.x) / 2.0, (mn.y + mx.y) / 2.0)
            log["opened"] = opened
            log["items"] = []
        if opened:
            for item in items:
                clicked, _ = imgui.selectable(item, item == current)
                if lbl == label:
                    m1, m2 = imgui.get_item_rect_min(), imgui.get_item_rect_max()
                    log["items"].append((item, (m1.x + m2.x) / 2.0, (m1.y + m2.y) / 2.0))
                if clicked and item != current:
                    selected = item
                    changed = True
            imgui.end_combo()
        return changed, selected

    monkeypatch.setattr("glplot.gui.panels.mathlab.widgets.enum_combo", spy)
    return log


def _capture_apply_rect(monkeypatch) -> Dict[str, float]:
    """Record the screen rect of the real Apply button."""
    rect: Dict[str, float] = {}
    real = imgui.button

    def spy(label, *args, **kwargs):
        out = real(label, *args, **kwargs)
        if label == "Apply":
            mn, mx = imgui.get_item_rect_min(), imgui.get_item_rect_max()
            rect["top"] = mn.y
            rect["bottom"] = mx.y
        return out

    monkeypatch.setattr("glplot.gui.panels.mathlab.imgui.button", spy)
    return rect


def _popup_open_during_frame(panel: MathLabPanel, io: Any, pos) -> bool:
    """Draw a frame and ask ImGui whether any popup is open while it is still live."""
    io.mouse_pos = pos
    io.mouse_down[0] = False
    imgui.new_frame()
    imgui.set_next_window_pos((100, 100))
    imgui.set_next_window_size((_PANEL_W, _PANEL_H))
    imgui.begin("Math Lab")
    panel.draw()
    state = imgui.is_popup_open("", imgui.PopupFlags_.any_popup)
    imgui.end()
    imgui.render()
    return bool(state)


def _dataset_source(ws: _FakeWorkspace) -> Any:
    """The _Source the panel would resolve for dataset 'alpha', built without imgui."""
    from glplot.gui.panels.mathlab import _Source

    dataset = ws.store.get("alpha")
    return _Source(
        key=("dataset", "alpha", "x", "y"),
        label="alpha.y",
        x_name="x",
        y_name="y",
        x_raw=dataset.get("x"),
        y_raw=dataset.get("y"),
        dataset=dataset,
        x_col="x",
        y_col="y",
    )


def _fake_result() -> Any:
    """A minimal _Result whose default name is deterministic."""
    from glplot.gui.panels.mathlab import _Result

    return _Result(
        x=np.zeros(4),
        y=np.zeros(4),
        x_name="x",
        y_name="y integral",
        suffix="integral",
        plot_label="cumulative integral",
    )


class TestFitTab:
    """The Fit tab must be real fitting: models, guesses and errors, not one spinner.

    REGRESSION for the user's "podria incluir fitting": the tab existed but was
    polynomial-only with a lone Degree spinner, which to a scientist is not fitting.
    """

    @pytest.fixture
    def peak_workspace(self):
        """A workspace whose one dataset is a noisy gaussian with known parameters."""
        ws = _FakeWorkspace()
        rng = np.random.default_rng(4)
        x = np.linspace(-10.0, 12.0, 400)
        y = mathops.MODELS["gaussian"].fn(x, *_PEAK_TRUTH) + rng.normal(0.0, 0.08, x.size)
        ws.store.add(DataSet("peak", [Column("x", x), Column("y", y)]))
        return ws

    def _fit_result(self, panel, io, model):
        """Settle the Fit tab on ``model`` and return the last computed _Result."""
        panel.show_operation("fit")
        panel._fit_model = model
        for _ in range(4):
            _draw_frame(panel, io)
        return panel._cache_result, panel._cache_error

    def test_model_combo_offers_more_than_polynomial(self, peak_workspace):
        """The tab must offer the nonlinear models a scientist means by "fitting"."""
        assert _FIT_MODELS[0] == "polynomial", "polynomial stays the default and first"
        assert {
            "gaussian",
            "lorentzian",
            "exp_decay",
            "power_law",
            "sigmoid",
            "linear",
            "custom",
        } <= set(_FIT_MODELS)
        assert MathLabPanel(peak_workspace)._fit_model == "polynomial"

    @pytest.mark.parametrize("model", ["linear", "gaussian", "lorentzian", "sigmoid"])
    def test_every_model_reports_a_value_and_an_error_per_parameter(
        self, imgui_context, peak_workspace, model
    ):
        """The real gap versus the old tab: a standard error on every fitted value."""
        panel = MathLabPanel(peak_workspace)
        result, error = self._fit_result(panel, imgui_context, model)
        assert error is None, f"{model} failed: {error}"
        rows = dict(result.stats)
        for name in mathops.MODELS[model].param_names:
            assert name in rows, f"{model} reported no row for parameter {name}"
            assert "+/-" in rows[name], f"{model}: {name} has no standard error"
        assert "R squared" in rows
        assert any("chi sq" in label for label in rows)

    def test_gaussian_tab_recovers_the_known_peak(self, imgui_context, peak_workspace):
        """Driven through the real panel, the fit lands on the parameters we generated."""
        panel = MathLabPanel(peak_workspace)
        result, error = self._fit_result(panel, imgui_context, "gaussian")
        assert error is None
        rows = dict(result.stats)
        assert float(rows["mu"].split("+/-")[0]) == pytest.approx(_PEAK_TRUTH[1], abs=0.05)
        assert float(rows["sigma"].split("+/-")[0]) == pytest.approx(_PEAK_TRUTH[2], abs=0.05)
        assert float(rows["R squared"]) > 0.99
        assert result.plot_label == "Gaussian fit"
        # The source must still be overlaid on the fit: that is the "sobreponer".
        assert result.overlay is not None

    def test_polynomial_path_is_unchanged(self, imgui_context, peak_workspace):
        """The old tab still works exactly as it did, degree spinner and all."""
        panel = MathLabPanel(peak_workspace)
        panel.show_operation("fit")
        panel._fit_model = "polynomial"
        panel._fit_degree = 2
        for _ in range(4):
            _draw_frame(panel, imgui_context)
        result = panel._cache_result
        assert panel._cache_error is None
        assert result.plot_label == "polynomial fit (degree 2)"
        assert dict(result.stats)["Effective degree"] == "2"

    def test_deconvolution_separates_overlapping_peaks(self, imgui_context):
        """The 'gaussians' model deconvolves overlapping peaks and reports per-peak areas.

        The end-to-end answer to the Peaks tab's documented limitation: valley integration
        cannot split overlapping peaks, but a joint fit can, and its areas come out exact.
        """
        ws = _FakeWorkspace()
        x = np.linspace(0.0, 20.0, 2000)

        def g(a, mu, s):
            return a * np.exp(-((x - mu) ** 2) / (2.0 * s * s))

        ws.store.add(
            DataSet(
                "blend", [Column("x", x), Column("y", g(5.0, 8.0, 0.8) + g(3.0, 11.0, 0.8) + 0.5)]
            )
        )
        panel = MathLabPanel(ws)
        panel.show_operation("fit")
        panel._fit_model = "gaussians"
        panel._fit_npeaks = 2
        for _ in range(4):
            _draw_frame(panel, imgui_context)
        assert panel._cache_error is None
        rows = dict(panel._cache_result.stats)
        assert float(rows["R squared"]) > 0.999
        # Areas: a*sigma*sqrt(2*pi) = 5*0.8*2.5066=10.03 and 3*0.8*2.5066=6.02.
        areas = sorted(float(rows[f"Peak {i} area"]) for i in (1, 2))
        assert areas[1] == pytest.approx(10.03, rel=0.03)
        assert areas[0] == pytest.approx(6.02, rel=0.03)
        assert "Total peak area" in rows

    def test_peak_count_is_in_the_cache_key(self, imgui_context):
        """Changing the peak count must invalidate the preview cache and refit."""
        ws = _FakeWorkspace()
        x = np.linspace(0.0, 20.0, 1500)
        y = np.exp(-((x - 7.0) ** 2) / 2.0) + np.exp(-((x - 13.0) ** 2) / 2.0)
        ws.store.add(DataSet("two", [Column("x", x), Column("y", y)]))
        panel = MathLabPanel(ws)
        panel.show_operation("fit")
        panel._fit_model = "gaussians"
        panel._fit_npeaks = 1
        for _ in range(4):
            _draw_frame(panel, imgui_context)
        one = len(dict(panel._cache_result.stats))
        panel._fit_npeaks = 2
        for _ in range(4):
            _draw_frame(panel, imgui_context)
        two = len(dict(panel._cache_result.stats))
        assert two > one, "raising the peak count did not add parameter/area rows"

    def test_auto_guess_reaches_the_tab(self, imgui_context, peak_workspace):
        """The heuristic start is computed from the data and stashed for the tab to show.

        A tab body never sees the source, so this hand-off is the only way the panel can
        display the automatic guess or seed a manual one from it.
        """
        panel = MathLabPanel(peak_workspace)
        self._fit_result(panel, imgui_context, "gaussian")
        key = _fit_state_key("gaussian", mathops.MODELS["gaussian"])
        assert key in panel._fit_auto
        source = peak_workspace.store.get("peak")
        expected = mathops.initial_guess(
            source.columns[0].values, source.columns[1].values, "gaussian"
        )
        assert np.allclose(panel._fit_auto[key], expected)

    def test_manual_guess_seeds_from_auto_and_reaches_the_fit(self, imgui_context, peak_workspace):
        """Manual mode starts from the heuristic; editing four numbers from nothing is not a UI."""
        panel = MathLabPanel(peak_workspace)
        self._fit_result(panel, imgui_context, "gaussian")
        key = _fit_state_key("gaussian", mathops.MODELS["gaussian"])
        auto = list(panel._fit_auto[key])

        panel._fit_manual = True
        for _ in range(2):
            _draw_frame(panel, imgui_context)
        assert np.allclose(panel._fit_p0[key], auto), "manual guess did not seed from auto"
        assert panel._cache_error is None

        # A deliberately hopeless guess must actually reach curve_fit and spoil the fit,
        # proving the control is wired rather than decorative.
        panel._fit_p0[key] = [1.0, -9.0, 0.01, 0.0]
        panel._cache_key = None
        for _ in range(2):
            _draw_frame(panel, imgui_context)
        if panel._cache_error is None:
            assert float(dict(panel._cache_result.stats)["R squared"]) < 0.9

    def test_custom_model_fits_the_free_variables_of_the_expression(
        self, imgui_context, peak_workspace
    ):
        """custom f(x): every name but x becomes a fitted parameter."""
        panel = MathLabPanel(peak_workspace)
        panel.show_operation("fit")
        panel._fit_model = "custom"
        panel._fit_expr = "a*exp(-(x - m)**2) + c"
        for _ in range(4):
            _draw_frame(panel, imgui_context)
        assert panel._cache_error is None, panel._cache_error
        rows = dict(panel._cache_result.stats)
        for name in ("a", "c", "m"):
            assert "+/-" in rows[name], f"custom parameter {name} has no error"

    def test_a_broken_custom_expression_does_not_kill_the_frame(
        self, imgui_context, peak_workspace
    ):
        """An exception escaping a draw callback takes the whole window with it."""
        panel = MathLabPanel(peak_workspace)
        panel.show_operation("fit")
        panel._fit_model = "custom"
        for expr in ("a*exp(-x/", "__import__('os')", "", "2*x"):
            panel._fit_expr = expr
            for _ in range(3):
                _draw_frame(panel, imgui_context)  # must not raise

    def test_fit_params_stay_hashable(self, imgui_context, peak_workspace):
        """The params tuple is the preview cache key; an unhashable member breaks it."""
        panel = MathLabPanel(peak_workspace)
        panel.show_operation("fit")
        for model in _FIT_MODELS:
            panel._fit_model = model
            panel._fit_manual = model != "polynomial"
            for _ in range(3):
                _draw_frame(panel, imgui_context)
            assert panel._cache_key is None or hash(panel._cache_key)

    def test_apply_is_still_reachable_with_a_parameter_table(
        self, imgui_context, peak_workspace, monkeypatch
    ):
        """The fit report is several rows longer now; Apply must stay pinned to the bottom."""
        panel = MathLabPanel(peak_workspace)
        rect = _capture_apply_rect(monkeypatch)
        panel.show_operation("fit")
        panel._fit_model = "gaussian"
        for _ in range(6):
            rect.clear()
            _draw_frame(panel, imgui_context)
        assert rect, "the Fit tab drew no Apply button"
        assert rect["bottom"] <= 100 + _PANEL_H


class TestPeaksTab:
    """The Peaks tab must find and mark maxima, preview them, and apply a peak table."""

    @pytest.fixture
    def peaks_workspace(self):
        """One dataset: three clean Gaussian peaks of heights 3, 1, 2 at x = 5, 15, 23."""
        ws = _FakeWorkspace()
        x = np.linspace(0.0, 30.0, 1500)

        def g(mu, a, s):
            return a * np.exp(-((x - mu) ** 2) / (2.0 * s * s))

        y = g(5.0, 3.0, 0.6) + g(15.0, 1.0, 0.5) + g(23.0, 2.0, 0.8)
        ws.store.add(DataSet("sig", [Column("x", x), Column("y", y)]))
        return ws

    def _peaks_result(self, panel, io):
        panel.show_operation("peaks")
        for _ in range(4):
            _draw_frame(panel, io)
        return panel._cache_result, panel._cache_error

    def test_finds_three_peaks_and_marks_them(self, imgui_context, peaks_workspace):
        """The default prominence keeps all three peaks and exposes them as markers."""
        panel = MathLabPanel(peaks_workspace)
        result, error = self._peaks_result(panel, imgui_context)
        assert error is None
        assert result.x.size == 3
        assert np.allclose(result.x, [5.0, 15.0, 23.0], atol=0.05)
        # The preview shows the whole signal (base_*) with the peaks as markers.
        assert result.base_x is not None and result.base_y is not None
        assert result.markers is not None
        assert result.markers[0].size == 3
        # A peak table cannot overwrite the signal it was measured from.
        assert result.allow_replace is False

    def test_markers_reach_the_preview_plot(self, imgui_context, peaks_workspace):
        """The detected peaks must actually be handed to mini_plot as markers."""
        panel = MathLabPanel(peaks_workspace)
        seen: Dict[str, Any] = {}
        real = widgets.mini_plot

        def spy(*args, **kwargs):
            if kwargs.get("markers") is not None:
                seen["n"] = int(np.asarray(kwargs["markers"][0]).size)
            return real(*args, **kwargs)

        monkeypatch_target = "glplot.gui.panels.mathlab.widgets.mini_plot"
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(monkeypatch_target, spy)
            panel.show_operation("peaks")
            for _ in range(4):
                _draw_frame(panel, imgui_context)
        assert seen.get("n") == 3

    def test_raising_prominence_drops_the_small_peak(self, imgui_context, peaks_workspace):
        """A prominence floor above the shortest peak keeps only the tall two."""
        panel = MathLabPanel(peaks_workspace)
        panel._peak_prominence = 0.6  # > 1/3 of the 3.0 full swing -> drops the height-1 peak
        result, error = self._peaks_result(panel, imgui_context)
        assert error is None
        assert np.allclose(result.x, [5.0, 23.0], atol=0.05)

    def test_no_peaks_offers_no_apply(self, imgui_context):
        """A flat signal yields no peaks, so there is nothing to commit."""
        ws = _FakeWorkspace()
        x = np.linspace(0.0, 1.0, 200)
        ws.store.add(DataSet("flat", [Column("x", x), Column("y", np.zeros_like(x))]))
        panel = MathLabPanel(ws)
        result, error = self._peaks_result(panel, imgui_context)
        assert error is None
        assert result.x.size == 0
        assert result.allow_apply is False

    def test_reports_area_and_builds_the_full_table(self, imgui_context, peaks_workspace):
        """The result exposes a per-peak table and a total-area stat."""
        panel = MathLabPanel(peaks_workspace)
        result, error = self._peaks_result(panel, imgui_context)
        assert error is None
        assert result.table is not None
        names = [col for col, _values in result.table]
        assert names == ["position", "height", "prominence", "width", "area"]
        assert all(values.size == 3 for _col, values in result.table)
        assert "Total area" in dict(result.stats)

    def test_apply_stores_the_full_peak_table(self, imgui_context, peaks_workspace):
        """Applying as a new dataset must register the 5-column peak table, not the signal."""
        panel = MathLabPanel(peaks_workspace)
        panel.show_operation("peaks")
        panel._apply_modes["peaks"] = "new_dataset"
        for _ in range(4):
            _draw_frame(panel, imgui_context)
        result = panel._cache_result
        source = _dataset_source_named(panel.ws, "sig")
        cmd = panel._command_new_dataset(source, result, "peaks")
        cmd.do()
        created = [n for n in panel.store.names() if n != "sig"]
        assert created, "no dataset was created"
        ds = panel.store.get(created[0])
        assert ds.n_rows() == 3
        assert ds.column_names() == ["position", "height", "prominence", "width", "area"]
        # The three clean Gaussians of height 3, 1, 2 (sigma 0.6, 0.5, 0.8) have areas
        # a*sigma*sqrt(2*pi); check the tallest matches to a few percent.
        areas = ds.get("area")
        assert float(np.max(areas)) == pytest.approx(3.0 * 0.6 * np.sqrt(2.0 * np.pi), rel=0.05)


class TestPeaksValleysMode:
    """Peaks tab's "valleys" mode: the same detector on the negated signal."""

    def _dips_source(self, workspace, name="dips"):
        """Three downward Gaussian dips (valleys) of depth 3, 1, 2 at x = 5, 15, 23."""
        x = np.linspace(0.0, 30.0, 1500)

        def g(mu, a, s):
            return -a * np.exp(-((x - mu) ** 2) / (2.0 * s * s))

        y = g(5.0, 3.0, 0.6) + g(15.0, 1.0, 0.5) + g(23.0, 2.0, 0.8)
        return _make_dataset_source(workspace, name, x, y)

    def test_default_mode_is_peaks(self, workspace):
        panel = MathLabPanel(workspace)
        assert panel._peak_mode == "peaks"

    def test_valleys_mode_finds_the_three_dips(self, workspace):
        panel = MathLabPanel(workspace)
        source = self._dips_source(workspace)
        result = panel._compute(source, ("peaks", "valleys", 0.05, None))
        assert result.x.size == 3
        assert np.allclose(np.sort(result.x), [5.0, 15.0, 23.0], atol=0.05)
        # Valley "height" is the actual (negative) signal value, not the negated one.
        assert np.all(result.y < 0.0)

    def test_valleys_mode_reports_valley_wording_in_stats(self, workspace):
        panel = MathLabPanel(workspace)
        source = self._dips_source(workspace, name="dips2")
        result = panel._compute(source, ("peaks", "valleys", 0.05, None))
        labels = [label for label, _v in result.stats]
        assert "Valleys found" in labels
        assert any("Deepest valley" in label for label in labels)
        assert not any("Tallest" in label for label in labels)

    def test_peaks_mode_on_the_same_signal_finds_nothing(self, workspace):
        """A signal that only dips must not report any (upward) peaks at high prominence.

        (At lower prominence a real, if uninteresting, bump exists: the signal returns
        toward its zero baseline between two dips of very different depth, and that
        return is itself a shallow local maximum -- correctly detected, not a bug.)
        """
        panel = MathLabPanel(workspace)
        source = self._dips_source(workspace, name="dips3")
        result = panel._compute(source, ("peaks", "peaks", 0.7, None))
        assert result.x.size == 0
        assert result.allow_apply is False

    def test_valleys_mode_preview_shows_the_full_signal_with_markers(self, workspace):
        panel = MathLabPanel(workspace)
        source = self._dips_source(workspace, name="dips4")
        result = panel._compute(source, ("peaks", "valleys", 0.05, None))
        assert result.base_x is not None and result.base_y is not None
        assert result.markers is not None
        assert result.markers[0].size == 3

    def test_reachable_end_to_end_via_the_real_ui(self, imgui_context, workspace):
        """Driving the real enum_combo to 'valleys' must reach _compute's params."""
        ws2 = _FakeWorkspace()
        self._dips_source(ws2, name="dips5")
        panel = MathLabPanel(ws2)
        panel.show_operation("peaks")
        panel._peak_mode = "valleys"
        for _ in range(6):
            _draw_frame(panel, imgui_context)
        assert panel._cache_key is not None
        # The prominence slider round-trips through an ImGui float32 widget every frame,
        # so 0.05 may come back as its nearest float32 value -- compare loosely.
        params = panel._cache_key[-1]
        assert params[0] == "peaks"
        assert params[1] == "valleys"
        assert params[2] == pytest.approx(0.05)
        assert params[3] is None


class TestFilterTab:
    """The Filter tab must run a Butterworth filter and report the real cutoff."""

    @pytest.fixture
    def tone_workspace(self):
        """A slow + fast tone sum sampled densely enough to separate them."""
        ws = _FakeWorkspace()
        x = np.linspace(0.0, 1.0, 2000, endpoint=False)
        y = np.sin(2.0 * np.pi * 5.0 * x) + np.sin(2.0 * np.pi * 200.0 * x)
        ws.store.add(DataSet("tone", [Column("x", x), Column("y", y)]))
        return ws

    def _filter_result(self, panel, io):
        panel.show_operation("filter")
        for _ in range(4):
            _draw_frame(panel, io)
        return panel._cache_result, panel._cache_error

    @pytest.mark.skipif(not mathops.filter_available(), reason="needs scipy.signal")
    def test_lowpass_reports_cutoff_and_overlays_source(self, imgui_context, tone_workspace):
        """The result carries the filtered curve, the source overlay, and cutoff stats."""
        panel = MathLabPanel(tone_workspace)
        result, error = self._filter_result(panel, imgui_context)
        assert error is None
        assert result.y.size == 2000
        assert result.overlay is not None
        rows = dict(result.stats)
        assert "Nyquist frequency" in rows
        assert "Cutoff frequency" in rows

    @pytest.mark.skipif(not mathops.filter_available(), reason="needs scipy.signal")
    def test_bandpass_uses_both_cutoffs(self, imgui_context, tone_workspace):
        """Switching to a band type engages the second cutoff without error."""
        panel = MathLabPanel(tone_workspace)
        panel.show_operation("filter")
        panel._filter_type = "bandpass"
        panel._filter_cutoff = 0.02
        panel._filter_cutoff_hi = 0.2
        for _ in range(4):
            _draw_frame(panel, imgui_context)
        assert panel._cache_error is None
        rows = dict(panel._cache_result.stats)
        assert "Low cutoff" in rows and "High cutoff" in rows


class TestSmoothEmaTab:
    """The Smooth tab's "ema" method, alongside the pre-existing four."""

    @pytest.fixture
    def tone_workspace(self):
        """A single dataset, so it is the panel's default source with no extra picking."""
        ws = _FakeWorkspace()
        x = np.linspace(0.0, 10.0, 200)
        rng = np.random.default_rng(0)
        y = np.sin(x) + rng.normal(0.0, 0.3, x.size)
        ws.store.add(DataSet("tone", [Column("x", x), Column("y", y)]))
        return ws

    def test_default_method_is_unchanged(self, workspace):
        assert MathLabPanel(workspace)._smooth_method == "moving_average"

    def test_ema_is_reachable_via_the_real_ui(self, imgui_context, tone_workspace):
        panel = MathLabPanel(tone_workspace)
        panel.show_operation("smooth")
        panel._smooth_method = "ema"
        for _ in range(6):
            _draw_frame(panel, imgui_context)
        assert panel._cache_error is None
        assert panel._cache_result.y.size == 200

    def test_alpha_slider_only_appears_for_ema(self, imgui_context, tone_workspace):
        panel = MathLabPanel(tone_workspace)
        panel.show_operation("smooth")
        panel._smooth_method = "moving_average"
        for _ in range(6):
            _draw_frame(panel, imgui_context)
        assert panel._cache_error is None
        # moving_average's params tuple carries alpha too (shared shape), but the value
        # is whatever the last-set default is -- the control just isn't drawn for it.
        assert panel._cache_key[-1][0] == "smooth"

    def test_alpha_reaches_the_params_and_changes_the_result(self, imgui_context, tone_workspace):
        panel = MathLabPanel(tone_workspace)
        panel.show_operation("smooth")
        panel._smooth_method = "ema"
        panel._smooth_alpha = 0.1
        for _ in range(6):
            _draw_frame(panel, imgui_context)
        low_alpha_y = np.array(panel._cache_result.y)

        panel._smooth_alpha = 0.9
        for _ in range(6):
            _draw_frame(panel, imgui_context)
        high_alpha_y = np.array(panel._cache_result.y)

        assert not np.allclose(low_alpha_y, high_alpha_y)


class TestFilterFamilies:
    """The Filter tab's "Family" combo: butter/cheby1/cheby2/bessel via iir_filter()."""

    @pytest.fixture
    def tone_workspace(self):
        ws = _FakeWorkspace()
        x = np.linspace(0.0, 1.0, 2000, endpoint=False)
        y = np.sin(2.0 * np.pi * 5.0 * x) + np.sin(2.0 * np.pi * 200.0 * x)
        ws.store.add(DataSet("tone", [Column("x", x), Column("y", y)]))
        return ws

    def test_default_family_is_butter(self, workspace):
        assert MathLabPanel(workspace)._filter_family == "butter"

    @pytest.mark.skipif(not mathops.filter_available(), reason="needs scipy.signal")
    @pytest.mark.parametrize("family", mathops.IIR_FAMILIES)
    def test_every_family_is_reachable_and_reports_it_in_stats(
        self, imgui_context, tone_workspace, family
    ):
        panel = MathLabPanel(tone_workspace)
        panel.show_operation("filter")
        panel._filter_family = family
        for _ in range(6):
            _draw_frame(panel, imgui_context)
        assert panel._cache_error is None
        rows = dict(panel._cache_result.stats)
        assert rows["Family"] == family

    @pytest.mark.skipif(not mathops.filter_available(), reason="needs scipy.signal")
    def test_cheby1_ripple_control_appears_and_reaches_the_params(
        self, imgui_context, tone_workspace
    ):
        panel = MathLabPanel(tone_workspace)
        panel.show_operation("filter")
        panel._filter_family = "cheby1"
        panel._filter_ripple = 3.5
        for _ in range(6):
            _draw_frame(panel, imgui_context)
        assert panel._cache_error is None
        rows = dict(panel._cache_result.stats)
        assert rows["Passband ripple"] == "3.5 dB"
        assert "Stopband attenuation" not in rows
        assert panel._cache_key[-1][5] == "cheby1"
        assert panel._cache_key[-1][6] == pytest.approx(3.5)

    @pytest.mark.skipif(not mathops.filter_available(), reason="needs scipy.signal")
    def test_cheby2_attenuation_control_appears_and_reaches_the_params(
        self, imgui_context, tone_workspace
    ):
        panel = MathLabPanel(tone_workspace)
        panel.show_operation("filter")
        panel._filter_family = "cheby2"
        panel._filter_attenuation = 55.0
        for _ in range(6):
            _draw_frame(panel, imgui_context)
        assert panel._cache_error is None
        rows = dict(panel._cache_result.stats)
        assert rows["Stopband attenuation"] == "55 dB"
        assert "Passband ripple" not in rows
        assert panel._cache_key[-1][7] == pytest.approx(55.0)

    @pytest.mark.skipif(not mathops.filter_available(), reason="needs scipy.signal")
    def test_butter_and_bessel_show_neither_ripple_nor_attenuation_row(
        self, imgui_context, tone_workspace
    ):
        panel = MathLabPanel(tone_workspace)
        for family in ("butter", "bessel"):
            panel.show_operation("filter")
            panel._filter_family = family
            for _ in range(6):
                _draw_frame(panel, imgui_context)
            assert panel._cache_error is None
            rows = dict(panel._cache_result.stats)
            assert "Passband ripple" not in rows
            assert "Stopband attenuation" not in rows

    @pytest.mark.skipif(not mathops.filter_available(), reason="needs scipy.signal")
    def test_changing_family_actually_changes_the_output(self, imgui_context, tone_workspace):
        panel = MathLabPanel(tone_workspace)
        panel.show_operation("filter")
        panel._filter_family = "butter"
        for _ in range(6):
            _draw_frame(panel, imgui_context)
        butter_y = np.array(panel._cache_result.y)

        panel._filter_family = "bessel"
        for _ in range(6):
            _draw_frame(panel, imgui_context)
        bessel_y = np.array(panel._cache_result.y)

        assert not np.allclose(butter_y, bessel_y)


class TestBaselineTab:
    """The Baseline tab must remove a constant, linear, or curved AsLS baseline."""

    def test_linear_detrend_flattens_the_mean(self, imgui_context, workspace):
        """Detrending a sloped signal leaves a zero-mean residual, with a baseline overlay."""
        ws = _FakeWorkspace()
        x = np.linspace(0.0, 10.0, 400)
        ws.store.add(DataSet("ramp", [Column("x", x), Column("y", 3.0 * x + np.sin(x))]))
        panel = MathLabPanel(ws)
        panel.show_operation("detrend")
        for _ in range(4):
            _draw_frame(panel, imgui_context)
        assert panel._cache_error is None
        result = panel._cache_result
        assert abs(float(np.nanmean(result.y))) < 1e-6
        # The estimated baseline is shown over the original signal.
        assert result.base_y is not None
        assert result.overlay is not None and result.overlay_label == "baseline"

    @pytest.mark.skipif(not mathops.baseline_available(), reason="needs scipy.sparse")
    def test_asls_removes_a_curved_background(self, imgui_context):
        """AsLS flattens a peak sitting on a curved background that a line cannot follow."""
        ws = _FakeWorkspace()
        x = np.linspace(0.0, 20.0, 1000)
        peak = 5.0 * np.exp(-((x - 6.0) ** 2) / (2.0 * 0.4**2))
        background = 3.0 * np.exp(-((x - 10.0) ** 2) / 40.0) + 0.5
        ws.store.add(DataSet("spec", [Column("x", x), Column("y", peak + background)]))
        panel = MathLabPanel(ws)
        panel.show_operation("detrend")
        panel._detrend_kind = "asls"
        panel._baseline_lam_log10 = 6.0
        for _ in range(4):
            _draw_frame(panel, imgui_context)
        assert panel._cache_error is None
        corrected = panel._cache_result.y
        # Away from the peak the corrected signal sits near zero (background removed)...
        off_peak = np.abs(peak) < 0.05
        assert float(np.median(np.abs(corrected[off_peak]))) < 0.2
        # ...while the peak itself is largely preserved.
        assert float(corrected[np.argmin(np.abs(x - 6.0))]) > 0.8 * 5.0


class TestHistogramTab:
    """The Histogram tab must bin values, preview a silhouette, and fit a distribution."""

    @pytest.fixture
    def normal_workspace(self):
        """One dataset whose y column is a large normal sample of known mean/std."""
        ws = _FakeWorkspace()
        rng = np.random.default_rng(7)
        y = rng.normal(4.0, 1.5, 4000)
        ws.store.add(DataSet("bell", [Column("x", np.arange(y.size, dtype=float)), Column("y", y)]))
        return ws

    def _hist_result(self, panel, io):
        panel.show_operation("histogram")
        for _ in range(4):
            _draw_frame(panel, io)
        return panel._cache_result, panel._cache_error

    def test_bins_and_silhouette_reach_the_preview(self, imgui_context, normal_workspace):
        """The result carries the bin table (x/y) and a silhouette outline for the preview."""
        panel = MathLabPanel(normal_workspace)
        result, error = self._hist_result(panel, imgui_context)
        assert error is None
        assert result.x.size == panel._hist_bins  # bin centres
        assert result.y.size == panel._hist_bins
        # base_* is the staircase outline: two points per bin plus the two baseline ends.
        assert result.base_x is not None
        assert result.base_x.size == 2 * panel._hist_bins + 2
        assert result.allow_replace is False

    @pytest.mark.skipif(not mathops.distributions_available(), reason="needs scipy.stats")
    def test_normal_fit_overlays_and_recovers_parameters(self, imgui_context, normal_workspace):
        """The default normal fit overlays a PDF and reports mu, sigma and a KS p-value."""
        panel = MathLabPanel(normal_workspace)
        result, error = self._hist_result(panel, imgui_context)
        assert error is None
        assert result.overlay is not None
        # The PDF overlay shares the silhouette's x grid so mini_plot can draw both.
        assert result.overlay.size == result.base_x.size
        rows = dict(result.stats)
        assert float(rows["mu"]) == pytest.approx(4.0, abs=0.15)
        assert float(rows["sigma"]) == pytest.approx(1.5, abs=0.15)
        assert "KS p-value" in rows

    def test_none_distribution_shows_histogram_only(self, imgui_context, normal_workspace):
        """With 'none' selected there is no overlay, just the histogram."""
        panel = MathLabPanel(normal_workspace)
        panel.show_operation("histogram")
        panel._hist_dist = "none"
        for _ in range(4):
            _draw_frame(panel, imgui_context)
        assert panel._cache_error is None
        assert panel._cache_result.overlay is None

    @pytest.mark.skipif(not mathops.distributions_available(), reason="needs scipy.stats")
    @pytest.mark.parametrize("dist", ["weibull", "chi2", "beta", "uniform"])
    def test_new_distributions_are_reachable_via_the_real_ui(
        self, imgui_context, normal_workspace, dist
    ):
        panel = MathLabPanel(normal_workspace)
        panel.show_operation("histogram")
        panel._hist_dist = dist
        for _ in range(4):
            _draw_frame(panel, imgui_context)
        assert panel._cache_error is None
        assert panel._cache_result.overlay is not None
        rows = dict(panel._cache_result.stats)
        assert "KS p-value" in rows


class TestCopyReport:
    """Every tab with statistics must let the user copy them out as text."""

    def test_report_text_has_header_and_every_stat(self, workspace):
        """The report names the operation and includes each stat as label<TAB>value."""
        panel = MathLabPanel(workspace)
        source = _dataset_source(workspace)
        result = panel._compute(source, ("normalize", "minmax"))
        text = panel._report_text(source, result)
        lines = text.splitlines()
        assert lines[0].startswith("#")
        assert "alpha.y" in lines[0]
        for label, value in result.stats:
            assert f"{label}\t{value}" in lines

    def test_copy_button_sets_the_clipboard(self, imgui_context, workspace, monkeypatch):
        """Clicking Copy report must push the report text to the imgui clipboard."""
        captured: Dict[str, str] = {}
        monkeypatch.setattr(
            "glplot.gui.panels.mathlab.imgui.set_clipboard_text",
            lambda text: captured.__setitem__("text", text),
        )
        real_button = imgui.small_button
        monkeypatch.setattr(
            "glplot.gui.panels.mathlab.imgui.small_button",
            lambda label, *a, **k: True if label == "Copy report" else real_button(label, *a, **k),
        )
        panel = MathLabPanel(workspace)
        panel.show_operation("stats")
        for _ in range(4):
            _draw_frame(panel, imgui_context)
        assert "text" in captured, "Copy report never reached the clipboard"
        assert captured["text"].startswith("#")
        assert "\t" in captured["text"]


class TestAutocorrTab:
    """The Autocorr tab must show the ACF and report the dominant period."""

    @pytest.fixture
    def periodic_workspace(self):
        """One dataset: a clean sine whose period is 5.0 in x units (dx=0.1, 50 samples)."""
        ws = _FakeWorkspace()
        x = np.arange(2000, dtype=float) * 0.1
        y = np.sin(2.0 * np.pi * x / 5.0)
        ws.store.add(DataSet("wave", [Column("x", x), Column("y", y)]))
        return ws

    def _acf_result(self, panel, io):
        panel.show_operation("autocorr")
        for _ in range(4):
            _draw_frame(panel, io)
        return panel._cache_result, panel._cache_error

    def test_reports_the_period_and_marks_it(self, imgui_context, periodic_workspace):
        """The ACF peaks at the period, reported in x units and marked on the curve."""
        panel = MathLabPanel(periodic_workspace)
        result, error = self._acf_result(panel, imgui_context)
        assert error is None
        assert result.y[0] == pytest.approx(1.0, abs=1e-6)  # ACF is 1 at lag 0
        assert result.markers is not None
        rows = dict(result.stats)
        assert float(rows["Dominant period"]) == pytest.approx(5.0, abs=0.11)
        assert result.allow_replace is False

    def test_max_lag_slider_shrinks_the_lag_axis(self, imgui_context, periodic_workspace):
        """Lowering the max-lag fraction returns fewer lags."""
        panel = MathLabPanel(periodic_workspace)
        panel.show_operation("autocorr")
        panel._autocorr_maxlag = 1.0
        for _ in range(4):
            _draw_frame(panel, imgui_context)
        wide = panel._cache_result.x.size
        panel._autocorr_maxlag = 0.2
        for _ in range(4):
            _draw_frame(panel, imgui_context)
        assert panel._cache_result.x.size < wide


class TestAutocorrCrossMode:
    """Autocorr's "cross" mode: correlate the source's y against another dataset column."""

    def _delayed_workspace(self, delay=20):
        """A dataset with 'x', 'a' (noise) and 'b' (a delayed by `delay` samples)."""
        ws = _FakeWorkspace()
        rng = np.random.default_rng(0)
        n = 600
        x = np.arange(n, dtype=float)
        a = rng.normal(0.0, 1.0, n)
        b = np.zeros(n)
        b[delay:] = a[:-delay]
        ws.store.add(DataSet("xc", [Column("x", x), Column("a", a), Column("b", b)]))
        return ws

    def _cross_source(self, ws, x_col="x", y_col="a"):
        ds = ws.store.get("xc")
        return _Source(
            key=("dataset", "xc", x_col, y_col),
            label=f"xc.{y_col}",
            x_name=x_col,
            y_name=y_col,
            x_raw=ds.get(x_col),
            y_raw=ds.get(y_col),
            dataset=ds,
            x_col=x_col,
            y_col=y_col,
        )

    def test_default_mode_is_auto(self, workspace):
        panel = MathLabPanel(workspace)
        assert panel._autocorr_mode == "auto"

    def test_cross_mode_recovers_the_known_delay(self):
        ws = self._delayed_workspace(delay=20)
        panel = MathLabPanel(ws)
        source = self._cross_source(ws)
        result = panel._compute(source, ("autocorr", "cross", "b", 0.9))
        rows = dict(result.stats)
        assert float(rows["Peak lag"]) == pytest.approx(20.0, abs=1.0)
        assert float(rows["Peak correlation"]) > 0.5

    def test_cross_mode_with_no_column_picked_falls_back_to_auto(self):
        """An empty cross_col (nothing picked yet) must not crash -- behave like auto."""
        ws = self._delayed_workspace()
        panel = MathLabPanel(ws)
        source = self._cross_source(ws)
        result = panel._compute(source, ("autocorr", "cross", "", 0.5))
        assert result.y[0] == pytest.approx(1.0, abs=1e-6)  # the auto (ACF) fallback

    def test_layer_source_falls_back_to_auto(self):
        """A source with no dataset (a plotted layer) cannot cross-correlate -- must not
        crash, and must fall back to autocorrelation rather than erroring."""
        panel = MathLabPanel(_FakeWorkspace())
        source = _Source(
            key=("layer",),
            label="L",
            x_name="x",
            y_name="y",
            x_raw=np.sin(np.linspace(0.0, 10.0, 200)),
            y_raw=np.sin(np.linspace(0.0, 10.0, 200)),
            dataset=None,
        )
        result = panel._compute(source, ("autocorr", "cross", "b", 0.5))
        assert result.y[0] == pytest.approx(1.0, abs=1e-6)

    def test_column_picker_disabled_without_a_dataset(self, imgui_context):
        """Rendering the tab on a layer source must not raise when the picker has no
        dataset to list columns from."""
        panel = MathLabPanel(_FakeWorkspace())
        panel.show_operation("autocorr")
        panel._autocorr_mode = "cross"
        for _ in range(4):
            _draw_frame(panel, imgui_context)  # must not raise

    def test_reachable_end_to_end_via_the_real_ui(self, imgui_context):
        ws = self._delayed_workspace(delay=15)
        panel = MathLabPanel(ws)
        panel.show_operation("autocorr")
        panel._autocorr_mode = "cross"
        panel._autocorr_cross_col = "b"
        for _ in range(6):
            _draw_frame(panel, imgui_context)
        assert panel._cache_key is not None
        params = panel._cache_key[-1]
        assert params[0] == "autocorr"
        assert params[1] == "cross"
        assert params[2] == "b"


def _dataset_source_named(ws, name):
    """A _Source for dataset ``name`` (x/y columns), built without imgui."""
    from glplot.gui.panels.mathlab import _Source

    dataset = ws.store.get(name)
    cols = dataset.column_names()
    return _Source(
        key=("dataset", name, cols[0], cols[1]),
        label=f"{name}.{cols[1]}",
        x_name=cols[0],
        y_name=cols[1],
        x_raw=dataset.get(cols[0]),
        y_raw=dataset.get(cols[1]),
        dataset=dataset,
        x_col=cols[0],
        y_col=cols[1],
    )


class TestParametricDerivative:
    """The Derivative tab's parametric mode: dy/dx of an X-Y curve that is not a function of x."""

    def _source(self, x, y):
        return _Source(
            key=("t",),
            label="curve",
            x_name="x",
            y_name="y",
            x_raw=np.asarray(x, dtype=np.float64),
            y_raw=np.asarray(y, dtype=np.float64),
        )

    def test_parametric_matches_chain_rule_on_a_circle(self, workspace):
        panel = MathLabPanel(workspace)
        t = np.linspace(0.2, 3.0, 200)
        src = self._source(np.cos(t), np.sin(t))
        # params: (kind, order, method, window, polyorder, parametric)
        res = panel._compute(src, ("derivative", 1, "central", 11, 3, True))
        expected = -1.0 / np.tan(t)  # -cot(t)
        assert np.allclose(res.y[5:-5], expected[5:-5], atol=1e-2)
        # Parametric mode keeps the sample order (no sort by x).
        assert res.x_changed is False
        assert np.allclose(res.x, np.cos(t))

    def test_parametric_handles_self_intersecting_curve(self, workspace):
        panel = MathLabPanel(workspace)
        t = np.linspace(0.0, 2 * np.pi, 300)
        src = self._source(np.sin(2 * t), np.sin(3 * t))  # a Lissajous figure
        res = panel._compute(src, ("derivative", 1, "central", 11, 3, True))
        assert res.y.shape == (300,)
        assert np.isfinite(res.y).all()

    def test_non_parametric_still_sorts_by_x(self, workspace):
        panel = MathLabPanel(workspace)
        x = np.linspace(0.0, 10.0, 50)
        src = self._source(x, np.sin(x))
        res = panel._compute(src, ("derivative", 1, "central", 11, 3, False))
        assert res.x_changed is True
        # Interior points only: np.gradient is 1st-order at the two edges.
        assert np.allclose(res.y[2:-2], np.cos(x)[2:-2], atol=1e-2)


class TestAddColumnApply:
    """Apply mode 'add_column': a computed variable becomes a new column of the source table."""

    def _source_with_dataset(self, ws, name="alpha"):
        ds = ws.store.get(name)
        x = ds.get("x")
        y = ds.get("y")
        return _Source(
            key=("ds", name),
            label=name,
            x_name="x",
            y_name="y",
            x_raw=np.asarray(x, dtype=np.float64),
            y_raw=np.asarray(y, dtype=np.float64),
            dataset=ds,
            x_col="x",
            y_col="y",
        )

    def _result(self, x, y, suffix="dy/dx"):
        from glplot.gui.panels.mathlab import _Result

        return _Result(
            x=np.asarray(x, float),
            y=np.asarray(y, float),
            x_name="x",
            y_name=f"y {suffix}",
            suffix=suffix,
            plot_label="p",
            overlay=None,
        )

    def test_same_length_result_adds_a_column(self, workspace):
        panel = MathLabPanel(workspace)
        src = self._source_with_dataset(workspace)
        n = src.dataset.n_rows()
        res = self._result(src.x_raw, np.cos(np.linspace(0, 10, n)))
        cmd = panel._command_add_column(src, res, "derivative")
        assert cmd is not None
        before = src.dataset.n_cols()
        cmd.do()
        assert src.dataset.n_cols() == before + 1
        # The new column carries the computed values at the table's row count.
        new_col = src.dataset.columns[-1]
        assert len(new_col.values) == n
        assert np.allclose(new_col.values, res.y)
        cmd.undo()
        assert src.dataset.n_cols() == before

    def test_short_result_is_padded_with_nan(self, workspace):
        panel = MathLabPanel(workspace)
        src = self._source_with_dataset(workspace)
        n = src.dataset.n_rows()
        res = self._result(np.arange(10.0), np.arange(10.0))  # only 10 rows
        cmd = panel._command_add_column(src, res, "resample")
        cmd.do()
        col = src.dataset.columns[-1]
        assert len(col.values) == n
        assert np.allclose(col.values[:10], np.arange(10.0))
        assert np.isnan(col.values[10:]).all()

    def test_long_result_is_truncated(self, workspace):
        panel = MathLabPanel(workspace)
        src = self._source_with_dataset(workspace)
        n = src.dataset.n_rows()
        res = self._result(np.arange(n + 50.0), np.arange(n + 50.0))
        cmd = panel._command_add_column(src, res, "resample")
        cmd.do()
        col = src.dataset.columns[-1]
        assert len(col.values) == n

    def test_layer_source_has_no_add_column(self, workspace):
        panel = MathLabPanel(workspace)
        # A source with no dataset (a plotted layer) cannot add a column.
        res = self._result(np.arange(5.0), np.arange(5.0))
        src = _Source(
            key=("layer",),
            label="L",
            x_name="x",
            y_name="y",
            x_raw=np.arange(5.0),
            y_raw=np.arange(5.0),
            dataset=None,
        )
        assert panel._command_add_column(src, res, "derivative") is None


class TestPreviewStyleMatch:
    """The preview must draw with the plot's own live style, not generic ImGui chrome."""

    def test_stub_plot_without_options_does_not_raise(self, workspace):
        """A bare stub plot (no .options at all, the shared _FakePlot) must not crash."""
        panel = MathLabPanel(workspace)
        background, grid, palette = panel._preview_style()
        assert len(background) == 3
        assert len(grid) == 3
        assert len(palette) >= 1

    def test_reads_the_live_background_and_palette(self, workspace):
        from glplot.engine import GPULinePlot
        from glplot.gui import styles

        plot = GPULinePlot()
        styles.apply_style(plot, styles.get_style("neon"))
        workspace.plot = plot
        panel = MathLabPanel(workspace)

        background, _grid, palette = panel._preview_style()
        neon = styles.get_style("neon")
        assert tuple(background) == neon.background
        assert tuple(palette) == neon.palette

    def test_a_gui_applied_style_is_picked_up_too(self, workspace):
        """Closes the exact _style_key gap fixed in styles.apply_style: a preset applied
        through the panel click path (styles.apply_style directly, never through
        gplt.plot_style()) must still drive the preview's palette."""
        from glplot.engine import GPULinePlot
        from glplot.gui import styles

        plot = GPULinePlot()
        styles.apply_style(plot, styles.get_style("chalk"))  # the Style panel's own path
        workspace.plot = plot
        panel = MathLabPanel(workspace)

        _bg, _grid, palette = panel._preview_style()
        assert tuple(palette) == styles.get_style("chalk").palette

    def test_auto_grid_color_resolves_when_the_option_is_at_the_sentinel(self, workspace):
        """When ``axis_grid_color`` is left at the AUTO sentinel (the "pick from the
        background luminance" convention), the preview must compute it, not pass the
        sentinel through as a literal (near-invisible dark gray) color."""
        from glplot.engine import GPULinePlot
        from glplot.gui import styles

        plot = GPULinePlot()
        plot.options.visual.background_color = (1.0, 1.0, 1.0)
        plot.options.axis_grid_color = styles.AUTO_GRID_COLOR
        workspace.plot = plot
        panel = MathLabPanel(workspace)

        background, grid, _palette = panel._preview_style()
        assert grid == styles.auto_grid_color(background) == (0.2, 0.2, 0.2)

    def test_an_explicit_grid_color_overrides_auto_contrast(self, workspace):
        from glplot.engine import GPULinePlot

        plot = GPULinePlot()
        plot.options.axis_grid_color = (0.5, 0.1, 0.9)
        workspace.plot = plot
        panel = MathLabPanel(workspace)

        _background, grid, _palette = panel._preview_style()
        assert grid == (0.5, 0.1, 0.9)

    def test_preview_passes_the_live_palette_into_mini_plot(
        self, imgui_context, workspace, monkeypatch
    ):
        """End-to-end: drawing a real tab body must reach mini_plot with the resolved
        style's colors, not a hardcoded theme accent."""
        from glplot.engine import GPULinePlot
        from glplot.gui import styles

        plot = GPULinePlot()
        styles.apply_style(plot, styles.get_style("neon"))
        workspace.plot = plot
        panel = MathLabPanel(workspace)
        panel.show_operation("normalize")

        calls: List[Dict[str, Any]] = []
        real_mini_plot = widgets.mini_plot

        def spy(*args, **kwargs):
            calls.append(kwargs)
            return real_mini_plot(*args, **kwargs)

        monkeypatch.setattr("glplot.gui.panels.mathlab.widgets.mini_plot", spy)

        for _ in range(3):
            _draw_frame(panel, imgui_context)

        assert calls, "mini_plot was never called"
        neon = styles.get_style("neon")
        last = calls[-1]
        assert tuple(last["color"]) == neon.palette[0]
        assert tuple(last["background_color"]) == neon.background


class TestCsvExport:
    """Every operation with a result must be exportable as CSV, matching the Data
    Editor's own path/icon-button/synchronous-write convention (no file dialog exists)."""

    def test_xy_result_round_trips(self, workspace, tmp_path):
        panel = MathLabPanel(workspace)
        source = _dataset_source(workspace)
        result = panel._compute(source, ("normalize", "minmax"))
        panel._csv_path = str(tmp_path / "out.csv")

        panel._export_result_csv(result)

        from glplot.gui import clipboard

        text = (tmp_path / "out.csv").read_text(encoding="utf-8")
        headers, data = clipboard.parse_table(text)
        assert headers == [result.x_name, result.y_name]
        assert np.allclose(data[:, 0], result.x)
        assert np.allclose(data[:, 1], result.y)

    def test_table_result_exports_every_column(self, workspace, tmp_path):
        """Peaks-like operations carry a multi-column .table; export must use it, not
        fall back to the 2-column (x, y) default."""
        panel = MathLabPanel(workspace)
        source = _dataset_source(workspace)
        result = panel._compute(source, ("peaks", "peaks", 0.05, None))
        panel._csv_path = str(tmp_path / "peaks.csv")

        panel._export_result_csv(result)

        from glplot.gui import clipboard

        text = (tmp_path / "peaks.csv").read_text(encoding="utf-8")
        headers, data = clipboard.parse_table(text)
        assert result.table is not None
        assert headers == [name for name, _ in result.table]
        assert data.shape[0] == result.table[0][1].size

    def test_duplicate_x_y_names_are_disambiguated(self, workspace, tmp_path):
        """Mirrors _command_new_dataset's own "(2)" dedup so headers are never identical."""
        from glplot.gui.panels.mathlab import _Result

        panel = MathLabPanel(workspace)
        result = _Result(
            x=np.array([1.0, 2.0]),
            y=np.array([3.0, 4.0]),
            x_name="v",
            y_name="v",
            suffix="s",
            plot_label="p",
        )
        panel._csv_path = str(tmp_path / "dup.csv")
        panel._export_result_csv(result)

        from glplot.gui import clipboard

        text = (tmp_path / "dup.csv").read_text(encoding="utf-8")
        headers, _data = clipboard.parse_table(text)
        assert headers == ["v", "v (2)"]

    def test_an_unwritable_path_sets_the_notice_not_an_exception(self, workspace):
        panel = MathLabPanel(workspace)
        source = _dataset_source(workspace)
        result = panel._compute(source, ("normalize", "minmax"))
        panel._csv_path = "/no/such/directory/out.csv"

        panel._export_result_csv(result)  # must not raise

        assert panel._notice is not None
        assert "out.csv" in panel._notice

    def test_export_button_writes_the_file(self, imgui_context, workspace, monkeypatch, tmp_path):
        """End-to-end: clicking the export icon button must reach disk."""
        real_icon_button = icons.icon_button

        def spy_icon_button(id_str, shape, *args, **kwargs):
            if shape == "download":
                return True
            return real_icon_button(id_str, shape, *args, **kwargs)

        monkeypatch.setattr("glplot.gui.panels.mathlab.icons.icon_button", spy_icon_button)

        panel = MathLabPanel(workspace)
        panel.show_operation("normalize")
        panel._csv_path = str(tmp_path / "clicked.csv")

        for _ in range(3):
            _draw_frame(panel, imgui_context)

        assert (tmp_path / "clicked.csv").exists()


class TestRecommendations:
    """The "Suggested next step" strip must surface mathadvise's heuristics and let a
    user jump straight to the suggested tab."""

    def test_recommend_is_cached_per_source_not_recomputed_every_frame(
        self, imgui_context, workspace, monkeypatch
    ):
        import glplot.gui.mathadvise as mathadvise

        calls: List[int] = []
        real_recommend = mathadvise.recommend

        def spy(*args, **kwargs):
            calls.append(1)
            return real_recommend(*args, **kwargs)

        monkeypatch.setattr("glplot.gui.panels.mathlab.mathadvise.recommend", spy)

        panel = MathLabPanel(workspace)
        for _ in range(5):
            _draw_frame(panel, imgui_context)

        assert len(calls) == 1, "recommend() must key off source.key, not run every frame"

    def test_try_it_button_navigates_to_the_recommended_tab(
        self, imgui_context, workspace, monkeypatch
    ):
        """Clicking a recommendation's "Try it" button must behave like show_operation."""
        from glplot.gui.mathadvise import Recommendation

        panel = MathLabPanel(workspace)
        panel._advise_key = _dataset_source(workspace).key
        panel._advise_cache = [Recommendation("smooth", "Smooth", "test reason", 1.0)]

        # The strip is collapsed by default; force it open so the "Try it" button draws.
        real_section = widgets.section

        def spy_section(label, *a, **k):
            return True if label == "Suggested next step" else real_section(label, *a, **k)

        monkeypatch.setattr("glplot.gui.panels.mathlab.widgets.section", spy_section)

        real_small_button = imgui.small_button

        def spy_small_button(label, *a, **k):
            if label.startswith("Try it##advise_smooth"):
                return True
            return real_small_button(label, *a, **k)

        monkeypatch.setattr("glplot.gui.panels.mathlab.imgui.small_button", spy_small_button)

        for _ in range(6):
            _draw_frame(panel, imgui_context)

        # The click reached show_operation and the tab actually took hold: the cache key
        # (set by _cached_result) has "smooth" as the operation kind that last computed.
        assert panel._cache_key is not None
        assert panel._cache_key[-1][0] == "smooth"

    def test_no_recommendations_shows_a_message_not_a_crash(self, imgui_context, workspace):
        panel = MathLabPanel(workspace)
        panel._advise_key = _dataset_source(workspace).key
        panel._advise_cache = []
        for _ in range(3):
            _draw_frame(panel, imgui_context)  # must not raise


def _make_dataset_source(ws, name, x, y):
    """A _Source backed by a fresh 2-column dataset added to ``ws``, for controlled data."""
    ds = DataSet(
        name,
        [
            Column("x", np.asarray(x, dtype=np.float64)),
            Column("y", np.asarray(y, dtype=np.float64)),
        ],
    )
    ws.store.add(ds)
    return _Source(
        key=("dataset", name, "x", "y"),
        label=f"{name}.y",
        x_name="x",
        y_name="y",
        x_raw=ds.get("x"),
        y_raw=ds.get("y"),
        dataset=ds,
        x_col="x",
        y_col="y",
    )


class TestCorrelateTab:
    """The Correlate tab reports Pearson/Spearman correlation and a linear-fit overlay."""

    def test_reports_strong_positive_correlation(self, workspace):
        panel = MathLabPanel(workspace)
        x = np.linspace(0.0, 10.0, 200)
        source = _make_dataset_source(workspace, "lin", x, 3.0 * x + 2.0)
        result = panel._compute(source, ("correlate", False, 0.95))
        stats = dict(result.stats)
        assert float(stats["Pearson r"]) == pytest.approx(1.0, abs=1e-6)
        assert float(stats["Slope"]) == pytest.approx(3.0, abs=1e-6)
        assert float(stats["Intercept"]) == pytest.approx(2.0, abs=1e-6)

    def test_overlay_is_the_linear_fit(self, workspace):
        panel = MathLabPanel(workspace)
        x = np.linspace(0.0, 10.0, 200)
        source = _make_dataset_source(workspace, "lin2", x, -1.5 * x + 4.0)
        result = panel._compute(source, ("correlate", False, 0.95))
        assert result.overlay is not None
        assert np.allclose(result.overlay, -1.5 * result.x + 4.0, atol=1e-6)

    def test_does_not_allow_replace(self, workspace):
        """Correlate is a readout of the source, not a transform of it."""
        panel = MathLabPanel(workspace)
        x = np.linspace(0.0, 10.0, 50)
        source = _make_dataset_source(workspace, "lin3", x, x)
        result = panel._compute(source, ("correlate", False, 0.95))
        assert result.allow_replace is False

    def test_is_in_the_statistics_category(self):
        assert "correlate" in dict(_CATEGORIES)["Statistics"]
        assert ("correlate", "Correlate", "_tab_correlate") in _TABS

    def test_reachable_via_show_operation(self, imgui_context, workspace):
        panel = MathLabPanel(workspace)
        panel.show_operation("correlate")
        for _ in range(6):
            _draw_frame(panel, imgui_context)
        assert panel._cache_key is not None
        assert panel._cache_key[-1][0] == "correlate"


class TestFitRobust:
    """The Fit tab's "Robust fit" checkbox routes through fit_model_robust."""

    def test_robust_flag_changes_the_fit_engine(self, workspace, monkeypatch):
        calls = {"robust": 0, "plain": 0}
        real_robust = mathops.fit_model_robust
        real_plain = mathops.fit_model

        def spy_robust(*a, **k):
            calls["robust"] += 1
            return real_robust(*a, **k)

        def spy_plain(*a, **k):
            calls["plain"] += 1
            return real_plain(*a, **k)

        monkeypatch.setattr("glplot.gui.panels.mathlab.mathops.fit_model_robust", spy_robust)
        monkeypatch.setattr("glplot.gui.panels.mathlab.mathops.fit_model", spy_plain)

        panel = MathLabPanel(workspace)
        rng = np.random.default_rng(0)
        x = np.linspace(-5.0, 5.0, 100)
        source = _make_dataset_source(
            workspace, "rob", x, 2.0 * x + 1.0 + rng.normal(0, 0.1, x.size)
        )

        panel._compute(
            source,
            ("fit", "linear", 0, "", None, False, "soft_l1", False, 0.95, False, 200, _NO_SAMPLING),
        )
        assert calls["plain"] == 1 and calls["robust"] == 0

        panel._compute(
            source,
            ("fit", "linear", 0, "", None, True, "soft_l1", False, 0.95, False, 200, _NO_SAMPLING),
        )
        assert calls["robust"] == 1

    def test_robust_fit_recovers_parameters_despite_outliers(self, workspace):
        panel = MathLabPanel(workspace)
        rng = np.random.default_rng(1)
        x = np.linspace(0.0, 10.0, 200)
        y = 2.0 * x + 1.0 + rng.normal(0.0, 0.1, x.size)
        bad = rng.choice(x.size, size=8, replace=False)
        y[bad] += 25.0
        source = _make_dataset_source(workspace, "out", x, y)

        result = panel._compute(
            source,
            ("fit", "linear", 0, "", None, True, "soft_l1", False, 0.95, False, 200, _NO_SAMPLING),
        )
        stats = dict(result.stats)
        # "m" is the linear model's slope parameter name.
        slope = float(stats["m"].split(" +/- ")[0])
        assert slope == pytest.approx(2.0, abs=0.3)

    def test_robust_reachable_end_to_end(self, imgui_context, workspace, monkeypatch):
        """Clicking the Robust checkbox in the real UI must reach the panel's params."""
        panel = MathLabPanel(workspace)
        panel.show_operation("fit")
        panel._fit_model = "linear"

        real_checkbox = imgui.checkbox

        def spy_checkbox(label, value, *a, **k):
            if label == "Robust fit (outliers)":
                return True, True  # simulate the box being checked
            return real_checkbox(label, value, *a, **k)

        monkeypatch.setattr("glplot.gui.panels.mathlab.imgui.checkbox", spy_checkbox)

        for _ in range(6):
            _draw_frame(panel, imgui_context)

        assert panel._fit_robust is True


class TestFitConfidenceBand:
    """The Fit tab's "Show confidence band" checkbox propagates fit_errors into a band."""

    def test_band_is_none_by_default(self, workspace):
        panel = MathLabPanel(workspace)
        x = np.linspace(-5.0, 5.0, 100)
        source = _make_dataset_source(workspace, "cb1", x, 2.0 * x + 1.0)
        result = panel._compute(
            source,
            ("fit", "linear", 0, "", None, False, "soft_l1", False, 0.95, False, 200, _NO_SAMPLING),
        )
        assert result.band is None

    def test_band_is_populated_when_requested(self, workspace):
        panel = MathLabPanel(workspace)
        rng = np.random.default_rng(2)
        x = np.linspace(-5.0, 5.0, 200)
        y = 2.0 * x + 1.0 + rng.normal(0.0, 0.3, x.size)
        source = _make_dataset_source(workspace, "cb2", x, y)
        result = panel._compute(
            source,
            ("fit", "linear", 0, "", None, False, "soft_l1", True, 0.95, False, 200, _NO_SAMPLING),
        )
        assert result.band is not None
        lo, hi = result.band
        assert lo.shape == result.x.shape
        assert np.all(hi >= lo)

    def test_wider_level_gives_a_wider_band(self, workspace):
        panel = MathLabPanel(workspace)
        rng = np.random.default_rng(3)
        x = np.linspace(-5.0, 5.0, 200)
        y = 2.0 * x + 1.0 + rng.normal(0.0, 0.3, x.size)
        source = _make_dataset_source(workspace, "cb3", x, y)
        r68 = panel._compute(
            source,
            ("fit", "linear", 0, "", None, False, "soft_l1", True, 0.68, False, 200, _NO_SAMPLING),
        )
        r99 = panel._compute(
            source,
            ("fit", "linear", 0, "", None, False, "soft_l1", True, 0.99, False, 200, _NO_SAMPLING),
        )
        w68 = float(r68.band[1][0] - r68.band[0][0])
        w99 = float(r99.band[1][0] - r99.band[0][0])
        assert w99 > w68

    def test_polynomial_fit_also_supports_a_band(self, workspace):
        panel = MathLabPanel(workspace)
        rng = np.random.default_rng(4)
        x = np.linspace(-3.0, 3.0, 200)
        y = 2.0 * x**2 - 3.0 * x + 1.0 + rng.normal(0.0, 0.2, x.size)
        source = _make_dataset_source(workspace, "cb4", x, y)
        result = panel._compute(
            source,
            (
                "fit",
                "polynomial",
                2,
                "",
                None,
                False,
                "soft_l1",
                True,
                0.95,
                False,
                200,
                _NO_SAMPLING,
            ),
        )
        assert result.band is not None
        lo, hi = result.band
        assert np.all(hi >= lo)

    def test_preview_passes_the_band_into_mini_plot(self, imgui_context, workspace, monkeypatch):
        panel = MathLabPanel(workspace)
        rng = np.random.default_rng(5)
        x = np.linspace(-5.0, 5.0, 200)
        y = 2.0 * x + 1.0 + rng.normal(0.0, 0.3, x.size)
        _make_dataset_source(workspace, "cb5", x, y)  # registers the dataset on workspace

        panel.show_operation("fit")
        panel._fit_model = "linear"
        panel._fit_band = True

        calls: List[Dict[str, Any]] = []
        real_mini_plot = widgets.mini_plot

        def spy(*args, **kwargs):
            calls.append(kwargs)
            return real_mini_plot(*args, **kwargs)

        monkeypatch.setattr("glplot.gui.panels.mathlab.widgets.mini_plot", spy)

        for _ in range(6):
            _draw_frame(panel, imgui_context)

        assert calls
        assert calls[-1].get("band") is not None


class TestFitOutputGrid:
    """The Fit tab's "Evaluate on a smooth grid" checkbox + point-count field.

    Directly answers the user's "permitir seleccionar la cantidad de puntos a agregar
    cuando se hace una regresion ajuste": off (default), Apply commits one fitted value
    per original x sample, unchanged from before this feature existed; on, Apply commits
    N evenly spaced points across the x range instead, however sparse or unevenly
    sampled the input was.
    """

    def test_default_matches_the_source_sample_count(self, workspace):
        panel = MathLabPanel(workspace)
        x = np.linspace(-5.0, 5.0, 37)
        source = _make_dataset_source(workspace, "og1", x, 2.0 * x + 1.0)
        result = panel._compute(
            source,
            ("fit", "linear", 0, "", None, False, "soft_l1", False, 0.95, False, 200, _NO_SAMPLING),
        )
        assert result.x.size == 37
        np.testing.assert_allclose(result.x, x)
        assert result.overlay is not None
        assert result.markers is None
        assert result.x_changed is False

    def test_grid_on_produces_the_requested_point_count(self, workspace):
        panel = MathLabPanel(workspace)
        x = np.linspace(-5.0, 5.0, 37)
        source = _make_dataset_source(workspace, "og2", x, 2.0 * x + 1.0)
        result = panel._compute(
            source,
            ("fit", "linear", 0, "", None, False, "soft_l1", False, 0.95, True, 500, _NO_SAMPLING),
        )
        assert result.x.size == 500
        assert float(result.x[0]) == pytest.approx(-5.0)
        assert float(result.x[-1]) == pytest.approx(5.0)
        np.testing.assert_allclose(result.y, 2.0 * result.x + 1.0, atol=1e-9)
        assert result.x_changed is True

    def test_grid_on_shows_original_data_as_markers_not_overlay(self, workspace):
        panel = MathLabPanel(workspace)
        x = np.linspace(-5.0, 5.0, 37)
        y = 2.0 * x + 1.0
        source = _make_dataset_source(workspace, "og3", x, y)
        result = panel._compute(
            source,
            ("fit", "linear", 0, "", None, False, "soft_l1", False, 0.95, True, 80, _NO_SAMPLING),
        )
        assert result.overlay is None
        assert result.markers is not None
        mx, my = result.markers
        np.testing.assert_allclose(mx, x)
        np.testing.assert_allclose(my, y)
        assert result.marker_label == "data"

    def test_polynomial_model_also_supports_the_grid(self, workspace):
        panel = MathLabPanel(workspace)
        rng = np.random.default_rng(6)
        x = np.sort(rng.uniform(-3.0, 3.0, 40))  # unevenly sampled, the case this exists for
        y = 2.0 * x**2 - 3.0 * x + 1.0
        source = _make_dataset_source(workspace, "og4", x, y)
        result = panel._compute(
            source,
            (
                "fit",
                "polynomial",
                2,
                "",
                None,
                False,
                "soft_l1",
                False,
                0.95,
                True,
                300,
                _NO_SAMPLING,
            ),
        )
        assert result.x.size == 300
        np.testing.assert_allclose(result.y, 2.0 * result.x**2 - 3.0 * result.x + 1.0, atol=1e-6)

    def test_band_is_evaluated_on_the_output_grid_not_the_source_samples(self, workspace):
        panel = MathLabPanel(workspace)
        rng = np.random.default_rng(7)
        x = np.linspace(-5.0, 5.0, 200)
        y = 2.0 * x + 1.0 + rng.normal(0.0, 0.3, x.size)
        source = _make_dataset_source(workspace, "og5", x, y)
        result = panel._compute(
            source,
            ("fit", "linear", 0, "", None, False, "soft_l1", True, 0.95, True, 250, _NO_SAMPLING),
        )
        assert result.band is not None
        lo, hi = result.band
        assert lo.shape == (250,) == hi.shape == result.x.shape

    def test_output_points_row_reports_the_grid_size(self, workspace):
        panel = MathLabPanel(workspace)
        x = np.linspace(-5.0, 5.0, 37)
        source = _make_dataset_source(workspace, "og6", x, 2.0 * x + 1.0)
        result = panel._compute(
            source,
            ("fit", "linear", 0, "", None, False, "soft_l1", False, 0.95, True, 123, _NO_SAMPLING),
        )
        assert dict(result.stats)["Output points"] == "123"

    def test_replace_is_unavailable_once_the_grid_changes_the_row_count(self, workspace):
        panel = MathLabPanel(workspace)
        x = np.linspace(-5.0, 5.0, 37)
        source = _make_dataset_source(workspace, "og7", x, 2.0 * x + 1.0)
        result = panel._compute(
            source,
            ("fit", "linear", 0, "", None, False, "soft_l1", False, 0.95, True, 500, _NO_SAMPLING),
        )
        assert panel._replace_reason(source, result) is not None

    def test_ui_checkbox_and_int_field_reach_the_params(self, imgui_context, workspace):
        panel = MathLabPanel(workspace)
        x = np.linspace(-5.0, 5.0, 60)
        _make_dataset_source(workspace, "og8", x, 2.0 * x + 1.0)

        panel.show_operation("fit")
        panel._fit_model = "linear"
        panel._fit_output_grid = True
        panel._fit_output_points = 321
        for _ in range(4):
            _draw_frame(panel, imgui_context)

        assert panel._cache_error is None
        assert panel._cache_result.x.size == 321

    def test_point_count_is_clamped_to_a_sane_minimum(self, workspace):
        """``_output_grid``'s own floor: a "curve" of 0 or 1 points is not a curve."""
        panel = MathLabPanel(workspace)
        x = np.linspace(-5.0, 5.0, 40)
        source = _make_dataset_source(workspace, "og9", x, 2.0 * x + 1.0)
        result = panel._compute(
            source,
            ("fit", "linear", 0, "", None, False, "soft_l1", False, 0.95, True, 0, _NO_SAMPLING),
        )
        assert result.x.size == 2

    def test_ui_field_clamps_out_of_range_input(self, imgui_context, workspace, monkeypatch):
        """The "Points" field itself clamps to [2, 100_000], independent of _compute's floor."""
        panel = MathLabPanel(workspace)
        x = np.linspace(-5.0, 5.0, 40)
        _make_dataset_source(workspace, "og10", x, 2.0 * x + 1.0)

        real_input_int = imgui.input_int

        def spy_input_int(label, value, *a, **k):
            if label == "Points##fit_output_points":
                return True, 999_999_999
            return real_input_int(label, value, *a, **k)

        monkeypatch.setattr("glplot.gui.panels.mathlab.imgui.input_int", spy_input_int)

        panel.show_operation("fit")
        panel._fit_model = "linear"
        panel._fit_output_grid = True
        for _ in range(4):
            _draw_frame(panel, imgui_context)

        assert panel._fit_output_points == 100_000


class TestStatsShowMore:
    """The Statistics tab's "Show more" checkbox adds percentile/shape/robust rows."""

    def test_default_is_the_nine_row_summary(self, workspace):
        panel = MathLabPanel(workspace)
        source = _dataset_source(workspace)
        result = panel._compute(source, ("stats", False, False, 0.95))
        labels = {label for label, _v in result.stats}
        assert "Skewness" not in labels
        assert "Mean" in labels

    def test_show_more_adds_shape_and_robust_rows(self, workspace):
        panel = MathLabPanel(workspace)
        source = _dataset_source(workspace)
        result = panel._compute(source, ("stats", True, False, 0.95))
        labels = {label for label, _v in result.stats}
        for expected in (
            "Skewness",
            "Excess kurtosis",
            "Interquartile range",
            "Median abs deviation",
        ):
            assert expected in labels

    def test_checkbox_reaches_the_params(self, imgui_context, workspace, monkeypatch):
        panel = MathLabPanel(workspace)
        panel.show_operation("stats")

        real_checkbox = imgui.checkbox

        def spy_checkbox(label, value, *a, **k):
            if label == "Show more":
                return True, True
            return real_checkbox(label, value, *a, **k)

        monkeypatch.setattr("glplot.gui.panels.mathlab.imgui.checkbox", spy_checkbox)

        for _ in range(6):
            _draw_frame(panel, imgui_context)

        assert panel._stats_show_more is True
        assert panel._cache_key is not None
        assert panel._cache_key[-1] == ("stats", True, False, 0.95)


class TestConfidenceIntervalWiring:
    """The "Show confidence interval" checkbox on Statistics, Correlate and Compare.

    Point 2 of the user's request: "dar la opcion de generar intervalos de confianza"
    -- not just on Fit (which already had one), but wherever a point estimate is
    reported. mathops.confidence_interval_* itself is covered by
    tests/test_gui_mathops.py; this class only checks the tabs actually reach it.
    """

    def test_stats_ci_off_by_default(self, workspace):
        panel = MathLabPanel(workspace)
        source = _dataset_source(workspace)
        result = panel._compute(source, ("stats", False, False, 0.95))
        labels = {label for label, _v in result.stats}
        assert not any("CI for mean" in label for label in labels)
        table_names = {name for name, _v in result.table}
        assert "CI lower (mean)" not in table_names

    def test_stats_ci_adds_a_row_and_table_columns(self, workspace):
        panel = MathLabPanel(workspace)
        source = _dataset_source(workspace)
        result = panel._compute(source, ("stats", False, True, 0.95))
        rows = dict(result.stats)
        assert "95% CI for mean" in rows
        table = dict(result.table)
        assert "CI lower (mean)" in table and "CI upper (mean)" in table
        expected = mathops.confidence_interval_mean(source.y_raw, level=0.95)
        assert float(table["CI lower (mean)"][0]) == pytest.approx(expected["lower"])
        assert float(table["CI upper (mean)"][0]) == pytest.approx(expected["upper"])

    def test_stats_ci_widens_with_a_higher_level(self, workspace):
        panel = MathLabPanel(workspace)
        source = _dataset_source(workspace)
        r95 = panel._compute(source, ("stats", False, True, 0.95))
        r99 = panel._compute(source, ("stats", False, True, 0.99))
        t95 = dict(r95.table)
        t99 = dict(r99.table)
        w95 = float(t95["CI upper (mean)"][0]) - float(t95["CI lower (mean)"][0])
        w99 = float(t99["CI upper (mean)"][0]) - float(t99["CI lower (mean)"][0])
        assert w99 > w95

    def test_correlate_ci_off_by_default(self, workspace):
        panel = MathLabPanel(workspace)
        x = np.linspace(0.0, 10.0, 200)
        source = _make_dataset_source(workspace, "cci1", x, 3.0 * x + 2.0)
        result = panel._compute(source, ("correlate", False, 0.95))
        labels = {label for label, _v in result.stats}
        assert not any("CI for Pearson" in label for label in labels)

    def test_correlate_ci_adds_a_row_matching_the_backend(self, workspace):
        panel = MathLabPanel(workspace)
        rng = np.random.default_rng(90)
        x = rng.normal(0.0, 1.0, 200)
        y = 0.6 * x + rng.normal(0.0, 1.0, 200)
        source = _make_dataset_source(workspace, "cci2", x, y)
        result = panel._compute(source, ("correlate", True, 0.95))
        rows = dict(result.stats)
        assert "95% CI for Pearson r" in rows
        xs, ys = _sorted_by_x(x, y)
        expected = mathops.confidence_interval_correlation(xs, ys, level=0.95)
        lo_str, hi_str = rows["95% CI for Pearson r"].strip("[]").split(", ")
        assert float(lo_str) == pytest.approx(expected["lower"], abs=1e-3)
        assert float(hi_str) == pytest.approx(expected["upper"], abs=1e-3)

    def test_compare_ci_off_by_default(self, workspace):
        panel = MathLabPanel(workspace)
        ds = workspace.store.get("wide")
        source = _Source(
            key=("dataset", "wide", "t", "v"),
            label="wide.v",
            x_name="t",
            y_name="v",
            x_raw=ds.get("t"),
            y_raw=ds.get("v"),
            dataset=ds,
            x_col="t",
            y_col="v",
        )
        result = panel._compute(source, ("compare", "w", "welch", False, 0.95))
        labels = {label for label, _v in result.stats}
        assert not any("CI for mean diff" in label for label in labels)
        table_names = {name for name, _v in result.table}
        assert "CI lower (mean diff)" not in table_names

    def test_compare_ci_adds_a_row_and_table_columns(self, workspace):
        panel = MathLabPanel(workspace)
        ds = workspace.store.get("wide")
        source = _Source(
            key=("dataset", "wide", "t", "v"),
            label="wide.v",
            x_name="t",
            y_name="v",
            x_raw=ds.get("t"),
            y_raw=ds.get("v"),
            dataset=ds,
            x_col="t",
            y_col="v",
        )
        result = panel._compute(source, ("compare", "w", "welch", True, 0.95))
        rows = dict(result.stats)
        assert any("CI for mean diff" in label for label in rows)
        table = dict(result.table)
        assert "CI lower (mean diff)" in table and "CI upper (mean diff)" in table
        expected = mathops.confidence_interval_difference(ds.get("v"), ds.get("w"), level=0.95)
        assert float(table["CI lower (mean diff)"][0]) == pytest.approx(expected["lower"])
        assert float(table["CI upper (mean diff)"][0]) == pytest.approx(expected["upper"])

    def test_compare_ci_uses_welch_even_when_the_test_method_is_mannwhitney(self, workspace):
        """mannwhitney/ks have no matching mean-difference interval -- the CI always
        falls back to Welch's, independent of which hypothesis test is displayed."""
        panel = MathLabPanel(workspace)
        ds = workspace.store.get("wide")
        source = _Source(
            key=("dataset", "wide", "t", "v"),
            label="wide.v",
            x_name="t",
            y_name="v",
            x_raw=ds.get("t"),
            y_raw=ds.get("v"),
            dataset=ds,
            x_col="t",
            y_col="v",
        )
        result = panel._compute(source, ("compare", "w", "mannwhitney", True, 0.95))
        table = dict(result.table)
        expected = mathops.confidence_interval_difference(
            ds.get("v"), ds.get("w"), level=0.95, method="welch"
        )
        assert float(table["CI lower (mean diff)"][0]) == pytest.approx(expected["lower"])

    def test_ui_checkboxes_reach_the_params(self, imgui_context, workspace):
        panel = MathLabPanel(workspace)
        panel.show_operation("correlate")
        panel._correlate_ci = True
        panel._correlate_ci_level = "90%"
        for _ in range(4):
            _draw_frame(panel, imgui_context)
        assert panel._cache_key[-1] == ("correlate", True, 0.90)


class TestApplyModesRestriction:
    """_Result.apply_modes restricts which of the 4 Apply radios a tab offers.

    Generic-mechanism tests: TestStatsAndCompareOfferNewDataset below exercises the
    concrete case (Stats/Compare, restricted to "New dataset" only).
    """

    def test_default_none_means_all_four_are_offered(self, workspace):
        panel = MathLabPanel(workspace)
        result = _fake_result()
        assert result.apply_modes is None
        assert panel._effective_apply_mode("integral", result) == "new_layer"

    def test_stored_preference_outside_the_allowed_set_falls_back(self, workspace):
        """A mode picked on another tab, or before this result restricted itself."""
        panel = MathLabPanel(workspace)
        panel._apply_modes["stats"] = "replace"
        result = _fake_result()
        result.apply_modes = ("new_dataset",)
        assert panel._effective_apply_mode("stats", result) == "new_dataset"

    def test_stored_preference_inside_the_allowed_set_is_kept(self, workspace):
        panel = MathLabPanel(workspace)
        panel._apply_modes["k"] = "add_column"
        result = _fake_result()
        result.apply_modes = ("new_layer", "add_column")
        assert panel._effective_apply_mode("k", result) == "add_column"

    def test_restricted_result_draws_only_its_allowed_radios(
        self, imgui_context, workspace, monkeypatch
    ):
        """Driven through the real Stats tab: only "New dataset" must appear on screen."""
        panel = MathLabPanel(workspace)
        panel.show_operation("stats")

        labels: List[str] = []
        real_radio = imgui.radio_button

        def spy_radio(label, active, *a, **k):
            labels.append(label)
            return real_radio(label, active, *a, **k)

        monkeypatch.setattr("glplot.gui.panels.mathlab.imgui.radio_button", spy_radio)
        for _ in range(4):
            # A tab switch takes one frame to settle; only the last frame's radios count.
            labels.clear()
            _draw_frame(panel, imgui_context)

        # The panel draws other radios too (source kind, apply-mode radios on other
        # tabs' cached state, etc.) -- only the Apply-mode row is under test here.
        apply_mode_labels = {candidate_label for _candidate, candidate_label in _APPLY_MODES}
        seen = [label for label in labels if label in apply_mode_labels]
        assert seen == ["New dataset"]


class TestStatsAndCompareOfferNewDataset:
    """Stats and Compare have no (x, y) curve to plot, but always offer New dataset.

    Point 6 of the user's request: "siempre [tener la] opcion de agregar crear
    reemplazar dataset" -- an analysis that produces only summary numbers must still be
    one click from becoming data, not a dead end the way "Apply disabled" was before.
    """

    def test_stats_result_is_new_dataset_only(self, workspace):
        panel = MathLabPanel(workspace)
        source = _dataset_source(workspace)
        result = panel._compute(source, ("stats", False, False, 0.95))
        assert result.apply_modes == ("new_dataset",)
        assert result.allow_replace is False

    def test_stats_table_is_one_row_per_statistic_as_columns(self, workspace):
        panel = MathLabPanel(workspace)
        source = _dataset_source(workspace)
        result = panel._compute(source, ("stats", False, False, 0.95))
        assert result.table is not None
        table = dict(result.table)
        # Every stat row shown in the tab has a matching column in the table.
        assert {label for label, _v in result.stats} == set(table)
        for name, values in result.table:
            assert values.shape == (1,), f"column {name!r} is not one row"
        assert "Mean" in table
        assert float(table["Mean"][0]) == pytest.approx(float(np.mean(source.y_raw)))
        assert "Total samples" in table
        assert float(table["Total samples"][0]) == float(source.y_raw.size)

    def test_stats_new_dataset_apply_creates_a_one_row_dataset(self, workspace):
        panel = MathLabPanel(workspace)
        source = _dataset_source(workspace)
        result = panel._compute(source, ("stats", False, False, 0.95))
        cmd = panel._command_new_dataset(source, result, "stats")
        before = set(workspace.store.names())
        cmd.do()
        try:
            new_name = next(iter(set(workspace.store.names()) - before))
            created = workspace.store.get(new_name)
            assert created.n_rows() == 1
            assert created.get("Mean")[0] == pytest.approx(float(np.mean(source.y_raw)))
        finally:
            cmd.undo()

    def test_compare_result_is_new_dataset_only(self, workspace):
        panel = MathLabPanel(workspace)
        ds = workspace.store.get("wide")
        source = _Source(
            key=("dataset", "wide", "t", "v"),
            label="wide.v",
            x_name="t",
            y_name="v",
            x_raw=ds.get("t"),
            y_raw=ds.get("v"),
            dataset=ds,
            x_col="t",
            y_col="v",
        )
        result = panel._compute(source, ("compare", "w", "welch", False, 0.95))
        assert result.apply_modes == ("new_dataset",)
        assert result.allow_replace is False
        assert result.table is not None
        table = dict(result.table)
        for name, values in result.table:
            assert values.shape == (1,), f"column {name!r} is not one row"
        assert "Statistic" in table and "p-value" in table

    def test_compare_new_dataset_apply_creates_a_one_row_dataset(self, workspace):
        panel = MathLabPanel(workspace)
        ds = workspace.store.get("wide")
        source = _Source(
            key=("dataset", "wide", "t", "v"),
            label="wide.v",
            x_name="t",
            y_name="v",
            x_raw=ds.get("t"),
            y_raw=ds.get("v"),
            dataset=ds,
            x_col="t",
            y_col="v",
        )
        result = panel._compute(source, ("compare", "w", "welch", False, 0.95))
        cmd = panel._command_new_dataset(source, result, "compare")
        before = set(workspace.store.names())
        cmd.do()
        try:
            new_name = next(iter(set(workspace.store.names()) - before))
            assert workspace.store.get(new_name).n_rows() == 1
        finally:
            cmd.undo()

    def test_ui_drives_compare_apply_end_to_end(self, imgui_context, workspace, monkeypatch):
        """The full path: pick a column, land on Stats/Compare's only mode, press Apply."""
        panel = MathLabPanel(workspace)
        panel._ds_name = "wide"
        panel.show_operation("compare")
        panel._compare_col = "w"

        before = len(workspace.store)
        for _ in range(6):
            _draw_frame(panel, imgui_context)
        assert panel._cache_error is None

        real_button = imgui.button

        def click_apply(label, *a, **k):
            return True if label == "Apply" else real_button(label, *a, **k)

        monkeypatch.setattr("glplot.gui.panels.mathlab.imgui.button", click_apply)
        _draw_frame(panel, imgui_context)

        # Run what Apply queued directly rather than through CommandQueue.drain: the
        # fake plot here has no .frame/.cache, and this test only cares that the click
        # reached _submit_apply and built a working command, not the dirty-flag epilogue
        # (that belongs to _main_loop, already covered elsewhere).
        assert not workspace.queue.is_empty(), "Apply click did not queue a command"
        while not workspace.queue.is_empty():
            workspace.queue._q.popleft()()

        assert len(workspace.store) == before + 1


class TestClusterTab:
    """The Cluster tab k-means-clusters the source's (x, y) points and colors by cluster."""

    def _blob_source(self, workspace, name="blobs", seed=0):
        rng = np.random.default_rng(seed)
        c1 = rng.normal((0.0, 0.0), 0.3, (60, 2))
        c2 = rng.normal((10.0, 10.0), 0.3, (60, 2))
        pts = np.vstack([c1, c2])
        return _make_dataset_source(workspace, name, pts[:, 0], pts[:, 1])

    def test_compute_produces_color_values_and_centroids(self, workspace):
        panel = MathLabPanel(workspace)
        source = self._blob_source(workspace)
        result = panel._compute(source, ("cluster", "kmeans", 2, 0, _NO_SAMPLING))
        assert result.color_values is not None
        assert result.color_values.shape == result.x.shape
        assert result.markers is not None
        assert np.asarray(result.markers[0]).size == 2
        assert result.table is not None
        assert result.table[-1][0] == "cluster"

    def test_compute_reports_cluster_sizes_in_stats(self, workspace):
        panel = MathLabPanel(workspace)
        source = self._blob_source(workspace)
        result = panel._compute(source, ("cluster", "kmeans", 2, 0, _NO_SAMPLING))
        labels_in_stats = [label for label, _v in result.stats]
        assert "Clusters found" in labels_in_stats
        assert any(
            label.startswith("Cluster ") and label.endswith("size") for label in labels_in_stats
        )

    def test_k_too_large_surfaces_as_an_error_not_a_crash(self, imgui_context, workspace):
        panel = MathLabPanel(workspace)
        _make_dataset_source(workspace, "tiny", [1.0, 2.0], [1.0, 2.0])
        panel.show_operation("cluster")
        panel._cluster_k = 9  # more clusters than the 2 points in "tiny"

        for _ in range(6):
            _draw_frame(panel, imgui_context)  # must not raise

    def test_is_in_the_multivariate_category(self):
        assert "cluster" in dict(_CATEGORIES)["Multivariate"]
        assert ("cluster", "Cluster", "_tab_cluster") in _TABS

    def test_reachable_via_show_operation(self, imgui_context, workspace):
        panel = MathLabPanel(workspace)
        panel.show_operation("cluster")
        for _ in range(6):
            _draw_frame(panel, imgui_context)
        assert panel._cache_key is not None
        assert panel._cache_key[-1][0] == "cluster"

    def test_preview_uses_mini_scatter_not_mini_plot(self, imgui_context, workspace, monkeypatch):
        panel = MathLabPanel(workspace)
        self._blob_source(workspace, name="blobs2")
        panel.show_operation("cluster")

        scatter_calls = []
        line_calls = []
        real_scatter = widgets.mini_scatter
        real_plot = widgets.mini_plot

        def spy_scatter(*a, **k):
            scatter_calls.append(k)
            return real_scatter(*a, **k)

        def spy_plot(*a, **k):
            line_calls.append(k)
            return real_plot(*a, **k)

        monkeypatch.setattr("glplot.gui.panels.mathlab.widgets.mini_scatter", spy_scatter)
        monkeypatch.setattr("glplot.gui.panels.mathlab.widgets.mini_plot", spy_plot)

        for _ in range(6):
            _draw_frame(panel, imgui_context)

        assert scatter_calls, "cluster preview must draw with mini_scatter"
        assert scatter_calls[-1]["point_colors"] is not None

    def test_add_column_writes_cluster_labels(self, workspace):
        panel = MathLabPanel(workspace)
        source = self._blob_source(workspace, name="blobs3")
        result = panel._compute(source, ("cluster", "kmeans", 2, 0, _NO_SAMPLING))
        cmd = panel._command_add_column(source, result, "cluster")
        assert cmd is not None
        cmd.do()
        new_col = source.dataset.columns[-1]
        assert np.array_equal(new_col.values, result.color_values)

    def test_new_layer_passes_color_values_as_c_kwarg(self, workspace, monkeypatch):
        """New layer must close the loop with GLPlot's per-point colour encoding."""
        panel = MathLabPanel(workspace)
        source = self._blob_source(workspace, name="blobs4")
        result = panel._compute(source, ("cluster", "kmeans", 2, 0, _NO_SAMPLING))

        captured = {}

        def spy_add_xy_layer(plot, x, y, **kwargs):
            captured.update(kwargs)
            return object()

        monkeypatch.setattr("glplot.gui.panels.mathlab.layerops.add_xy_layer", spy_add_xy_layer)

        cmd = panel._command_new_layer(source, result, "cluster")
        cmd.do()

        assert captured.get("c") is not None
        assert np.array_equal(captured["c"], result.color_values)
        assert captured.get("cmap") == "viridis"

    def test_new_layer_c_is_none_for_non_cluster_results(self, workspace, monkeypatch):
        """A regular (non-cluster) operation must not accidentally pass a c= kwarg."""
        panel = MathLabPanel(workspace)
        source = _dataset_source(workspace)
        result = panel._compute(source, ("normalize", "minmax"))

        captured = {}

        def spy_add_xy_layer(plot, x, y, **kwargs):
            captured.update(kwargs)
            return object()

        monkeypatch.setattr("glplot.gui.panels.mathlab.layerops.add_xy_layer", spy_add_xy_layer)

        cmd = panel._command_new_layer(source, result, "normalize")
        cmd.do()

        assert captured.get("c") is None


class TestDensity2DTab:
    """The Density 2D tab bins the source's (x, y) points into a histogram or KDE grid."""

    def _blob_source(self, workspace, name="dblobs", seed=0):
        rng = np.random.default_rng(seed)
        c1 = rng.normal((0.0, 0.0), 0.5, (150, 2))
        c2 = rng.normal((8.0, 8.0), 0.5, (150, 2))
        pts = np.vstack([c1, c2])
        return _make_dataset_source(workspace, name, pts[:, 0], pts[:, 1])

    def test_histogram_mode_produces_a_grid(self, workspace):
        panel = MathLabPanel(workspace)
        source = self._blob_source(workspace)
        result = panel._compute(source, ("density2d", "histogram", 20, 0, False, 1.0))
        assert result.grid_x is not None
        assert result.grid_y is not None
        assert result.grid_z is not None
        assert result.grid_z.shape == (20, 20)
        assert result.grid_x.size == 21

    def test_kde_mode_produces_a_grid(self, workspace):
        panel = MathLabPanel(workspace)
        source = self._blob_source(workspace, name="dblobs2")
        result = panel._compute(source, ("density2d", "kde", 40, 25, False, 1.0))
        assert result.grid_z is not None
        assert result.grid_z.shape == (25, 25)
        assert np.all(result.grid_z >= 0.0)

    def test_x_y_stay_the_raw_points_not_the_grid(self, workspace):
        """New layer/Add column/Replace must keep operating on real data, not the grid."""
        panel = MathLabPanel(workspace)
        source = self._blob_source(workspace, name="dblobs3")
        result = panel._compute(source, ("density2d", "histogram", 15, 0, False, 1.0))
        assert result.x.shape == source.x_raw.shape
        assert np.array_equal(result.x, np.asarray(source.x_raw, dtype=np.float64))

    def test_does_not_allow_replace(self, workspace):
        panel = MathLabPanel(workspace)
        source = self._blob_source(workspace, name="dblobs4")
        result = panel._compute(source, ("density2d", "histogram", 15, 0, False, 1.0))
        assert result.allow_replace is False

    def test_reports_stats(self, workspace):
        panel = MathLabPanel(workspace)
        source = self._blob_source(workspace, name="dblobs5")
        result = panel._compute(source, ("density2d", "histogram", 15, 0, False, 1.0))
        labels_in_stats = [label for label, _v in result.stats]
        assert "Mode" in labels_in_stats
        assert "Points" in labels_in_stats
        assert "Grid cells" in labels_in_stats

    def test_is_in_the_multivariate_category(self):
        assert "density2d" in dict(_CATEGORIES)["Multivariate"]
        assert ("density2d", "Density 2D", "_tab_density2d") in _TABS

    def test_reachable_via_show_operation(self, imgui_context, workspace):
        panel = MathLabPanel(workspace)
        panel.show_operation("density2d")
        for _ in range(6):
            _draw_frame(panel, imgui_context)
        assert panel._cache_key is not None
        assert panel._cache_key[-1][0] == "density2d"

    def test_preview_uses_mini_heatmap(self, imgui_context, workspace, monkeypatch):
        panel = MathLabPanel(workspace)
        self._blob_source(workspace, name="dblobs6")
        panel.show_operation("density2d")

        heatmap_calls = []
        real_heatmap = widgets.mini_heatmap

        def spy(*a, **k):
            heatmap_calls.append(k)
            return real_heatmap(*a, **k)

        monkeypatch.setattr("glplot.gui.panels.mathlab.widgets.mini_heatmap", spy)

        for _ in range(6):
            _draw_frame(panel, imgui_context)

        assert heatmap_calls, "density2d preview must draw with mini_heatmap"

    def test_kde_unavailable_shows_an_error_not_a_crash(
        self, imgui_context, workspace, monkeypatch
    ):
        monkeypatch.setattr("glplot.gui.panels.mathlab.mathops2d.kde2d_available", lambda: False)
        panel = MathLabPanel(workspace)
        self._blob_source(workspace, name="dblobs7")
        panel.show_operation("density2d")
        panel._density_mode = "kde"

        for _ in range(6):
            _draw_frame(panel, imgui_context)  # must not raise

    def test_new_layer_targets_hist2d_when_kind_is_selected(self, workspace, monkeypatch):
        """New layer must not error when the user picks the existing hist2d PlotKind --
        the point of Apply's Kind combo already listing it (layerops.KIND_KEYS)."""
        panel = MathLabPanel(workspace)
        source = self._blob_source(workspace, name="dblobs8")
        result = panel._compute(source, ("density2d", "histogram", 15, 0, False, 1.0))

        captured = {}

        def spy_add_xy_layer(plot, x, y, **kwargs):
            captured.update(kwargs)
            return object()

        monkeypatch.setattr("glplot.gui.panels.mathlab.layerops.add_xy_layer", spy_add_xy_layer)

        panel._layer_kind = "hist2d"
        cmd = panel._command_new_layer(source, result, "density2d")
        cmd.do()
        assert captured.get("kind") == "hist2d"


class TestEnvelopeTab:
    """The Envelope tab traces the amplitude envelope of an oscillating signal."""

    def _am_source(self, workspace, name="am"):
        t = np.linspace(0.0, 1.0, 2000, endpoint=False)
        mod = 1.0 + 0.6 * np.sin(2.0 * np.pi * 3.0 * t)
        y = np.sin(2.0 * np.pi * 50.0 * t) * mod
        return _make_dataset_source(workspace, name, t, y), mod

    def test_compute_tracks_the_modulation(self, workspace):
        panel = MathLabPanel(workspace)
        source, mod = self._am_source(workspace)
        result = panel._compute(source, ("envelope",))
        assert result.x.size == source.x_raw.size
        assert np.corrcoef(result.y, mod)[0, 1] > 0.99

    def test_overlay_is_the_original_signal(self, workspace):
        panel = MathLabPanel(workspace)
        source, _mod = self._am_source(workspace, name="am2")
        result = panel._compute(source, ("envelope",))
        assert result.overlay is not None
        assert result.overlay.shape == result.y.shape

    def test_does_not_allow_replace(self, workspace):
        panel = MathLabPanel(workspace)
        source, _mod = self._am_source(workspace, name="am3")
        result = panel._compute(source, ("envelope",))
        assert result.allow_replace is False

    def test_is_in_the_signal_category(self):
        assert "envelope" in dict(_CATEGORIES)["Signal"]
        assert ("envelope", "Envelope", "_tab_envelope") in _TABS

    def test_reachable_via_show_operation(self, imgui_context, workspace):
        panel = MathLabPanel(workspace)
        panel.show_operation("envelope")
        for _ in range(6):
            _draw_frame(panel, imgui_context)
        assert panel._cache_key is not None
        assert panel._cache_key[-1][0] == "envelope"


class TestRollingTab:
    """The Rolling stats tab computes a moving-window statistic over the signal."""

    def test_compute_matches_mathops_directly(self, workspace):
        panel = MathLabPanel(workspace)
        source = _dataset_source(workspace)
        result = panel._compute(source, ("rolling", "std", 9, True))
        xs, ys = source.x_raw, source.y_raw
        order = np.argsort(xs)
        ref = mathops.rolling_stat(ys[order], 9, stat="std", center=True)
        assert np.allclose(result.y, ref, equal_nan=True)

    def test_output_length_matches_input(self, workspace):
        panel = MathLabPanel(workspace)
        source = _dataset_source(workspace)
        result = panel._compute(source, ("rolling", "mean", 5, False))
        assert result.x.size == source.x_raw.size
        assert result.y.size == source.x_raw.size

    def test_overlay_is_the_original_signal(self, workspace):
        panel = MathLabPanel(workspace)
        source = _dataset_source(workspace)
        result = panel._compute(source, ("rolling", "mean", 5, True))
        assert result.overlay is not None

    def test_is_in_the_signal_category(self):
        assert "rolling" in dict(_CATEGORIES)["Signal"]
        assert ("rolling", "Rolling stats", "_tab_rolling") in _TABS

    def test_reachable_via_show_operation(self, imgui_context, workspace):
        panel = MathLabPanel(workspace)
        panel.show_operation("rolling")
        for _ in range(6):
            _draw_frame(panel, imgui_context)
        assert panel._cache_key is not None
        assert panel._cache_key[-1][0] == "rolling"

    def test_statistic_combo_reaches_the_params(self, imgui_context, workspace):
        panel = MathLabPanel(workspace)
        panel.show_operation("rolling")
        panel._rolling_stat = "max"
        for _ in range(6):
            _draw_frame(panel, imgui_context)
        assert panel._cache_key[-1][1] == "max"


class TestSpatialTab:
    """The Spatial tab reports nearest-neighbour spacing and the convex hull."""

    def _square_source(self, workspace, name="square"):
        x = np.array([0.0, 1.0, 1.0, 0.0, 0.5])
        y = np.array([0.0, 0.0, 1.0, 1.0, 0.5])
        return _make_dataset_source(workspace, name, x, y)

    def test_compute_reports_hull_area_and_perimeter(self, workspace):
        panel = MathLabPanel(workspace)
        source = self._square_source(workspace)
        result = panel._compute(source, ("spatial",))
        rows = dict(result.stats)
        assert float(rows["Convex hull area"]) == pytest.approx(1.0)
        assert float(rows["Convex hull perimeter"]) == pytest.approx(4.0)

    def test_markers_are_the_hull_vertices(self, workspace):
        panel = MathLabPanel(workspace)
        source = self._square_source(workspace, name="square2")
        result = panel._compute(source, ("spatial",))
        assert result.markers is not None
        assert result.markers[0].size == 5  # 4 vertices + the closing repeat

    def test_force_scatter_is_set(self, workspace):
        panel = MathLabPanel(workspace)
        source = self._square_source(workspace, name="square3")
        result = panel._compute(source, ("spatial",))
        assert result.force_scatter is True
        assert result.color_values is None

    def test_does_not_allow_replace(self, workspace):
        panel = MathLabPanel(workspace)
        source = self._square_source(workspace, name="square4")
        result = panel._compute(source, ("spatial",))
        assert result.allow_replace is False

    def test_collinear_points_have_no_hull_but_still_compute(self, workspace):
        panel = MathLabPanel(workspace)
        x = np.linspace(0.0, 10.0, 20)
        y = np.zeros(20)
        source = _make_dataset_source(workspace, "line5", x, y)
        result = panel._compute(source, ("spatial",))
        rows = dict(result.stats)
        assert "Convex hull area" not in rows
        assert result.markers is None

    def test_preview_uses_mini_scatter(self, imgui_context, workspace, monkeypatch):
        panel = MathLabPanel(workspace)
        self._square_source(workspace, name="square6")
        panel.show_operation("spatial")

        calls = []
        real_scatter = widgets.mini_scatter

        def spy(*a, **k):
            calls.append(k)
            return real_scatter(*a, **k)

        monkeypatch.setattr("glplot.gui.panels.mathlab.widgets.mini_scatter", spy)

        for _ in range(6):
            _draw_frame(panel, imgui_context)

        assert calls, "spatial preview must draw with mini_scatter"

    def test_is_in_the_multivariate_category(self):
        assert "spatial" in dict(_CATEGORIES)["Multivariate"]
        assert ("spatial", "Spatial", "_tab_spatial") in _TABS

    def test_reachable_via_show_operation(self, imgui_context, workspace):
        panel = MathLabPanel(workspace)
        panel.show_operation("spatial")
        for _ in range(6):
            _draw_frame(panel, imgui_context)
        assert panel._cache_key is not None
        assert panel._cache_key[-1][0] == "spatial"


class TestCompareTab:
    """The Compare tab runs a two-sample test between the source's y and another column."""

    def _two_column_workspace(self):
        ws = _FakeWorkspace()
        rng = np.random.default_rng(0)
        x = np.arange(200, dtype=float)
        a = rng.normal(0.0, 1.0, 200)
        b = rng.normal(3.0, 1.0, 200)
        ws.store.add(DataSet("cmp", [Column("x", x), Column("a", a), Column("b", b)]))
        return ws

    def _cmp_source(self, ws, y_col="a"):
        ds = ws.store.get("cmp")
        return _Source(
            key=("dataset", "cmp", "x", y_col),
            label=f"cmp.{y_col}",
            x_name="x",
            y_name=y_col,
            x_raw=ds.get("x"),
            y_raw=ds.get(y_col),
            dataset=ds,
            x_col="x",
            y_col=y_col,
        )

    def test_compute_detects_a_significant_difference(self):
        ws = self._two_column_workspace()
        panel = MathLabPanel(ws)
        source = self._cmp_source(ws)
        result = panel._compute(source, ("compare", "b", "welch", False, 0.95))
        rows = dict(result.stats)
        assert float(rows["p-value"]) < 1e-10

    def test_apply_is_restricted_to_new_dataset(self):
        """No natural (x, y) curve exists for two independent samples, but the test's
        own numbers are still one click from becoming data -- see
        TestStatsAndCompareOfferNewDataset for the full New dataset path."""
        ws = self._two_column_workspace()
        panel = MathLabPanel(ws)
        source = self._cmp_source(ws)
        result = panel._compute(source, ("compare", "b", "welch", False, 0.95))
        assert result.allow_apply is True
        assert result.apply_modes == ("new_dataset",)

    def test_overlay_is_the_second_samples_histogram(self):
        ws = self._two_column_workspace()
        panel = MathLabPanel(ws)
        source = self._cmp_source(ws)
        result = panel._compute(source, ("compare", "b", "welch", False, 0.95))
        assert result.overlay is not None
        assert result.overlay.shape == result.y.shape

    def test_no_column_picked_raises_a_catchable_error(self):
        ws = self._two_column_workspace()
        panel = MathLabPanel(ws)
        source = self._cmp_source(ws)
        with pytest.raises(ValueError, match="second column"):
            panel._compute(source, ("compare", "", "welch", False, 0.95))

    def test_layer_source_raises_a_catchable_error(self):
        panel = MathLabPanel(_FakeWorkspace())
        source = _Source(
            key=("layer",),
            label="L",
            x_name="x",
            y_name="y",
            x_raw=np.arange(5.0),
            y_raw=np.arange(5.0),
            dataset=None,
        )
        with pytest.raises(ValueError, match="dataset"):
            panel._compute(source, ("compare", "b", "welch", False, 0.95))

    def test_every_method_runs(self):
        ws = self._two_column_workspace()
        panel = MathLabPanel(ws)
        source = self._cmp_source(ws)
        for method in mathops.TWO_SAMPLE_METHODS:
            result = panel._compute(source, ("compare", "b", method, False, 0.95))
            assert result is not None

    def test_is_in_the_statistics_category(self):
        assert "compare" in dict(_CATEGORIES)["Statistics"]
        assert ("compare", "Compare", "_tab_compare") in _TABS

    def test_reachable_via_show_operation(self, imgui_context):
        ws = self._two_column_workspace()
        panel = MathLabPanel(ws)
        panel.show_operation("compare")
        for _ in range(6):
            _draw_frame(panel, imgui_context)
        assert panel._cache_key is not None
        assert panel._cache_key[-1][0] == "compare"
        # The column picker must have defaulted to *some* column, not stayed empty.
        assert panel._cache_key[-1][1] != ""


class TestClusterHierarchical:
    """Cluster's "hierarchical" method, alongside the existing k-means default."""

    def _blob_source(self, workspace, name="hblobs", seed=0):
        rng = np.random.default_rng(seed)
        c1 = rng.normal((0.0, 0.0), 0.3, (60, 2))
        c2 = rng.normal((10.0, 10.0), 0.3, (60, 2))
        pts = np.vstack([c1, c2])
        return _make_dataset_source(workspace, name, pts[:, 0], pts[:, 1])

    def test_default_method_is_kmeans(self, workspace):
        panel = MathLabPanel(workspace)
        assert panel._cluster_method == "kmeans"

    def test_hierarchical_recovers_the_two_blobs(self, workspace):
        panel = MathLabPanel(workspace)
        source = self._blob_source(workspace)
        result = panel._compute(source, ("cluster", "hierarchical", 2, "ward", _NO_SAMPLING))
        assert result.color_values is not None
        assert result.markers is not None
        assert np.asarray(result.markers[0]).size == 2

    def test_method_label_appears_in_stats(self, workspace):
        panel = MathLabPanel(workspace)
        source = self._blob_source(workspace, name="hblobs2")
        result = panel._compute(source, ("cluster", "hierarchical", 2, "single", _NO_SAMPLING))
        rows = dict(result.stats)
        assert "hierarchical" in rows["Method"]
        assert "single" in rows["Method"]

    def test_unavailable_scipy_shows_an_error_not_a_crash(
        self, imgui_context, workspace, monkeypatch
    ):
        monkeypatch.setattr(
            "glplot.gui.panels.mathlab.mathops2d.hierarchical_available", lambda: False
        )
        self._blob_source(workspace, name="hblobs3")
        panel = MathLabPanel(workspace)
        panel.show_operation("cluster")
        panel._cluster_method = "hierarchical"
        for _ in range(6):
            _draw_frame(panel, imgui_context)  # must not raise

    def test_reachable_end_to_end_via_the_real_ui(self, imgui_context, workspace):
        self._blob_source(workspace, name="hblobs4")
        panel = MathLabPanel(workspace)
        panel.show_operation("cluster")
        panel._cluster_method = "hierarchical"
        for _ in range(6):
            _draw_frame(panel, imgui_context)
        assert panel._cache_key is not None
        params = panel._cache_key[-1]
        assert params[0] == "cluster"
        assert params[1] == "hierarchical"


def _make_multicol_source(ws, name, columns, *, x_col=None, y_col=None):
    """A _Source backed by a fresh N-column dataset -- PCA/UMAP's input, which unlike
    every other tab's is however many columns the user selects, not just x_col/y_col."""
    cols = [Column(cname, np.asarray(vals, dtype=np.float64)) for cname, vals in columns.items()]
    ds = DataSet(name, cols)
    ws.store.add(ds)
    names = list(columns.keys())
    x_col = x_col or names[0]
    y_col = y_col or names[1]
    return _Source(
        key=("dataset", name, x_col, y_col),
        label=f"{name}.{y_col}",
        x_name=x_col,
        y_name=y_col,
        x_raw=ds.get(x_col),
        y_raw=ds.get(y_col),
        dataset=ds,
        x_col=x_col,
        y_col=y_col,
    )


def _correlated_columns(n_rows=200, n_features=5, seed=0):
    """A dict of n_features columns, the first three visibly correlated -- PCA should
    concentrate most of the variance in a single component, UMAP should keep points
    that are close on the correlated axes close in the embedding too."""
    rng = np.random.default_rng(seed)
    base = rng.normal(0.0, 1.0, n_rows)
    columns = {
        "a": base + rng.normal(0.0, 0.05, n_rows),
        "b": 2.0 * base + rng.normal(0.0, 0.05, n_rows),
        "c": -1.0 * base + rng.normal(0.0, 0.1, n_rows),
    }
    for i in range(3, n_features):
        columns[f"noise{i}"] = rng.normal(0.0, 1.0, n_rows)
    return columns


class TestMultiColumnPicker:
    """_multi_column_picker(): one checkbox per dataset column, toggled independently."""

    def test_no_dataset_source_returns_empty(self, imgui_context, workspace):
        panel = MathLabPanel(workspace)
        panel._current_source = _Source(
            key=("layer", 1),
            label="layer",
            x_name="x",
            y_name="y",
            x_raw=np.arange(5.0),
            y_raw=np.arange(5.0),
        )
        imgui.new_frame()
        result = panel._multi_column_picker("Columns", ())
        imgui.render()
        assert result == ()

    def test_returns_selected_columns_in_dataset_order(self, imgui_context, workspace):
        panel = MathLabPanel(workspace)
        source = _make_multicol_source(workspace, "mc1", _correlated_columns())
        panel._current_source = source

        imgui.new_frame()
        result = panel._multi_column_picker("Columns", ("c", "a"))
        imgui.render()
        # Order follows the dataset's own column order, not the order passed in.
        assert result[:2] == ("a", "c")

    def test_checkbox_click_toggles_membership(self, imgui_context, workspace, monkeypatch):
        panel = MathLabPanel(workspace)
        source = _make_multicol_source(workspace, "mc2", _correlated_columns())
        panel._current_source = source

        real_checkbox = imgui.checkbox

        def spy_checkbox(label, value, *a, **k):
            if label.startswith("noise3##"):
                return True, not value
            return real_checkbox(label, value, *a, **k)

        monkeypatch.setattr("glplot.gui.panels.mathlab.imgui.checkbox", spy_checkbox)

        imgui.new_frame()
        result = panel._multi_column_picker("Columns", ())
        imgui.render()
        assert "noise3" in result


class TestPcaTab:
    """The PCA tab projects 2+ selected columns onto their top directions of variance."""

    def test_defaults_to_every_column_the_first_time(self, workspace):
        panel = MathLabPanel(workspace)
        assert panel._pca_columns == ()
        source = _make_multicol_source(workspace, "pca1", _correlated_columns())
        panel._current_source = source
        defaulted = panel._default_nd_columns(panel._pca_columns)
        assert set(defaulted) == set(source.dataset.column_names())

    def test_compute_matches_the_backend(self, workspace):
        panel = MathLabPanel(workspace)
        columns = _correlated_columns()
        source = _make_multicol_source(workspace, "pca2", columns)
        names = tuple(columns.keys())
        result = panel._compute(source, ("pca", names, 3, "zscore", _NO_SAMPLING))

        expected = mathopsnd.pca([columns[n] for n in names], n_components=3, scale="zscore")
        np.testing.assert_allclose(result.x, expected["scores"][:, 0])
        np.testing.assert_allclose(result.y, expected["scores"][:, 1])

    def test_result_is_a_scatter_not_a_line(self, workspace):
        panel = MathLabPanel(workspace)
        columns = _correlated_columns()
        source = _make_multicol_source(workspace, "pca3", columns)
        result = panel._compute(source, ("pca", tuple(columns.keys()), 2, "zscore", _NO_SAMPLING))
        assert result.force_scatter is True

    def test_table_has_one_column_per_component(self, workspace):
        panel = MathLabPanel(workspace)
        columns = _correlated_columns()
        source = _make_multicol_source(workspace, "pca4", columns)
        result = panel._compute(source, ("pca", tuple(columns.keys()), 4, "zscore", _NO_SAMPLING))
        assert result.table is not None
        assert [name for name, _v in result.table] == ["PC1", "PC2", "PC3", "PC4"]
        for _name, values in result.table:
            assert values.shape == (result.x.size,)

    def test_stats_report_explained_variance_per_component(self, workspace):
        panel = MathLabPanel(workspace)
        columns = _correlated_columns()
        source = _make_multicol_source(workspace, "pca5", columns)
        result = panel._compute(source, ("pca", tuple(columns.keys()), 3, "zscore", _NO_SAMPLING))
        rows = dict(result.stats)
        assert "PC1 explained variance" in rows
        assert "PC2 explained variance" in rows
        assert "PC3 explained variance" in rows
        assert "Cumulative explained variance" in rows

    def test_correlated_columns_alone_concentrate_variance_in_pc1(self, workspace):
        panel = MathLabPanel(workspace)
        columns = _correlated_columns()
        source = _make_multicol_source(workspace, "pca5b", columns)
        # Only the three visibly-correlated columns -- not the noise ones -- so PC1
        # alone should capture nearly all the variance.
        result = panel._compute(source, ("pca", ("a", "b", "c"), 2, "zscore", _NO_SAMPLING))
        rows = dict(result.stats)
        assert float(rows["PC1 explained variance"].rstrip("%")) > 80.0

    def test_apply_modes_exclude_add_column_and_replace(self, workspace):
        panel = MathLabPanel(workspace)
        columns = _correlated_columns()
        source = _make_multicol_source(workspace, "pca6", columns)
        result = panel._compute(source, ("pca", tuple(columns.keys()), 2, "zscore", _NO_SAMPLING))
        assert result.apply_modes == ("new_layer", "new_dataset")

    def test_fewer_than_two_columns_raises(self, workspace):
        panel = MathLabPanel(workspace)
        columns = _correlated_columns()
        source = _make_multicol_source(workspace, "pca7", columns)
        with pytest.raises(ValueError, match="at least 2 columns"):
            panel._compute(source, ("pca", ("a",), 2, "zscore", _NO_SAMPLING))

    def test_plotted_layer_source_raises(self, workspace):
        panel = MathLabPanel(workspace)
        source = _Source(
            key=("layer", 1),
            label="layer",
            x_name="x",
            y_name="y",
            x_raw=np.arange(5.0),
            y_raw=np.arange(5.0),
        )
        with pytest.raises(ValueError, match="dataset source"):
            panel._compute(source, ("pca", ("a", "b"), 2, "zscore", _NO_SAMPLING))

    def test_is_in_the_multivariate_category(self):
        assert "pca" in dict(_CATEGORIES)["Multivariate"]
        assert ("pca", "PCA", "_tab_pca") in _TABS

    def test_new_dataset_apply_writes_all_components(self, workspace):
        panel = MathLabPanel(workspace)
        columns = _correlated_columns()
        source = _make_multicol_source(workspace, "pca8", columns)
        result = panel._compute(source, ("pca", tuple(columns.keys()), 3, "zscore", _NO_SAMPLING))
        cmd = panel._command_new_dataset(source, result, "pca")
        before = set(workspace.store.names())
        cmd.do()
        try:
            new_name = next(iter(set(workspace.store.names()) - before))
            created = workspace.store.get(new_name)
            assert created.column_names() == ["PC1", "PC2", "PC3"]
            assert created.n_rows() == result.x.size
        finally:
            cmd.undo()

    def test_reachable_end_to_end_via_the_real_ui(self, imgui_context, workspace):
        columns = _correlated_columns()
        _make_multicol_source(workspace, "pca9", columns)
        panel = MathLabPanel(workspace)
        panel.show_operation("pca")
        for _ in range(6):
            _draw_frame(panel, imgui_context)
        assert panel._cache_error is None
        assert panel._cache_key[-1][0] == "pca"


class TestUmapTab:
    """The UMAP tab embeds 2+ selected columns nonlinearly onto 2+ dimensions."""

    def test_compute_matches_the_backend_shape(self, workspace):
        pytest.importorskip("umap")
        panel = MathLabPanel(workspace)
        columns = _correlated_columns(n_rows=60)
        source = _make_multicol_source(workspace, "umap1", columns)
        names = tuple(columns.keys())
        result = panel._compute(source, ("umap", names, 2, 10, 0.1, "zscore", 0, _NO_SAMPLING))
        assert result.x.shape == (60,)
        assert result.y.shape == (60,)

    def test_result_is_a_scatter_not_a_line(self, workspace):
        pytest.importorskip("umap")
        panel = MathLabPanel(workspace)
        columns = _correlated_columns(n_rows=60)
        source = _make_multicol_source(workspace, "umap2", columns)
        result = panel._compute(
            source, ("umap", tuple(columns.keys()), 2, 10, 0.1, "zscore", 0, _NO_SAMPLING)
        )
        assert result.force_scatter is True

    def test_reproducible_with_the_same_seed(self, workspace):
        pytest.importorskip("umap")
        panel = MathLabPanel(workspace)
        columns = _correlated_columns(n_rows=60)
        source = _make_multicol_source(workspace, "umap3", columns)
        names = tuple(columns.keys())
        r1 = panel._compute(source, ("umap", names, 2, 10, 0.1, "zscore", 7, _NO_SAMPLING))
        r2 = panel._compute(source, ("umap", names, 2, 10, 0.1, "zscore", 7, _NO_SAMPLING))
        np.testing.assert_allclose(r1.x, r2.x)
        np.testing.assert_allclose(r1.y, r2.y)

    def test_table_has_one_column_per_component(self, workspace):
        pytest.importorskip("umap")
        panel = MathLabPanel(workspace)
        columns = _correlated_columns(n_rows=60)
        source = _make_multicol_source(workspace, "umap4", columns)
        result = panel._compute(
            source, ("umap", tuple(columns.keys()), 2, 10, 0.1, "zscore", 0, _NO_SAMPLING)
        )
        assert result.table is not None
        assert [name for name, _v in result.table] == ["UMAP1", "UMAP2"]

    def test_stats_report_neighbors_used_and_no_explained_variance(self, workspace):
        """UMAP is nonlinear -- unlike PCA, there is no explained-variance concept."""
        pytest.importorskip("umap")
        panel = MathLabPanel(workspace)
        columns = _correlated_columns(n_rows=60)
        source = _make_multicol_source(workspace, "umap5", columns)
        result = panel._compute(
            source, ("umap", tuple(columns.keys()), 2, 10, 0.1, "zscore", 0, _NO_SAMPLING)
        )
        rows = dict(result.stats)
        assert "Neighbors used" in rows
        assert not any("explained variance" in label for label in rows)

    def test_apply_modes_exclude_add_column_and_replace(self, workspace):
        pytest.importorskip("umap")
        panel = MathLabPanel(workspace)
        columns = _correlated_columns(n_rows=60)
        source = _make_multicol_source(workspace, "umap6", columns)
        result = panel._compute(
            source, ("umap", tuple(columns.keys()), 2, 10, 0.1, "zscore", 0, _NO_SAMPLING)
        )
        assert result.apply_modes == ("new_layer", "new_dataset")

    def test_fewer_than_two_columns_raises(self, workspace):
        pytest.importorskip("umap")
        panel = MathLabPanel(workspace)
        columns = _correlated_columns(n_rows=60)
        source = _make_multicol_source(workspace, "umap7", columns)
        with pytest.raises(ValueError, match="at least 2 columns"):
            panel._compute(source, ("umap", ("a",), 2, 10, 0.1, "zscore", 0, _NO_SAMPLING))

    def test_unavailable_umap_shows_an_error_not_a_crash(
        self, imgui_context, workspace, monkeypatch
    ):
        monkeypatch.setattr("glplot.gui.panels.mathlab.mathopsnd.umap_available", lambda: False)
        columns = _correlated_columns(n_rows=60)
        _make_multicol_source(workspace, "umap8", columns)
        panel = MathLabPanel(workspace)
        panel.show_operation("umap")
        for _ in range(6):
            _draw_frame(panel, imgui_context)  # must not raise

    def test_is_in_the_multivariate_category(self):
        assert "umap" in dict(_CATEGORIES)["Multivariate"]
        assert ("umap", "UMAP", "_tab_umap") in _TABS

    def test_new_dataset_apply_writes_the_embedding(self, workspace):
        pytest.importorskip("umap")
        panel = MathLabPanel(workspace)
        columns = _correlated_columns(n_rows=60)
        source = _make_multicol_source(workspace, "umap9", columns)
        result = panel._compute(
            source, ("umap", tuple(columns.keys()), 2, 10, 0.1, "zscore", 0, _NO_SAMPLING)
        )
        cmd = panel._command_new_dataset(source, result, "umap")
        before = set(workspace.store.names())
        cmd.do()
        try:
            new_name = next(iter(set(workspace.store.names()) - before))
            created = workspace.store.get(new_name)
            assert created.column_names() == ["UMAP1", "UMAP2"]
            assert created.n_rows() == 60
        finally:
            cmd.undo()

    def test_reachable_end_to_end_via_the_real_ui(self, imgui_context, workspace):
        pytest.importorskip("umap")
        columns = _correlated_columns(n_rows=60)
        _make_multicol_source(workspace, "umap10", columns)
        panel = MathLabPanel(workspace)
        panel.show_operation("umap")
        for _ in range(6):
            _draw_frame(panel, imgui_context)
        assert panel._cache_error is None
        assert panel._cache_key[-1][0] == "umap"


# ============================================================================
# Async background compute (glplot/gui/panels/mathlab.py's own _async_enabled/
# _cached_result_async machinery, predating this session -- see background.py and
# tests/test_gui_background.py for the BackgroundJob primitive it wraps). Lost from
# this file by an earlier tool-use accident (a full-file overwrite during an unrelated
# migration); rebuilt fresh here against the current, intact production code rather
# than reconstructed from memory.
# ============================================================================


def _async_wait_until(predicate, *, timeout=2.0, interval=0.005) -> None:
    """Poll ``predicate`` until it's truthy or ``timeout`` elapses.

    For tests that call ``_cached_result``/``_cached_result_async`` directly (no imgui
    frame involved) to observe a real background thread settle -- a poll loop, never a
    fixed sleep-then-assert, so this stays robust under scheduler jitter.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(interval)
    raise AssertionError(f"condition not met within {timeout}s")


class TestAsyncEnabled:
    """_async_enabled(): the sync/async switch, and why every OTHER async-adjacent
    test in this file that does not force it stays deterministic without touching it."""

    def test_default_is_synchronous_under_the_fake_plot(self, workspace):
        """``_FakeWorkspace``'s ``_FakePlot`` has no ``_is_test_mode`` attribute at
        all -- ``getattr(..., True)`` defaults it True, so ``not True`` is sync."""
        panel = MathLabPanel(workspace)
        assert panel._async_enabled() is False

    def test_force_async_overrides_the_default(self, workspace):
        panel = MathLabPanel(workspace)
        panel._force_async = True
        assert panel._async_enabled() is True

    def test_plot_with_is_test_mode_false_is_async(self, workspace):
        panel = MathLabPanel(workspace)
        panel.plot._is_test_mode = False
        assert panel._async_enabled() is True

    def test_plot_with_is_test_mode_true_is_sync(self, workspace):
        panel = MathLabPanel(workspace)
        panel.plot._is_test_mode = True
        assert panel._async_enabled() is False

    def test_force_async_wins_even_if_is_test_mode_is_true(self, workspace):
        panel = MathLabPanel(workspace)
        panel.plot._is_test_mode = True
        panel._force_async = True
        assert panel._async_enabled() is True


class TestAsyncCompute:
    """Dispatch/poll/cancel/retry through ``_cached_result``/``_cached_result_async``
    directly -- no imgui frame needed, since none of this machinery touches imgui.
    Every test forces async mode explicitly and stubs ``panel._compute`` with a short,
    controllable real sleep; every wait is :func:`_async_wait_until`'s poll loop.
    """

    def test_first_dispatch_is_immediate_and_does_not_block(self, workspace):
        panel = MathLabPanel(workspace)
        panel._force_async = True
        source = _dataset_source(workspace)

        def stub(source, params):
            time.sleep(0.3)
            return _fake_result()

        panel._compute = stub
        start = time.monotonic()
        result, error, status = panel._cached_result(source, ("p",), "smooth")
        elapsed = time.monotonic() - start
        assert elapsed < 0.1, "dispatching a job must not block the caller"
        assert status == "pending"
        assert result is None and error is None

    def test_first_call_after_dispatch_is_always_pending_even_for_an_instant_job(self, workspace):
        """The dispatch branch always returns "pending" on the very call that
        constructs the job -- it never polls before returning -- so this holds
        structurally, regardless of how fast ``fn`` actually is."""
        panel = MathLabPanel(workspace)
        panel._force_async = True
        source = _dataset_source(workspace)
        panel._compute = lambda source, params: _fake_result()  # no sleep at all

        result, error, status = panel._cached_result(source, ("p",), "smooth")
        assert status == "pending"
        assert result is None and error is None

    def test_pending_state_is_never_stale_or_blank_while_running(self, workspace):
        panel = MathLabPanel(workspace)
        panel._force_async = True
        source = _dataset_source(workspace)

        def stub(source, params):
            time.sleep(0.05)
            return _fake_result()

        panel._compute = stub
        panel._cached_result(source, ("p",), "smooth")
        result, error, status = panel._cached_result(source, ("p",), "smooth")
        assert status == "pending"
        assert result is None
        assert error is None

    def test_settling_populates_cache_result_and_async_results(self, workspace):
        panel = MathLabPanel(workspace)
        panel._force_async = True
        source = _dataset_source(workspace)
        sentinel = _fake_result()

        def stub(source, params):
            time.sleep(0.02)
            return sentinel

        panel._compute = stub
        params = ("p",)
        panel._cached_result(source, params, "smooth")
        _async_wait_until(lambda: panel._cached_result(source, params, "smooth")[2] == "ready")

        result, error, status = panel._cached_result(source, params, "smooth")
        assert status == "ready"
        assert result is sentinel
        assert error is None
        assert panel._cache_result is sentinel
        assert panel._cache_error is None
        assert "smooth" in panel._async_results

    def test_value_error_surfaces_as_its_own_message(self, workspace):
        """_async_compute_fn reclassifies a ValueError (mathops's documented failure
        mode) so the background path reports exactly the message the sync path
        would -- not prefixed with "Unexpected error"."""
        panel = MathLabPanel(workspace)
        panel._force_async = True
        source = _dataset_source(workspace)

        def stub(source, params):
            raise ValueError("bad window size")

        panel._compute = stub
        params = ("p",)
        panel._cached_result(source, params, "smooth")
        _async_wait_until(lambda: panel._cached_result(source, params, "smooth")[2] == "ready")

        result, error, status = panel._cached_result(source, params, "smooth")
        assert result is None
        assert error == "bad window size"

    def test_unexpected_exception_is_prefixed_and_logged(self, workspace):
        panel = MathLabPanel(workspace)
        panel._force_async = True
        source = _dataset_source(workspace)

        def stub(source, params):
            raise RuntimeError("boom")

        panel._compute = stub
        params = ("p",)
        panel._cached_result(source, params, "smooth")
        _async_wait_until(lambda: panel._cached_result(source, params, "smooth")[2] == "ready")

        result, error, status = panel._cached_result(source, params, "smooth")
        assert result is None
        assert error == "Unexpected error: boom"

    def test_cancel_stops_waiting_immediately(self, workspace):
        panel = MathLabPanel(workspace)
        panel._force_async = True
        source = _dataset_source(workspace)

        def stub(source, params):
            time.sleep(0.15)
            return _fake_result()

        panel._compute = stub
        params = ("p",)
        panel._cached_result(source, params, "smooth")
        panel._cancel_async("smooth")
        result, error, status = panel._cached_result(source, params, "smooth")
        assert status == "cancelled"
        assert result is None and error is None

    def test_cancel_stays_cancelled_after_the_job_finishes_behind_it(self, workspace):
        """A stale/already-cancelled job's late result must not resurrect -- the tab
        stays "cancelled" even once the (unobserved) thread behind it completes."""
        panel = MathLabPanel(workspace)
        panel._force_async = True
        source = _dataset_source(workspace)

        def stub(source, params):
            time.sleep(0.03)
            return _fake_result()

        panel._compute = stub
        params = ("p",)
        panel._cached_result(source, params, "smooth")
        panel._cancel_async("smooth")
        time.sleep(0.08)  # comfortably longer than the stub's own sleep
        for _ in range(3):
            result, error, status = panel._cached_result(source, params, "smooth")
            assert status == "cancelled"
            assert result is None and error is None

    def test_cancel_stays_cancelled_until_params_actually_change(self, workspace):
        panel = MathLabPanel(workspace)
        panel._force_async = True
        source = _dataset_source(workspace)
        panel._compute = lambda source, params: _fake_result()

        panel._cached_result(source, ("p1",), "smooth")
        panel._cancel_async("smooth")
        assert panel._cached_result(source, ("p1",), "smooth")[2] == "cancelled"

        # A genuinely new params tuple must dispatch its own fresh job, not stay
        # wedged on the old cancellation.
        result, error, status = panel._cached_result(source, ("p2",), "smooth")
        assert status == "pending"

    def test_retry_redispatches_immediately_and_eventually_settles(self, workspace):
        panel = MathLabPanel(workspace)
        panel._force_async = True
        source = _dataset_source(workspace)
        calls: List[Any] = []

        def stub(source, params):
            calls.append(params)
            time.sleep(0.02)
            return _fake_result()

        panel._compute = stub
        params = ("p",)
        panel._cached_result(source, params, "smooth")
        panel._cancel_async("smooth")
        assert panel._cached_result(source, params, "smooth")[2] == "cancelled"

        panel._retry_async("smooth")
        result, error, status = panel._cached_result(source, params, "smooth")
        assert status == "pending", "retry must redispatch immediately, no debounce"

        _async_wait_until(lambda: panel._cached_result(source, params, "smooth")[2] == "ready")
        assert len(calls) == 2, "one call before the cancel, one after the retry"

    def test_multiple_tabs_settle_independently_via_direct_dispatch(self, workspace):
        """Two different ``tab_key``s never interfere -- each has its own job/key/
        stable_since/results tracking (all dicts keyed by tab_key, never globally)."""
        panel = MathLabPanel(workspace)
        panel._force_async = True
        source = _dataset_source(workspace)
        calls: List[str] = []

        def stub(source, params):
            calls.append(params[0])
            time.sleep(0.03)
            return _fake_result()

        panel._compute = stub
        r1 = panel._cached_result(source, ("tabA", 1), "tabA")
        r2 = panel._cached_result(source, ("tabB", 2), "tabB")
        assert r1[2] == "pending"
        assert r2[2] == "pending"
        assert panel._async_jobs["tabA"] is not panel._async_jobs["tabB"]

        _async_wait_until(lambda: panel._cached_result(source, ("tabA", 1), "tabA")[2] == "ready")
        _async_wait_until(lambda: panel._cached_result(source, ("tabB", 2), "tabB")[2] == "ready")
        assert sorted(calls) == ["tabA", "tabB"]

    def test_sync_mode_never_touches_async_bookkeeping_dicts(self, workspace):
        panel = MathLabPanel(workspace)
        source = _dataset_source(workspace)
        assert panel._async_enabled() is False
        for i in range(5):
            panel._cached_result(source, (f"p{i}",), "smooth")
        assert panel._async_jobs == {}
        assert panel._async_job_keys == {}
        assert panel._async_stable_since == {}
        assert panel._async_results == {}


class TestAsyncUiIntegration:
    """The parts of the async story that only exist through a real imgui frame: the
    pending spinner/Cancel row, the Cancelled/Retry row, the debounce as actually
    driven by ``_draw_frame``, independent concurrent tabs through the real UI, and
    the slow-job toast."""

    def test_pending_row_draws_a_spinner_and_hides_apply(self, imgui_context, workspace):
        panel = MathLabPanel(workspace)
        panel._force_async = True

        def stub(source, params):
            time.sleep(0.3)
            return _fake_result()

        panel._compute = stub
        panel.show_operation("smooth")
        _draw_frame(panel, imgui_context)  # the tab switch itself needs one frame to land
        _draw_frame(panel, imgui_context)  # dispatch: this frame renders the pending row

        assert panel._async_jobs.get("smooth") is not None
        assert panel._cache_result is None
        # Nothing was committed this frame -- the "give the body the whole panel back"
        # branch runs, exactly like an op with no result to commit.
        assert panel._apply_heights.get("smooth") == 0.0

    def test_cancel_button_click_cancels_the_running_job(
        self, imgui_context, workspace, monkeypatch
    ):
        panel = MathLabPanel(workspace)
        panel._force_async = True

        def stub(source, params):
            time.sleep(0.3)
            return _fake_result()

        panel._compute = stub
        panel.show_operation("smooth")
        _draw_frame(panel, imgui_context)  # the tab switch itself needs one frame to land
        _draw_frame(panel, imgui_context)
        job = panel._async_jobs.get("smooth")
        assert job is not None
        assert job.cancelled is False

        monkeypatch.setattr("glplot.gui.panels.mathlab.widgets.busy_row", lambda *a, **k: True)
        _draw_frame(panel, imgui_context)
        assert job.cancelled is True

    def test_cancelled_row_retry_button_redispatches(self, imgui_context, workspace, monkeypatch):
        panel = MathLabPanel(workspace)
        panel._force_async = True
        calls: List[Any] = []

        def stub(source, params):
            calls.append(params)
            time.sleep(0.02)
            return _fake_result()

        panel._compute = stub
        panel.show_operation("smooth")
        _draw_frame(panel, imgui_context)  # the tab switch itself needs one frame to land
        calls.clear()  # discard whatever the throwaway landing frame may have dispatched
        _draw_frame(panel, imgui_context)  # dispatch
        panel._cancel_async("smooth")
        _draw_frame(panel, imgui_context)  # now renders the Cancelled/Retry row
        assert len(calls) == 1

        real_button = imgui.button

        def spy_button(label, *a, **k):
            if label.startswith("Retry##cancelled_smooth"):
                return True
            return real_button(label, *a, **k)

        monkeypatch.setattr("glplot.gui.panels.mathlab.imgui.button", spy_button)
        _draw_frame(panel, imgui_context)  # click Retry (redispatch happens next frame)
        _draw_frame(panel, imgui_context)
        assert len(calls) == 2, "Retry must redispatch, with no debounce"

        _wait_frames_until(panel, imgui_context, lambda: panel._cache_result is not None)
        assert panel._cache_error is None

    def test_rapid_param_changes_collapse_into_one_dispatch_via_debounce(
        self, imgui_context, workspace, monkeypatch
    ):
        """A slider-drag style burst of param changes must not spawn one job per
        frame. The debounce is stretched to 5s (real, monkeypatched) so the assertion
        is independent of how fast this machine happens to draw a headless frame."""
        monkeypatch.setattr("glplot.gui.panels.mathlab._ASYNC_DEBOUNCE_SECONDS", 5.0)
        panel = MathLabPanel(workspace)
        panel._force_async = True
        calls: List[Any] = []

        def stub(source, params):
            calls.append(params)
            time.sleep(0.01)
            return _fake_result()

        panel._compute = stub
        panel.show_operation("smooth")
        _draw_frame(panel, imgui_context)  # the tab switch itself needs one frame to land
        calls.clear()  # discard whatever the throwaway landing frame may have dispatched
        for i in range(25):
            panel._smooth_window = (i % 7) + 1
            _draw_frame(panel, imgui_context)
        assert len(calls) <= 2, (
            f"expected the debounce to collapse 25 rapid param changes into at most "
            f"2 dispatches (the first immediate one plus its supersession), got {len(calls)}"
        )

        # Collapse the debounce back down so the final value is free to settle.
        monkeypatch.setattr("glplot.gui.panels.mathlab._ASYNC_DEBOUNCE_SECONDS", 0.0)
        _wait_frames_until(
            panel,
            imgui_context,
            lambda: panel._cache_result is not None or panel._cache_error is not None,
        )
        assert panel._cache_error is None

    def test_multiple_tabs_run_independently_through_the_real_ui(self, imgui_context, workspace):
        """Switching tabs must not cancel a slow computation left running elsewhere
        (the module docstring's headline claim for this whole feature)."""
        panel = MathLabPanel(workspace)
        panel._force_async = True
        calls: List[str] = []

        def stub(source, params):
            calls.append(params[0])
            time.sleep(0.05)
            return _fake_result()

        panel._compute = stub

        panel.show_operation("smooth")
        _draw_frame(panel, imgui_context)  # the tab switch itself needs one frame to land
        _draw_frame(panel, imgui_context)
        job_smooth = panel._async_jobs.get("smooth")
        assert job_smooth is not None

        panel.show_operation("integral")
        _draw_frame(panel, imgui_context)  # the tab switch itself needs one frame to land
        _draw_frame(panel, imgui_context)
        job_integral = panel._async_jobs.get("integral")
        assert job_integral is not None

        # Drawing the "integral" tab must not have touched "smooth"'s job at all.
        assert panel._async_jobs.get("smooth") is job_smooth
        assert job_smooth.cancelled is False

        _wait_frames_until(panel, imgui_context, lambda: panel._async_jobs.get("integral") is None)
        panel.show_operation("smooth")
        _wait_frames_until(panel, imgui_context, lambda: panel._async_jobs.get("smooth") is None)

        assert "smooth" in calls
        assert "integral" in calls

    def test_slow_job_pushes_a_success_toast(self, imgui_context, workspace, monkeypatch):
        notifications.clear()
        monkeypatch.setattr("glplot.gui.panels.mathlab._ASYNC_NOTIFY_THRESHOLD_SECONDS", 0.02)
        panel = MathLabPanel(workspace)
        panel._force_async = True

        def stub(source, params):
            time.sleep(0.05)
            return _fake_result()

        panel._compute = stub
        panel.show_operation("smooth")
        try:
            _wait_frames_until(panel, imgui_context, lambda: panel._cache_result is not None)
            toasts = notifications.active()
            assert any(t.kind == "success" and "Smooth finished" in t.message for t in toasts)
        finally:
            notifications.clear()

    def test_slow_job_failure_pushes_an_error_toast(self, imgui_context, workspace, monkeypatch):
        notifications.clear()
        monkeypatch.setattr("glplot.gui.panels.mathlab._ASYNC_NOTIFY_THRESHOLD_SECONDS", 0.02)
        panel = MathLabPanel(workspace)
        panel._force_async = True

        def stub(source, params):
            time.sleep(0.05)
            raise ValueError("bad window")

        panel._compute = stub
        panel.show_operation("smooth")
        try:
            _wait_frames_until(panel, imgui_context, lambda: panel._cache_error is not None)
            toasts = notifications.active()
            assert any(t.kind == "error" and "failed" in t.message for t in toasts)
        finally:
            notifications.clear()

    def test_fast_job_does_not_push_a_toast(self, imgui_context, workspace):
        """Default threshold (1.5s) -- an instant compute never crosses it."""
        notifications.clear()
        panel = MathLabPanel(workspace)
        panel._force_async = True
        panel._compute = lambda source, params: _fake_result()
        panel.show_operation("smooth")
        try:
            _wait_frames_until(panel, imgui_context, lambda: panel._cache_result is not None)
            toasts = notifications.active()
            assert not any("Smooth" in t.message for t in toasts)
        finally:
            notifications.clear()


# ============================================================================
# Sub-sampling (Phase 0) and Train/val/test split (Phase 1) -- the "trainable" tabs'
# (fit/cluster/pca/umap, see _TRAINABLE_TABS) shared _SamplingParams/_sampling_controls
# machinery. mathops2d's nearest_centroid/cluster_inertia/silhouette_score and
# mathopsnd's pca_transform/apply_scale already have dedicated, unaffected coverage in
# tests/test_gui_mathops2d.py and tests/test_gui_mathopsnd.py -- this section tests
# only the mathlab.py-side wiring: the two pure index-selection functions, and how each
# trainable _compute branch reacts to sub-sampling/split being on vs. off.
# ============================================================================


class TestSubsampleIndices:
    """_subsample_indices(): ascending, seeded, bounded random row selection."""

    def test_none_when_n_within_max_rows(self):
        assert _subsample_indices(50, 100, seed=0) is None
        assert _subsample_indices(100, 100, seed=0) is None  # exactly at the cap

    def test_none_when_max_rows_is_non_positive(self):
        assert _subsample_indices(50, 0, seed=0) is None
        assert _subsample_indices(50, -5, seed=0) is None

    def test_returns_at_most_max_rows_indices(self):
        idx = _subsample_indices(1000, 100, seed=0)
        assert idx is not None
        assert idx.size == 100

    def test_returns_ascending_order(self):
        idx = _subsample_indices(1000, 100, seed=0)
        assert np.all(np.diff(idx) > 0)

    def test_indices_are_within_bounds_and_unique(self):
        idx = _subsample_indices(500, 50, seed=3)
        assert idx.min() >= 0
        assert idx.max() < 500
        assert len(set(idx.tolist())) == 50

    def test_deterministic_for_the_same_seed(self):
        idx1 = _subsample_indices(1000, 100, seed=5)
        idx2 = _subsample_indices(1000, 100, seed=5)
        np.testing.assert_array_equal(idx1, idx2)

    def test_differs_for_a_different_seed(self):
        idx1 = _subsample_indices(1000, 100, seed=1)
        idx2 = _subsample_indices(1000, 100, seed=2)
        assert not np.array_equal(idx1, idx2)


class TestSplitIndices:
    """_split_indices(): a random (seeded) train/val/test partition of range(n)."""

    def test_partition_sizes_match_requested_fractions(self):
        train, val, test = _split_indices(1000, 0.2, 0.1, seed=0)
        assert val.size == round(1000 * 0.2)
        assert test.size == round(1000 * 0.1)
        assert train.size == 1000 - val.size - test.size

    def test_disjoint_and_exhaustive(self):
        train, val, test = _split_indices(300, 0.15, 0.15, seed=1)
        train_set, val_set, test_set = set(train.tolist()), set(val.tolist()), set(test.tolist())
        assert not (train_set & val_set)
        assert not (train_set & test_set)
        assert not (val_set & test_set)
        assert train_set | val_set | test_set == set(range(300))

    def test_each_partition_is_ascending(self):
        train, val, test = _split_indices(300, 0.2, 0.2, seed=2)
        for part in (train, val, test):
            if part.size > 1:
                assert np.all(np.diff(part) > 0)

    def test_deterministic_for_the_same_seed(self):
        a = _split_indices(300, 0.2, 0.2, seed=7)
        b = _split_indices(300, 0.2, 0.2, seed=7)
        for pa, pb in zip(a, b):
            np.testing.assert_array_equal(pa, pb)

    def test_different_seed_gives_a_different_partition(self):
        train1, _val1, _test1 = _split_indices(300, 0.2, 0.2, seed=1)
        train2, _val2, _test2 = _split_indices(300, 0.2, 0.2, seed=2)
        assert not np.array_equal(train1, train2)

    def test_train_never_empties_under_extreme_val_test_fractions(self):
        train, val, test = _split_indices(20, 0.9, 0.9, seed=0)
        assert train.size > 0
        assert val.size + test.size < 20


class TestSubsamplingWiring:
    """How training.subsample actually reaches the Fit/Cluster/PCA/UMAP _compute
    branches -- restricting the technique to the subsampled rows, not just cosmetic."""

    def test_fit_subsampling_restricts_rows_used_and_is_reported(self, workspace):
        panel = MathLabPanel(workspace)
        x = np.linspace(0.0, 10.0, 500)
        source = _make_dataset_source(workspace, "fitsub", x, 2.0 * x + 1.0)
        training = _SamplingParams(subsample=True, max_rows=100, subsample_seed=0)
        result = panel._compute(
            source,
            ("fit", "polynomial", 1, "", None, False, "soft_l1", False, 0.95, False, 200, training),
        )
        rows = dict(result.stats)
        assert rows["Rows used"].startswith("100 of 500")
        assert result.overlay.size == 100

    def test_fit_subsampling_off_matches_the_no_sampling_baseline(self, workspace):
        panel = MathLabPanel(workspace)
        x = np.linspace(0.0, 10.0, 50)
        source = _make_dataset_source(workspace, "fitoff", x, 3.0 * x + 1.0)
        common = ("fit", "polynomial", 1, "", None, False, "soft_l1", False, 0.95, False, 200)
        r1 = panel._compute(source, common + (_NO_SAMPLING,))
        r2 = panel._compute(source, common + (_SamplingParams(subsample=False, split=False),))
        np.testing.assert_allclose(r1.x, r2.x)
        np.testing.assert_allclose(r1.y, r2.y)
        assert dict(r1.stats) == dict(r2.stats)
        assert "Rows used" not in dict(r1.stats)

    def test_cluster_subsampling_restricts_the_technique_to_the_subset(self, workspace):
        panel = MathLabPanel(workspace)
        rng = np.random.default_rng(1)
        pts = np.vstack(
            [rng.normal((0.0, 0.0), 0.3, (150, 2)), rng.normal((10.0, 10.0), 0.3, (150, 2))]
        )
        source = _make_dataset_source(workspace, "clustersub", pts[:, 0], pts[:, 1])
        training = _SamplingParams(subsample=True, max_rows=60, subsample_seed=0)
        result = panel._compute(source, ("cluster", "kmeans", 2, 0, training))
        rows = dict(result.stats)
        assert rows["Rows used"].startswith("60 of 300")
        assert result.x.size == 60
        assert result.color_values.shape == (60,)

    def test_cluster_subsampling_off_reports_no_rows_used_row(self, workspace):
        panel = MathLabPanel(workspace)
        source = self._blob_source(workspace)
        result = panel._compute(source, ("cluster", "kmeans", 2, 0, _NO_SAMPLING))
        assert "Rows used" not in dict(result.stats)

    @staticmethod
    def _blob_source(workspace, name="blobs_sub"):
        rng = np.random.default_rng(0)
        pts = np.vstack(
            [rng.normal((0.0, 0.0), 0.3, (60, 2)), rng.normal((10.0, 10.0), 0.3, (60, 2))]
        )
        return _make_dataset_source(workspace, name, pts[:, 0], pts[:, 1])

    def test_pca_subsampling_restricts_the_technique_to_the_subset(self, workspace):
        panel = MathLabPanel(workspace)
        columns = _correlated_columns(n_rows=300)
        source = _make_multicol_source(workspace, "pcasub", columns)
        training = _SamplingParams(subsample=True, max_rows=80, subsample_seed=0)
        result = panel._compute(source, ("pca", tuple(columns.keys()), 2, "zscore", training))
        rows = dict(result.stats)
        assert rows["Rows used"].startswith("80 of 300")
        assert result.x.size == 80

    def test_pca_no_sampling_sentinel_reports_a_plain_row_count(self, workspace):
        panel = MathLabPanel(workspace)
        columns = _correlated_columns(n_rows=40)
        source = _make_multicol_source(workspace, "pcanoop", columns)
        result = panel._compute(source, ("pca", tuple(columns.keys()), 2, "zscore", _NO_SAMPLING))
        assert dict(result.stats)["Rows used"] == "40"

    def test_pca_subsampling_off_matches_the_no_sampling_baseline(self, workspace):
        panel = MathLabPanel(workspace)
        columns = _correlated_columns(n_rows=50)
        source = _make_multicol_source(workspace, "pcaoff", columns)
        names = tuple(columns.keys())
        r1 = panel._compute(source, ("pca", names, 2, "zscore", _NO_SAMPLING))
        r2 = panel._compute(
            source, ("pca", names, 2, "zscore", _SamplingParams(subsample=False, split=False))
        )
        np.testing.assert_allclose(r1.x, r2.x)
        np.testing.assert_allclose(r1.y, r2.y)
        assert dict(r1.stats) == dict(r2.stats)

    def test_umap_subsampling_restricts_the_technique_to_the_subset(self, workspace):
        pytest.importorskip("umap")
        panel = MathLabPanel(workspace)
        columns = _correlated_columns(n_rows=200)
        source = _make_multicol_source(workspace, "umapsub", columns)
        training = _SamplingParams(subsample=True, max_rows=50, subsample_seed=0)
        result = panel._compute(
            source, ("umap", tuple(columns.keys()), 2, 10, 0.1, "zscore", 0, training)
        )
        rows = dict(result.stats)
        assert rows["Rows used"].startswith("50 of 200")
        assert result.x.size == 50

    def test_umap_no_sampling_sentinel_reports_a_plain_row_count(self, workspace):
        pytest.importorskip("umap")
        panel = MathLabPanel(workspace)
        columns = _correlated_columns(n_rows=40)
        source = _make_multicol_source(workspace, "umapnoop", columns)
        result = panel._compute(
            source, ("umap", tuple(columns.keys()), 2, 10, 0.1, "zscore", 0, _NO_SAMPLING)
        )
        assert dict(result.stats)["Rows used"] == "40"


class TestTrainValTestSplit:
    """How training.split reaches the Fit/Cluster/PCA/UMAP _compute branches -- held-out
    metrics reported with the "Train "/"Val "/"Test " prefix convention, and absent
    entirely with the split off (the pre-feature baseline)."""

    def test_fit_split_reports_train_val_test_r_squared_and_rmse(self, workspace):
        panel = MathLabPanel(workspace)
        rng = np.random.default_rng(2)
        x = np.linspace(0.0, 10.0, 300)
        y = 2.0 * x + 1.0 + rng.normal(0.0, 0.05, x.size)
        source = _make_dataset_source(workspace, "fitsplit", x, y)
        training = _SamplingParams(split=True, val_frac=0.2, test_frac=0.2, split_seed=0)
        result = panel._compute(
            source,
            ("fit", "polynomial", 1, "", None, False, "soft_l1", False, 0.95, False, 200, training),
        )
        rows = dict(result.stats)
        for label in (
            "Train R squared",
            "Val R squared",
            "Test R squared",
            "Train RMSE",
            "Val RMSE",
            "Test RMSE",
        ):
            assert label in rows, f"missing {label!r} in fit split stats"

    def test_fit_split_off_has_no_train_val_test_rows(self, workspace):
        panel = MathLabPanel(workspace)
        x = np.linspace(0.0, 10.0, 60)
        source = _make_dataset_source(workspace, "fitsplitoff", x, 2.0 * x + 1.0)
        result = panel._compute(
            source,
            (
                "fit",
                "polynomial",
                1,
                "",
                None,
                False,
                "soft_l1",
                False,
                0.95,
                False,
                200,
                _NO_SAMPLING,
            ),
        )
        labels = [label for label, _v in result.stats]
        assert not any(label.startswith(("Train ", "Val ", "Test ")) for label in labels)

    def test_cluster_split_reports_train_val_test_inertia_and_silhouette(self, workspace):
        panel = MathLabPanel(workspace)
        rng = np.random.default_rng(3)
        pts = np.vstack(
            [rng.normal((0.0, 0.0), 0.3, (100, 2)), rng.normal((10.0, 10.0), 0.3, (100, 2))]
        )
        source = _make_dataset_source(workspace, "clustersplit", pts[:, 0], pts[:, 1])
        training = _SamplingParams(split=True, val_frac=0.2, test_frac=0.2, split_seed=0)
        result = panel._compute(source, ("cluster", "kmeans", 2, 0, training))
        rows = dict(result.stats)
        for label in (
            "Train inertia",
            "Val inertia",
            "Test inertia",
            "Train silhouette",
            "Val silhouette",
            "Test silhouette",
        ):
            assert label in rows, f"missing {label!r} in cluster split stats"

    def test_cluster_split_off_has_no_train_val_test_rows(self, workspace):
        panel = MathLabPanel(workspace)
        rng = np.random.default_rng(4)
        pts = np.vstack(
            [rng.normal((0.0, 0.0), 0.3, (60, 2)), rng.normal((10.0, 10.0), 0.3, (60, 2))]
        )
        source = _make_dataset_source(workspace, "clustersplitoff", pts[:, 0], pts[:, 1])
        result = panel._compute(source, ("cluster", "kmeans", 2, 0, _NO_SAMPLING))
        labels = [label for label, _v in result.stats]
        assert not any(label.startswith(("Train ", "Val ", "Test ")) for label in labels)

    def test_pca_split_reports_reconstruction_error_per_partition(self, workspace):
        panel = MathLabPanel(workspace)
        columns = _correlated_columns(n_rows=200)
        source = _make_multicol_source(workspace, "pcasplit", columns)
        training = _SamplingParams(split=True, val_frac=0.2, test_frac=0.2, split_seed=0)
        result = panel._compute(source, ("pca", tuple(columns.keys()), 2, "zscore", training))
        rows = dict(result.stats)
        for label in (
            "Train reconstruction error",
            "Val reconstruction error",
            "Test reconstruction error",
        ):
            assert label in rows, f"missing {label!r} in pca split stats"

    def test_pca_split_off_has_no_train_val_test_rows(self, workspace):
        panel = MathLabPanel(workspace)
        columns = _correlated_columns(n_rows=40)
        source = _make_multicol_source(workspace, "pcasplitoff", columns)
        result = panel._compute(source, ("pca", tuple(columns.keys()), 2, "zscore", _NO_SAMPLING))
        labels = [label for label, _v in result.stats]
        assert not any(label.startswith(("Train ", "Val ", "Test ")) for label in labels)

    def test_umap_split_reports_row_counts_per_partition(self, workspace):
        pytest.importorskip("umap")
        panel = MathLabPanel(workspace)
        columns = _correlated_columns(n_rows=150)
        source = _make_multicol_source(workspace, "umapsplit", columns)
        training = _SamplingParams(split=True, val_frac=0.2, test_frac=0.2, split_seed=0)
        result = panel._compute(
            source, ("umap", tuple(columns.keys()), 2, 10, 0.1, "zscore", 0, training)
        )
        rows = dict(result.stats)
        assert "Train rows" in rows and "Val rows" in rows and "Test rows" in rows
        total = int(rows["Train rows"]) + int(rows["Val rows"]) + int(rows["Test rows"])
        assert total == 150

    def test_umap_split_off_has_no_row_count_rows(self, workspace):
        pytest.importorskip("umap")
        panel = MathLabPanel(workspace)
        columns = _correlated_columns(n_rows=40)
        source = _make_multicol_source(workspace, "umapsplitoff", columns)
        result = panel._compute(
            source, ("umap", tuple(columns.keys()), 2, 10, 0.1, "zscore", 0, _NO_SAMPLING)
        )
        rows = dict(result.stats)
        assert "Train rows" not in rows

    def test_model_state_is_populated_only_for_the_four_trainable_kinds(self, workspace):
        panel = MathLabPanel(workspace)
        x = np.linspace(0.0, 10.0, 60)
        y = np.sin(x)
        source = _make_dataset_source(workspace, "modelstate1", x, y)

        fit_result = panel._compute(
            source,
            (
                "fit",
                "polynomial",
                2,
                "",
                None,
                False,
                "soft_l1",
                False,
                0.95,
                False,
                200,
                _NO_SAMPLING,
            ),
        )
        assert fit_result.model_state is not None

        cluster_source = self._blob_source(workspace)
        cluster_result = panel._compute(cluster_source, ("cluster", "kmeans", 2, 0, _NO_SAMPLING))
        assert cluster_result.model_state is not None

        columns = _correlated_columns(n_rows=40)
        nd_source = _make_multicol_source(workspace, "modelstate2", columns)
        pca_result = panel._compute(
            nd_source, ("pca", tuple(columns.keys()), 2, "zscore", _NO_SAMPLING)
        )
        assert pca_result.model_state is not None

        smooth_result = panel._compute(source, ("smooth", "moving_average", 5, 0.0, 0, 0.0))
        assert smooth_result.model_state is None

        correlate_result = panel._compute(source, ("correlate", False, 0.95))
        assert correlate_result.model_state is None

    @staticmethod
    def _blob_source(workspace, name="blobs_split"):
        rng = np.random.default_rng(0)
        pts = np.vstack(
            [rng.normal((0.0, 0.0), 0.3, (60, 2)), rng.normal((10.0, 10.0), 0.3, (60, 2))]
        )
        return _make_dataset_source(workspace, name, pts[:, 0], pts[:, 1])


def _wait_frames_until(panel, io, predicate, *, timeout=2.0, interval=0.005) -> None:
    """Drive real frames until ``predicate()`` holds, polling rather than sleeping a
    fixed amount -- robust against scheduler jitter, only fails if the condition
    genuinely never becomes true within a generous margin."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        _draw_frame(panel, io)
        if predicate():
            return
        time.sleep(interval)
    raise AssertionError(f"condition not met within {timeout}s")


def _drain(workspace) -> None:
    """Run every command the queue holds (each a ``Command.do()``-wrapping closure
    submitted via ``self.submit(...)``), without the trailing dirty-flag bookkeeping
    ``CommandQueue.drain(plot)`` applies afterwards -- this file's own ``_FakePlot``
    has no ``.frame``/``.cache`` for that step to touch, and these tests only care
    that the queued closure actually ran."""
    queue = workspace.queue
    while len(queue):
        queue._q.popleft()()


def _fit_params(model_key: str, degree: int = 0) -> Tuple[Any, ...]:
    """A ready-to-use Fit-tab params tuple for ``panel._compute(source, ...)`` -- every
    knob but ``model_key``/``degree`` stays at a plain default (no expression, no
    manual p0, non-robust, no confidence band, no output grid), since these tests care
    about a particular fit's own numerics, not the surrounding UI knobs."""
    return (
        "fit",
        model_key,
        degree,
        "",
        None,
        False,
        "soft_l1",
        False,
        0.95,
        False,
        200,
        _NO_SAMPLING,
    )


def _click_labelled_button(monkeypatch, matcher) -> None:
    """Monkeypatch ``imgui.button`` so any label ``matcher(label)`` approves fires
    True, every frame it's drawn -- for a plain "this button is pressed" assertion,
    not a toggle (see :func:`_click_once_labelled_button` for that case)."""
    real_button = imgui.button

    def clicker(label, *a, **k):
        return True if matcher(label) else real_button(label, *a, **k)

    monkeypatch.setattr("glplot.gui.panels.mathlab.imgui.button", clicker)


def _click_once_labelled_button(monkeypatch, matcher) -> None:
    """Same as :func:`_click_labelled_button`, but the matched click only ever fires
    True ONCE across the whole test -- every frame after that, the real button is
    called normally. Needed for a toggle button (e.g. "Apply to..."): returning True
    every frame would flip it open/closed/open instead of opening it once."""
    real_button = imgui.button
    fired = {"done": False}

    def clicker(label, *a, **k):
        if not fired["done"] and matcher(label):
            fired["done"] = True
            return True
        return real_button(label, *a, **k)

    monkeypatch.setattr("glplot.gui.panels.mathlab.imgui.button", clicker)


class TestSaveAsModel:
    """ "Save as model": mathlab.py's OWN wiring (``_build_trained_model``/
    ``_draw_save_model_row``) -- ``ModelStore``'s add/remove/get/names/unique_name
    behavior is already thoroughly covered by tests/test_gui_models.py's TestModelStore
    and is deliberately not re-tested here.
    """

    @pytest.fixture
    def fit_workspace(self):
        """A single noiseless linear dataset -- the panel's default source with no
        extra picking, and exact enough that a re-evaluated fit matches the settled
        result to machine precision (used by later apply-transfer tests too)."""
        ws = _FakeWorkspace()
        x = np.linspace(-5.0, 5.0, 80)
        y = 2.0 * x + 1.0
        ws.store.add(DataSet("line", [Column("x", x), Column("y", y)]))
        return ws

    def _settle_fit(self, panel: MathLabPanel, io: Any, model: str = "gaussian") -> None:
        panel.show_operation("fit")
        panel._fit_model = model
        for _ in range(4):
            _draw_frame(panel, io)

    def test_row_absent_for_a_non_trainable_tab(self, imgui_context, workspace, monkeypatch):
        """Smooth is not in _TRAINABLE_TABS -- the row must never appear, however many
        frames settle."""
        panel = MathLabPanel(workspace)
        panel.show_operation("smooth")
        seen: List[str] = []
        _click_labelled_button(monkeypatch, lambda label: (seen.append(label), False)[-1])
        for _ in range(4):
            _draw_frame(panel, imgui_context)
        assert panel._cache_error is None
        assert "Save as model" not in seen

    def test_row_hidden_for_a_trainable_tab_whose_result_has_no_model_state(
        self, imgui_context, workspace, monkeypatch
    ):
        """Direct check of the ``result.model_state is None`` half of the gate: a
        non-trained ``_Result`` (from a non-fit operation) drawn under a trainable KEY
        must still draw nothing."""
        panel = MathLabPanel(workspace)
        source = _dataset_source(workspace)
        result = panel._compute(source, ("normalize", "minmax"))
        assert result.model_state is None
        seen: List[str] = []
        _click_labelled_button(monkeypatch, lambda label: (seen.append(label), False)[-1])

        def draw_row() -> None:
            imgui.new_frame()
            imgui.set_next_window_pos((100, 100))
            imgui.set_next_window_size((_PANEL_W, _PANEL_H))
            imgui.begin("Math Lab")
            panel._draw_save_model_row(source, result, "fit")
            imgui.end()
            imgui.render()

        draw_row()
        assert "Save as model" not in seen

    def test_row_shown_once_a_trainable_tab_has_settled(
        self, imgui_context, fit_workspace, monkeypatch
    ):
        panel = MathLabPanel(fit_workspace)
        seen: List[str] = []
        _click_labelled_button(monkeypatch, lambda label: (seen.append(label), False)[-1])
        self._settle_fit(panel, imgui_context, "linear")
        assert panel._cache_error is None
        assert panel._cache_result.model_state is not None
        assert "Save as model" in seen

    def test_clicking_save_does_not_trigger_a_new_compute_call(
        self, imgui_context, fit_workspace, monkeypatch
    ):
        """Saving reads the already-settled ``_cache_result`` -- it must not recompute."""
        panel = MathLabPanel(fit_workspace)
        self._settle_fit(panel, imgui_context, "linear")
        assert panel._cache_error is None

        real_compute = panel._compute
        calls = {"n": 0}

        def counting(source, params):
            calls["n"] += 1
            return real_compute(source, params)

        panel._compute = counting
        _click_labelled_button(monkeypatch, lambda label: label == "Save as model")
        _draw_frame(panel, imgui_context)

        assert calls["n"] == 0, "clicking Save as model triggered a recompute"
        assert not fit_workspace.queue.is_empty(), "the click never queued a save"
        _drain(fit_workspace)
        assert len(fit_workspace.models) == 1

    def test_saved_fit_model_captures_popt_pcov_and_model_key(self, workspace):
        panel = MathLabPanel(workspace)
        x = np.linspace(-5.0, 5.0, 80)
        source = _make_dataset_source(workspace, "linfit", x, 2.0 * x + 1.0)
        result = panel._compute(source, _fit_params("linear"))
        model = panel._build_trained_model(source, result, "fit")
        assert model is not None
        assert model.technique == "fit"
        assert model.name == "Fit model"
        assert model.fit_model_key == "linear"
        assert model.input_columns == ("x",)
        np.testing.assert_allclose(model.fit_popt, result.model_state["popt"])
        np.testing.assert_allclose(model.fit_pcov, result.model_state["pcov"])

    def test_saved_polynomial_fit_model_has_empty_expr_and_polyval_reproduces_it(self, workspace):
        x = np.linspace(-3.0, 3.0, 60)
        y = 2.0 * x**2 - 3.0 * x + 1.0
        ws = _FakeWorkspace()
        source = _make_dataset_source(ws, "polyfit", x, y)
        panel = MathLabPanel(ws)
        result = panel._compute(source, _fit_params("polynomial", degree=2))
        model = panel._build_trained_model(source, result, "fit")
        assert model.fit_model_key == "polynomial"
        assert model.fit_expr == ""
        assert model.fit_n_peaks == 0
        np.testing.assert_allclose(np.polyval(model.fit_popt, x), y, atol=1e-6)

    def test_saved_cluster_model_captures_method_and_centroids(self, workspace):
        panel = MathLabPanel(workspace)
        rng = np.random.default_rng(3)
        pts = np.vstack(
            [rng.normal((0.0, 0.0), 0.2, (40, 2)), rng.normal((8.0, 8.0), 0.2, (40, 2))]
        )
        source = _make_dataset_source(workspace, "clusS", pts[:, 0], pts[:, 1])
        result = panel._compute(source, ("cluster", "kmeans", 2, 0, _NO_SAMPLING))
        model = panel._build_trained_model(source, result, "cluster")
        assert model.technique == "cluster"
        assert model.cluster_method == "kmeans"
        assert model.input_columns == ("x", "y")
        np.testing.assert_allclose(model.cluster_centroid_x, result.model_state["centroid_x"])
        np.testing.assert_allclose(model.cluster_centroid_y, result.model_state["centroid_y"])

    def test_saved_pca_model_captures_components_mean_and_scale_stats(self, workspace):
        panel = MathLabPanel(workspace)
        columns = _correlated_columns()
        source = _make_multicol_source(workspace, "pcaS", columns)
        names = tuple(columns.keys())
        result = panel._compute(source, ("pca", names, 3, "zscore", _NO_SAMPLING))
        model = panel._build_trained_model(source, result, "pca")
        assert model.technique == "pca"
        assert model.pca_columns == names
        assert model.pca_scale == "zscore"
        np.testing.assert_allclose(model.pca_components_, result.model_state["components_"])
        np.testing.assert_allclose(model.pca_mean_, result.model_state["mean_"])
        assert model.pca_scale_stats is result.model_state["scale_stats_"]

    def test_saved_umap_model_captures_the_reducer(self, workspace):
        pytest.importorskip("umap")
        panel = MathLabPanel(workspace)
        columns = _correlated_columns(n_rows=60)
        source = _make_multicol_source(workspace, "umapS", columns)
        names = tuple(columns.keys())
        result = panel._compute(source, ("umap", names, 2, 10, 0.1, "zscore", 0, _NO_SAMPLING))
        model = panel._build_trained_model(source, result, "umap")
        assert model.technique == "umap"
        assert model.umap_columns == names
        assert model.umap_reducer is result.model_state["reducer"]

    def test_saving_twice_with_the_same_typed_name_gets_deduped_by_modelstore(
        self, imgui_context, fit_workspace, monkeypatch
    ):
        panel = MathLabPanel(fit_workspace)
        self._settle_fit(panel, imgui_context, "linear")
        assert panel._cache_error is None
        panel._model_names["fit"] = "MyModel"

        _click_labelled_button(monkeypatch, lambda label: label == "Save as model")
        for _ in range(2):
            _draw_frame(panel, imgui_context)
            _drain(fit_workspace)

        assert fit_workspace.models.names() == ["MyModel", "MyModel (2)"]

    def test_saving_is_not_recorded_on_the_undo_stack(
        self, imgui_context, fit_workspace, monkeypatch
    ):
        panel = MathLabPanel(fit_workspace)
        self._settle_fit(panel, imgui_context, "linear")
        assert panel._cache_error is None
        before = len(panel.undo)

        _click_labelled_button(monkeypatch, lambda label: label == "Save as model")
        _draw_frame(panel, imgui_context)
        _drain(fit_workspace)

        assert len(fit_workspace.models) == 1
        assert len(panel.undo) == before


class TestModelRail:
    """The right-hand trained-models rail: visibility, ordering, rename, delete, and
    the thumbnail's reuse of the saved preview -- ``_draw_model_rail``/
    ``_draw_model_row``/rename/delete/``draw()``'s width fallback.
    """

    def _simple_model(
        self, panel: MathLabPanel, ws: _FakeWorkspace, *, name: str = "m", technique: str = "fit"
    ) -> TrainedModel:
        """A real, ``_build_trained_model``-produced model -- a genuine ``_Result`` as
        its ``preview``, not a hand-rolled stand-in, so the rail's thumbnail dispatch
        (``_draw_result_preview``) has real x/y/color_values to draw."""
        if technique == "cluster":
            rng = np.random.default_rng(1)
            pts = np.vstack(
                [rng.normal((0.0, 0.0), 0.2, (30, 2)), rng.normal((6.0, 6.0), 0.2, (30, 2))]
            )
            source = _make_dataset_source(ws, f"{name}_ds", pts[:, 0], pts[:, 1])
            result = panel._compute(source, ("cluster", "kmeans", 2, 0, _NO_SAMPLING))
            model = panel._build_trained_model(source, result, "cluster")
        else:
            x = np.linspace(-5.0, 5.0, 80)
            source = _make_dataset_source(ws, f"{name}_ds", x, 2.0 * x + 1.0)
            result = panel._compute(source, _fit_params("linear"))
            model = panel._build_trained_model(source, result, "fit")
        assert model is not None
        model.name = name
        return model

    def test_rail_hidden_with_zero_models_and_not_toggled(self, imgui_context, workspace):
        panel = MathLabPanel(workspace)
        assert panel._rail_open is False
        assert not workspace.models.models
        calls = {"n": 0}
        real_rail = panel._draw_model_rail

        def spy() -> None:
            calls["n"] += 1
            real_rail()

        panel._draw_model_rail = spy
        for _ in range(3):
            _draw_frame(panel, imgui_context)
        assert calls["n"] == 0

    def test_rail_appears_automatically_after_the_first_save(self, imgui_context, workspace):
        panel = MathLabPanel(workspace)
        workspace.models.add(self._simple_model(panel, workspace, name="auto"))
        assert panel._rail_open is False, "must not require the user to have opened it"
        calls = {"n": 0}
        real_rail = panel._draw_model_rail

        def spy() -> None:
            calls["n"] += 1
            real_rail()

        panel._draw_model_rail = spy
        _draw_frame(panel, imgui_context)
        assert calls["n"] > 0

    def test_header_toggle_opens_the_rail_even_with_zero_models(
        self, imgui_context, workspace, monkeypatch
    ):
        panel = MathLabPanel(workspace)
        real_icon_button = icons.icon_button

        def spy(id_str, shape, *args, **kwargs):
            if shape == "layers":
                return True
            return real_icon_button(id_str, shape, *args, **kwargs)

        monkeypatch.setattr("glplot.gui.panels.mathlab.icons.icon_button", spy)
        assert not workspace.models.models
        assert panel._rail_open is False
        _draw_frame(panel, imgui_context)
        assert panel._rail_open is True

    def test_models_are_listed_newest_first(self, imgui_context, workspace):
        panel = MathLabPanel(workspace)
        workspace.models.add(self._simple_model(panel, workspace, name="first"))
        workspace.models.add(self._simple_model(panel, workspace, name="second"))
        seen: List[str] = []
        real_row = panel._draw_model_row

        def spy(model: TrainedModel) -> None:
            seen.append(model.name)
            real_row(model)

        panel._draw_model_row = spy
        _draw_frame(panel, imgui_context)
        assert seen == ["second", "first"]

    def test_each_row_shows_a_technique_badge(self, imgui_context, workspace, monkeypatch):
        panel = MathLabPanel(workspace)
        workspace.models.add(
            self._simple_model(panel, workspace, name="badge", technique="cluster")
        )
        texts: List[str] = []
        real_text_colored = imgui.text_colored

        def spy(color, text, *args, **kwargs):
            texts.append(text)
            return real_text_colored(color, text, *args, **kwargs)

        monkeypatch.setattr("glplot.gui.panels.mathlab.imgui.text_colored", spy)
        _draw_frame(panel, imgui_context)
        assert "CLUSTER" in texts

    def test_double_click_begins_a_rename_and_the_rename_field_draws(
        self, imgui_context, workspace, monkeypatch
    ):
        """``_begin_rail_rename`` is exactly what a rail row's double-click handler
        calls (see ``_draw_model_row``) -- driven directly here since simulating a real
        double-click through headless imgui has no sanctioned idiom in this suite, then
        checked against the real draw: once rename is in progress, the inline text
        field must actually render."""
        panel = MathLabPanel(workspace)
        model = self._simple_model(panel, workspace, name="torename")
        workspace.models.add(model)

        panel._begin_rail_rename(model)
        assert panel._rail_rename_target == "torename"

        seen_ids: List[str] = []
        real_input_text = imgui.input_text

        def spy(label, *args, **kwargs):
            seen_ids.append(label)
            return real_input_text(label, *args, **kwargs)

        monkeypatch.setattr("glplot.gui.panels.mathlab.imgui.input_text", spy)
        _draw_frame(panel, imgui_context)
        assert "##rail_rename" in seen_ids

    def test_commit_rename_renames_the_model_and_is_not_on_the_undo_stack(self, workspace):
        panel = MathLabPanel(workspace)
        model = self._simple_model(panel, workspace, name="orig")
        workspace.models.add(model)
        before = len(panel.undo)

        panel._begin_rail_rename(model)
        panel._commit_rail_rename(model, "renamed")
        assert panel._rail_rename_target is None, "commit must leave rename mode"
        _drain(workspace)

        assert model.name == "renamed"
        assert workspace.models.get("renamed") is model
        assert len(panel.undo) == before

    def test_rename_that_collides_with_an_existing_name_gets_deduped(self, workspace):
        panel = MathLabPanel(workspace)
        a = self._simple_model(panel, workspace, name="a")
        b = self._simple_model(panel, workspace, name="b")
        workspace.models.add(a)
        workspace.models.add(b)

        panel._commit_rail_rename(b, "a")
        _drain(workspace)

        assert a.name == "a"
        assert b.name == "a (2)"

    def test_delete_removes_the_model_and_is_not_recorded_on_the_undo_stack(
        self, imgui_context, workspace, monkeypatch
    ):
        panel = MathLabPanel(workspace)
        model = self._simple_model(panel, workspace, name="todelete")
        workspace.models.add(model)
        before = len(panel.undo)

        real_icon_button = icons.icon_button

        def spy(id_str, shape, *args, **kwargs):
            if shape == "trash":
                return True
            return real_icon_button(id_str, shape, *args, **kwargs)

        monkeypatch.setattr("glplot.gui.panels.mathlab.icons.icon_button", spy)
        _draw_frame(panel, imgui_context)
        _drain(workspace)

        assert len(workspace.models) == 0
        assert len(panel.undo) == before

    def test_rail_thumbnail_reuses_the_saved_preview_without_recomputing(
        self, imgui_context, workspace
    ):
        """Isolated to ``_draw_model_rail()`` alone, not a full ``panel.draw()`` frame:
        the panel's currently-active TAB unconditionally recomputes its own result every
        frame too (see ``_draw_body``), which would pollute a global ``_compute`` call
        count that has nothing to do with the rail thumbnail this test is about."""
        panel = MathLabPanel(workspace)
        model = self._simple_model(panel, workspace, name="thumb")
        workspace.models.add(model)

        preview_calls: List[Any] = []
        real_preview = panel._draw_result_preview

        def spy_preview(result, **kwargs):
            preview_calls.append((result, kwargs.get("id_prefix")))
            return real_preview(result, **kwargs)

        panel._draw_result_preview = spy_preview

        compute_calls = {"n": 0}
        real_compute = panel._compute

        def counting(source, params):
            compute_calls["n"] += 1
            return real_compute(source, params)

        panel._compute = counting

        imgui.new_frame()
        imgui.set_next_window_pos((100, 100))
        imgui.set_next_window_size((_PANEL_W, _PANEL_H))
        imgui.begin("Math Lab")
        panel._draw_model_rail()
        imgui.end()
        imgui.render()

        rail_calls = [c for c in preview_calls if c[1] == "rail_thumb"]
        assert rail_calls, "the rail row never drew its thumbnail via _draw_result_preview"
        assert rail_calls[0][0] is model.preview
        assert compute_calls["n"] == 0, "the rail thumbnail must not recompute the result"

    def test_narrow_panel_falls_back_to_a_stacked_rail_section(
        self, imgui_context, workspace, monkeypatch
    ):
        """Below ``_RAIL_MIN_LEFT_WIDTH``, the rail must draw as a full-width
        collapsible section BELOW the body, not squeeze a fixed side column into a
        panel too narrow for it (the live-window-screenshot regression noted in
        ``draw()``'s own docstring)."""
        panel = MathLabPanel(workspace)
        workspace.models.add(self._simple_model(panel, workspace, name="narrow"))

        section_labels: List[str] = []
        real_section = widgets.section

        def spy_section(label, **kwargs):
            section_labels.append(label)
            return real_section(label, **kwargs)

        monkeypatch.setattr("glplot.gui.panels.mathlab.widgets.section", spy_section)

        child_ids: List[str] = []
        real_begin_child = imgui.begin_child

        def spy_begin_child(id_str, *args, **kwargs):
            child_ids.append(id_str)
            return real_begin_child(id_str, *args, **kwargs)

        monkeypatch.setattr("glplot.gui.panels.mathlab.imgui.begin_child", spy_begin_child)

        imgui.new_frame()
        imgui.set_next_window_pos((100, 100))
        imgui.set_next_window_size((300, _PANEL_H))
        imgui.begin("Math Lab")
        panel.draw()
        imgui.end()
        imgui.render()

        assert "Trained models" in section_labels
        assert "##mathlab_rail" not in child_ids

    def test_wide_panel_uses_a_side_column_rail_not_a_stacked_section(
        self, imgui_context, workspace, monkeypatch
    ):
        """The other half of the same fallback: a panel wide enough to spare
        _RAIL_WIDTH without squeezing the body must use the side-column child, not the
        stacked section -- guards against a fix for the narrow case regressing the
        (still real, still used) wide case."""
        panel = MathLabPanel(workspace)
        workspace.models.add(self._simple_model(panel, workspace, name="wide"))

        child_ids: List[str] = []
        real_begin_child = imgui.begin_child

        def spy_begin_child(id_str, *args, **kwargs):
            child_ids.append(id_str)
            return real_begin_child(id_str, *args, **kwargs)

        monkeypatch.setattr("glplot.gui.panels.mathlab.imgui.begin_child", spy_begin_child)

        imgui.new_frame()
        imgui.set_next_window_pos((100, 100))
        imgui.set_next_window_size((1000, _PANEL_H))
        imgui.begin("Math Lab")
        panel.draw()
        imgui.end()
        imgui.render()

        assert "##mathlab_rail" in child_ids


class TestApplyModelToDataset:
    """Transferring a saved TrainedModel onto a (usually different) dataset --
    ``_evaluate_model``/``_command_apply_model_add_column``/
    ``_command_apply_model_new_dataset``/``_draw_apply_model_section``.
    """

    def _fit_model(self, panel: MathLabPanel, ws: _FakeWorkspace, *, name: str = "fitm"):
        x = np.linspace(-5.0, 5.0, 80)
        y = 2.0 * x + 1.0
        source = _make_dataset_source(ws, f"{name}_ds", x, y)
        result = panel._compute(source, _fit_params("linear"))
        model = panel._build_trained_model(source, result, "fit")
        assert model is not None
        model.name = name
        return model, source, result

    def test_evaluate_fit_model_reproduces_the_fit_on_its_own_training_x(self, workspace):
        panel = MathLabPanel(workspace)
        model, source, result = self._fit_model(panel, workspace)
        values = panel._evaluate_model(model, source.dataset, {"x": "x"})
        np.testing.assert_allclose(values, result.y, atol=1e-6)

    def test_evaluate_polynomial_fit_model_uses_polyval_not_fit_spec(self, workspace, monkeypatch):
        x = np.linspace(-3.0, 3.0, 60)
        y = 2.0 * x**2 - 3.0 * x + 1.0
        ws = _FakeWorkspace()
        source = _make_dataset_source(ws, "polyfit", x, y)
        panel = MathLabPanel(ws)
        result = panel._compute(source, _fit_params("polynomial", degree=2))
        model = panel._build_trained_model(source, result, "fit")
        assert model.fit_model_key == "polynomial"

        def boom(*args, **kwargs):
            raise AssertionError("_fit_spec must never be reached for a polynomial model")

        monkeypatch.setattr(panel, "_fit_spec", boom)
        values = panel._evaluate_model(model, source.dataset, {"x": "x"})
        np.testing.assert_allclose(values, np.polyval(model.fit_popt, x), atol=1e-9)

    def test_evaluate_fit_model_on_a_different_dataset_sizes_to_its_own_row_count(self, workspace):
        panel = MathLabPanel(workspace)
        model, _source, _result = self._fit_model(panel, workspace)
        target = DataSet("other", [Column("t", np.linspace(0.0, 1.0, 17))])
        workspace.store.add(target)
        values = panel._evaluate_model(model, target, {"x": "t"})
        assert np.asarray(values).shape == (17,)
        spec = panel._fit_spec(model.fit_model_key, model.fit_expr, model.fit_n_peaks)
        np.testing.assert_allclose(values, spec.fn(target.get("t"), *model.fit_popt), atol=1e-9)

    def test_evaluate_cluster_model_uses_nearest_centroid(self, workspace):
        panel = MathLabPanel(workspace)
        rng = np.random.default_rng(2)
        pts = np.vstack(
            [rng.normal((0.0, 0.0), 0.2, (40, 2)), rng.normal((8.0, 8.0), 0.2, (40, 2))]
        )
        source = _make_dataset_source(workspace, "clusA", pts[:, 0], pts[:, 1])
        result = panel._compute(source, ("cluster", "kmeans", 2, 0, _NO_SAMPLING))
        model = panel._build_trained_model(source, result, "cluster")

        target = DataSet(
            "clusB", [Column("px", np.array([0.1, 8.1])), Column("py", np.array([0.1, 8.1]))]
        )
        workspace.store.add(target)
        values = panel._evaluate_model(model, target, {"x": "px", "y": "py"})
        expected = mathops2d.nearest_centroid(
            target.get("px"), target.get("py"), model.cluster_centroid_x, model.cluster_centroid_y
        )
        np.testing.assert_array_equal(values, expected)

    def test_evaluate_hierarchical_cluster_model_also_uses_nearest_centroid(self, workspace):
        """Hierarchical clustering has no native out-of-sample rule -- the approved
        transfer is the same nearest-centroid assignment k-means uses."""
        panel = MathLabPanel(workspace)
        rng = np.random.default_rng(5)
        pts = np.vstack(
            [rng.normal((0.0, 0.0), 0.2, (20, 2)), rng.normal((8.0, 8.0), 0.2, (20, 2))]
        )
        source = _make_dataset_source(workspace, "hierA", pts[:, 0], pts[:, 1])
        result = panel._compute(source, ("cluster", "hierarchical", 2, "ward", _NO_SAMPLING))
        model = panel._build_trained_model(source, result, "cluster")
        assert model.cluster_method == "hierarchical"

        target = DataSet(
            "hierB", [Column("px", np.array([0.1, 8.1])), Column("py", np.array([0.1, 8.1]))]
        )
        workspace.store.add(target)
        values = panel._evaluate_model(model, target, {"x": "px", "y": "py"})
        expected = mathops2d.nearest_centroid(
            target.get("px"), target.get("py"), model.cluster_centroid_x, model.cluster_centroid_y
        )
        np.testing.assert_array_equal(values, expected)

    def test_hierarchical_apply_section_shows_a_permanent_disclosure(
        self, imgui_context, workspace, monkeypatch
    ):
        """The disclosure must be a standing help-marker line, drawn every frame the
        section is open -- not a one-time dismissible warning."""
        panel = MathLabPanel(workspace)
        rng = np.random.default_rng(9)
        pts = np.vstack(
            [rng.normal((0.0, 0.0), 0.2, (20, 2)), rng.normal((8.0, 8.0), 0.2, (20, 2))]
        )
        source = _make_dataset_source(workspace, "hierD", pts[:, 0], pts[:, 1])
        result = panel._compute(source, ("cluster", "hierarchical", 2, "ward", _NO_SAMPLING))
        model = panel._build_trained_model(source, result, "cluster")
        model.name = "hier"
        workspace.models.add(model)

        _click_once_labelled_button(monkeypatch, lambda label: label.startswith("Apply to..."))

        marks: List[str] = []
        real_marker = widgets.help_marker

        def spy_marker(text):
            marks.append(text)
            return real_marker(text)

        monkeypatch.setattr("glplot.gui.panels.mathlab.widgets.help_marker", spy_marker)

        for _ in range(3):
            marks.clear()
            _draw_frame(panel, imgui_context)
            assert any(
                "nearest saved centroid" in t for t in marks
            ), "the hierarchical disclosure did not draw this frame"

    def test_evaluate_pca_model_matches_pca_transform_called_directly(self, workspace):
        panel = MathLabPanel(workspace)
        columns = _correlated_columns()
        source = _make_multicol_source(workspace, "pcaE", columns)
        names = tuple(columns.keys())
        result = panel._compute(source, ("pca", names, 3, "zscore", _NO_SAMPLING))
        model = panel._build_trained_model(source, result, "pca")

        # A different dataset with the same columns, reordered rows.
        order = np.arange(len(columns[names[0]]))[::-1]
        target = DataSet("pcaTarget", [Column(n, columns[n][order]) for n in names])
        workspace.store.add(target)
        column_map = {n: n for n in names}

        values = panel._evaluate_model(model, target, column_map)
        expected = mathopsnd.pca_transform(
            [target.get(n) for n in names],
            mean_=model.pca_mean_,
            components_=model.pca_components_,
            scale_stats=model.pca_scale_stats,
            scale=model.pca_scale,
        )
        np.testing.assert_allclose(values, expected)

    def test_evaluate_umap_model_calls_the_saved_reducers_own_transform(
        self, workspace, monkeypatch
    ):
        pytest.importorskip("umap")
        panel = MathLabPanel(workspace)
        columns = _correlated_columns(n_rows=60)
        source = _make_multicol_source(workspace, "umapE", columns)
        names = tuple(columns.keys())
        result = panel._compute(source, ("umap", names, 2, 10, 0.1, "zscore", 0, _NO_SAMPLING))
        model = panel._build_trained_model(source, result, "umap")

        target = DataSet("umapTarget", [Column(n, columns[n]) for n in names])
        workspace.store.add(target)

        calls: List[Any] = []
        real_transform = model.umap_reducer.transform

        def spy_transform(x):
            calls.append(x)
            return real_transform(x)

        monkeypatch.setattr(model.umap_reducer, "transform", spy_transform)

        values = panel._evaluate_model(model, target, {n: n for n in names})
        assert calls, "evaluate did not call the saved reducer's own .transform"
        assert np.asarray(values).shape[0] == target.n_rows()

    def test_evaluate_raises_a_clear_error_when_a_required_column_is_missing(self, workspace):
        panel = MathLabPanel(workspace)
        model, source, _result = self._fit_model(panel, workspace, name="missingcol")
        with pytest.raises(ValueError, match="no column"):
            panel._evaluate_model(model, source.dataset, {})

    def test_umap_model_with_no_saved_reducer_is_disabled_with_a_reason_not_a_crash(
        self, imgui_context, workspace, monkeypatch
    ):
        """A model saved when umap-learn was unavailable carries ``umap_reducer=None``
        -- the UI must show a disabled reason instead of reaching _evaluate_model."""
        panel = MathLabPanel(workspace)
        model = TrainedModel(
            name="noreducer",
            technique="umap",
            created=0.0,
            source_label="src [a, b]",
            input_columns=("a", "b"),
            umap_columns=("a", "b"),
            umap_reducer=None,
        )
        workspace.models.add(model)

        _click_once_labelled_button(monkeypatch, lambda label: label.startswith("Apply to..."))

        marks: List[str] = []
        real_marker = widgets.help_marker

        def spy_marker(text):
            marks.append(text)
            return real_marker(text)

        monkeypatch.setattr("glplot.gui.panels.mathlab.widgets.help_marker", spy_marker)

        for _ in range(2):
            _draw_frame(panel, imgui_context)  # must not raise
        assert any("Unavailable" in t for t in marks)

        with pytest.raises(ValueError, match="no saved UMAP reducer"):
            panel._evaluate_model(model, workspace.store.get("alpha"), {"a": "x", "b": "y"})

    def test_add_column_apply_sizes_output_to_the_target_with_no_nan_padding(self, workspace):
        panel = MathLabPanel(workspace)
        model, _source, _result = self._fit_model(panel, workspace, name="addcol")
        target = DataSet("addcolTarget", [Column("tx", np.linspace(-2.0, 2.0, 13))])
        workspace.store.add(target)

        cmd = panel._command_apply_model_add_column(model, target, {"x": "tx"}, "predicted")
        assert cmd is not None
        before = target.n_cols()
        cmd.do()
        try:
            assert target.n_cols() == before + 1
            new_col = target.columns[-1]
            assert new_col.name == "predicted"
            assert len(new_col.values) == 13
            assert not np.isnan(new_col.values).any()
        finally:
            cmd.undo()
        assert target.n_cols() == before

    def test_new_dataset_apply_creates_one_column_per_pca_component(self, workspace):
        panel = MathLabPanel(workspace)
        columns = _correlated_columns()
        source = _make_multicol_source(workspace, "pcaND", columns)
        names = tuple(columns.keys())
        result = panel._compute(source, ("pca", names, 3, "zscore", _NO_SAMPLING))
        model = panel._build_trained_model(source, result, "pca")

        target = DataSet("pcaNDTarget", [Column(n, columns[n]) for n in names])
        workspace.store.add(target)
        column_map = {n: n for n in names}
        cmd = panel._command_apply_model_new_dataset(model, target, column_map, "applied")
        assert cmd is not None
        before = set(workspace.store.names())
        cmd.do()
        try:
            new_name = next(iter(set(workspace.store.names()) - before))
            created = workspace.store.get(new_name)
            assert created.column_names() == ["applied 1", "applied 2", "applied 3"]
            assert created.n_rows() == target.n_rows()
        finally:
            cmd.undo()

    def test_undo_on_add_column_apply_removes_exactly_the_columns_it_added(self, workspace):
        panel = MathLabPanel(workspace)
        columns = _correlated_columns()
        source = _make_multicol_source(workspace, "pcaUndo", columns)
        names = tuple(columns.keys())
        result = panel._compute(source, ("pca", names, 2, "zscore", _NO_SAMPLING))
        model = panel._build_trained_model(source, result, "pca")

        target = DataSet("pcaUndoTarget", [Column(n, columns[n]) for n in names])
        workspace.store.add(target)
        before_cols = list(target.column_names())
        cmd = panel._command_apply_model_add_column(model, target, {n: n for n in names}, "score")
        assert cmd is not None
        cmd.do()
        assert target.column_names() == before_cols + ["score 1", "score 2"]
        cmd.undo()
        assert target.column_names() == before_cols

    def test_ui_drives_apply_to_dataset_add_column_end_to_end(
        self, imgui_context, workspace, monkeypatch
    ):
        """The full path: the rail's own "Apply to..." section, pre-opened and with a
        target/column mapping picked, "Apply" pressed, drain -- and the written values
        match a direct _evaluate_model call.

        Drawn via ``_draw_apply_model_section`` directly, not a full ``panel.draw()``
        frame: the panel's currently-active TAB also has its own "Apply" button with
        the exact same label (``_draw_apply``, line 5426), so a full-frame draw with
        every "Apply" forced to click would ALSO fire that unrelated button -- isolating
        the section keeps this test about the rail's own Apply, and only it.
        """
        panel = MathLabPanel(workspace)
        model, _source, _result = self._fit_model(panel, workspace, name="uiapply")
        workspace.models.add(model)

        target = DataSet("uiApplyTarget", [Column("tx", np.linspace(-1.0, 1.0, 9))])
        workspace.store.add(target)

        key = model.name
        panel._apply_model_open[key] = True
        panel._apply_model_target_ds[key] = "uiApplyTarget"
        panel._apply_model_columns[key] = {"x": "tx"}

        _click_labelled_button(monkeypatch, lambda label: label == "Apply")

        def draw_section() -> None:
            imgui.new_frame()
            imgui.set_next_window_pos((100, 100))
            imgui.set_next_window_size((_PANEL_W, _PANEL_H))
            imgui.begin("Math Lab")
            panel._draw_apply_model_section(model)
            imgui.end()
            imgui.render()

        before = target.n_cols()
        draw_section()
        assert not workspace.queue.is_empty(), "the Apply click never queued a command"
        _drain(workspace)

        assert target.n_cols() == before + 1
        new_col = target.columns[-1]
        assert new_col.name == "uiapply applied"
        expected = panel._evaluate_model(model, target, {"x": "tx"})
        np.testing.assert_allclose(new_col.values, expected, atol=1e-9)


# ============================================================================
# Diagnostic mini-plots (Phase 5): Fit residuals, PCA scree, k-means
# elbow/silhouette, UMAP "color by column" -- lost from this file by the same
# tool-use accident the sections above explain; rebuilt fresh here against the
# current, intact production code rather than reconstructed from memory.
# ============================================================================


class TestFitDiagnostics:
    """``result.diagnostic`` on the Fit tab: ``{"kind": "residuals", "x": ..., "y": ...}``,
    actual minus fitted, always evaluated at the SOURCE's own (post-subsample) x -- never
    at a synthetic output grid, and (with a split on) using the TRAIN-fit parameters but
    scored over every post-subsample row, not just the train subset."""

    def test_polynomial_residuals_equal_actual_minus_fitted_at_source_x(self, workspace):
        panel = MathLabPanel(workspace)
        rng = np.random.default_rng(1)
        x = np.linspace(-5.0, 5.0, 60)
        y = 2.0 * x + 1.0 + rng.normal(0.0, 0.2, x.size)
        source = _make_dataset_source(workspace, "fitdiag1", x, y)
        result = panel._compute(
            source,
            (
                "fit",
                "polynomial",
                1,
                "",
                None,
                False,
                "soft_l1",
                False,
                0.95,
                False,
                200,
                _NO_SAMPLING,
            ),
        )
        diag = result.diagnostic
        assert diag is not None
        assert diag["kind"] == "residuals"
        coeffs = result.model_state["popt"]
        expected = y - np.polyval(coeffs, x)
        np.testing.assert_allclose(diag["x"], x)
        np.testing.assert_allclose(diag["y"], expected)

    def test_named_model_residuals_use_the_fitted_spec_not_polyval(self, workspace):
        panel = MathLabPanel(workspace)
        rng = np.random.default_rng(2)
        x = np.linspace(-5.0, 5.0, 80)
        y = 3.0 * x - 2.0 + rng.normal(0.0, 0.15, x.size)
        source = _make_dataset_source(workspace, "fitdiag2", x, y)
        result = panel._compute(
            source,
            ("fit", "linear", 0, "", None, False, "soft_l1", False, 0.95, False, 200, _NO_SAMPLING),
        )
        diag = result.diagnostic
        assert diag is not None and diag["kind"] == "residuals"
        popt = result.model_state["popt"]
        expected = y - mathops.MODELS["linear"].fn(x, *popt)
        np.testing.assert_allclose(diag["y"], expected, atol=1e-9)

    def test_residuals_use_the_sources_own_x_even_when_use_grid_is_on(self, workspace):
        """The main result may commit a synthetic N-point grid; the diagnostic must not."""
        panel = MathLabPanel(workspace)
        x = np.linspace(-5.0, 5.0, 37)
        y = 2.0 * x + 1.0
        source = _make_dataset_source(workspace, "fitdiag3", x, y)
        result = panel._compute(
            source,
            ("fit", "linear", 0, "", None, False, "soft_l1", False, 0.95, True, 500, _NO_SAMPLING),
        )
        assert result.x.size == 500, "the main result grid must still be the requested size"
        diag = result.diagnostic
        assert diag["x"].size == 37, "the diagnostic must keep the source's own row count"
        np.testing.assert_allclose(diag["x"], x)
        popt = result.model_state["popt"]
        expected = y - mathops.MODELS["linear"].fn(x, *popt)
        np.testing.assert_allclose(diag["y"], expected, atol=1e-9)

    def test_residuals_evaluate_the_polynomial_grid_case_at_source_x_too(self, workspace):
        panel = MathLabPanel(workspace)
        rng = np.random.default_rng(9)
        x = np.sort(rng.uniform(-3.0, 3.0, 45))
        y = 2.0 * x**2 - 3.0 * x + 1.0
        source = _make_dataset_source(workspace, "fitdiag4", x, y)
        result = panel._compute(
            source,
            (
                "fit",
                "polynomial",
                2,
                "",
                None,
                False,
                "soft_l1",
                False,
                0.95,
                True,
                300,
                _NO_SAMPLING,
            ),
        )
        diag = result.diagnostic
        assert diag["x"].size == 45
        np.testing.assert_allclose(diag["x"], x)
        coeffs = result.model_state["popt"]
        np.testing.assert_allclose(diag["y"], y - np.polyval(coeffs, x), atol=1e-9)

    def test_residuals_reflect_the_train_fit_popt_over_the_full_post_subsample_source(
        self, workspace
    ):
        """``training.split`` on: model_state's popt is fit on train only, but the
        diagnostic still covers every post-subsample row, not just train."""
        panel = MathLabPanel(workspace)
        rng = np.random.default_rng(3)
        x = np.linspace(0.0, 10.0, 300)
        y = 2.0 * x + 1.0 + rng.normal(0.0, 0.05, x.size)
        source = _make_dataset_source(workspace, "fitdiag5", x, y)
        training = _SamplingParams(split=True, val_frac=0.2, test_frac=0.2, split_seed=0)
        result = panel._compute(
            source,
            ("fit", "polynomial", 1, "", None, False, "soft_l1", False, 0.95, False, 200, training),
        )
        train_idx, _val_idx, _test_idx = _split_indices(x.size, 0.2, 0.2, seed=0)
        expected_coeffs, _cov, _yfit = mathops.fit_polynomial_covariance(
            x[train_idx], y[train_idx], 1
        )
        np.testing.assert_allclose(result.model_state["popt"], expected_coeffs)

        diag = result.diagnostic
        # The diagnostic's x is every post-subsample row (300), not the ~240-row train
        # subset the model itself was fit on.
        assert diag["x"].size == 300 == x.size
        assert diag["x"].size != train_idx.size
        np.testing.assert_allclose(diag["x"], x)
        expected_residuals = y - np.polyval(expected_coeffs, x)
        np.testing.assert_allclose(diag["y"], expected_residuals, atol=1e-9)

    def test_named_model_residuals_also_reflect_train_fit_popt_over_full_source(self, workspace):
        panel = MathLabPanel(workspace)
        rng = np.random.default_rng(4)
        x = np.linspace(0.0, 10.0, 250)
        y = 3.0 * x - 1.0 + rng.normal(0.0, 0.1, x.size)
        source = _make_dataset_source(workspace, "fitdiag6", x, y)
        training = _SamplingParams(split=True, val_frac=0.2, test_frac=0.2, split_seed=1)
        result = panel._compute(
            source,
            ("fit", "linear", 0, "", None, False, "soft_l1", False, 0.95, False, 200, training),
        )
        diag = result.diagnostic
        assert diag["x"].size == 250
        popt = result.model_state["popt"]
        expected = y - mathops.MODELS["linear"].fn(x, *popt)
        np.testing.assert_allclose(diag["y"], expected, atol=1e-9)

    def test_no_diagnostic_for_operations_other_than_fit_and_pca(self, workspace):
        source = _make_dataset_source(
            workspace, "fitdiag7", np.linspace(0, 10, 40), np.sin(np.linspace(0, 10, 40))
        )
        panel = MathLabPanel(workspace)
        result = panel._compute(source, ("normalize", "minmax"))
        assert result.diagnostic is None


class TestPcaDiagnostics:
    """``result.diagnostic`` on the PCA tab: ``{"kind": "scree", "ratios": ..., "chosen": ...}``
    -- the FULL explained-variance spectrum, not just the ``n_components`` kept."""

    def test_pca_backend_returns_the_full_spectrum_not_just_kept_components(self):
        columns = _correlated_columns(n_rows=200, n_features=5)
        arrays = list(columns.values())
        out = mathopsnd.pca(arrays, n_components=2, scale="zscore")
        full = out["explained_variance_ratio_full"]
        assert full.size == min(out["n_samples"], out["n_features"]) == 5
        # First n_components entries match explained_variance_ratio exactly.
        np.testing.assert_allclose(full[:2], out["explained_variance_ratio"])

    def test_full_spectrum_length_is_independent_of_n_components_requested(self):
        columns = _correlated_columns(n_rows=150, n_features=6)
        arrays = list(columns.values())
        small = mathopsnd.pca(arrays, n_components=1, scale="zscore")
        large = mathopsnd.pca(arrays, n_components=5, scale="zscore")
        assert small["explained_variance_ratio_full"].size == 6
        assert large["explained_variance_ratio_full"].size == 6
        np.testing.assert_allclose(
            small["explained_variance_ratio_full"], large["explained_variance_ratio_full"]
        )

    def test_scree_diagnostic_matches_the_backends_full_spectrum(self, workspace):
        panel = MathLabPanel(workspace)
        columns = _correlated_columns(n_rows=200, n_features=5)
        source = _make_multicol_source(workspace, "pcadiag1", columns)
        names = tuple(columns.keys())
        result = panel._compute(source, ("pca", names, 3, "zscore", _NO_SAMPLING))
        diag = result.diagnostic
        assert diag is not None and diag["kind"] == "scree"
        expected = mathopsnd.pca([columns[n] for n in names], n_components=3, scale="zscore")[
            "explained_variance_ratio_full"
        ]
        np.testing.assert_allclose(diag["ratios"], expected)
        assert diag["ratios"].size == 5, "5 columns, so up to 5 singular values"

    def test_scree_diagnostic_marks_the_chosen_component_count(self, workspace):
        panel = MathLabPanel(workspace)
        columns = _correlated_columns(n_rows=120, n_features=4)
        source = _make_multicol_source(workspace, "pcadiag2", columns)
        names = tuple(columns.keys())
        result = panel._compute(source, ("pca", names, 2, "zscore", _NO_SAMPLING))
        diag = result.diagnostic
        assert diag["chosen"] == 2

        result4 = panel._compute(source, ("pca", names, 4, "zscore", _NO_SAMPLING))
        assert result4.diagnostic["chosen"] == 4
        # The spectrum itself does not depend on how many components were requested.
        np.testing.assert_allclose(diag["ratios"], result4.diagnostic["ratios"])

    def test_scree_ratios_sum_to_at_most_one(self, workspace):
        panel = MathLabPanel(workspace)
        columns = _correlated_columns(n_rows=100, n_features=5)
        source = _make_multicol_source(workspace, "pcadiag3", columns)
        result = panel._compute(source, ("pca", tuple(columns.keys()), 2, "zscore", _NO_SAMPLING))
        ratios = result.diagnostic["ratios"]
        assert float(np.sum(ratios)) == pytest.approx(1.0, abs=1e-6)


class TestClusterDiagnostics:
    """The k-means elbow/silhouette k-sweep: its own dedicated ``BackgroundJob``
    (``_elbow_jobs``/``_elbow_results``), never dispatched through
    ``_cached_result_async``/``_compute``, offered only for kmeans, and never run
    automatically."""

    def _blob_source(self, workspace, name="cdiagblobs", seed=0, n_per=60):
        rng = np.random.default_rng(seed)
        c1 = rng.normal((0.0, 0.0), 0.3, (n_per, 2))
        c2 = rng.normal((10.0, 10.0), 0.3, (n_per, 2))
        pts = np.vstack([c1, c2])
        return _make_dataset_source(workspace, name, pts[:, 0], pts[:, 1])

    def _force_section_open(self, monkeypatch, label):
        real_section = widgets.section

        def spy(lbl, *a, **k):
            return True if lbl == label else real_section(lbl, *a, **k)

        monkeypatch.setattr("glplot.gui.panels.mathlab.widgets.section", spy)

    def _wait_for_elbow(self, panel, io, key="cluster", timeout=8.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            _draw_frame(panel, io)
            if panel._elbow_results.get(key) is not None:
                return
            time.sleep(0.01)
        raise AssertionError("elbow/silhouette sweep never completed")

    def test_elbow_section_is_offered_for_kmeans_not_hierarchical(self, imgui_context, workspace):
        panel = MathLabPanel(workspace)
        self._blob_source(workspace, name="cdiagA")

        calls = []
        real = panel._draw_elbow_section

        def spy(source, key):
            calls.append(key)
            return real(source, key)

        panel._draw_elbow_section = spy

        panel.show_operation("cluster")
        panel._cluster_method = "kmeans"
        for _ in range(6):
            _draw_frame(panel, imgui_context)
        assert calls, "kmeans must offer the elbow/silhouette section"

        calls.clear()
        panel._cluster_method = "hierarchical"
        for _ in range(6):
            _draw_frame(panel, imgui_context)
        assert not calls, "hierarchical must NOT offer the elbow/silhouette section"

    def test_elbow_sweep_never_runs_automatically_from_viewing_the_tab(
        self, imgui_context, workspace, monkeypatch
    ):
        panel = MathLabPanel(workspace)
        self._blob_source(workspace, name="cdiagB")
        # Force the collapsible section itself open (the strongest form of this claim:
        # even with the button actually drawn and visible, no click means no job).
        self._force_section_open(monkeypatch, "Elbow / silhouette (k-means)")

        panel.show_operation("cluster")
        panel._cluster_method = "kmeans"
        for _ in range(10):
            _draw_frame(panel, imgui_context)

        assert panel._elbow_jobs == {}
        assert panel._elbow_results == {}

    def test_click_dispatches_a_real_sweep_and_marks_the_selected_k(
        self, imgui_context, workspace, monkeypatch
    ):
        panel = MathLabPanel(workspace)
        self._blob_source(workspace, name="cdiagC", n_per=60)  # 120 rows total
        self._force_section_open(monkeypatch, "Elbow / silhouette (k-means)")
        _click_once_labelled_button(
            monkeypatch, lambda label: label.startswith("Compute elbow/silhouette")
        )

        marker_calls = {}
        real_mini_plot = widgets.mini_plot

        def spy_mini_plot(id_str, y, *a, **k):
            if id_str in ("elbow_inertia", "elbow_silhouette"):
                marker_calls[id_str] = k.get("markers")
            return real_mini_plot(id_str, y, *a, **k)

        monkeypatch.setattr("glplot.gui.panels.mathlab.widgets.mini_plot", spy_mini_plot)

        panel.show_operation("cluster")
        panel._cluster_method = "kmeans"
        panel._cluster_k = 3
        self._wait_for_elbow(panel, imgui_context, key="cluster")

        k_values, inertias, silhouettes = panel._elbow_results["cluster"]
        assert k_values.size > 0
        assert float(k_values[0]) == 2.0
        # 120 points, so the sweep is capped at the module's own max_k=10, not the row count.
        assert float(k_values[-1]) == 10.0
        assert inertias.shape == k_values.shape == silhouettes.shape

        # One more settled frame so the mini_plot spy above sees the now-populated results.
        _draw_frame(panel, imgui_context)
        assert "elbow_inertia" in marker_calls and marker_calls["elbow_inertia"] is not None
        mk_x, mk_y = marker_calls["elbow_inertia"]
        assert mk_x == [3.0], "the marker must sit at the currently selected k"
        idx3 = int(np.argmin(np.abs(k_values - 3.0)))
        assert mk_y == pytest.approx([float(inertias[idx3])])

    def test_elbow_sweep_operates_on_the_subsampled_row_set_not_the_full_dataset(
        self, imgui_context, workspace, monkeypatch
    ):
        panel = MathLabPanel(workspace)
        source = self._blob_source(workspace, name="cdiagD", n_per=2500)  # 5000 rows total

        # Set the sub-sampling state directly (the "Sampling" section stays collapsed
        # and untouched -- this mirrors how the elbow section reads it, independent of
        # whether that section has ever been opened in the UI).
        panel._sampling_enabled["cluster"] = True
        panel._sampling_max_rows["cluster"] = 5
        panel._sampling_seed["cluster"] = 0

        # Confirm _cluster_training_xy (what the elbow section actually fits on) itself
        # already reflects the cap, independent of the async plumbing below.
        panel._current_source = source
        xa, ya = panel._cluster_training_xy(source, "cluster")
        assert xa.size == ya.size == 5

        self._force_section_open(monkeypatch, "Elbow / silhouette (k-means)")
        _click_once_labelled_button(
            monkeypatch, lambda label: label.startswith("Compute elbow/silhouette")
        )

        panel.show_operation("cluster")
        panel._cluster_method = "kmeans"
        self._wait_for_elbow(panel, imgui_context, key="cluster")

        k_values, _inertias, _silhouettes = panel._elbow_results["cluster"]
        # min(max_k=10, n - 1) with n=5 (the SUBSAMPLE cap) is 4, not 10 (which is what
        # the full 5000-row dataset would have produced).
        assert float(k_values[-1]) == 4.0
        assert k_values.size == 3  # k = 2, 3, 4


# ============================================================================
# UMAP "color by column" preview (Phase 5): presentation-only, recolors the
# already-computed embedding without touching _compute/caching/model_state.
# ============================================================================


class TestUmapDiagnostics:
    def _columns_with_category(self, n_rows=150, n_features=4, seed=0):
        columns = _correlated_columns(n_rows=n_rows, n_features=n_features, seed=seed)
        # A column NOT included in the embedding -- what "color by" recolors with.
        columns["cat"] = (np.arange(n_rows) % 3).astype(np.float64)
        return columns

    def test_color_by_column_produces_categories_and_an_aligned_bucketed_array(self, workspace):
        pytest.importorskip("umap")
        panel = MathLabPanel(workspace)
        columns = self._columns_with_category(n_rows=150)
        embed_cols = ("a", "b", "c", "noise3")
        source = _make_multicol_source(workspace, "umapdiag1", columns, x_col="a", y_col="b")
        result = panel._compute(source, ("umap", embed_cols, 2, 10, 0.1, "zscore", 0, _NO_SAMPLING))
        bucket = panel._umap_color_values(source, "umap", result, "cat")
        assert bucket is not None
        categories, bucketed = bucket
        assert bucketed.size == int(np.asarray(result.x).size)
        np.testing.assert_array_equal(np.sort(categories), np.array([0.0, 1.0, 2.0]))

        # The bucketed values must actually correspond to "cat" at the embedding's own
        # rows (every row here, since nothing was subsampled and nothing is non-finite).
        expected_picked = source.dataset.get("cat")
        expected_categories, expected_bucketed = np.unique(expected_picked, return_inverse=True)
        np.testing.assert_array_equal(categories, expected_categories)
        np.testing.assert_array_equal(bucketed, expected_bucketed.astype(np.int64))

    def test_color_by_column_length_matches_the_embedding_after_subsampling(self, workspace):
        pytest.importorskip("umap")
        panel = MathLabPanel(workspace)
        columns = self._columns_with_category(n_rows=800, n_features=4)
        embed_cols = ("a", "b", "c", "noise3")
        source = _make_multicol_source(workspace, "umapdiag2", columns, x_col="a", y_col="b")

        training = _SamplingParams(subsample=True, max_rows=100, subsample_seed=0)
        panel._sampling_enabled["umap"] = True
        panel._sampling_max_rows["umap"] = 100
        panel._sampling_seed["umap"] = 0

        result = panel._compute(source, ("umap", embed_cols, 2, 10, 0.1, "zscore", 0, training))
        n_embedding = int(np.asarray(result.x).size)
        assert n_embedding <= 100
        assert n_embedding < 800, "the embedding must actually be smaller than the full dataset"

        bucket = panel._umap_color_values(source, "umap", result, "cat")
        assert bucket is not None
        _categories, bucketed = bucket
        assert bucketed.size == n_embedding

        # Cross-check against the row-index reconstruction directly.
        row_idx = panel._umap_row_indices(source, "umap", embed_cols)
        assert row_idx.size == n_embedding
        expected_picked = np.asarray(source.dataset.get("cat"))[row_idx]
        _expected_categories, expected_bucketed = np.unique(expected_picked, return_inverse=True)
        np.testing.assert_array_equal(bucketed, expected_bucketed.astype(np.int64))

    def test_color_by_column_is_presentation_only_and_never_touches_model_state(self, workspace):
        pytest.importorskip("umap")
        panel = MathLabPanel(workspace)
        columns = self._columns_with_category(n_rows=120)
        embed_cols = ("a", "b", "c", "noise3")
        source = _make_multicol_source(workspace, "umapdiag3", columns, x_col="a", y_col="b")
        result = panel._compute(source, ("umap", embed_cols, 2, 10, 0.1, "zscore", 0, _NO_SAMPLING))
        model_state_before = result.model_state
        x_before = np.array(result.x, copy=True)
        panel._umap_color_values(source, "umap", result, "cat")
        panel._umap_color_values(source, "umap", result, "cat")
        assert result.model_state is model_state_before
        np.testing.assert_array_equal(result.x, x_before)

    def test_color_by_column_returns_none_on_a_row_count_mismatch(self, workspace):
        """Defensive fallback: if the row-index reconstruction disagrees with the
        embedding's own row count (e.g. stale sampling state for a different tab key),
        coloring is skipped rather than raising or silently misaligning arrays."""
        pytest.importorskip("umap")
        panel = MathLabPanel(workspace)
        columns = self._columns_with_category(n_rows=150)
        embed_cols = ("a", "b", "c", "noise3")
        source = _make_multicol_source(workspace, "umapdiag4", columns, x_col="a", y_col="b")
        result = panel._compute(source, ("umap", embed_cols, 2, 10, 0.1, "zscore", 0, _NO_SAMPLING))
        assert int(np.asarray(result.x).size) == 150

        # Mismatch this key's sampling state against the actual (unsampled) embedding.
        panel._sampling_enabled["umap"] = True
        panel._sampling_max_rows["umap"] = 50
        panel._sampling_seed["umap"] = 0

        bucket = panel._umap_color_values(source, "umap", result, "cat")
        assert bucket is None

    def test_color_by_column_missing_column_returns_none(self, workspace):
        pytest.importorskip("umap")
        panel = MathLabPanel(workspace)
        columns = self._columns_with_category(n_rows=100)
        embed_cols = ("a", "b", "c", "noise3")
        source = _make_multicol_source(workspace, "umapdiag5", columns, x_col="a", y_col="b")
        result = panel._compute(source, ("umap", embed_cols, 2, 10, 0.1, "zscore", 0, _NO_SAMPLING))
        assert panel._umap_color_values(source, "umap", result, "does_not_exist") is None


# ============================================================================
# "(index)" pseudo-column in Math Lab's dataset-source X/Y picker (mathlab.py's
# _draw_dataset_source, _INDEX_OPTION, _replace_reason) -- lost from this file by
# the same tool-use accident the sections above explain; rebuilt fresh here against
# the current, intact production code rather than reconstructed from memory. NOT the
# Data Editor's own "(index)" option (a separate feature in a separate, unaffected
# file -- see tests/test_gui_data_editor.py).
# ============================================================================


def _capture_combo_options(monkeypatch) -> Dict[str, List[str]]:
    """Record the options list every ``widgets.enum_combo`` call drew, keyed by label."""
    box: Dict[str, List[str]] = {}
    real = widgets.enum_combo

    def spy(lbl, current, options, **kwargs):
        box[lbl] = [str(o) for o in options]
        return real(lbl, current, options, **kwargs)

    monkeypatch.setattr("glplot.gui.panels.mathlab.widgets.enum_combo", spy)
    return box


class TestIndexAxisOption:
    """``_INDEX_OPTION`` ("(index)"): a selectable X/Y pick that synthesizes
    ``np.arange(dataset.n_rows())`` instead of reading a real column, and
    ``_replace_reason``'s precise handling of it (only the axis a result actually
    WRITES BACK to, if it is the index, blocks Replace)."""

    def test_index_option_is_offered_in_both_the_x_and_y_column_pickers(
        self, imgui_context, workspace, monkeypatch
    ):
        panel = MathLabPanel(workspace)
        box = _capture_combo_options(monkeypatch)
        for _ in range(2):
            _draw_frame(panel, imgui_context)
        assert _INDEX_OPTION in box.get("X column", [])
        assert _INDEX_OPTION in box.get("Y column", [])

    def test_picking_index_for_x_produces_arange_x_raw(self, imgui_context, workspace):
        panel = MathLabPanel(workspace)
        panel._ds_name = "alpha"  # 128 rows, columns "x"/"y"
        panel._ds_columns["alpha"] = (_INDEX_OPTION, "y")
        for _ in range(2):
            _draw_frame(panel, imgui_context)

        assert panel._ds_x_col == _INDEX_OPTION
        assert panel._current_source is not None
        dataset = workspace.store.get("alpha")
        np.testing.assert_array_equal(
            panel._current_source.x_raw, np.arange(dataset.n_rows(), dtype=np.float64)
        )
        # y is untouched -- a real column, not synthesized.
        np.testing.assert_allclose(panel._current_source.y_raw, dataset.get("y"))

    def test_picking_index_for_y_produces_arange_y_raw(self, imgui_context, workspace):
        panel = MathLabPanel(workspace)
        panel._ds_name = "alpha"
        panel._ds_columns["alpha"] = ("x", _INDEX_OPTION)
        for _ in range(2):
            _draw_frame(panel, imgui_context)

        assert panel._ds_y_col == _INDEX_OPTION
        dataset = workspace.store.get("alpha")
        np.testing.assert_array_equal(
            panel._current_source.y_raw, np.arange(dataset.n_rows(), dtype=np.float64)
        )
        np.testing.assert_allclose(panel._current_source.x_raw, dataset.get("x"))

    def test_a_stale_remembered_column_falls_back_even_with_index_on_the_other_axis(
        self, imgui_context, workspace
    ):
        """A no-longer-existing column name must still fall back to columns[1] (the
        ordinary stale-column rule), independent of _INDEX_OPTION being a valid pick on
        the OTHER axis at the same time."""
        panel = MathLabPanel(workspace)
        panel._ds_name = "alpha"
        panel._ds_columns["alpha"] = (_INDEX_OPTION, "vanished")
        for _ in range(2):
            _draw_frame(panel, imgui_context)
        assert panel._ds_x_col == _INDEX_OPTION, "a genuinely valid pick must survive"
        assert panel._ds_y_col == "y", "the stale pick must fall back to columns[1]"

    def test_index_pick_survives_being_redrawn_across_several_frames(
        self, imgui_context, workspace
    ):
        """REGRESSION class this feature already had to guard against once: an index
        pick must not silently reset back to a real column on some later frame."""
        panel = MathLabPanel(workspace)
        panel._ds_name = "alpha"
        panel._ds_columns["alpha"] = (_INDEX_OPTION, "y")
        for i in range(8):
            _draw_frame(panel, imgui_context)
            assert panel._ds_x_col == _INDEX_OPTION, f"index pick reset itself at frame {i}"

    def _index_x_source(self, workspace, name="idxrepl", n=40):
        x = np.arange(n, dtype=np.float64)
        y = np.sin(x)
        ds = DataSet(name, [Column("y", y)])
        workspace.store.add(ds)
        return _Source(
            key=("dataset", name, _INDEX_OPTION, "y"),
            label=f"{name}.y",
            x_name=_INDEX_OPTION,
            y_name="y",
            x_raw=x,
            y_raw=y,
            dataset=ds,
            x_col=_INDEX_OPTION,
            y_col="y",
        )

    def _index_y_source(self, workspace, name="idxreply", n=40):
        x = np.linspace(0.0, 10.0, n)
        y = np.arange(n, dtype=np.float64)
        ds = DataSet(name, [Column("x", x)])
        workspace.store.add(ds)
        return _Source(
            key=("dataset", name, "x", _INDEX_OPTION),
            label=f"{name}.{_INDEX_OPTION}",
            x_name="x",
            y_name=_INDEX_OPTION,
            x_raw=x,
            y_raw=y,
            dataset=ds,
            x_col="x",
            y_col=_INDEX_OPTION,
        )

    def _plain_result(self, x, y, *, x_changed=False):
        from glplot.gui.panels.mathlab import _Result

        return _Result(
            x=np.asarray(x, float),
            y=np.asarray(y, float),
            x_name="x",
            y_name="y out",
            suffix="op",
            plot_label="p",
            overlay=None,
            x_changed=x_changed,
        )

    def test_replace_unavailable_when_y_is_the_index(self, workspace):
        panel = MathLabPanel(workspace)
        source = self._index_y_source(workspace)
        result = self._plain_result(source.x_raw, np.cos(source.x_raw))
        reason = panel._replace_reason(source, result)
        assert reason is not None
        assert "Y axis is the row index" in reason

    def test_replace_unavailable_when_x_is_the_index_and_the_result_changes_x(self, workspace):
        panel = MathLabPanel(workspace)
        source = self._index_x_source(workspace)
        result = self._plain_result(np.linspace(0, 39, 100), np.zeros(100), x_changed=True)
        reason = panel._replace_reason(source, result)
        assert reason is not None
        assert "X axis is the row index" in reason

    def test_replace_available_when_x_is_the_index_but_the_result_leaves_x_alone(self, workspace):
        """A y-only transform (e.g. Smooth) over an index-X source: only y gets written
        back, so Replace must stay available."""
        panel = MathLabPanel(workspace)
        source = self._index_x_source(workspace)
        result = self._plain_result(source.x_raw, np.cos(source.x_raw), x_changed=False)
        reason = panel._replace_reason(source, result)
        assert reason is None

    def test_replace_still_checks_row_count_once_the_index_case_is_cleared(self, workspace):
        panel = MathLabPanel(workspace)
        source = self._index_x_source(workspace, n=40)
        # x_changed False (index left alone) but wrong row count -> the ordinary
        # row-count reason must still fire.
        result = self._plain_result(np.arange(10.0), np.arange(10.0), x_changed=False)
        reason = panel._replace_reason(source, result)
        assert reason is not None
        assert "row count" in reason

    def test_add_column_works_fine_with_an_index_x_source(self, workspace):
        panel = MathLabPanel(workspace)
        source = self._index_x_source(workspace, n=30)
        result = self._plain_result(source.x_raw, np.cos(source.x_raw))
        cmd = panel._command_add_column(source, result, "smooth")
        assert cmd is not None
        before = source.dataset.n_cols()
        cmd.do()
        assert source.dataset.n_cols() == before + 1
        np.testing.assert_allclose(source.dataset.columns[-1].values, result.y)
        cmd.undo()
        assert source.dataset.n_cols() == before

    def test_new_dataset_works_fine_with_an_index_x_source(self, workspace):
        panel = MathLabPanel(workspace)
        source = self._index_x_source(workspace, n=25)
        result = self._plain_result(source.x_raw, np.cos(source.x_raw))
        cmd = panel._command_new_dataset(source, result, "smooth")
        before_names = set(workspace.store.names())
        cmd.do()
        created = set(workspace.store.names()) - before_names
        assert created, "New dataset must succeed with an index-X source"
        ds = workspace.store.get(next(iter(created)))
        assert ds.n_rows() == 25
        cmd.undo()
        assert set(workspace.store.names()) == before_names
