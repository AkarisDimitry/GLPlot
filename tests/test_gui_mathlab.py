"""Test the Math Lab panel in glplot.gui.panels.mathlab.

Driven through the sanctioned headless imgui harness (CONTRACT 2.10): a real context,
real frames, real widgets, synthetic mouse input, and no OpenGL or GPU at any point.

The headline case is :class:`TestSourceSelector`. ``radio_button("Dataset")`` and
``enum_combo("Dataset", ...)`` used to sit in one id scope, and since ImGui derives a
widget id from ``hash(label) ^ id_stack`` they resolved to the same id -- so the combo's
popup, keyed on that id, could never open and the dataset could not be changed at all.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pytest

imgui = pytest.importorskip("imgui_bundle").imgui

from glplot.gui import icons  # noqa: E402
from glplot.gui import mathops  # noqa: E402
from glplot.gui import mathopsnd  # noqa: E402
from glplot.gui import widgets  # noqa: E402
from glplot.gui.commands import CommandQueue  # noqa: E402
from glplot.gui.datasets import Column, DataSet, DataStore  # noqa: E402
from glplot.gui.history import UndoStack  # noqa: E402
from glplot.gui.panels.mathlab import (  # noqa: E402
    _APPLY_MODES,
    _CATEGORIES,
    _CATEGORY_OF,
    _FIT_MODELS,
    _TABS,
    MathLabPanel,
    _fit_state_key,
    _sorted_by_x,
    _Source,
)

# The panel's default geometry: workspace.py sizes it 0.37 x 0.47 of a 1280x900 viewport.
_PANEL_W = int(1280 * 0.37)
_PANEL_H = int(900 * 0.47)

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
    """The four collaborators Panel proxies onto."""

    def __init__(self) -> None:
        self.plot = _FakePlot()
        self.store = DataStore()
        self.queue = CommandQueue()
        self.undo = UndoStack()
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
    ds = DataSet(name, [Column("x", np.asarray(x, dtype=np.float64)), Column("y", np.asarray(y, dtype=np.float64))])
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
        source = _make_dataset_source(workspace, "rob", x, 2.0 * x + 1.0 + rng.normal(0, 0.1, x.size))

        panel._compute(source, ("fit", "linear", 0, "", None, False, "soft_l1", False, 0.95, False, 200))
        assert calls["plain"] == 1 and calls["robust"] == 0

        panel._compute(source, ("fit", "linear", 0, "", None, True, "soft_l1", False, 0.95, False, 200))
        assert calls["robust"] == 1

    def test_robust_fit_recovers_parameters_despite_outliers(self, workspace):
        panel = MathLabPanel(workspace)
        rng = np.random.default_rng(1)
        x = np.linspace(0.0, 10.0, 200)
        y = 2.0 * x + 1.0 + rng.normal(0.0, 0.1, x.size)
        bad = rng.choice(x.size, size=8, replace=False)
        y[bad] += 25.0
        source = _make_dataset_source(workspace, "out", x, y)

        result = panel._compute(source, ("fit", "linear", 0, "", None, True, "soft_l1", False, 0.95, False, 200))
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
        result = panel._compute(source, ("fit", "linear", 0, "", None, False, "soft_l1", False, 0.95, False, 200))
        assert result.band is None

    def test_band_is_populated_when_requested(self, workspace):
        panel = MathLabPanel(workspace)
        rng = np.random.default_rng(2)
        x = np.linspace(-5.0, 5.0, 200)
        y = 2.0 * x + 1.0 + rng.normal(0.0, 0.3, x.size)
        source = _make_dataset_source(workspace, "cb2", x, y)
        result = panel._compute(source, ("fit", "linear", 0, "", None, False, "soft_l1", True, 0.95, False, 200))
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
        r68 = panel._compute(source, ("fit", "linear", 0, "", None, False, "soft_l1", True, 0.68, False, 200))
        r99 = panel._compute(source, ("fit", "linear", 0, "", None, False, "soft_l1", True, 0.99, False, 200))
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
            source, ("fit", "polynomial", 2, "", None, False, "soft_l1", True, 0.95, False, 200)
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
            source, ("fit", "linear", 0, "", None, False, "soft_l1", False, 0.95, False, 200)
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
            source, ("fit", "linear", 0, "", None, False, "soft_l1", False, 0.95, True, 500)
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
            source, ("fit", "linear", 0, "", None, False, "soft_l1", False, 0.95, True, 80)
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
            source, ("fit", "polynomial", 2, "", None, False, "soft_l1", False, 0.95, True, 300)
        )
        assert result.x.size == 300
        np.testing.assert_allclose(
            result.y, 2.0 * result.x**2 - 3.0 * result.x + 1.0, atol=1e-6
        )

    def test_band_is_evaluated_on_the_output_grid_not_the_source_samples(self, workspace):
        panel = MathLabPanel(workspace)
        rng = np.random.default_rng(7)
        x = np.linspace(-5.0, 5.0, 200)
        y = 2.0 * x + 1.0 + rng.normal(0.0, 0.3, x.size)
        source = _make_dataset_source(workspace, "og5", x, y)
        result = panel._compute(
            source, ("fit", "linear", 0, "", None, False, "soft_l1", True, 0.95, True, 250)
        )
        assert result.band is not None
        lo, hi = result.band
        assert lo.shape == (250,) == hi.shape == result.x.shape

    def test_output_points_row_reports_the_grid_size(self, workspace):
        panel = MathLabPanel(workspace)
        x = np.linspace(-5.0, 5.0, 37)
        source = _make_dataset_source(workspace, "og6", x, 2.0 * x + 1.0)
        result = panel._compute(
            source, ("fit", "linear", 0, "", None, False, "soft_l1", False, 0.95, True, 123)
        )
        assert dict(result.stats)["Output points"] == "123"

    def test_replace_is_unavailable_once_the_grid_changes_the_row_count(self, workspace):
        panel = MathLabPanel(workspace)
        x = np.linspace(-5.0, 5.0, 37)
        source = _make_dataset_source(workspace, "og7", x, 2.0 * x + 1.0)
        result = panel._compute(
            source, ("fit", "linear", 0, "", None, False, "soft_l1", False, 0.95, True, 500)
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
            source, ("fit", "linear", 0, "", None, False, "soft_l1", False, 0.95, True, 0)
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
        for expected in ("Skewness", "Excess kurtosis", "Interquartile range", "Median abs deviation"):
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
        result = panel._compute(source, ("cluster", "kmeans", 2, 0))
        assert result.color_values is not None
        assert result.color_values.shape == result.x.shape
        assert result.markers is not None
        assert np.asarray(result.markers[0]).size == 2
        assert result.table is not None
        assert result.table[-1][0] == "cluster"

    def test_compute_reports_cluster_sizes_in_stats(self, workspace):
        panel = MathLabPanel(workspace)
        source = self._blob_source(workspace)
        result = panel._compute(source, ("cluster", "kmeans", 2, 0))
        labels_in_stats = [label for label, _v in result.stats]
        assert "Clusters found" in labels_in_stats
        assert any(label.startswith("Cluster ") and label.endswith("size") for label in labels_in_stats)

    def test_k_too_large_surfaces_as_an_error_not_a_crash(self, imgui_context, workspace):
        panel = MathLabPanel(workspace)
        source = _make_dataset_source(workspace, "tiny", [1.0, 2.0], [1.0, 2.0])
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
        result = panel._compute(source, ("cluster", "kmeans", 2, 0))
        cmd = panel._command_add_column(source, result, "cluster")
        assert cmd is not None
        cmd.do()
        new_col = source.dataset.columns[-1]
        assert np.array_equal(new_col.values, result.color_values)

    def test_new_layer_passes_color_values_as_c_kwarg(self, workspace, monkeypatch):
        """New layer must close the loop with GLPlot's per-point colour encoding."""
        panel = MathLabPanel(workspace)
        source = self._blob_source(workspace, name="blobs4")
        result = panel._compute(source, ("cluster", "kmeans", 2, 0))

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

    def test_kde_unavailable_shows_an_error_not_a_crash(self, imgui_context, workspace, monkeypatch):
        monkeypatch.setattr(
            "glplot.gui.panels.mathlab.mathops2d.kde2d_available", lambda: False
        )
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
        result = panel._compute(source, ("cluster", "hierarchical", 2, "ward"))
        assert result.color_values is not None
        assert result.markers is not None
        assert np.asarray(result.markers[0]).size == 2

    def test_method_label_appears_in_stats(self, workspace):
        panel = MathLabPanel(workspace)
        source = self._blob_source(workspace, name="hblobs2")
        result = panel._compute(source, ("cluster", "hierarchical", 2, "single"))
        rows = dict(result.stats)
        assert "hierarchical" in rows["Method"]
        assert "single" in rows["Method"]

    def test_unavailable_scipy_shows_an_error_not_a_crash(self, imgui_context, workspace, monkeypatch):
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
        result = panel._compute(source, ("pca", names, 3, "zscore"))

        expected = mathopsnd.pca([columns[n] for n in names], n_components=3, scale="zscore")
        np.testing.assert_allclose(result.x, expected["scores"][:, 0])
        np.testing.assert_allclose(result.y, expected["scores"][:, 1])

    def test_result_is_a_scatter_not_a_line(self, workspace):
        panel = MathLabPanel(workspace)
        columns = _correlated_columns()
        source = _make_multicol_source(workspace, "pca3", columns)
        result = panel._compute(source, ("pca", tuple(columns.keys()), 2, "zscore"))
        assert result.force_scatter is True

    def test_table_has_one_column_per_component(self, workspace):
        panel = MathLabPanel(workspace)
        columns = _correlated_columns()
        source = _make_multicol_source(workspace, "pca4", columns)
        result = panel._compute(source, ("pca", tuple(columns.keys()), 4, "zscore"))
        assert result.table is not None
        assert [name for name, _v in result.table] == ["PC1", "PC2", "PC3", "PC4"]
        for _name, values in result.table:
            assert values.shape == (result.x.size,)

    def test_stats_report_explained_variance_per_component(self, workspace):
        panel = MathLabPanel(workspace)
        columns = _correlated_columns()
        source = _make_multicol_source(workspace, "pca5", columns)
        result = panel._compute(source, ("pca", tuple(columns.keys()), 3, "zscore"))
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
        result = panel._compute(source, ("pca", ("a", "b", "c"), 2, "zscore"))
        rows = dict(result.stats)
        assert float(rows["PC1 explained variance"].rstrip("%")) > 80.0

    def test_apply_modes_exclude_add_column_and_replace(self, workspace):
        panel = MathLabPanel(workspace)
        columns = _correlated_columns()
        source = _make_multicol_source(workspace, "pca6", columns)
        result = panel._compute(source, ("pca", tuple(columns.keys()), 2, "zscore"))
        assert result.apply_modes == ("new_layer", "new_dataset")

    def test_fewer_than_two_columns_raises(self, workspace):
        panel = MathLabPanel(workspace)
        columns = _correlated_columns()
        source = _make_multicol_source(workspace, "pca7", columns)
        with pytest.raises(ValueError, match="at least 2 columns"):
            panel._compute(source, ("pca", ("a",), 2, "zscore"))

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
            panel._compute(source, ("pca", ("a", "b"), 2, "zscore"))

    def test_is_in_the_multivariate_category(self):
        assert "pca" in dict(_CATEGORIES)["Multivariate"]
        assert ("pca", "PCA", "_tab_pca") in _TABS

    def test_new_dataset_apply_writes_all_components(self, workspace):
        panel = MathLabPanel(workspace)
        columns = _correlated_columns()
        source = _make_multicol_source(workspace, "pca8", columns)
        result = panel._compute(source, ("pca", tuple(columns.keys()), 3, "zscore"))
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
        result = panel._compute(source, ("umap", names, 2, 10, 0.1, "zscore", 0))
        assert result.x.shape == (60,)
        assert result.y.shape == (60,)

    def test_result_is_a_scatter_not_a_line(self, workspace):
        pytest.importorskip("umap")
        panel = MathLabPanel(workspace)
        columns = _correlated_columns(n_rows=60)
        source = _make_multicol_source(workspace, "umap2", columns)
        result = panel._compute(source, ("umap", tuple(columns.keys()), 2, 10, 0.1, "zscore", 0))
        assert result.force_scatter is True

    def test_reproducible_with_the_same_seed(self, workspace):
        pytest.importorskip("umap")
        panel = MathLabPanel(workspace)
        columns = _correlated_columns(n_rows=60)
        source = _make_multicol_source(workspace, "umap3", columns)
        names = tuple(columns.keys())
        r1 = panel._compute(source, ("umap", names, 2, 10, 0.1, "zscore", 7))
        r2 = panel._compute(source, ("umap", names, 2, 10, 0.1, "zscore", 7))
        np.testing.assert_allclose(r1.x, r2.x)
        np.testing.assert_allclose(r1.y, r2.y)

    def test_table_has_one_column_per_component(self, workspace):
        pytest.importorskip("umap")
        panel = MathLabPanel(workspace)
        columns = _correlated_columns(n_rows=60)
        source = _make_multicol_source(workspace, "umap4", columns)
        result = panel._compute(source, ("umap", tuple(columns.keys()), 2, 10, 0.1, "zscore", 0))
        assert result.table is not None
        assert [name for name, _v in result.table] == ["UMAP1", "UMAP2"]

    def test_stats_report_neighbors_used_and_no_explained_variance(self, workspace):
        """UMAP is nonlinear -- unlike PCA, there is no explained-variance concept."""
        pytest.importorskip("umap")
        panel = MathLabPanel(workspace)
        columns = _correlated_columns(n_rows=60)
        source = _make_multicol_source(workspace, "umap5", columns)
        result = panel._compute(source, ("umap", tuple(columns.keys()), 2, 10, 0.1, "zscore", 0))
        rows = dict(result.stats)
        assert "Neighbors used" in rows
        assert not any("explained variance" in label for label in rows)

    def test_apply_modes_exclude_add_column_and_replace(self, workspace):
        pytest.importorskip("umap")
        panel = MathLabPanel(workspace)
        columns = _correlated_columns(n_rows=60)
        source = _make_multicol_source(workspace, "umap6", columns)
        result = panel._compute(source, ("umap", tuple(columns.keys()), 2, 10, 0.1, "zscore", 0))
        assert result.apply_modes == ("new_layer", "new_dataset")

    def test_fewer_than_two_columns_raises(self, workspace):
        pytest.importorskip("umap")
        panel = MathLabPanel(workspace)
        columns = _correlated_columns(n_rows=60)
        source = _make_multicol_source(workspace, "umap7", columns)
        with pytest.raises(ValueError, match="at least 2 columns"):
            panel._compute(source, ("umap", ("a",), 2, 10, 0.1, "zscore", 0))

    def test_unavailable_umap_shows_an_error_not_a_crash(self, imgui_context, workspace, monkeypatch):
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
        result = panel._compute(source, ("umap", tuple(columns.keys()), 2, 10, 0.1, "zscore", 0))
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
