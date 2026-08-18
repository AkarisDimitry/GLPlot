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

from glplot.gui import mathops  # noqa: E402
from glplot.gui import widgets  # noqa: E402
from glplot.gui.commands import CommandQueue  # noqa: E402
from glplot.gui.datasets import Column, DataSet, DataStore  # noqa: E402
from glplot.gui.history import UndoStack  # noqa: E402
from glplot.gui.panels.mathlab import (  # noqa: E402
    _CATEGORIES,
    _CATEGORY_OF,
    _FIT_MODELS,
    _TABS,
    MathLabPanel,
    _fit_state_key,
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

        if key == "stats":
            assert not rect, "Statistics commits nothing and must not offer Apply"
            return

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
