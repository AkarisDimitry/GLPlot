"""Test the interactive-controls model (glplot.controls) and the layer data setters.

Pure logic: no window, no GL, no imgui. The two dispatch modes are both exercised --
script mode (callbacks run inline, exceptions propagate) and live mode (changes are
coalesced until ``flush``, exceptions are recorded) -- because the second is what the
window runs and the first is what every script and test sees.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pytest

import glplot.pyplot as gplt
from glplot import controls as ctl
from glplot.core.layers import FunctionLayer, PolylineLayer
from glplot.gui import notifications


@pytest.fixture(autouse=True)
def clean_state():
    gplt._cleanup_pyplot_state()
    notifications.clear()
    yield
    gplt._cleanup_pyplot_state()
    notifications.clear()


@pytest.fixture
def fig():
    return gplt.figure()


@pytest.fixture
def live(fig):
    """The figure's controls in live (window-open) mode."""
    fig.controls._set_live(True)
    yield fig.controls
    fig.controls._live = False  # no flush-on-exit: each test flushes what it means to


# ======================================================================================
# Individual controls
# ======================================================================================


class TestSlider:
    def test_default_is_the_middle_of_the_range(self):
        assert ctl.Slider("a", 0, 10).value == 5.0

    def test_log_default_is_the_geometric_middle(self):
        assert ctl.Slider("f", 1, 100, log=True).value == pytest.approx(10.0)

    def test_values_are_clamped(self):
        s = ctl.Slider("a", 0, 1)
        s.value = 7
        assert s.value == 1.0
        s.value = -3
        assert s.value == 0.0

    def test_step_snaps_without_float_noise(self):
        s = ctl.Slider("a", 0, 1, step=0.1)
        s.value = 0.31
        assert s.value == 0.3  # exactly, not 0.30000000000000004

    def test_integer_slider_holds_ints(self):
        s = ctl.Slider("n", 0.5, 10.5, integer=True)
        assert (s.min, s.max) == (1.0, 10.0)
        s.value = 4.6
        assert s.value == 5 and isinstance(s.value, int)

    def test_bad_definitions_raise(self):
        with pytest.raises(ctl.ControlError, match="less than max"):
            ctl.Slider("a", 5, 1)
        with pytest.raises(ctl.ControlError, match="log slider needs min > 0"):
            ctl.Slider("a", 0, 1, log=True)
        with pytest.raises(ctl.ControlError, match="step must be positive"):
            ctl.Slider("a", 0, 1, step=0)
        with pytest.raises(ctl.ControlError, match="finite"):
            ctl.Slider("a", 0, float("inf"))

    def test_a_rejected_value_keeps_the_old_one(self):
        s = ctl.Slider("a", 0, 1, 0.25)
        with pytest.raises(ctl.ControlError, match="expected a number"):
            s.value = "lots"
        assert s.value == 0.25

    def test_display_format_follows_the_step_and_unit(self):
        assert ctl.Slider("a", 0, 1, step=0.25).fmt == "%.2f"
        assert ctl.Slider("n", 0, 9, integer=True).fmt == "%d"
        assert ctl.Slider("f", 1, 100, unit="Hz").fmt.endswith(" Hz")


class TestOtherControls:
    def test_range_slider_orders_and_snaps_its_ends(self):
        r = ctl.RangeSlider("band", 0, 100, step=5)
        assert r.value == (0.0, 100.0)
        r.value = (72, 13)
        assert r.value == (15.0, 70.0)
        with pytest.raises(ctl.ControlError, match="pair"):
            r.value = 3

    def test_checkbox_refuses_strings(self):
        c = ctl.Checkbox("on")
        c.value = 1
        assert c.value is True
        with pytest.raises(ctl.ControlError):
            c.value = "false"  # bool("false") is True: never what was meant

    def test_dropdown_accepts_values_and_labels(self):
        d = ctl.Dropdown("f", {"Sine": np.sin, "Cosine": np.cos})
        assert d.value is np.sin and d.selected_label == "Sine"
        d.value = "Cosine"
        assert d.value is np.cos and d.index == 1
        d.value = np.sin
        assert d.to_json() == "Sine"
        with pytest.raises(ctl.ControlError, match="not one of Sine, Cosine"):
            d.value = "Tan"

    def test_dropdown_option_forms(self):
        assert ctl.Dropdown("a", [1, 2, 3]).labels == ["1", "2", "3"]
        pairs = ctl.Dropdown("a", [("low", 0.1), ("high", 10.0)], "high")
        assert pairs.value == 10.0
        with pytest.raises(ctl.ControlError, match="unique"):
            ctl.Dropdown("a", [("x", 1), ("x", 2)])
        with pytest.raises(ctl.ControlError, match="at least one"):
            ctl.Dropdown("a", [])
        with pytest.raises(ctl.ControlError):
            ctl.Dropdown("a", ["x"], style="wheel")

    def test_textbox_limits_its_length(self):
        t = ctl.TextBox("t", None)
        assert t.value == ""
        with pytest.raises(ctl.ControlError, match="limit"):
            t.value = "x" * (ctl.MAX_TEXT_LENGTH + 1)

    def test_number_clamps_to_optional_bounds(self):
        n = ctl.Number("k", 5, min=0)
        n.value = -2
        assert n.value == 0.0
        n.value = 1e9
        assert n.value == 1e9
        i = ctl.Number("i", 2.6, integer=True)
        assert i.value == 3

    def test_color_picker_spellings(self):
        c = ctl.ColorPicker("c", "#ff0000")
        assert c.value == (1.0, 0.0, 0.0, 1.0)
        c.value = (0, 128, 255)
        assert c.hex == "#0080ff"
        c.value = (0.0, 0.0, 1.0, 0.5)
        assert c.hex == "#0000ff80"
        c.value = "tab:orange"
        assert c.value[3] == 1.0
        with pytest.raises(ctl.ControlError):
            c.value = "not-a-colour"

    def test_keys_are_slugged_and_must_be_identifiers(self):
        assert ctl.Slider("Cut-off (Hz)", 0, 1).key == "cut_off_hz"
        assert ctl.Slider("3rd", 0, 1).key == "_3rd"
        with pytest.raises(ctl.ControlError, match="identifier"):
            ctl.Slider("a", 0, 1, key="not valid")
        with pytest.raises(ctl.ControlError, match="non-empty"):
            ctl.Slider("  ", 0, 1)

    def test_observer_arity_is_detected(self):
        seen = []
        s = ctl.Slider("a", 0, 10)
        s.on_change(lambda: seen.append("no-arg"))
        s.on_change(lambda v: seen.append(v))
        s.value = 3
        assert seen == ["no-arg", 3.0]

    def test_disconnect(self):
        seen = []
        s = ctl.Slider("a", 0, 10)
        fn = s.on_change(seen.append)
        assert s.disconnect(fn)
        s.value = 1
        assert seen == [] and not s.disconnect(fn)


class TestButton:
    def test_each_click_runs_the_callback(self, fig):
        hits = []
        b = gplt.button("Go", lambda: hits.append(1))
        b.click()
        b.click()
        assert hits == [1, 1] and b.clicks == 2

    def test_decorator_forms(self, fig):
        hits = []

        @gplt.button("Named")
        def first():
            hits.append("named")

        @gplt.button
        def export_csv(btn):
            hits.append(btn.label)

        assert callable(first) and callable(export_csv)
        fig.controls["named"].click()
        fig.controls["export_csv"].click()
        assert hits == ["named", "Export csv"]

    def test_invalid_shortcut_is_refused_at_definition(self):
        with pytest.raises(ctl.ControlError, match="invalid shortcut"):
            ctl.Button("Go", shortcut="Mod+Nope")

    def test_a_disabled_button_does_not_fire(self):
        hits = []
        b = ctl.Button("Go", lambda: hits.append(1), enabled=False)
        b.click()
        assert hits == []

    def test_a_button_has_no_settable_value(self):
        with pytest.raises(ctl.ControlError, match="no settable value"):
            ctl.Button("Go").value = 3

    def test_clicks_are_not_coalesced_in_live_mode(self, fig, live):
        hits = []
        b = gplt.button("Go", lambda: hits.append(1))
        b.click()
        b.click()
        b.click()
        assert hits == []
        live.flush()
        assert hits == [1, 1, 1]


# ======================================================================================
# The set
# ======================================================================================


class TestControlSet:
    def test_clashing_keys_get_a_suffix(self, fig):
        a = gplt.slider("Gain", 0, 1)
        b = gplt.slider("Gain", 0, 1)
        assert (a.key, b.key) == ("gain", "gain_2")
        assert fig.controls["gain_2"] is b and len(fig.controls) == 2

    def test_a_control_belongs_to_one_figure(self, fig):
        s = gplt.slider("a", 0, 1)
        other = gplt.figure()
        with pytest.raises(ctl.ControlError, match="another figure"):
            other.controls.add(s)

    def test_groups_keep_first_appearance_order(self, fig):
        gplt.slider("loose", 0, 1)
        with gplt.control_group("Filter"):
            gplt.slider("cut", 0, 1)
        with gplt.control_group("Filter"):
            gplt.checkbox("on")
        groups = [(name, [c.key for c in members]) for name, members in fig.controls.groups()]
        assert groups == [(ctl.DEFAULT_GROUP, ["loose"]), ("Filter", ["cut", "on"])]

    def test_unknown_key_lists_the_known_ones(self, fig):
        gplt.slider("amp", 0, 1)
        with pytest.raises(KeyError, match="have: amp"):
            fig.controls["freq"]

    def test_remove_drops_dependent_bindings(self, fig):
        amp = gplt.slider("amp", 0, 1)

        @gplt.interact
        def f(amp):
            return None

        assert len(fig.controls.bindings) == 1
        fig.controls.remove("amp")
        assert fig.controls.bindings == [] and amp.owner is None

    def test_set_values_runs_each_binding_once(self, fig):
        gplt.slider("a", 0, 10)
        gplt.slider("b", 0, 10)
        runs = []

        @gplt.interact
        def f(a, b):
            runs.append((a, b))

        fig.controls.set_values(a=1, b=2)
        assert runs[-1] == (1.0, 2.0) and len(runs) == 2  # initial run + one

    def test_reset_returns_to_defaults(self, fig):
        s = gplt.slider("a", 0, 10, 2)
        c = gplt.checkbox("c")
        s.value, c.value = 9, True
        fig.controls.reset()
        assert (s.value, c.value) == (2.0, False)
        s.value = 7
        fig.controls.reset("a")
        assert s.value == 2.0

    def test_snapshot_restore_and_presets(self, fig):
        s = gplt.slider("a", 0, 10)
        d = gplt.dropdown("f", {"Sine": np.sin, "Cosine": np.cos})
        gplt.button("Go")
        gplt.readout("r", 1)
        s.value, d.value = 3, np.cos
        snap = fig.controls.snapshot()
        assert snap == {"a": 3.0, "f": "Cosine"}
        json.dumps(snap)
        fig.controls.save_preset("mine")
        s.value, d.value = 8, np.sin
        fig.controls.load_preset("mine")
        assert (s.value, d.value) == (3.0, np.cos)
        assert fig.controls.restore({"a": 1, "gone": 5}) == ["gone"]
        with pytest.raises(KeyError):
            fig.controls.restore({"gone": 5}, strict=True)
        with pytest.raises(KeyError, match="no preset"):
            fig.controls.load_preset("nope")

    def test_save_and_load_round_trip(self, fig, tmp_path):
        s = gplt.slider("a", 0, 10, 4)
        fig.controls.save_preset("p")
        path = tmp_path / "controls.json"
        fig.controls.save(str(path))
        s.value = 9
        fig.controls.presets.clear()
        assert fig.controls.load(str(path)) == []
        assert s.value == 4.0 and "p" in fig.controls.presets

    def test_readouts_never_trigger_bindings(self, fig):
        r = gplt.readout("rms")
        runs = []

        @gplt.interact(a=(0, 1))
        def f(a=0.5):
            runs.append(a)

        r.value = 0.25
        assert len(runs) == 1 and r.text == "0.25"


class TestDispatch:
    def test_script_mode_runs_inline_and_propagates(self, fig):
        s = gplt.slider("a", 0, 1)

        @s.on_change
        def boom(value):
            raise RuntimeError("nope")

        with pytest.raises(RuntimeError, match="nope"):
            s.value = 0.9
        assert s.value == 0.9  # the value itself was accepted

    def test_live_mode_coalesces_until_flush(self, fig, live):
        s = gplt.slider("a", 0, 100)
        seen = []
        s.on_change(seen.append)
        for v in range(1, 41):
            s.value = v
        assert seen == [] and live.pending
        assert live.flush() == 1
        assert seen == [40.0] and not live.pending

    def test_live_mode_records_errors_once_and_keeps_running(self, fig, live, caplog):
        s = gplt.slider("a", 0, 10)
        ok = []

        @gplt.interact
        def bad(a):
            raise ZeroDivisionError("oops")

        @gplt.interact
        def good(a):
            ok.append(a)

        with caplog.at_level(logging.ERROR, logger="glplot.controls"):
            s.value = 1
            live.flush()
            s.value = 2
            live.flush()
        assert bad.error == "ZeroDivisionError: oops" and "Traceback" in bad.traceback
        assert ok[-1] == 2.0
        assert len([r for r in caplog.records if "oops" in r.getMessage()]) == 1
        assert [t.kind for t in notifications.active()] == ["error"]
        assert fig.controls.errors() == [("Bad", "ZeroDivisionError: oops")]

    def test_an_error_clears_when_the_callback_recovers(self, fig, live):
        s = gplt.slider("a", 0, 10)
        s.on_change(lambda v: 1 / (v - 3))
        s.value = 3
        live.flush()
        assert s.error.startswith("ZeroDivisionError")
        s.value = 4
        live.flush()
        assert s.error is None

    def test_a_button_moving_sliders_reruns_a_binding_once(self, fig, live):
        a = gplt.slider("a", 0, 10)
        b = gplt.slider("b", 0, 10)
        runs = []

        @gplt.interact
        def f(a, b):
            runs.append((a, b))

        @gplt.button("Both")
        def both():
            a.value = 1
            b.value = 2

        runs.clear()
        fig.controls["both"].click()
        live.flush()
        assert runs == [(1.0, 2.0)]

    def test_callbacks_feeding_each_other_are_stopped(self, fig, live, caplog):
        a = gplt.slider("a", 0, 1000, 0)
        b = gplt.slider("b", 0, 1000, 0)
        a.on_change(lambda v: b.set(v + 1))
        b.on_change(lambda v: a.set(v + 1))
        a.value = 1
        with caplog.at_level(logging.WARNING, logger="glplot.controls"):
            live.flush()
        assert "kept changing each other" in caplog.text
        assert not live.pending

    def test_leaving_live_mode_flushes_the_last_changes(self, fig, live):
        seen = []
        s = gplt.slider("a", 0, 10)
        s.on_change(seen.append)
        s.value = 6
        live._set_live(False)
        assert seen == [6.0]

    def test_the_engine_flush_hook(self, fig, live):
        seen = []
        s = gplt.slider("a", 0, 10)
        s.on_change(seen.append)
        s.value = 2
        fig._flush_controls()
        assert seen == [2.0]
        fig._flush_controls()  # nothing pending: a no-op
        assert seen == [2.0]

    def test_a_flush_marks_the_figure_dirty(self, fig, live):
        gplt.slider("a", 0, 10)
        fig.frame.dirty_scene = False
        fig.controls["a"].value = 1
        live.flush()
        assert fig.frame.dirty_scene and fig.cache.refresh_requested


# ======================================================================================
# interact
# ======================================================================================


class TestInteract:
    def test_abbreviations(self):
        make = ctl.control_from_spec
        assert isinstance(make("a", (0, 1)), ctl.Slider) and not make("a", (0, 1)).integer
        assert make("n", (1, 100, 1)).integer
        assert not make("x", (0, 1, 0.5)).integer
        assert isinstance(make("on", True), ctl.Checkbox)
        assert isinstance(make("w", ["sin", "cos"]), ctl.Dropdown)
        assert isinstance(make("w", {"a": 1}), ctl.Dropdown)
        assert isinstance(make("t", "hello"), ctl.TextBox)
        assert isinstance(make("k", 2.5), ctl.Number) and make("k", 3).integer
        with pytest.raises(ctl.ControlError, match="cannot make a control"):
            make("bad", object())

    def test_function_defaults_seed_range_controls(self, fig):
        @gplt.interact(amp=(0, 5), wave=["sin", "cos"], on=True)
        def f(amp=1.0, wave="cos", on=False):
            return None

        values = fig.controls.values()
        # A range carries no value, so the default wins; a bool spec *is* a value.
        assert values == {"amp": 1.0, "wave": "cos", "on": True}
        assert fig.controls["amp"].default == 1.0

    def test_missing_and_unknown_parameters_fail_early(self, fig):
        with pytest.raises(ctl.ControlError, match="no control for parameter"):

            @gplt.interact(a=(0, 1))
            def f(a, b):
                return None

        with pytest.raises(ctl.ControlError, match="has no parameter 'zz'"):
            gplt.interact(lambda a: None, a=(0, 1), zz=(0, 1))

    def test_parameters_match_existing_controls_by_key(self, fig):
        amp = gplt.slider("amp", 0, 10, 3)

        @gplt.interact
        def f(amp):
            return amp * 2

        assert f.params["amp"] is amp and f.result == 6.0
        amp.value = 4
        assert f.result == 8.0

    def test_created_controls_are_grouped_under_the_function(self, fig):
        @gplt.interact(freq=(0.1, 10))
        def wave_shape(freq=1.0):
            return None

        assert fig.controls["freq"].group == "Wave shape"

    def test_mapping_results_become_outputs(self, fig):
        @gplt.interact(a=(0, 4))
        def stats(a=2.0):
            return {"square": a * a}

        assert stats.outputs == {"square": 4.0}

    def test_xy_results_draw_one_curve_updated_in_place(self, fig):
        x = np.linspace(0, 1, 50)

        @gplt.interact(k=(0, 5))
        def curve(k=1.0):
            return x, k * x

        layer = curve.layer
        assert isinstance(layer, PolylineLayer) and layer.label == "Curve"
        assert layer.metadata["controls_binding"] == "Curve"
        n_layers = len(fig.scene.layers)
        fig.controls["k"].value = 3
        assert curve.layer is layer and len(fig.scene.layers) == n_layers
        np.testing.assert_allclose(layer.get_ydata(), 3 * x, rtol=1e-6)

    def test_a_removed_curve_is_recreated(self, fig):
        @gplt.interact(k=(0, 5))
        def curve(k=1.0):
            return np.arange(4) * k

        first = curve.layer
        fig.scene.layers.remove(first)
        fig.controls["k"].value = 2
        assert curve.layer is not first and curve.layer in fig.scene.layers

    def test_scatter_kind_and_plot_kwargs(self, fig):
        @gplt.interact(kind="scatter", plot_kwargs={"s": 30}, k=(0, 5))
        def pts(k=1.0):
            return np.arange(5.0), np.arange(5.0) * k

        assert pts.layer.layer_type == "scatter"
        fig.controls["k"].value = 2
        np.testing.assert_allclose(pts.layer.get_offsets()[:, 1], np.arange(5.0) * 2)

    def test_autoscale_follows_the_data(self, fig, monkeypatch):
        calls = []
        monkeypatch.setattr(fig, "autoscale", lambda *a, **k: calls.append(1))

        @gplt.interact(autoscale=True, k=(0, 5))
        def curve(k=1.0):
            return np.arange(3.0) * k

        assert calls == [1]

    def test_paused_bindings_do_not_run_on_change(self, fig):
        runs = []

        @gplt.interact(k=(0, 5))
        def f(k=1.0):
            runs.append(k)

        f.enabled = False
        fig.controls["k"].value = 4
        assert runs == [1.0]
        assert f() is None and runs == [1.0, 4.0]  # calling still runs it

    def test_function_layers_are_invalidated_on_change(self, fig):
        amp = gplt.slider("amp", 0, 5, 1)
        layer = gplt.function(lambda x: amp.value * np.sin(x))
        layer.sampled_window = (0.0, 1.0)
        amp.value = 2
        assert layer.sampled_window is None


# ======================================================================================
# Layer setters
# ======================================================================================


class TestLayerSetters:
    def test_set_ydata_keeps_x_and_marks_dirty(self, fig):
        (line,) = gplt.plot([0, 1, 2], [0, 0, 0])
        line.dirty.clear()
        line.set_ydata([1, 2, 3])
        np.testing.assert_allclose(line.get_xdata(), [0, 1, 2])
        np.testing.assert_allclose(line.get_ydata(), [1, 2, 3])
        assert line.dirty.gpu_dirty and line.dirty.bounds_dirty and line.dirty.data_dirty

    def test_set_data_forms_and_length_change(self, fig):
        (line,) = gplt.plot([0, 1, 2], [0, 0, 0])
        line.set_data(np.array([[0, 1], [1, 2], [2, 3], [3, 4]]))
        assert line.pts.shape == (4, 2) and line.pts.dtype == np.float32
        line.set_data(np.array([[0, 1, 2], [5, 6, 7]]))  # (2, N)
        np.testing.assert_allclose(line.get_ydata(), [5, 6, 7])
        with pytest.raises(ValueError, match="same length"):
            line.set_data([0, 1], [0])

    def test_single_value_broadcasts_like_axhline(self, fig):
        (line,) = gplt.plot([0, 1, 2], [0, 0, 0])
        line.set_ydata([4])
        np.testing.assert_allclose(line.get_ydata(), [4, 4, 4])
        with pytest.raises(ValueError, match="use set_data"):
            line.set_ydata([1, 2])

    def test_per_point_colours_follow_a_length_change(self, fig):
        layer = PolylineLayer(pts=np.zeros((4, 2), np.float32))
        layer.colors = np.tile([1, 0, 0, 1], (4, 1)).astype(np.float32)
        layer.set_data(np.arange(10.0), np.arange(10.0))
        assert layer.colors.shape == (10, 4)

    def test_derived_geometry_is_refused(self, fig):
        (step,) = gplt.step([0, 1, 2], [1, 2, 1])
        with pytest.raises(ValueError, match="derived"):
            step.set_ydata([0, 0, 0])

    def test_function_layers_refuse_set_data(self):
        with pytest.raises(ValueError, match="samples its own points"):
            FunctionLayer(np.sin).set_data([0, 1], [0, 1])

    def test_scatter_offsets_and_sizes(self, fig):
        sc = gplt.scatter([0, 1, 2], [0, 1, 2])
        sc.set_offsets([[5, 5], [6, 6]])
        np.testing.assert_allclose(sc.get_offsets(), [[5, 5], [6, 6]])
        sc.set_sizes([3, 9])
        np.testing.assert_allclose(sc.sizes, [3, 9])
        sc.set_sizes(12)
        assert sc.sizes is None and sc.style.point_size == 12
        with pytest.raises(ValueError, match="sizes for 2 points"):
            sc.set_sizes([1, 2, 3])

    def test_the_legacy_mirror_follows_a_replaced_array(self, fig):
        (line,) = gplt.plot([0, 1, 2], [0, 0, 0])
        mirror = next(m for m in fig.scene.strips if m.pts is line.pts)
        line.set_data([0, 1, 2, 3], [1, 1, 1, 1])
        ctl.repoint_legacy_mirrors(fig)
        assert mirror.pts is line.pts and line._superseded_pts is None


class TestEngineAndPyplot:
    def test_controls_are_per_figure_and_lazy(self):
        a = gplt.figure()
        assert a._controls is None
        gplt.slider("x", 0, 1)
        b = gplt.figure()
        assert len(a.controls) == 1 and len(b.controls) == 0
        assert gplt.controls(a) is a.controls

    def test_pyplot_builders_return_live_controls(self, fig):
        made = [
            gplt.slider("s", 0, 1),
            gplt.range_slider("r", 0, 1),
            gplt.checkbox("c"),
            gplt.dropdown("d", ["a", "b"]),
            gplt.radio("o", ["a", "b"]),
            gplt.textbox("t"),
            gplt.number("n"),
            gplt.color_picker("col"),
            gplt.readout("ro"),
            gplt.button("b"),
        ]
        assert [c.owner for c in made] == [fig.controls] * len(made)
        assert fig.controls["o"].style == "radio"
        with pytest.raises(TypeError, match="needs a label"):
            gplt.button()

    def test_controls_do_not_disturb_a_headless_export(self, fig, tmp_path):
        x = np.linspace(0, 1, 20)

        @gplt.interact(k=(0, 3))
        def curve(k=2.0):
            return x, k * x

        path = tmp_path / "out.png"
        gplt.savefig(str(path))
        assert path.exists() and path.stat().st_size > 0


def test_a_failed_run_leaves_no_stale_result(fig):
    live = fig.controls
    k = gplt.slider("k", 0, 10, 1)

    @gplt.interact
    def f(k):
        if k > 5:
            raise ValueError("too big")
        return k * 2

    live._set_live(True)
    k.value = 9
    live.flush()
    assert f.result is None and f.error == "ValueError: too big"
    k.value = 2
    live.flush()
    assert f.result == 4.0 and f.error is None
    live._live = False
