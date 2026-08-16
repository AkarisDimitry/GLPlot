"""Test the pyplot 3D axes surface: projection, camera, limits, labels and aspect.

No window and no GL: ``GPULinePlot`` constructs headless and every function under test is
state manipulation. ``gplt.show()`` is never called.
"""

from __future__ import annotations

import numpy as np
import pytest

import glplot.pyplot as gplt


@pytest.fixture(autouse=True)
def fresh_figure():
    """Each test starts from a clean pyplot state.

    ``figure(num=...)`` *reuses* a figure registered under that number, which is
    matplotlib's documented behaviour and exactly what makes a shared ``"p"`` leak a 3D
    mode from one test into the next. The module-level registries are therefore reset
    rather than relying on ``close()``.
    """

    def reset() -> None:
        gplt._CURRENT_PLOT = None
        gplt._ALL_PLOTS.clear()
        gplt._FIGURES_BY_NUM.clear()

    reset()
    yield
    reset()


def _curve(n: int = 200):
    t = np.linspace(0.0, 10.0, n)
    return np.cos(t), np.sin(t), t / 5.0


class TestProjectionKeyword:
    """``projection="3d"`` — matplotlib's spelling, honoured everywhere it appears."""

    def test_figure_projection_pins_an_empty_figure_to_3d(self):
        """The point of an explicit mode: 3D tools before there is any 3D data."""
        fig = gplt.figure("p", projection="3d")
        assert fig.is_3d_scene() is True
        assert fig.ndim == 3

    def test_no_projection_keeps_the_inferred_behaviour(self):
        fig = gplt.figure("p")
        assert fig.is_3d_scene() is False
        gplt.plot3d(*_curve())
        assert fig.is_3d_scene() is True

    def test_axes_projection_switches_the_current_panel(self):
        fig = gplt.figure("p")
        gplt.axes(projection="3d")
        assert fig.is_3d_scene() is True
        gplt.axes(projection="rectilinear")
        assert fig.is_3d_scene() is False

    def test_subplot_projection_is_per_panel(self):
        fig = gplt.figure("p")
        gplt.subplot(1, 2, 1)
        gplt.subplot(1, 2, 2, projection="3d")
        assert [panel.is_3d() for panel in fig.panels] == [False, True]

    def test_subplots_subplot_kw_applies_to_every_panel(self):
        fig, _axs = gplt.subplots(2, 2, subplot_kw={"projection": "3d"})
        assert all(panel.is_3d() for panel in fig.panels)

    def test_an_unsupported_projection_is_refused_not_ignored(self):
        with pytest.raises(ValueError, match="not supported"):
            gplt.axes(projection="polar")

    def test_set_projection_flips_either_way(self):
        fig = gplt.figure("p")
        gplt.set_projection("3d")
        assert fig.ndim == 3
        gplt.set_projection("2d")
        assert fig.ndim == 2


class TestCamera:
    """view_init, projection type and the presets."""

    def test_view_init_sets_the_angles(self):
        fig = gplt.figure("p", projection="3d")
        gplt.view_init(elev=33.0, azim=-77.0, roll=12.0)
        assert fig.camera3d.elev == pytest.approx(33.0)
        assert fig.camera3d.azim == pytest.approx(-77.0)
        assert fig.camera3d.roll == pytest.approx(12.0)

    def test_view_init_leaves_omitted_angles_alone(self):
        fig = gplt.figure("p", projection="3d")
        gplt.view_init(elev=10.0, azim=20.0)
        gplt.view_init(elev=50.0)
        assert fig.camera3d.azim == pytest.approx(20.0)

    def test_view_init_vertical_axis(self):
        fig = gplt.figure("p", projection="3d")
        gplt.view_init(vertical_axis="y")
        assert fig.camera3d.up_axis == "y"

    def test_view_init_share_reaches_every_panel(self):
        fig, _axs = gplt.subplots(1, 3, subplot_kw={"projection": "3d"})
        gplt.view_init(elev=44.0, share=True)
        assert all(panel.camera3d.elev == pytest.approx(44.0) for panel in fig.panels)

    def test_view_init_without_share_touches_only_the_current_panel(self):
        fig, _axs = gplt.subplots(1, 2, subplot_kw={"projection": "3d"})
        gplt.view_init(elev=44.0)
        assert fig.panels[0].camera3d.elev == pytest.approx(44.0)
        assert fig.panels[1].camera3d.elev != pytest.approx(44.0)

    def test_proj_type_round_trips(self):
        gplt.figure("p", projection="3d")
        gplt.set_proj_type("ortho")
        assert gplt.get_proj_type() == "ortho"
        gplt.set_proj_type("persp")
        assert gplt.get_proj_type() == "persp"

    def test_focal_length_converts_to_a_field_of_view(self):
        fig = gplt.figure("p", projection="3d")
        gplt.set_proj_type("persp", focal_length=1.0)
        assert fig.camera3d.fov == pytest.approx(90.0, abs=1e-6)

    def test_a_non_positive_focal_length_is_refused(self):
        """Refused with matplotlib's own wording, not GLPlot's.

        The message is matched verbatim on purpose: a script that catches this and reads
        the text should not be able to tell the two libraries apart. matplotlib raises
        ``focal_length = 0.0 must be greater than 0``; so does this.
        """
        gplt.figure("p", projection="3d")
        with pytest.raises(ValueError, match="must be greater than 0"):
            gplt.set_proj_type("persp", focal_length=0.0)
        with pytest.raises(ValueError, match="must be greater than 0"):
            gplt.set_proj_type("persp", focal_length=-1.0)

    def test_view_preset_snaps_the_orientation(self):
        fig = gplt.figure("p", projection="3d")
        gplt.view_preset("top")
        assert fig.camera3d.elev == pytest.approx(90.0)

    def test_auto_rotate_sets_and_clears_the_spin(self):
        fig = gplt.figure("p", projection="3d")
        gplt.auto_rotate(45.0)
        assert fig.camera3d.auto_spin == pytest.approx(45.0)
        gplt.auto_rotate(0.0)
        assert fig.camera3d.auto_spin == 0.0

    def test_dist3d_reads_and_writes(self):
        gplt.figure("p", projection="3d")
        gplt.plot3d(*_curve())
        automatic = gplt.dist3d()
        assert automatic > 0.0
        assert gplt.dist3d(12.0) == pytest.approx(12.0)


class TestBoxAspect:
    def test_setting_and_clearing(self):
        fig = gplt.figure("p", projection="3d")
        gplt.set_box_aspect((1, 1, 1))
        assert gplt.get_box_aspect() == (1.0, 1.0, 1.0)
        gplt.set_box_aspect(None)
        assert gplt.get_box_aspect() is None
        assert fig.camera3d.box_aspect is None

    def test_a_wrong_length_is_refused(self):
        gplt.figure("p", projection="3d")
        with pytest.raises(ValueError, match="three components"):
            gplt.set_box_aspect((1, 1))

    def test_a_non_positive_component_is_refused(self):
        gplt.figure("p", projection="3d")
        with pytest.raises(ValueError, match="positive and finite"):
            gplt.set_box_aspect((1, 0, 1))

    def test_zoom_tightens_the_framing(self):
        gplt.figure("p", projection="3d")
        gplt.plot3d(*_curve())
        gplt.set_box_aspect((1, 1, 1))
        wide = gplt.dist3d()
        gplt.set_box_aspect((1, 1, 1), zoom=2.0)
        assert gplt.dist3d() < wide


class TestLimits:
    """A 3D limit clips as well as framing — that is the whole difference from 2D."""

    def test_reading_the_limits_without_setting_them(self):
        gplt.figure("p", projection="3d")
        gplt.plot3d(*_curve())
        lo, hi = gplt.zlim()
        assert hi > lo

    def test_setting_z_limits_clips(self):
        fig = gplt.figure("p", projection="3d")
        gplt.plot3d(*_curve())
        assert fig.clip_3d_bounds() is None
        assert gplt.set_zlim(0.5, 1.5) == (0.5, 1.5)
        clip = fig.clip_3d_bounds()
        assert clip is not None
        assert clip[4:] == (0.5, 1.5)

    def test_an_unbounded_axis_gets_the_inverted_sentinel(self):
        """The shader reads lo >= hi as "this axis is unbounded"."""
        fig = gplt.figure("p", projection="3d")
        gplt.plot3d(*_curve())
        gplt.set_zlim(0.0, 1.0)
        clip = fig.clip_3d_bounds()
        assert clip[0] >= clip[1] and clip[2] >= clip[3]

    def test_an_explicit_limit_is_not_padded(self):
        """A tick at the limit must land on the wall, not inside it."""
        fig = gplt.figure("p", projection="3d")
        gplt.plot3d(*_curve())
        gplt.set_zlim(-3.0, 7.0)
        assert fig.padded_3d_bounds()[4:] == (-3.0, 7.0)

    def test_x_and_y_limits_too(self):
        fig = gplt.figure("p", projection="3d")
        gplt.plot3d(*_curve())
        gplt.set_xlim3d(-0.5, 0.5)
        gplt.set_ylim3d(-0.25, 0.25)
        clip = fig.clip_3d_bounds()
        assert clip[:2] == (-0.5, 0.5)
        assert clip[2:4] == (-0.25, 0.25)

    def test_set_zlim3d_is_an_alias(self):
        gplt.figure("p", projection="3d")
        gplt.plot3d(*_curve())
        assert gplt.set_zlim3d(1.0, 2.0) == (1.0, 2.0)

    def test_a_pair_may_be_passed_as_one_argument(self):
        gplt.figure("p", projection="3d")
        gplt.plot3d(*_curve())
        assert gplt.set_zlim((0.25, 0.75)) == (0.25, 0.75)

    def test_zlim_setter_and_getter_forms(self):
        gplt.figure("p", projection="3d")
        gplt.plot3d(*_curve())
        gplt.zlim(0.1, 0.9)
        assert gplt.zlim() == (0.1, 0.9)

    def test_an_inverted_limit_is_refused(self):
        gplt.figure("p", projection="3d")
        gplt.plot3d(*_curve())
        with pytest.raises(ValueError, match="hi > lo"):
            gplt.set_zlim(2.0, 1.0)

    def test_autoscale3d_clears_every_limit(self):
        fig = gplt.figure("p", projection="3d")
        gplt.plot3d(*_curve())
        gplt.set_zlim(0.0, 1.0)
        gplt.set_xlim3d(0.0, 1.0)
        gplt.autoscale3d()
        assert fig.clip_3d_bounds() is None
        assert fig.axes3d.zlim is None


class TestLabelsAndDecoration:
    def test_axis_titles_reach_both_homes(self):
        """The GL renderer reads ``axes3d``; the matplotlib bridge reads ``plot.*label``."""
        fig = gplt.figure("p", projection="3d")
        gplt.plot3d(*_curve())
        gplt.xlabel("X")
        gplt.ylabel("Y")
        gplt.zlabel("Z")
        assert (fig.axes3d.xlabel, fig.axes3d.ylabel, fig.axes3d.zlabel) == ("X", "Y", "Z")
        assert (fig.xlabel, fig.ylabel, fig.zlabel) == ("X", "Y", "Z")
        assert gplt.get_zlabel() == "Z"

    def test_zlabel_accepts_the_matplotlib_keyword(self):
        gplt.figure("p", projection="3d")
        gplt.zlabel(zlabel="height")
        assert gplt.get_zlabel() == "height"

    def test_titles_produce_label_anchors(self):
        fig = gplt.figure("p", projection="3d")
        gplt.plot3d(*_curve())
        gplt.zlabel("Z")
        titles = [a for a in fig._axes3d_labels if a.kind == "title"]
        assert any(a.text == "Z" for a in titles)

    def test_grid3d_toggles_the_walls(self):
        fig = gplt.figure("p", projection="3d")
        gplt.plot3d(*_curve())
        gplt.grid3d(False)
        assert fig.axes3d.show_grid is False
        assert not [layer for layer in fig.scene.layers if layer.metadata.get("artist") == "grid3d"]
        gplt.grid3d(True)
        assert fig.axes3d.show_grid is True

    def test_axis_off_and_on(self):
        fig = gplt.figure("p", projection="3d")
        gplt.plot3d(*_curve())
        gplt.set_axis3d_off()
        assert fig.axes3d.show_axes is False
        assert not [
            layer
            for layer in fig.scene.layers
            if layer.metadata.get("artist") in fig._SYSTEM_3D_ARTISTS
        ]
        gplt.set_axis3d_on()
        assert fig.axes3d.show_axes is True


class TestTicksFollowTheCamera:
    """The 3D ticks re-scale with the view, the way the 2D ones re-scale with the zoom."""

    def _figure(self):
        fig = gplt.figure("p", projection="3d")
        rng = np.random.default_rng(3)
        gplt.scatter3d(rng.normal(size=400) * 4, rng.normal(size=400) * 12, rng.normal(size=400))
        fig.set_3d_view()
        return fig

    def _counts(self, fig, dolly):
        fig.camera3d.distance = fig._camera_metadata(fig.get_3d_bounds())["distance"] * dolly
        fig.set_3d_view()
        return tuple(len(values) for values, _ in fig._axes3d_ticks)

    def test_dollying_in_subdivides_the_axes(self):
        """The complaint this answers: the same five numbers at every distance."""
        fig = self._figure()
        assert sum(self._counts(fig, 0.4)) > sum(self._counts(fig, 3.0))

    def test_the_tick_values_actually_change(self):
        fig = self._figure()
        self._counts(fig, 3.0)
        far = [tuple(values) for values, _ in fig._axes3d_ticks]
        self._counts(fig, 0.4)
        near = [tuple(values) for values, _ in fig._axes3d_ticks]
        assert far != near

    def test_a_pinned_count_stops_adapting(self):
        fig = self._figure()
        fig.axes3d.tick_count = 5
        assert self._counts(fig, 0.4) == self._counts(fig, 3.0)

    def test_zticks_reports_what_is_drawn_not_a_re_derivation(self):
        fig = self._figure()
        self._counts(fig, 0.4)
        values, labels = gplt.zticks()
        drawn, drawn_labels = fig._axes3d_ticks[2]
        assert np.allclose(values, drawn) and labels == list(drawn_labels)

    def test_labels_are_placed_with_the_camera_the_scene_is_drawn_through(self):
        """``camera3d`` stays in auto mode, so projecting through it misses by ~90 px."""
        from glplot.renderers.geometry3d import _layer_bounds, _layer_camera

        fig = self._figure()
        layer = next(
            layer
            for layer in fig.scene.layers
            if layer.metadata.get("artist") not in fig._SYSTEM_3D_ARTISTS
            and hasattr(layer, "vertices")
        )
        aspect = max(fig.width, 1) / max(fig.height, 1)
        placed = np.asarray(fig._axes3d_camera.mvp(aspect, fig._axes3d_bounds))
        drawn = np.asarray(_layer_camera(layer).mvp(aspect, _layer_bounds(layer)))
        assert np.allclose(placed, drawn)

    def test_the_resolved_camera_is_not_the_bare_panel_camera(self):
        """Guards the regression: they agreed only because both were unresolved."""
        fig = self._figure()
        assert fig.camera3d.distance is None
        assert fig._axes3d_camera.distance is not None


class TestAxesProxyDelegation:
    """``ax.view_init(...)`` must activate that panel and then call the same function."""

    def test_a_proxy_method_targets_its_own_panel(self):
        fig, axs = gplt.subplots(1, 2, subplot_kw={"projection": "3d"})
        axs[1].view_init(elev=61.0)
        assert fig.panels[1].camera3d.elev == pytest.approx(61.0)
        assert fig.panels[0].camera3d.elev != pytest.approx(61.0)

    def test_a_proxy_can_set_its_own_limits(self):
        fig, axs = gplt.subplots(1, 2, subplot_kw={"projection": "3d"})
        axs[0].plot3d(*_curve())
        axs[0].set_zlim(0.0, 1.0)
        assert fig.panels[0].axes3d.zlim == (0.0, 1.0)
        assert fig.panels[1].axes3d.zlim is None
