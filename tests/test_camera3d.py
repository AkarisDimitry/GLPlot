"""Test the 3D camera: matrices, poles, projections, box aspect and the compat facade.

Pure numpy. No OpenGL, no window, no imgui — :class:`glplot.core.camera3d.Camera3D` is
plain state with a matrix builder attached, which is exactly what makes these assertions
possible at all. Before it existed the equivalent code was three floats inlined in a GL
renderer and could only be checked by looking at a picture.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from glplot.core.camera3d import (
    DEFAULT_DISTANCE_FACTOR,
    MAX_ELEV,
    MIN_ELEV,
    STANDARD_VIEW_LABELS,
    STANDARD_VIEWS,
    SYSTEM_3D_ARTISTS,
    Axes3DOptions,
    Camera3D,
    View3DProxy,
    bounds_centre_radius,
    look_at,
    orthographic,
    perspective,
)

UNIT_BOX = (-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)


def _project(camera: Camera3D, bounds, point, aspect: float = 1.6):
    """Project a world point to NDC through ``camera``. None when it is behind the eye."""
    mvp = np.asarray(camera.mvp(aspect, bounds), dtype=np.float64)
    clip = mvp @ np.array([point[0], point[1], point[2], 1.0])
    if clip[3] <= 1e-9:
        return None
    return clip[:3] / clip[3]


class TestMatrixPrimitives:
    """The three matrix builders, which everything else is composed from."""

    def test_perspective_maps_the_near_plane_to_minus_one(self):
        near, far = 0.5, 10.0
        proj = perspective(60.0, 1.0, near, far)
        for depth, expected in ((near, -1.0), (far, 1.0)):
            clip = proj @ np.array([0.0, 0.0, -depth, 1.0])
            assert clip[3] != 0
            assert clip[2] / clip[3] == pytest.approx(expected, abs=1e-5)

    def test_perspective_divides_the_horizontal_by_the_aspect(self):
        """A wide viewport must not stretch the scene; x is scaled by 1/aspect."""
        square = perspective(45.0, 1.0, 0.1, 100.0)
        wide = perspective(45.0, 2.0, 0.1, 100.0)
        assert wide[0, 0] == pytest.approx(square[0, 0] / 2.0)
        assert wide[1, 1] == pytest.approx(square[1, 1])

    def test_orthographic_is_affine_in_depth(self):
        """Ortho depth is linear, which is why it can afford a symmetric near/far box."""
        proj = orthographic(-1.0, 1.0, -1.0, 1.0, -5.0, 5.0)
        # Evenly spaced depths: a linear map sends them to evenly spaced NDC, so the
        # second difference vanishes. (A perspective matrix would not.)
        depths = np.linspace(-4.0, 4.0, 5)
        ndc = [(proj @ np.array([0.0, 0.0, d, 1.0]))[2] for d in depths]
        assert np.allclose(np.diff(np.diff(ndc)), 0.0, atol=1e-9)

    def test_orthographic_accepts_a_negative_near_plane(self):
        """A box, not a cone: pulling near behind the eye is legal and must not blow up."""
        proj = orthographic(-1.0, 1.0, -1.0, 1.0, -3.0, 3.0)
        assert np.all(np.isfinite(proj))

    def test_look_at_puts_the_eye_at_the_origin(self):
        eye = np.array([3.0, 4.0, 5.0], dtype=np.float32)
        view = look_at(eye, np.zeros(3, dtype=np.float32), np.array([0, 0, 1], np.float32))
        transformed = view @ np.array([eye[0], eye[1], eye[2], 1.0])
        assert np.allclose(transformed[:3], 0.0, atol=1e-5)

    def test_bounds_centre_radius_handles_none_and_nonfinite(self):
        for bounds in (None, (np.nan, 1.0, 0.0, 1.0, 0.0, 1.0)):
            centre, radius = bounds_centre_radius(bounds)
            assert np.allclose(centre, 0.0)
            assert radius == pytest.approx(1.0)


class TestOrientation:
    """Elevation, azimuth, roll and the pole — the case the old code could not reach."""

    def test_default_eye_matches_the_historical_expression(self):
        """The pre-camera renderer inlined this; a default camera must reproduce it."""
        camera = Camera3D()
        centre, radius = bounds_centre_radius(UNIT_BOX)
        el, az = math.radians(camera.elev), math.radians(camera.azim)
        expected = centre + radius * DEFAULT_DISTANCE_FACTOR * np.array(
            [math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)]
        )
        assert np.allclose(camera.eye(centre, radius), expected, atol=1e-5)

    @pytest.mark.parametrize("elev", [-90.0, -89.9, 0.0, 45.0, 89.9, 90.0])
    def test_every_elevation_including_the_poles_builds_a_finite_matrix(self, elev):
        """``look_at`` degenerates at the pole; :meth:`basis` is what keeps it finite."""
        camera = Camera3D(elev=elev)
        assert np.all(np.isfinite(camera.mvp(1.6, UNIT_BOX)))

    def test_basis_is_orthonormal_at_the_pole(self):
        right, up, forward = Camera3D(elev=90.0).basis()
        for vector in (right, up, forward):
            assert float(np.linalg.norm(vector)) == pytest.approx(1.0, abs=1e-5)
        assert float(np.dot(right, up)) == pytest.approx(0.0, abs=1e-5)
        assert float(np.dot(right, forward)) == pytest.approx(0.0, abs=1e-5)

    def test_basis_stays_orthonormal_under_roll(self):
        right, up, forward = Camera3D(roll=37.0).basis()
        assert float(np.dot(right, up)) == pytest.approx(0.0, abs=1e-5)
        assert float(np.linalg.norm(np.cross(right, up))) == pytest.approx(1.0, abs=1e-5)
        assert float(np.dot(np.cross(right, up), forward)) == pytest.approx(-1.0, abs=1e-4)

    def test_roll_rotates_the_projected_image(self):
        """A 90-degree roll must move a point that was up to one that is sideways."""
        upright = _project(Camera3D(elev=0.0, azim=-90.0), UNIT_BOX, (0.0, 0.0, 1.0))
        rolled = _project(Camera3D(elev=0.0, azim=-90.0, roll=90.0), UNIT_BOX, (0.0, 0.0, 1.0))
        assert upright is not None and rolled is not None
        assert abs(upright[1]) > abs(upright[0])
        assert abs(rolled[0]) > abs(rolled[1])

    def test_orbit_wraps_azimuth_and_clamps_elevation(self):
        camera = Camera3D(elev=80.0, azim=170.0)
        camera.orbit(30.0, 40.0)
        assert -180.0 < camera.azim <= 180.0
        assert camera.elev == pytest.approx(MAX_ELEV)
        camera.orbit(0.0, -400.0)
        assert camera.elev == pytest.approx(MIN_ELEV)

    def test_spin_wraps_roll(self):
        camera = Camera3D(roll=170.0)
        camera.spin(30.0)
        assert -180.0 < camera.roll <= 180.0

    @pytest.mark.parametrize("name", sorted(STANDARD_VIEWS))
    def test_every_preset_is_labelled_and_renders(self, name):
        camera = Camera3D()
        camera.apply_view(name)
        assert name in STANDARD_VIEW_LABELS
        assert np.all(np.isfinite(camera.mvp(1.6, UNIT_BOX)))

    def test_unknown_preset_raises(self):
        with pytest.raises(ValueError, match="unknown view"):
            Camera3D().apply_view("sideways")

    def test_preset_keeps_the_dolly_and_the_pan(self):
        """ "Look from the top" is about orientation; re-framing would discard a zoom."""
        camera = Camera3D(distance=7.0, pan=(1.0, 2.0, 3.0))
        camera.apply_view("top")
        assert camera.distance == pytest.approx(7.0)
        assert camera.pan == (1.0, 2.0, 3.0)

    def test_top_view_looks_down_the_vertical_axis(self):
        camera = Camera3D()
        camera.apply_view("top")
        assert camera.direction()[2] == pytest.approx(1.0, abs=1e-6)


class TestProjection:
    """Perspective versus orthographic, and the framing they share."""

    def test_perspective_foreshortens_and_orthographic_does_not(self):
        """Two points at the same lateral offset but different depths.

        With ``elev=0, azim=0`` the eye sits on +x, so *depth* is x and the lateral
        offset that maps to screen height is z. The probes therefore share z and differ
        in x — the near one projects further from the centre under perspective, and to
        exactly the same place under orthographic.
        """
        near_point, far_point = (0.9, 0.0, 0.5), (-0.9, 0.0, 0.5)
        persp = Camera3D(elev=0.0, azim=0.0)
        ortho = Camera3D(elev=0.0, azim=0.0, projection="orthographic")

        p_near = _project(persp, UNIT_BOX, near_point)
        p_far = _project(persp, UNIT_BOX, far_point)
        o_near = _project(ortho, UNIT_BOX, near_point)
        o_far = _project(ortho, UNIT_BOX, far_point)
        assert abs(p_near[1]) > abs(p_far[1]) * 1.05
        assert abs(o_near[1]) == pytest.approx(abs(o_far[1]), rel=1e-6)

    def test_projections_agree_at_the_target_plane(self):
        """The ortho box is derived from fov and distance so switching does not jump."""
        on_plane = (0.0, 0.0, 0.0)
        persp = _project(Camera3D(), UNIT_BOX, on_plane)
        ortho = _project(Camera3D(projection="orthographic"), UNIT_BOX, on_plane)
        assert np.allclose(persp[:2], ortho[:2], atol=1e-5)

    def test_set_projection_accepts_matplotlib_spellings(self):
        camera = Camera3D()
        camera.set_projection("ortho")
        assert camera.projection == "orthographic"
        camera.set_projection("persp")
        assert camera.projection == "perspective"

    def test_set_projection_rejects_anything_else(self):
        with pytest.raises(ValueError, match="projection must be one of"):
            Camera3D().set_projection("fisheye")

    def test_orthographic_clip_box_encloses_the_scene(self):
        """Ortho must never clip: the box is symmetric about the target."""
        camera = Camera3D(projection="orthographic")
        near, far = camera.clip_planes(1.0)
        distance = camera.resolve_distance(1.0)
        assert near < distance - 1.0
        assert far > distance + 1.0

    def test_perspective_far_plane_follows_the_dolly(self):
        """Pulling back used to clip: the far plane was pinned at 12x the radius."""
        close = Camera3D(distance=3.0).clip_planes(1.0)[1]
        far_away = Camera3D(distance=500.0).clip_planes(1.0)[1]
        assert far_away > close
        assert far_away > 500.0

    def test_up_axis_y_moves_the_vertical(self):
        camera = Camera3D()
        camera.set_up_axis("y")
        assert np.allclose(camera.world_up(), [0.0, 1.0, 0.0])
        assert np.all(np.isfinite(camera.mvp(1.6, UNIT_BOX)))

    def test_up_axis_rejects_anything_else(self):
        with pytest.raises(ValueError, match="up_axis must be"):
            Camera3D().set_up_axis("w")


class TestFramingAndPan:
    """Distance, pan and the auto-fit that keeps a plot centred as its data changes."""

    def test_distance_none_resolves_against_the_scene(self):
        camera = Camera3D()
        assert camera.distance is None
        assert camera.resolve_distance(2.0) == pytest.approx(2.0 * DEFAULT_DISTANCE_FACTOR)

    def test_dolly_multiplies_and_clamps(self):
        camera = Camera3D(distance=10.0)
        camera.dolly(0.5)
        assert camera.distance == pytest.approx(5.0)
        camera.dolly(0.0)
        assert camera.distance > 0.0

    def test_pan_moves_the_target_in_the_camera_plane(self):
        """A horizontal drag must not move the target along the view direction."""
        camera = Camera3D()
        camera.pan_pixels(100.0, 0.0, 800.0, 1.0)
        offset = np.asarray(camera.pan)
        assert float(np.linalg.norm(offset)) > 0.0
        assert float(np.dot(offset, camera.direction())) == pytest.approx(0.0, abs=1e-5)

    def test_pan_is_reversible(self):
        camera = Camera3D()
        camera.pan_pixels(37.0, -12.0, 800.0, 2.0)
        camera.pan_pixels(-37.0, 12.0, 800.0, 2.0)
        assert np.allclose(camera.pan, 0.0, atol=1e-6)

    def test_frame_clears_the_pan_and_sets_a_distance(self):
        camera = Camera3D(pan=(5.0, 5.0, 5.0))
        camera.frame(UNIT_BOX)
        assert camera.pan == (0.0, 0.0, 0.0)
        assert camera.distance is not None and camera.distance > 0.0

    def test_frame_on_no_data_returns_to_auto(self):
        camera = Camera3D(distance=42.0)
        camera.frame(None)
        assert camera.distance is None

    def test_reset_restores_the_default_isometric(self):
        camera = Camera3D(elev=1.0, azim=2.0, roll=3.0, distance=4.0, pan=(5.0, 6.0, 7.0))
        camera.reset()
        assert (camera.elev, camera.azim, camera.roll) == STANDARD_VIEWS["iso"]
        assert camera.distance is None
        assert camera.pan == (0.0, 0.0, 0.0)

    def test_advance_spin_turns_only_when_enabled(self):
        camera = Camera3D()
        assert camera.advance_spin(0.1) is False
        camera.auto_spin = 90.0
        assert camera.advance_spin(0.5) is True
        assert camera.azim == pytest.approx(-45.0 + 45.0)


class TestBoxAspect:
    """Axis normalisation — the control that makes unrelated quantities comparable."""

    def test_none_is_the_identity(self):
        camera = Camera3D()
        assert np.allclose(camera.model_matrix(UNIT_BOX), np.eye(4))
        assert camera.transform_bounds(UNIT_BOX) == UNIT_BOX

    def test_cube_equalises_wildly_different_extents(self):
        camera = Camera3D(box_aspect=(1.0, 1.0, 1.0))
        bounds = (0.0, 1.0, 0.0, 10.0, 0.0, 1e6)
        transformed = camera.transform_bounds(bounds)
        spans = [
            transformed[1] - transformed[0],
            transformed[3] - transformed[2],
            transformed[5] - transformed[4],
        ]
        assert spans[0] == pytest.approx(spans[1], rel=1e-6)
        assert spans[1] == pytest.approx(spans[2], rel=1e-6)

    def test_normalisation_preserves_the_overall_size(self):
        """Without the geometric-mean renormalisation a 1e6 axis would shrink the model."""
        bounds = (0.0, 1.0, 0.0, 10.0, 0.0, 1e6)
        scale = Camera3D(box_aspect=(1.0, 1.0, 1.0)).scale_factors(bounds)
        assert float(np.exp(np.mean(np.log(scale)))) == pytest.approx(1.0, rel=1e-9)

    def test_aspect_ratios_other_than_one_are_honoured(self):
        camera = Camera3D(box_aspect=(2.0, 1.0, 1.0))
        transformed = camera.transform_bounds(UNIT_BOX)
        x_span = transformed[1] - transformed[0]
        y_span = transformed[3] - transformed[2]
        assert x_span == pytest.approx(2.0 * y_span, rel=1e-6)

    def test_model_matrix_keeps_the_centre_fixed(self):
        bounds = (2.0, 4.0, 10.0, 20.0, 0.0, 1.0)
        camera = Camera3D(box_aspect=(1.0, 1.0, 1.0))
        centre, _ = bounds_centre_radius(bounds)
        moved = camera.model_matrix(bounds) @ np.array([*centre, 1.0])
        assert np.allclose(moved[:3], centre, atol=1e-4)

    def test_mvp_includes_the_model_matrix(self):
        bounds = (0.0, 1.0, 0.0, 1.0, 0.0, 1000.0)
        plain = Camera3D().mvp(1.6, bounds)
        cubed = Camera3D(box_aspect=(1.0, 1.0, 1.0)).mvp(1.6, bounds)
        assert not np.allclose(plain, cubed)


class TestSerialisation:
    """``to_dict`` / ``from_dict`` — the bridge to the renderer's layer metadata."""

    def test_round_trip_preserves_every_field(self):
        camera = Camera3D(
            elev=12.0,
            azim=-33.0,
            roll=5.0,
            fov=61.0,
            distance=9.0,
            projection="orthographic",
            up_axis="y",
            pan=(1.0, 2.0, 3.0),
            box_aspect=(1.0, 2.0, 3.0),
        )
        restored = Camera3D.from_dict(camera.to_dict())
        for field in ("elev", "azim", "roll", "fov", "distance", "projection", "up_axis"):
            assert getattr(restored, field) == getattr(camera, field)
        assert restored.pan == camera.pan
        assert restored.box_aspect == camera.box_aspect

    def test_from_dict_tolerates_a_legacy_three_float_camera(self):
        """Metadata written before this module existed must still render identically."""
        camera = Camera3D.from_dict({"elev": 40.0, "azim": -20.0, "fov": 55.0})
        assert camera.elev == 40.0
        assert camera.distance is None
        assert camera.projection == "perspective"

    def test_from_dict_ignores_junk_rather_than_raising(self):
        camera = Camera3D.from_dict(
            {"elev": "nonsense", "projection": "fisheye", "pan": "no", "box_aspect": 5}
        )
        assert camera.elev == Camera3D().elev
        assert camera.projection == "perspective"
        assert camera.box_aspect is None

    def test_from_dict_of_none_is_the_default(self):
        assert Camera3D.from_dict(None).to_dict() == Camera3D().to_dict()

    def test_copy_is_independent(self):
        camera = Camera3D(pan=(1.0, 1.0, 1.0))
        clone = camera.copy()
        clone.elev = 80.0
        clone.pan = (9.0, 9.0, 9.0)
        assert camera.elev != clone.elev
        assert camera.pan == (1.0, 1.0, 1.0)


class TestView3DProxy:
    """The dict facade. Every historical ``view3d[...]`` call site goes through it."""

    def _proxy(self):
        camera, axes = Camera3D(), Axes3DOptions()
        return View3DProxy(camera, axes), camera, axes

    def test_reads_and_writes_reach_the_camera(self):
        proxy, camera, _ = self._proxy()
        assert proxy["elev"] == camera.elev
        proxy["elev"] = 41.0
        assert camera.elev == pytest.approx(41.0)

    def test_writes_reach_the_axes_options(self):
        proxy, _, axes = self._proxy()
        proxy["show_axes"] = False
        assert axes.show_axes is False
        proxy["tick_count"] = 9
        assert axes.tick_count == 9
        proxy["zlabel"] = "height"
        assert axes.zlabel == "height"

    def test_distance_none_round_trips(self):
        proxy, camera, _ = self._proxy()
        proxy["distance"] = 5.0
        assert camera.distance == pytest.approx(5.0)
        proxy["distance"] = None
        assert camera.distance is None

    def test_projection_is_validated_through_the_camera(self):
        proxy, camera, _ = self._proxy()
        proxy["projection"] = "ortho"
        assert camera.projection == "orthographic"
        with pytest.raises(ValueError):
            proxy["projection"] = "fisheye"

    def test_unknown_keys_are_stored_rather_than_rejected(self):
        proxy, _, _ = self._proxy()
        proxy["embedder_state"] = 7
        assert proxy["embedder_state"] == 7
        assert "embedder_state" in dict(proxy)

    def test_builtin_keys_cannot_be_deleted(self):
        proxy, _, _ = self._proxy()
        with pytest.raises(KeyError):
            del proxy["elev"]

    def test_it_behaves_as_a_mapping(self):
        proxy, _, _ = self._proxy()
        as_dict = dict(proxy)
        assert "elev" in as_dict and "show_axes" in as_dict
        assert len(proxy) == len(as_dict)
        assert proxy.get("azim") == proxy["azim"]

    def test_rebind_re_points_at_another_panel(self):
        proxy, _, _ = self._proxy()
        other_camera, other_axes = Camera3D(elev=77.0), Axes3DOptions()
        proxy._rebind(other_camera, other_axes)
        assert proxy["elev"] == pytest.approx(77.0)


class TestAxes3DOptions:
    """The decoration flags and the axis limits that clip."""

    def test_limits_default_to_none(self):
        axes = Axes3DOptions()
        assert axes.limits() == (None, None, None)
        assert axes.has_limits() is False

    def test_has_limits_notices_any_axis(self):
        axes = Axes3DOptions()
        axes.ylim = (0.0, 1.0)
        assert axes.has_limits() is True

    def test_copy_is_independent(self):
        axes = Axes3DOptions()
        clone = axes.copy()
        clone.show_grid = False
        assert axes.show_grid is True


def test_system_artists_cover_every_decoration_layer():
    """The one canonical set; consumers that cannot import the engine read it."""
    assert SYSTEM_3D_ARTISTS == {"axis3d", "floor3d", "grid3d", "ticks3d"}
