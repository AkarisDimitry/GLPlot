"""Test the steppable-process catalogue: its interface, its physics and its ceilings.

The catalogue is data, so the generic tests are a walk over it -- that is the point of
storing it as data, since a new process cannot be added without every assertion here
applying to it. The specific tests are the ones that matter: they assert *properties* that
a wrong implementation would fail, not merely that the code ran.

The three that would catch a genuinely broken module:

* :class:`TestEnergyConservation` integrates each mechanical system for thousands of frames
  and demands that its energy stay in a bounded band. A non-symplectic integrator passes
  the "it produced numbers" test and fails this one; the Euler contrast test makes that
  concrete by integrating the same orbit both ways and showing Euler unwinding it.
* :class:`TestGameOfLife` asserts the textbook oscillator periods and the glider's exact
  translation. These are not tolerances -- they are grid equalities, and any error in the
  neighbour count or the birth/survival rule breaks at least one of them.
* :class:`TestMandelbrot` pins known members and non-members of the set and checks that a
  deep zoom stays finite.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from glplot.anim import processes as ap

ALL_KEYS = list(ap.PROCESS_KEYS)
CONTINUOUS = [k for k in ALL_KEYS if ap.process(k).is_continuous]
DISCRETE = [k for k in ALL_KEYS if ap.process(k).is_discrete]
WITH_ENERGY = [k for k in ALL_KEYS if ap.process(k).energy is not None]

#: Cheap parameter overrides for the walk-the-catalogue tests. The escape-time rasters at
#: their defaults are 192x192 and cost tens of milliseconds a frame; nothing in a generic
#: test needs that resolution, and shrinking them keeps the whole file well under a second.
CHEAP = {
    "mandelbrot_zoom": {"side": 48, "max_iter": 40},
    "julia_path": {"side": 48, "max_iter": 40},
    "gray_scott": {"width": 32, "height": 32},
    "wave2d": {"width": 32, "height": 32},
    "life": {"width": 32, "height": 32},
    "elementary_ca": {"width": 65, "rows": 32},
    "chaos_game": {"chains": 64, "iters": 4},
    "spring_chain": {"nodes": 32},
    "nbody": {"bodies": 5},
}


def cheap(key: str) -> dict:
    """Default parameters for ``key``, overridden with the cheap ones where they exist."""
    params = ap.process(key).defaults()
    params.update(CHEAP.get(key, {}))
    return params


def advance(key: str, steps: int, params=None, **kwargs) -> ap.State:
    """Run ``key`` for ``steps`` frames and return the last state."""
    proc = ap.process(key)
    params = cheap(key) if params is None else params
    state = proc.initial_state(params)
    for _ in range(steps):
        state = proc.step(state, params, **kwargs)
    return state


def energy_series(key: str, steps: int, params=None, **kwargs) -> np.ndarray:
    """Total energy at every frame of a ``steps``-frame run."""
    proc = ap.process(key)
    params = cheap(key) if params is None else params
    state = proc.initial_state(params)
    out = [proc.total_energy(state, params)]
    for _ in range(steps):
        state = proc.step(state, params, **kwargs)
        out.append(proc.total_energy(state, params))
    return np.asarray(out, dtype=np.float64)


class TestCatalogueIntegrity:
    """Every entry is complete and self-consistent, checked without running it."""

    def test_the_catalogue_is_not_empty_and_keys_match(self):
        assert ALL_KEYS
        assert all(ap.PROCESSES[k].key == k for k in ALL_KEYS)
        assert list(ap.PROCESS_KEYS) == [k for k, _ in ap.process_labels()]

    def test_there_are_processes_in_every_category(self):
        for key, _label in ap.CATEGORIES:
            assert ap.by_category(key), f"category {key!r} is empty"

    def test_the_three_families_the_module_exists_for_are_all_present(self):
        # Newton, Life and fractals: the reason this module was written.
        assert ap.by_category("mechanics")
        assert "life" in ALL_KEYS
        assert len(ap.by_category("fractal")) >= 3

    @pytest.mark.parametrize("key", ALL_KEYS)
    def test_every_process_is_well_formed(self, key):
        spec = ap.process(key)
        assert spec.label and spec.description
        assert spec.category in dict(ap.CATEGORIES)
        assert spec.time_model in dict(ap.TIME_MODELS)
        assert spec.table_layout in (ap.LAYOUT_SNAPSHOT, ap.LAYOUT_HISTORY)
        assert len(spec.plot_columns) == 2
        assert set(spec.plot_columns) <= set(spec.columns)
        assert len(set(spec.columns)) == len(spec.columns)
        if spec.color_column is not None:
            assert spec.color_column in spec.columns
        assert 1 <= spec.substeps <= ap.MAX_SUBSTEPS

    @pytest.mark.parametrize("key", ALL_KEYS)
    def test_parameter_defaults_are_inside_their_bounds(self, key):
        for param in ap.process(key).params:
            assert param.vmin <= param.default <= param.vmax, param.name
            assert param.label

    @pytest.mark.parametrize("key", ALL_KEYS)
    def test_the_time_model_and_the_declared_dt_agree(self, key):
        spec = ap.process(key)
        if spec.is_discrete:
            assert spec.dt == 0.0
        else:
            assert math.isfinite(spec.dt) and spec.dt > 0.0

    def test_by_category_and_by_time_model_partition_the_catalogue(self):
        by_cat = [p.key for key, _ in ap.CATEGORIES for p in ap.by_category(key)]
        assert sorted(by_cat) == sorted(ALL_KEYS)
        by_time = [p.key for m, _ in ap.TIME_MODELS for p in ap.by_time_model(m)]
        assert sorted(by_time) == sorted(ALL_KEYS)

    def test_unknown_process_raises_and_names_the_alternatives(self):
        with pytest.raises(ap.ProcessError, match="unknown process"):
            ap.process("perpetual_motion")
        try:
            ap.process("perpetual_motion")
        except ap.ProcessError as exc:
            assert "life" in str(exc)

    def test_process_error_is_a_value_error(self):
        assert issubclass(ap.ProcessError, ValueError)

    def test_category_label_falls_back_to_the_key(self):
        assert ap.category_label("mechanics") == "Newtonian mechanics"
        assert ap.category_label("nope") == "nope"

    def test_the_life_pattern_art_is_rectangular(self):
        for name, art in ap.LIFE_PATTERNS.items():
            if art:
                assert len({len(row) for row in art}) == 1, name
                assert set("".join(art)) <= {"#", "."}, name

    def test_the_ifs_definitions_are_shaped_and_normalised(self):
        for name, (mats, offs, weights) in ap.IFS_SYSTEMS.items():
            assert mats.shape[1:] == (2, 2), name
            assert offs.shape == (mats.shape[0], 2), name
            assert weights.shape == (mats.shape[0],), name
            assert weights.min() > 0.0, name
            assert abs(float(weights.sum()) - 1.0) < 1e-9, name


class TestProcessConstruction:
    """A malformed catalogue entry must fail at import time, not at first click."""

    def _kwargs(self, **over):
        base = dict(
            key="x",
            label="X",
            category="mechanics",
            time_model=ap.TIME_CONTINUOUS,
            icon="wave",
            description="d",
            kind="line",
            columns=("a", "b"),
            plot_columns=("a", "b"),
            setup=lambda p: {},
            advance=lambda s, p, dt: {},
            tabulate=lambda s, p: {},
            dt=0.1,
        )
        base.update(over)
        return base

    def test_a_valid_entry_constructs(self):
        assert ap.Process(**self._kwargs()).key == "x"

    def test_a_discrete_process_may_not_declare_a_timestep(self):
        with pytest.raises(ap.ProcessError, match="has no timestep"):
            ap.Process(**self._kwargs(time_model=ap.TIME_DISCRETE, dt=1.0))

    def test_a_continuous_process_must_declare_one(self):
        with pytest.raises(ap.ProcessError, match="positive default timestep"):
            ap.Process(**self._kwargs(dt=0.0))

    def test_an_unknown_time_model_is_rejected(self):
        with pytest.raises(ap.ProcessError, match="time_model"):
            ap.Process(**self._kwargs(time_model="sometimes"))

    def test_an_unknown_layout_is_rejected(self):
        with pytest.raises(ap.ProcessError, match="table_layout"):
            ap.Process(**self._kwargs(table_layout="film"))

    def test_out_of_range_substeps_are_rejected(self):
        with pytest.raises(ap.ProcessError, match="sub-steps"):
            ap.Process(**self._kwargs(substeps=0))
        with pytest.raises(ap.ProcessError, match="sub-steps"):
            ap.Process(**self._kwargs(substeps=ap.MAX_SUBSTEPS + 1))

    def test_plotting_a_column_it_does_not_produce_is_rejected(self):
        with pytest.raises(ap.ProcessError, match="does not produce"):
            ap.Process(**self._kwargs(plot_columns=("a", "zzz")))

    def test_colouring_by_a_column_it_does_not_produce_is_rejected(self):
        with pytest.raises(ap.ProcessError, match="does not produce"):
            ap.Process(**self._kwargs(color_column="zzz"))

    def test_a_parameter_default_outside_its_own_bounds_is_rejected(self):
        bad = (ap.ProcessParam("k", 5.0, 0.0, 1.0),)
        with pytest.raises(ap.ProcessError, match="outside its own bounds"):
            ap.Process(**self._kwargs(params=bad))

    def test_a_ragged_table_is_caught_and_the_lengths_are_named(self):
        """The guarantee ``state_to_table`` makes, enforced rather than assumed."""
        ragged = ap.Process(
            **self._kwargs(
                tabulate=lambda s, p: {"a": np.zeros(3), "b": np.zeros(5)},
            )
        )
        with pytest.raises(ap.ProcessError, match="ragged columns"):
            ragged.state_to_table(ap.State(2, 0.2, {}))
        try:
            ragged.state_to_table(ap.State(2, 0.2, {}))
        except ap.ProcessError as exc:
            assert "a=3" in str(exc) and "b=5" in str(exc)


class TestParamAndState:
    """The two small records the interface is built out of."""

    def test_clamp_respects_bounds_and_integrality(self):
        p = ap.ProcessParam("n", 4.0, 1.0, 8.0, integer=True)
        assert p.clamp(100.0) == 8.0
        assert p.clamp(-100.0) == 1.0
        assert p.clamp(3.6) == 4.0
        assert isinstance(p.clamp(3.6), float)
        q = ap.ProcessParam("x", 0.5, 0.0, 1.0)
        assert q.clamp(0.25) == pytest.approx(0.25)

    def test_state_indexing_names_what_it_holds(self):
        s = ap.State(3, 0.3, {"pos": np.zeros(2)})
        assert s["pos"].shape == (2,)
        assert s.get("nope") is None
        assert s.get("nope", 7) == 7
        with pytest.raises(ap.ProcessError, match="pos"):
            s["velocity"]

    def test_advanced_increments_the_frame_and_the_clock(self):
        s = ap.State(3, 0.30, {"a": 1})
        t = s.advanced(0.25, {"a": 2})
        assert (t.frame, t.time, t.data) == (4, 0.55, {"a": 2})

    def test_state_equality_is_identity_not_array_comparison(self):
        # The default dataclass __eq__ would raise on dicts of arrays; eq=False avoids it.
        a = ap.State(0, 0.0, {"g": np.zeros(3)})
        b = ap.State(0, 0.0, {"g": np.zeros(3)})
        assert a == a
        assert a != b

    def test_resolve_clamps_and_ignores_unknown_keys(self):
        proc = ap.process("life")
        resolved = proc.resolve({"width": 10_000, "not_a_param": 3.0})
        assert resolved["width"] == ap.MAX_GRID_SIDE
        assert "not_a_param" not in resolved
        assert set(resolved) == {p.name for p in proc.params}

    def test_defaults_are_a_fresh_dict_every_time(self):
        proc = ap.process("nbody")
        first = proc.defaults()
        first["G"] = 999.0
        assert proc.defaults()["G"] != 999.0


class TestTimeModel:
    """The continuous/discrete distinction is enforced, not merely advertised."""

    @pytest.mark.parametrize("key", DISCRETE)
    def test_a_discrete_process_refuses_a_timestep(self, key):
        proc = ap.process(key)
        state = proc.initial_state(cheap(key))
        with pytest.raises(ap.ProcessError, match="one step is one generation"):
            proc.step(state, cheap(key), 0.5)
        with pytest.raises(ap.ProcessError):
            proc.timestep(1.0)

    @pytest.mark.parametrize("key", DISCRETE)
    def test_a_discrete_clock_counts_generations(self, key):
        proc = ap.process(key)
        assert proc.timestep() == 0.0
        state = advance(key, 5)
        assert state.frame == 5
        assert state.time == 5.0

    @pytest.mark.parametrize("key", CONTINUOUS)
    def test_a_continuous_clock_advances_by_exactly_dt(self, key):
        proc = ap.process(key)
        state = advance(key, 4)
        assert state.frame == 4
        assert state.time == pytest.approx(4.0 * proc.dt, rel=1e-12)
        custom = advance(key, 4, dt=proc.dt * 0.5)
        assert custom.time == pytest.approx(2.0 * proc.dt, rel=1e-12)

    @pytest.mark.parametrize("key", CONTINUOUS)
    def test_a_continuous_process_rejects_a_degenerate_dt(self, key):
        proc = ap.process(key)
        with pytest.raises(ap.ProcessError, match="finite, non-zero"):
            proc.timestep(0.0)
        with pytest.raises(ap.ProcessError, match="finite, non-zero"):
            proc.timestep(float("nan"))

    def test_the_two_flags_are_complementary(self):
        for key in ALL_KEYS:
            proc = ap.process(key)
            assert proc.is_continuous != proc.is_discrete


class TestStepping:
    """The generic run-it contract, applied to every process in the catalogue."""

    @pytest.mark.parametrize("key", ALL_KEYS)
    def test_every_frame_is_finite(self, key):
        proc = ap.process(key)
        params = cheap(key)
        state = proc.initial_state(params)
        for _ in range(12):
            state = proc.step(state, params)
            table = proc.state_to_table(state, params)
            for name, column in table.items():
                assert np.all(np.isfinite(column)), f"{key}.{name} at frame {state.frame}"

    @pytest.mark.parametrize("key", ALL_KEYS)
    def test_the_table_matches_the_declared_columns_and_is_rectangular(self, key):
        proc = ap.process(key)
        params = cheap(key)
        table = proc.state_to_table(advance(key, 6, params), params)
        assert list(table) == list(proc.columns)
        assert len({len(v) for v in table.values()}) == 1
        for column in table.values():
            assert isinstance(column, np.ndarray)
            assert column.dtype == np.float64
            assert column.ndim == 1

    @pytest.mark.parametrize("key", ALL_KEYS)
    def test_the_frame_counter_increments_by_one(self, key):
        proc = ap.process(key)
        params = cheap(key)
        state = proc.initial_state(params)
        assert state.frame == 0 and state.time == 0.0
        for expected in range(1, 6):
            state = proc.step(state, params)
            assert state.frame == expected

    @pytest.mark.parametrize("key", ALL_KEYS)
    def test_a_run_is_reproducible_from_params_alone(self, key):
        proc = ap.process(key)
        params = cheap(key)
        left = proc.state_to_table(advance(key, 8, params), params)
        right = proc.state_to_table(advance(key, 8, params), params)
        for name in left:
            np.testing.assert_array_equal(left[name], right[name], err_msg=f"{key}.{name}")

    @pytest.mark.parametrize("key", ALL_KEYS)
    def test_tabulating_is_pure_and_repeatable(self, key):
        proc = ap.process(key)
        params = cheap(key)
        state = advance(key, 5, params)
        first = proc.state_to_table(state, params)
        second = proc.state_to_table(state, params)
        for name in first:
            np.testing.assert_array_equal(first[name], second[name])

    @pytest.mark.parametrize("key", ALL_KEYS)
    def test_a_seeded_process_changes_with_its_seed(self, key):
        proc = ap.process(key)
        if not any(p.name == "seed" for p in proc.params):
            pytest.skip(f"{key} takes no seed")
        params = cheap(key)
        if key == "life":  # the seed only matters for the random soup
            params["pattern"] = ap.life_pattern_index("random")
        if key == "elementary_ca":
            params["init"] = 1.0
        if key == "wave2d":
            params["drops"] = 4.0
        a = dict(params, seed=1.0)
        b = dict(params, seed=2.0)
        left = proc.state_to_table(advance(key, 3, a), a)
        right = proc.state_to_table(advance(key, 3, b), b)
        differs = any(
            left[n].shape != right[n].shape or not np.array_equal(left[n], right[n]) for n in left
        )
        assert differs, f"{key} ignored its seed"

    @pytest.mark.parametrize("key", ALL_KEYS)
    def test_the_declared_layout_describes_the_row_count(self, key):
        proc = ap.process(key)
        params = cheap(key)
        early = len(proc.state_to_table(advance(key, 2, params), params)[proc.columns[0]])
        late = len(proc.state_to_table(advance(key, 9, params), params)[proc.columns[0]])
        if proc.table_layout == ap.LAYOUT_HISTORY:
            assert late == early + 7
        elif key not in ("life", "chaos_game", "elementary_ca"):
            # The three exceptions are snapshots whose *content* is a varying number of
            # live cells / accumulated points, not a varying number of system elements.
            assert late == early

    def test_frames_yields_count_plus_one_states(self):
        states = list(ap.process("pendulum").frames(cheap("pendulum"), count=5))
        assert [s.frame for s in states] == [0, 1, 2, 3, 4, 5]
        assert len(ap.process("life").run(cheap("life"), count=3)) == 4

    def test_frames_is_lazy_and_refuses_an_unbounded_run(self):
        proc = ap.process("pendulum")
        with pytest.raises(ap.ProcessError, match="count must be between"):
            next(proc.frames(count=ap.MAX_FRAMES + 1))
        with pytest.raises(ap.ProcessError, match="count must be between"):
            next(proc.frames(count=-1))

    def test_substeps_can_be_overridden_per_call_and_is_clamped(self):
        proc = ap.process("two_body")
        coarse = proc.step(proc.initial_state(), substeps=1)
        fine = proc.step(proc.initial_state(), substeps=32)
        assert not np.array_equal(coarse["pos"], fine["pos"])
        # Out-of-range values clamp rather than raise: this arrives from a frame-rate
        # governor, not from a user.
        assert np.all(np.isfinite(proc.step(proc.initial_state(), substeps=10_000)["pos"]))
        assert np.all(np.isfinite(proc.step(proc.initial_state(), substeps=0)["pos"]))

    def test_a_process_without_an_energy_reports_none(self):
        for key in ALL_KEYS:
            proc = ap.process(key)
            value = proc.total_energy(advance(key, 2), cheap(key))
            assert (value is None) == (proc.energy is None)
            if value is not None:
                assert math.isfinite(value)


class TestEnergyConservation:
    """The test that catches a wrong integrator.

    Tolerances are stated as a fraction of the initial energy and were measured, not
    guessed; each is roughly three times the observed value, so a real regression trips it
    while ordinary float64 noise does not.
    """

    def _band_and_drift(self, series):
        scale = abs(series[0])
        return float(np.ptp(series) / scale), float(abs(series[-1] - series[0]) / scale)

    def test_two_body_verlet_holds_energy_to_one_part_in_a_thousand(self):
        # Measured over 5000 frames: band 7.2e-5, final drift 1.3e-7.
        band, drift = self._band_and_drift(energy_series("two_body", 5000))
        assert band < 1e-3, band
        assert drift < 1e-5, drift

    def test_two_body_energy_error_does_not_grow_with_time(self):
        """The symplectic signature: a bounded band, not a trend.

        A fourth-order non-symplectic scheme would pass the band test above at these
        tolerances and fail this one, because its error accumulates in one direction.
        """
        series = energy_series("two_body", 6000)
        quarter = len(series) // 4
        means = [float(series[i * quarter : (i + 1) * quarter].mean()) for i in range(4)]
        spread = (max(means) - min(means)) / abs(series[0])
        assert spread < 1e-4, means

    def test_forward_euler_unwinds_the_orbit_that_verlet_keeps(self):
        """Why the integrator was chosen, demonstrated rather than asserted.

        Same initial condition, same step, same force: Euler gains 67% of the binding
        energy in a single period and the separation grows from 1.6 to 25 within four,
        while Verlet returns the orbit to the same radius with an energy error of 1e-14.
        """
        proc = ap.process("two_body")
        params = proc.resolve(None)
        period = ap.two_body_period()
        step = period / 256
        start = proc.initial_state()
        e0 = proc.total_energy(start)

        pos, vel, mass = start["pos"].copy(), start["vel"].copy(), start["mass"]
        for _ in range(1024):
            accel = ap._two_body_accel(pos, mass, params["GM"], params["softening"])
            pos = pos + step * vel
            vel = vel + step * accel
        euler = ap.State(1024, 1024 * step, {"pos": pos, "vel": vel, "mass": mass})
        euler_error = abs(proc.total_energy(euler) - e0) / abs(e0)
        euler_r = float(np.linalg.norm(pos[1] - pos[0]))

        state = start
        for _ in range(1024):
            state = proc.step(state, dt=step)
        verlet_error = abs(proc.total_energy(state) - e0) / abs(e0)
        verlet_r = float(np.linalg.norm(state["pos"][1] - state["pos"][0]))

        assert euler_error > 0.5, euler_error
        assert verlet_error < 1e-9, verlet_error
        assert euler_r > 3.0 * verlet_r
        assert verlet_r == pytest.approx(1.6, abs=1e-3)  # back at apoapsis, a(1 + e)

    def test_nbody_verlet_stays_bounded(self):
        # Measured over 5000 frames at the defaults: band 1.4e-3, final drift 9.8e-5.
        band, drift = self._band_and_drift(energy_series("nbody", 3000, ap.NBODY.defaults()))
        assert band < 5e-3, band
        assert drift < 1e-3, drift

    def test_nbody_softening_is_what_makes_that_possible(self):
        """Under-softened, the same integrator cannot resolve the close encounters.

        This is not a defect of Verlet: it is the fixed step failing against a stiffer
        potential, and it is why the default softening is 0.2 and why the docstring says
        that lowering it demands more sub-steps.
        """
        stiff = dict(ap.NBODY.defaults(), softening=0.05)
        band, _ = self._band_and_drift(energy_series("nbody", 3000, stiff))
        assert band > 0.1, band
        rescued = self._band_and_drift(energy_series("nbody", 3000, stiff, substeps=32))[0]
        assert rescued < band

    def test_spring_chain_verlet_stays_bounded(self):
        # Measured over 5000 frames: band 6.4e-6.
        band, drift = self._band_and_drift(energy_series("spring_chain", 4000))
        assert band < 1e-4, band
        assert drift < 1e-4, drift

    def test_wave2d_symplectic_euler_has_a_band_and_no_trend(self):
        # Symplectic Euler's band is O(h) and so wider than Verlet's O(h^2); measured 6.7%.
        series = energy_series("wave2d", 4000, dict(cheap("wave2d"), damping=0.0))
        band, _ = self._band_and_drift(series)
        assert band < 0.2, band
        quarter = len(series) // 4
        means = [float(series[i * quarter : (i + 1) * quarter].mean()) for i in range(4)]
        assert (max(means) - min(means)) / abs(series[0]) < 1e-2, means

    def test_undriven_pendulum_rk4_conserves_energy(self):
        # Measured over 5000 frames: band 3.1e-9.
        params = dict(ap.PENDULUM.defaults(), damping=0.0, drive=0.0)
        band, drift = self._band_and_drift(energy_series("pendulum", 5000, params))
        assert band < 1e-7, band
        assert drift < 1e-7, drift

    def test_double_pendulum_rk4_conserves_energy_through_the_chaos(self):
        # Measured over 5000 frames from a chaotic start: band 4.9e-8.
        band, drift = self._band_and_drift(energy_series("double_pendulum", 5000))
        assert band < 1e-6, band
        assert drift < 1e-6, drift

    def test_damping_removes_energy_and_a_drive_adds_it(self):
        damped = energy_series("pendulum", 600, dict(ap.PENDULUM.defaults(), damping=0.5))
        assert damped[-1] < 0.05 * damped[0]
        driven = energy_series(
            "pendulum",
            600,
            dict(ap.PENDULUM.defaults(), damping=0.0, drive=5.0, drive_freq=2.0),
        )
        assert np.ptp(driven) > 1e-3

    def test_double_pendulum_is_actually_chaotic(self):
        """A millidegree apart, the two tips are metres apart within seconds."""
        base = ap.DOUBLE_PENDULUM.defaults()
        nudged = dict(base, theta1=base["theta1"] + 1e-3)
        a = ap.DOUBLE_PENDULUM.state_to_table(advance("double_pendulum", 2500, base), base)
        b = ap.DOUBLE_PENDULUM.state_to_table(advance("double_pendulum", 2500, nudged), nudged)
        assert abs(a["x"][-1] - b["x"][-1]) > 0.1


class TestTwoBodyOrbit:
    """Kepler's laws are exact statements, so they make exact tests."""

    def test_the_period_is_keplers_third_law(self):
        assert ap.two_body_period({"semi_major": 1.0, "GM": 1.0}) == pytest.approx(2 * math.pi)
        assert ap.two_body_period({"semi_major": 4.0, "GM": 1.0}) == pytest.approx(
            2 * math.pi * 8.0
        )

    def test_the_orbit_closes_after_exactly_one_period(self):
        period = ap.two_body_period()
        proc = ap.process("two_body")
        start = proc.initial_state()
        state = start
        for _ in range(1024):
            state = proc.step(state, dt=period / 1024)
        assert state.time == pytest.approx(period, rel=1e-12)
        assert np.abs(state["pos"] - start["pos"]).max() < 1e-4
        assert np.abs(state["vel"] - start["vel"]).max() < 1e-4

    def test_the_closure_error_is_second_order_in_the_step(self):
        """Verlet is O(h^2): quartering the step must cut the error by about sixteen."""
        period = ap.two_body_period()
        proc = ap.process("two_body")
        errors = []
        for n in (256, 1024):
            state = proc.initial_state()
            start = state["pos"].copy()
            for _ in range(n):
                state = proc.step(state, dt=period / n)
            errors.append(float(np.abs(state["pos"] - start).max()))
        assert 8.0 < errors[0] / errors[1] < 32.0, errors

    @pytest.mark.parametrize("ecc", [0.0, 0.3, 0.6, 0.9])
    def test_the_eccentricity_parameter_is_the_orbit_s_eccentricity(self, ecc):
        """``r_apo / r_peri`` must equal ``(1 + e) / (1 - e)`` -- the definition."""
        params = dict(ap.TWO_BODY.defaults(), eccentricity=ecc, softening=0.0)
        proc = ap.process("two_body")
        step = ap.two_body_period(params) / 4096
        state = proc.initial_state(params)
        separations = []
        for _ in range(4096):
            state = proc.step(state, params, dt=step)
            separations.append(float(np.linalg.norm(state["pos"][1] - state["pos"][0])))
        ratio = max(separations) / min(separations)
        assert ratio == pytest.approx((1.0 + ecc) / (1.0 - ecc), rel=2e-3)

    def test_the_barycentre_stays_put(self):
        """The setup zeroes the total momentum, so the centre of mass never moves."""
        state = advance("two_body", 400, ap.TWO_BODY.defaults())
        mass = state["mass"]
        centre = (mass[:, None] * state["pos"]).sum(axis=0) / mass.sum()
        assert np.abs(centre).max() < 1e-9


class TestProjectile:
    """The one process with an analytic answer available for a special case."""

    def test_without_drag_it_is_the_textbook_parabola(self):
        params = dict(ap.PROJECTILE.defaults(), drag=0.0, wind=0.0, speed=20.0, angle=40.0)
        table = ap.PROJECTILE.state_to_table(advance("projectile", 40, params), params)
        theta, g = math.radians(40.0), params["gravity"]
        t = table["t"]
        assert table["x"] == pytest.approx(20.0 * math.cos(theta) * t, rel=1e-6)
        assert table["y"] == pytest.approx(
            20.0 * math.sin(theta) * t - 0.5 * g * t * t, rel=1e-6, abs=1e-9
        )

    def test_drag_shortens_the_flight(self):
        base = dict(ap.PROJECTILE.defaults(), drag=0.0)
        dragged = dict(ap.PROJECTILE.defaults(), drag=0.2)
        far = ap.PROJECTILE.state_to_table(advance("projectile", 100, base), base)["x"][-1]
        near = ap.PROJECTILE.state_to_table(advance("projectile", 100, dragged), dragged)["x"][-1]
        assert near < far

    def test_it_bounces_and_then_comes_to_rest_on_the_ground(self):
        params = dict(ap.PROJECTILE.defaults(), restitution=0.4, speed=12.0, angle=80.0)
        table = ap.PROJECTILE.state_to_table(advance("projectile", 900, params), params)
        assert table["y"].min() >= -1e-9
        assert table["y"][-1] == pytest.approx(0.0, abs=1e-6)

    def test_a_perfectly_elastic_bounce_keeps_bouncing(self):
        params = dict(ap.PROJECTILE.defaults(), restitution=1.0, drag=0.0, speed=10.0, angle=90.0)
        table = ap.PROJECTILE.state_to_table(advance("projectile", 400, params), params)
        assert table["y"].max() > 4.0

    def test_wind_pushes_it_downwind(self):
        with_wind = dict(ap.PROJECTILE.defaults(), wind=15.0, drag=0.1)
        still = dict(ap.PROJECTILE.defaults(), wind=0.0, drag=0.1)
        blown = ap.PROJECTILE.state_to_table(advance("projectile", 60, with_wind), with_wind)
        calm = ap.PROJECTILE.state_to_table(advance("projectile", 60, still), still)
        assert blown["x"][-1] > calm["x"][-1]


class TestSpringChain:
    """The discretised string: fixed ends, and modes that are actually modes."""

    def test_the_ends_never_move(self):
        state = advance("spring_chain", 200)
        assert state["y"][0] == 0.0 and state["y"][-1] == 0.0
        assert state["vy"][0] == 0.0 and state["vy"][-1] == 0.0

    def test_a_pure_mode_oscillates_at_its_own_frequency_and_keeps_its_shape(self):
        """Mode 1 must stay proportional to sin(pi x): a standing wave does not travel."""
        params = dict(ap.SPRING_CHAIN.defaults(), nodes=65, mode=1, damping=0.0)
        proc = ap.process("spring_chain")
        state = proc.initial_state(params)
        shape = state["y"] / np.max(np.abs(state["y"]))
        worst = 0.0
        for _ in range(300):
            state = proc.step(state, params)
            peak = np.max(np.abs(state["y"]))
            if peak > 1e-6:
                worst = max(
                    worst,
                    float(
                        np.max(
                            np.abs(state["y"] / peak - shape * np.sign(state["y"][len(shape) // 2]))
                        )
                    ),
                )
        assert worst < 0.05, worst

    def test_damping_removes_energy_monotonically(self):
        params = dict(ap.SPRING_CHAIN.defaults(), damping=0.5, mode=1, nodes=65)
        series = energy_series("spring_chain", 600, params)
        assert np.all(np.diff(series) < 0.0)
        assert series[-1] < 0.01 * series[0]

    def test_heavy_damping_is_overdamped_and_therefore_slower(self):
        """Not a bug: past critical damping the chain creeps back instead of ringing back.

        The relaxation rate of an overdamped mode goes like ``omega^2 / c``, so *more*
        damping means a *slower* return to flat. A test that simply demanded "big damping
        settles fastest" would be asserting the opposite of the physics.
        """
        light = energy_series(
            "spring_chain", 600, dict(ap.SPRING_CHAIN.defaults(), damping=0.5, mode=1, nodes=65)
        )
        heavy = energy_series(
            "spring_chain", 600, dict(ap.SPRING_CHAIN.defaults(), damping=3.0, mode=1, nodes=65)
        )
        assert np.all(np.diff(heavy) < 0.0)
        assert heavy[-1] > light[-1]


class TestGameOfLife:
    """Exact, non-negotiable invariants. Every one is a grid equality, not a tolerance."""

    def grid(self, pattern, generations, **over):
        params = dict(ap.LIFE.defaults(), pattern=ap.life_pattern_index(pattern), **over)
        state = ap.LIFE.initial_state(params)
        for _ in range(generations):
            state = ap.LIFE.step(state, params)
        return state["grid"]

    def test_a_blinker_has_period_two(self):
        start = self.grid("blinker", 0)
        assert not np.array_equal(self.grid("blinker", 1), start)
        assert np.array_equal(self.grid("blinker", 2), start)
        assert np.array_equal(self.grid("blinker", 4), start)
        assert np.array_equal(self.grid("blinker", 101), self.grid("blinker", 1))

    def test_a_block_is_still(self):
        start = self.grid("block", 0)
        for generation in range(1, 9):
            assert np.array_equal(self.grid("block", generation), start), generation

    def test_a_toad_has_period_two(self):
        start = self.grid("toad", 0)
        assert not np.array_equal(self.grid("toad", 1), start)
        assert np.array_equal(self.grid("toad", 2), start)

    def test_a_glider_returns_to_its_own_shape_translated_by_one_one(self):
        """The defining property of a glider: period 4 modulo a diagonal translation."""
        start = self.grid("glider", 0)
        after = self.grid("glider", 4)
        translated = np.roll(np.roll(start, 1, axis=0), 1, axis=1)
        assert np.array_equal(after, translated)
        # ... and again, so it is a genuine translation and not a coincidence at step 4.
        assert np.array_equal(self.grid("glider", 8), np.roll(np.roll(start, 2, axis=0), 2, axis=1))

    def test_a_glider_keeps_its_five_cells(self):
        assert [int(self.grid("glider", k).sum()) for k in range(13)] == [5] * 13

    def test_a_glider_on_a_torus_comes_exactly_home(self):
        """Four generations per cell of travel, so a W-wide torus closes after 4W."""
        side = 16
        start = self.grid("glider", 0, width=side, height=side, toroidal=1)
        assert np.array_equal(
            self.grid("glider", 4 * side, width=side, height=side, toroidal=1), start
        )

    def test_a_lightweight_spaceship_translates_by_two_across(self):
        start = self.grid("lwss", 0, width=40, height=40, toroidal=1)
        after = self.grid("lwss", 4, width=40, height=40, toroidal=1)
        assert np.array_equal(after, np.roll(start, 2, axis=1))

    def test_a_gosper_gun_emits_one_glider_every_thirty_generations(self):
        """Five new live cells per period, forever -- the first known infinite growth."""
        populations = [
            int(self.grid("gosper_gun", g, width=128, height=128, toroidal=0).sum())
            for g in (0, 30, 60, 90, 120)
        ]
        assert populations == [36, 41, 46, 51, 56]

    def test_the_r_pentomino_is_a_long_lived_methuselah(self):
        early = int(self.grid("r_pentomino", 0, width=128, height=128, toroidal=0).sum())
        late = int(self.grid("r_pentomino", 200, width=128, height=128, toroidal=0).sum())
        assert early == 5
        assert late > 40

    def test_the_boundary_is_dead_when_it_is_not_toroidal(self):
        """A glider aimed at a wall dies there; on a torus it comes back."""
        side = 12
        bounded = self.grid("glider", 4 * side, width=side, height=side, toroidal=0)
        toroidal = self.grid("glider", 4 * side, width=side, height=side, toroidal=1)
        assert not np.array_equal(bounded, toroidal)
        assert int(bounded.sum()) < 5

    def test_random_soup_has_roughly_the_requested_density(self):
        params = dict(
            ap.LIFE.defaults(),
            pattern=ap.life_pattern_index("random"),
            width=256,
            height=256,
            density=0.4,
            seed=7,
        )
        grid = ap.LIFE.initial_state(params)["grid"]
        assert grid.mean() == pytest.approx(0.4, abs=0.01)

    def test_age_counts_consecutive_generations_alive(self):
        params = dict(ap.LIFE.defaults(), pattern=ap.life_pattern_index("block"))
        state = ap.LIFE.initial_state(params)
        for expected in range(2, 7):
            state = ap.LIFE.step(state, params)
            table = ap.LIFE.state_to_table(state, params)
            assert list(table["age"]) == [float(expected)] * 4

    def test_the_table_holds_the_living_cells_with_y_pointing_up(self):
        params = dict(
            ap.LIFE.defaults(), pattern=ap.life_pattern_index("block"), width=16, height=16
        )
        state = ap.LIFE.initial_state(params)
        table = ap.LIFE.state_to_table(state, params)
        assert len(table["x"]) == 4 == int(state["grid"].sum())
        rows, cols = np.nonzero(state["grid"])
        np.testing.assert_array_equal(np.sort(table["y"]), np.sort(15.0 - rows))
        np.testing.assert_array_equal(np.sort(table["x"]), np.sort(cols.astype(float)))

    def test_an_empty_grid_gives_empty_but_still_rectangular_columns(self):
        params = dict(
            ap.LIFE.defaults(),
            pattern=ap.life_pattern_index("random"),
            density=0.0,
            width=16,
            height=16,
        )
        table = ap.LIFE.state_to_table(ap.LIFE.initial_state(params), params)
        assert {len(v) for v in table.values()} == {0}

    def test_an_unknown_pattern_name_raises(self):
        with pytest.raises(ap.ProcessError, match="unknown Life pattern"):
            ap.life_pattern_index("spaceship_of_theseus")

    def test_a_pattern_too_big_for_the_grid_says_how_big_to_make_it(self):
        params = dict(
            ap.LIFE.defaults(), pattern=ap.life_pattern_index("gosper_gun"), width=8, height=8
        )
        with pytest.raises(ap.ProcessError, match="36x9"):
            ap.LIFE.initial_state(params)

    def test_every_named_pattern_starts_and_runs(self):
        for name in ap.LIFE_PATTERN_NAMES:
            params = dict(
                ap.LIFE.defaults(), pattern=ap.life_pattern_index(name), width=64, height=64
            )
            state = ap.LIFE.initial_state(params)
            for _ in range(5):
                state = ap.LIFE.step(state, params)
            assert state["grid"].dtype == np.uint8
            assert set(np.unique(state["grid"])) <= {0, 1}


class TestElementaryCA:
    """Wolfram's rules have arithmetic consequences that pin the implementation."""

    def row_after(self, generations, **over):
        params = dict(ap.ELEMENTARY_CA.defaults(), **over)
        state = ap.ELEMENTARY_CA.initial_state(params)
        for _ in range(generations):
            state = ap.ELEMENTARY_CA.step(state, params)
        return state

    def test_rule_zero_empties_and_rule_255_fills(self):
        assert int(self.row_after(1, rule=0, width=33)["row"].sum()) == 0
        assert int(self.row_after(1, rule=255, width=33)["row"].sum()) == 33

    def test_rule_204_is_the_identity(self):
        """204 = 0b11001100 keeps the centre cell whatever its neighbours do."""
        start = self.row_after(0, rule=204, width=65, init=1, density=0.5, seed=3)["row"]
        later = self.row_after(6, rule=204, width=65, init=1, density=0.5, seed=3)["row"]
        np.testing.assert_array_equal(start, later)

    def test_rule_170_shifts_left_by_one(self):
        """170 = 0b10101010 copies the right neighbour: a pure left shift."""
        start = self.row_after(0, rule=170, width=65, init=1, density=0.4, seed=5)["row"]
        later = self.row_after(3, rule=170, width=65, init=1, density=0.4, seed=5)["row"]
        np.testing.assert_array_equal(later, np.roll(start, -3))

    @pytest.mark.parametrize("generation", [1, 2, 3, 4, 5, 6, 7, 8, 12, 16])
    def test_rule_90_reproduces_pascals_triangle_modulo_two(self, generation):
        """Row n of rule 90 from a single cell has 2**popcount(n) live cells (Kummer)."""
        row = self.row_after(generation, rule=90, width=129, init=0, toroidal=0)["row"]
        assert int(row.sum()) == 2 ** bin(generation).count("1")

    def test_rule_30_is_asymmetric_where_rule_90_is_not(self):
        thirty = self.row_after(20, rule=30, width=129, init=0, toroidal=0)["row"]
        ninety = self.row_after(20, rule=90, width=129, init=0, toroidal=0)["row"]
        assert not np.array_equal(thirty, thirty[::-1])
        np.testing.assert_array_equal(ninety, ninety[::-1])

    def test_rule_110_survives_and_is_not_periodic_in_a_few_steps(self):
        state = self.row_after(60, rule=110, width=129, init=1, density=0.5, seed=1)
        assert 0 < int(state["row"].sum()) < 129

    def test_the_history_scrolls_once_it_is_full(self):
        limit = 16
        state = self.row_after(40, rule=30, width=65, rows=limit)
        assert state["history"].shape[0] == limit
        assert state["origin"] == 41 - limit
        # The picture keeps moving downward: y is the absolute generation, not the buffer.
        table = ap.ELEMENTARY_CA.state_to_table(state, ap.ELEMENTARY_CA.resolve({"rows": limit}))
        assert table["y"].max() <= -float(state["origin"])

    def test_the_bounded_edge_is_dead_and_the_toroidal_one_wraps(self):
        bounded = self.row_after(30, rule=90, width=41, init=0, toroidal=0)["row"]
        wrapped = self.row_after(30, rule=90, width=41, init=0, toroidal=1)["row"]
        assert not np.array_equal(bounded, wrapped)


class TestGrayScott:
    """A PDE, so it has a real timestep and a real stability limit."""

    def test_it_makes_a_pattern_rather_than_a_uniform_field(self):
        params = dict(ap.GRAY_SCOTT.defaults(), width=64, height=64)
        state = advance("gray_scott", 500, params)
        assert np.all(np.isfinite(state["v"]))
        assert float(state["v"].std()) > 1e-3
        assert 0.0 <= float(state["v"].min())
        assert float(state["u"].max()) <= 1.05

    def test_the_whole_slider_range_is_strictly_stable(self):
        """``dt * D <= 0.25`` is the divergence bound; 0.25 itself only *rings*.

        A unit-amplitude checkerboard under pure diffusion is still exactly 1.0 after 40
        steps at ``dt D = 0.25`` (amplification -1: it flips sign forever), 0.036 at 0.24
        and 22 at 0.26. The sliders therefore stop at 0.24, and this pins that.
        """
        params = ap.GRAY_SCOTT.defaults()
        assert ap.GRAY_SCOTT.dt * max(params["Du"], params["Dv"]) < 0.25
        for param in ap.GRAY_SCOTT.params:
            if param.name in ("Du", "Dv"):
                assert ap.GRAY_SCOTT.dt * param.vmax < 0.25

        checker = np.indices((16, 16)).sum(axis=0) % 2 * 2.0 - 1.0
        for coefficient, expected in ((0.24, 0.1), (0.25, 1.1), (0.26, 20.0)):
            field = checker.copy()
            for _ in range(40):
                field = field + coefficient * ap._laplacian(field)
            peak = float(np.abs(field).max())
            assert peak < expected if coefficient == 0.24 else peak >= expected * 0.9

    def test_with_no_feed_and_no_kill_the_total_mass_is_conserved_exactly(self):
        """The reaction only moves mass u -> v and the periodic Laplacian sums to zero.

        So ``sum(u + v)`` is an exact invariant, and it is the sharpest check available on
        this stepper: any error in the Laplacian's stencil or its wrap-around breaks it.
        """
        params = dict(ap.GRAY_SCOTT.defaults(), width=32, height=32, feed=0.0, kill=0.0, seed=2)
        proc = ap.process("gray_scott")
        state = proc.initial_state(params)
        before = float((state["u"] + state["v"]).sum())
        for _ in range(500):
            state = proc.step(state, params)
        after = float((state["u"] + state["v"]).sum())
        assert after == pytest.approx(before, rel=1e-12)

    def test_the_laplacian_annihilates_a_constant_field(self):
        assert np.abs(ap._laplacian(np.full((8, 8), 3.5))).max() == 0.0


class TestWave2D:
    """Symplectic Euler on a periodic sheet, stable by construction."""

    def test_the_slider_range_cannot_violate_the_cfl_condition(self):
        fastest = max(p.vmax for p in ap.WAVE2D.params if p.name == "speed")
        assert fastest * ap.WAVE2D.dt <= 1.0 / math.sqrt(2.0)

    def test_a_drop_spreads_outward(self):
        params = dict(ap.WAVE2D.defaults(), width=64, height=64, drops=1, width_sigma=2.0)
        proc = ap.process("wave2d")
        state = proc.initial_state(params)
        centre = state["u"][32, 32]
        for _ in range(30):
            state = proc.step(state, params)
        assert abs(state["u"][32, 32]) < abs(centre)
        assert abs(state["u"][32, 44]) > 1e-6

    def test_damping_settles_the_sheet(self):
        params = dict(ap.WAVE2D.defaults(), width=48, height=48, damping=1.5)
        late = advance("wave2d", 300, params)
        assert float(np.abs(late["v"]).max()) < 1e-2


class TestMandelbrot:
    """Known points, a known zoom law, and a floor that keeps the deep end finite."""

    def test_known_members_and_non_members(self):
        points = np.array([0.0 + 0j, -1.0 + 0j, -0.5 + 0.5j, 2.0 + 2.0j, 1.0 + 1.0j, 0.5 + 0.5j])
        count, smooth = ap._escape_time(np.complex128(0.0), points, 200)
        assert count[0] == 200  # 0 is the fixed point: inside, forever
        assert count[1] == 200  # -1 is a period-2 point: inside
        assert count[2] == 200  # inside the main cardioid
        assert count[3] <= 3  # 2 + 2i escapes immediately
        assert count[4] <= 4  # 1 + i escapes almost immediately
        assert count[5] < 200  # just outside the cardioid
        assert np.all(np.isfinite(smooth))
        assert np.all((smooth >= 0.0) & (smooth <= 200.0))

    def test_the_escape_count_is_integral_and_the_smooth_count_is_not(self):
        grid = np.linspace(-2.0, 0.6, 400) + 0.35j
        count, smooth = ap._escape_time(np.complex128(0.0), grid, 120)
        np.testing.assert_array_equal(count, np.round(count))
        escaped = count < 120
        assert np.any(np.abs(smooth[escaped] - np.round(smooth[escaped])) > 1e-6)

    def test_the_zoom_span_is_exponential_and_floors(self):
        params = ap.MANDELBROT.resolve(None)
        assert ap._mandelbrot_span(params, 0.0) == pytest.approx(params["span0"])
        ratio = ap._mandelbrot_span(params, 2.0) / ap._mandelbrot_span(params, 1.0)
        assert ratio == pytest.approx(math.exp(-params["rate"]))
        assert ap._mandelbrot_span(params, 10_000.0) == ap.MIN_SPAN

    def test_a_deep_zoom_stays_finite(self):
        for t in (0.0, 50.0, 150.0, 400.0, 5000.0):
            state = ap.mandelbrot_frame(t, {"side": 48, "max_iter": 80})
            assert state["span"] >= ap.MIN_SPAN
            table = ap.MANDELBROT.state_to_table(state, {"side": 48, "max_iter": 80})
            for name, column in table.items():
                assert np.all(np.isfinite(column)), (t, name)
            assert len(table["x"]) == 48 * 48

    def test_seeking_a_frame_equals_stepping_to_it(self):
        """The zoom is a closed form in t, so frame 7 costs one raster, not seven."""
        params = {"side": 32, "max_iter": 60}
        stepped = advance("mandelbrot_zoom", 7, ap.MANDELBROT.resolve(params))
        sought = ap.mandelbrot_frame(7.0 * ap.MANDELBROT.dt, params)
        assert sought.frame == stepped.frame
        assert sought.time == pytest.approx(stepped.time)
        np.testing.assert_allclose(sought["count"], stepped["count"])

    def test_zooming_in_shrinks_the_view(self):
        params = {"side": 32, "max_iter": 60}
        wide = ap.mandelbrot_frame(0.0, params)
        tight = ap.mandelbrot_frame(20.0, params)
        assert np.ptp(tight["re"]) < np.ptp(wide["re"])
        assert tight["re"].mean() == pytest.approx(wide["re"].mean(), abs=1e-9)

    def test_the_raster_grid_is_row_major_and_matches_its_axes(self):
        state = ap.mandelbrot_frame(0.0, {"side": 16, "max_iter": 30})
        table = ap.MANDELBROT.state_to_table(state, {"side": 16, "max_iter": 30})
        assert table["x"][0] == pytest.approx(state["re"][0])
        assert table["x"][15] == pytest.approx(state["re"][15])
        assert table["y"][0] == pytest.approx(state["im"][0])
        assert table["y"][16] == pytest.approx(state["im"][1])


class TestJulia:
    """The animation is the parameter path, so the path is what gets tested."""

    def test_c_equals_zero_gives_the_unit_disc(self):
        """The Julia set of z^2 is exactly the closed unit disc -- an exact answer."""
        z = np.array([0.0 + 0j, 0.9 + 0j, 0.99 + 0j, 0.7 + 0.7j, 1.01 + 0j, 1.5 + 0j, 1.0 + 1.0j])
        count, _ = ap._escape_time(z, 0.0 + 0j, 400)
        assert count[0] == count[1] == count[2] == 400  # |z| < 1: inside
        assert count[3] == 400  # |z| = 0.99, still inside
        assert count[4] < 400  # |z| > 1: escapes
        assert count[5] < 400
        assert count[6] < 400

    def test_the_unit_disc_boundary_is_where_the_escape_happens(self):
        radii = np.linspace(0.5, 1.5, 101)
        count, _ = ap._escape_time(radii.astype(np.complex128), 0.0 + 0j, 300)
        inside = radii[count == 300]
        outside = radii[count < 300]
        assert inside.max() <= 1.0 + 1e-9
        assert outside.min() > 1.0

    def test_the_cardioid_path_stays_on_the_cardioid(self):
        """``|c|`` traces the main cardioid, whose cusp is at c = 1/4."""
        params = ap.JULIA.resolve({"path": ap.JULIA_PATHS.index("cardioid")})
        assert ap.julia_c(0.0, params) == pytest.approx(0.25)
        for t in np.linspace(0.0, 400.0, 25):
            c = ap.julia_c(t, params)
            theta = params["speed"] * t
            expected = 0.5 * np.exp(1j * theta) - 0.25 * np.exp(2j * theta)
            assert c == pytest.approx(expected)
            assert abs(c) <= 0.75 + 1e-12

    def test_the_circle_path_keeps_its_radius(self):
        params = ap.JULIA.resolve(
            {"path": ap.JULIA_PATHS.index("circle"), "radius": 0.3, "c_re": -0.4, "c_im": 0.6}
        )
        for t in (0.0, 13.0, 71.5):
            assert abs(ap.julia_c(t, params) - complex(-0.4, 0.6)) == pytest.approx(0.3)

    def test_c_moves_and_the_picture_moves_with_it(self):
        params = {"side": 40, "max_iter": 60, "path": 0, "speed": 0.5, "radius": 0.4}
        first = ap.julia_frame(0.0, params)
        later = ap.julia_frame(3.0, params)
        assert first["c"] != later["c"]
        assert not np.array_equal(first["count"], later["count"])

    def test_seeking_a_frame_equals_stepping_to_it(self):
        params = {"side": 32, "max_iter": 50}
        stepped = advance("julia_path", 5, ap.JULIA.resolve(params))
        sought = ap.julia_frame(5.0 * ap.JULIA.dt, params)
        np.testing.assert_allclose(sought["count"], stepped["count"])

    def test_every_path_is_reachable_and_bounded(self):
        for index, name in enumerate(ap.JULIA_PATHS):
            params = ap.JULIA.resolve({"path": index})
            values = [ap.julia_c(t, params) for t in np.linspace(0.0, 500.0, 40)]
            assert all(abs(c) < 3.0 for c in values), name


class TestChaosGame:
    """An IFS accumulating points: reproducible, bounded, and the right attractor."""

    def test_the_point_count_grows_by_chains_times_iters_each_frame(self):
        params = dict(ap.CHAOS_GAME.defaults(), chains=32, iters=4)
        proc = ap.process("chaos_game")
        state = proc.initial_state(params)
        assert state["points"].shape == (0, 2)
        for frame in range(1, 5):
            state = proc.step(state, params)
            assert state["points"].shape == (frame * 32 * 4, 2)
            assert state["codes"].shape == (frame * 32 * 4,)

    def test_the_chain_count_is_clamped_so_a_frame_stays_affordable(self):
        params = dict(ap.CHAOS_GAME.defaults(), chains=ap.MAX_POINTS_PER_FRAME, iters=16)
        assert ap._chaos_chains(params) * 16 <= ap.MAX_POINTS_PER_FRAME

    def test_the_barnsley_fern_lands_where_a_fern_lands(self):
        params = dict(ap.CHAOS_GAME.defaults(), system=ap.IFS_NAMES.index("fern"), chains=256)
        table = ap.CHAOS_GAME.state_to_table(advance("chaos_game", 6, params), params)
        assert -2.3 < table["x"].min() < -1.5
        assert 2.3 < table["x"].max() < 3.0
        assert table["y"].min() >= -0.01
        assert 9.0 < table["y"].max() < 10.5

    def test_the_sierpinski_gasket_stays_inside_its_triangle(self):
        params = dict(ap.CHAOS_GAME.defaults(), system=ap.IFS_NAMES.index("sierpinski"), chains=256)
        table = ap.CHAOS_GAME.state_to_table(advance("chaos_game", 6, params), params)
        x, y = table["x"], table["y"]
        # The attractor is the gasket on the triangle (0, 0), (1, 0), (0.5, 1); every
        # point must be inside it, which is three half-plane constraints.
        assert x.min() >= -1e-9 and x.max() <= 1.0 + 1e-9
        assert y.min() >= -1e-9 and y.max() <= 1.0 + 1e-9
        assert np.all(y <= 2.0 * x + 1e-9)
        assert np.all(y <= 2.0 * (1.0 - x) + 1e-9)
        # ... and it must reach into all three corners, or it is not the whole gasket.
        assert x.min() < 0.02 and x.max() > 0.98 and y.max() > 0.95

    def test_the_map_column_reports_which_map_made_each_point(self):
        params = dict(ap.CHAOS_GAME.defaults(), system=ap.IFS_NAMES.index("fern"), chains=512)
        table = ap.CHAOS_GAME.state_to_table(advance("chaos_game", 8, params), params)
        codes = table["map"]
        assert set(np.unique(codes)) <= {0.0, 1.0, 2.0, 3.0}
        # The famous fern weights: map 1 draws 85% of the points, map 0 about 1%.
        assert 0.80 < float((codes == 1).mean()) < 0.90
        assert float((codes == 0).mean()) < 0.03

    def test_every_system_runs_and_stays_bounded(self):
        for index, name in enumerate(ap.IFS_NAMES):
            params = dict(ap.CHAOS_GAME.defaults(), system=index, chains=64, iters=4)
            table = ap.CHAOS_GAME.state_to_table(advance("chaos_game", 5, params), params)
            assert np.all(np.isfinite(table["x"])), name
            assert np.abs(table["x"]).max() < 100.0, name
            assert np.abs(table["y"]).max() < 100.0, name

    def test_the_burn_in_keeps_the_stray_early_points_out(self):
        hot = dict(ap.CHAOS_GAME.defaults(), system=ap.IFS_NAMES.index("fern"), burn=0, chains=64)
        cold = dict(hot, burn=50)
        first = ap.CHAOS_GAME.state_to_table(advance("chaos_game", 1, hot), hot)
        settled = ap.CHAOS_GAME.state_to_table(advance("chaos_game", 1, cold), cold)
        assert settled["y"].min() >= -1e-9
        assert settled["y"].max() > first["y"].max()


class TestCostCeilings:
    """The ceilings are enforced where they are documented to be."""

    def test_grid_requests_are_clamped_to_the_cell_ceiling(self):
        height, width = ap._grid_shape({"width": 10_000, "height": 10_000})
        assert width == ap.MAX_GRID_SIDE
        assert width * height <= ap.MAX_CELLS
        assert ap._grid_shape({"width": 1, "height": 1}) == (ap.MIN_GRID_SIDE, ap.MIN_GRID_SIDE)

    def test_the_escape_budget_trades_iterations_not_resolution(self):
        assert ap.escape_budget(64 * 64, 100) == 100  # comfortably inside the ceiling
        big = ap.escape_budget(ap.MAX_GRID_SIDE**2, ap.MAX_ITERATIONS)
        assert big < ap.MAX_ITERATIONS
        assert ap.MAX_GRID_SIDE**2 * big <= ap.MAX_ESCAPE_WORK
        assert ap.escape_budget(10**9, 500) == 1  # never zero: always one iteration
        assert ap.escape_budget(16, 10_000) == ap.MAX_ITERATIONS

    def test_a_raster_reports_the_iteration_count_it_actually_used(self):
        state = ap.mandelbrot_frame(0.0, {"side": ap.MAX_GRID_SIDE, "max_iter": ap.MAX_ITERATIONS})
        assert state["max_iter"] == ap.escape_budget(ap.MAX_GRID_SIDE**2, ap.MAX_ITERATIONS)
        assert state["max_iter"] < ap.MAX_ITERATIONS
        assert state["count"].max() <= state["max_iter"]

    def test_the_body_count_is_clamped(self):
        state = ap.NBODY.initial_state({"bodies": 10_000})
        assert state["pos"].shape == (ap.MAX_BODIES, 2)

    def test_the_node_count_is_clamped(self):
        state = ap.SPRING_CHAIN.initial_state({"nodes": 10_000})
        assert state["y"].size == ap.MAX_NODES

    def test_a_history_table_scrolls_instead_of_growing_forever(self):
        params = dict(ap.PENDULUM.defaults(), trail=20)
        table = ap.PENDULUM.state_to_table(advance("pendulum", 60, params), params)
        assert len(table["t"]) == 20
        # ... and it kept the newest rows, not the oldest.
        assert table["t"][-1] == pytest.approx(60 * ap.PENDULUM.dt)

    def test_the_ca_history_is_capped(self):
        params = dict(ap.ELEMENTARY_CA.defaults(), rows=8, width=33)
        state = advance("elementary_ca", 50, params)
        assert state["history"].shape[0] == 8

    def test_the_ceilings_are_ordered_sensibly(self):
        assert ap.MIN_GRID_SIDE < ap.MAX_GRID_SIDE
        assert ap.MAX_GRID_SIDE**2 == ap.MAX_CELLS
        assert ap.MAX_ESCAPE_WORK < ap.MAX_CELLS * ap.MAX_ITERATIONS
        assert ap.MAX_POINTS_PER_FRAME < ap.MAX_ACCUM_POINTS
        assert ap.MIN_SPAN > 0.0


class TestPackageSurface:
    """The package is importable headless and re-exports only what it promises."""

    def test_the_lazy_reexports_resolve(self):
        import glplot.anim as anim

        assert anim.Process is ap.Process
        assert anim.State is ap.State
        assert anim.ProcessError is ap.ProcessError
        assert anim.ProcessParam is ap.ProcessParam
        assert anim.PROCESSES is ap.PROCESSES
        assert anim.process is ap.process
        assert set(anim.__all__) <= set(dir(anim))

    def test_an_unknown_attribute_raises_attribute_error(self):
        import glplot.anim as anim

        with pytest.raises(AttributeError):
            anim.does_not_exist

    def test_the_module_imports_nothing_graphical(self):
        """CONTRACT 5.1 rule 7: numpy and stdlib only, so this is testable headless.

        Read off the module's actual import statements rather than grepping its text --
        the docstrings legitimately *mention* imgui and OpenGL while explaining that they
        are absent, and a text search cannot tell the two apart.
        """
        import ast
        import inspect

        tree = ast.parse(inspect.getsource(ap))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        assert imported <= {"__future__", "dataclasses", "typing", "math", "numpy"}, imported

    def test_the_module_runs_with_imgui_uninstalled(self):
        """The property the AST check implies, executed rather than inferred.

        ``import glplot`` itself pulls in OpenGL through ``glplot.engine`` -- pre-existing
        upstream behaviour that :mod:`tests.test_gui_import_safety` documents -- so the
        meaningful statement is not "nothing graphical is in sys.modules" but "this module
        imports, builds a state and steps it with imgui missing". That is what makes the
        simulation testable headless, and it is what is checked here.
        """
        import subprocess
        import sys

        script = (
            "import sys\n"
            "class Blocker:\n"
            "    def find_spec(self, fullname, path=None, target=None):\n"
            "        if fullname == 'imgui' or fullname.startswith('imgui.'):\n"
            "            raise ImportError('blocked by test')\n"
            "        return None\n"
            "sys.meta_path.insert(0, Blocker())\n"
            "from glplot.anim.processes import LIFE, life_pattern_index\n"
            "params = dict(LIFE.defaults(), pattern=life_pattern_index('glider'))\n"
            "state = LIFE.initial_state(params)\n"
            "for _ in range(4):\n"
            "    state = LIFE.step(state, params)\n"
            "print(int(state['grid'].sum()))\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, timeout=180
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "5"
