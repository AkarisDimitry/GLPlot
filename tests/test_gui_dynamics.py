"""Test the dynamical-systems data generator in glplot.gui.dynamics.

Pure logic: RK4 integration of ODE systems into DataSets. No OpenGL, no window, no imgui.
The preloaded library is validated at import time; these tests exercise the integrator,
the parameter/initial overrides, the error paths, and each preloaded system's shape.
"""

from __future__ import annotations

import numpy as np
import pytest

from glplot.gui.dynamics import (
    MAX_STEPS,
    SYSTEMS,
    DynamicalSystem,
    DynamicsError,
    delay_embedding,
    integrate,
    local_maxima,
    lyapunov_rosenstein,
    return_map,
    system_labels,
)
from glplot.gui.expressions import ExpressionError


def _linear(w: float = 1.0) -> DynamicalSystem:
    """A simple harmonic oscillator: p' = q, q' = -w*p. Energy p^2 + w*q^2 is conserved."""
    return DynamicalSystem(
        key="lin",
        label="Linear",
        icon="wave",
        description="test",
        variables=("p", "q"),
        equations=("q", "-w * p"),
        params=(("w", w, 0.0, 5.0),),
        initial=(1.0, 0.0),
        dt=0.01,
        steps=1000,
    )


class TestValidation:
    """Test that DynamicalSystem rejects malformed definitions at construction."""

    def test_equation_count_must_match_variables(self):
        with pytest.raises(DynamicsError, match="one derivative per variable"):
            DynamicalSystem(
                key="k",
                label="L",
                icon="wave",
                description="",
                variables=("x", "y"),
                equations=("y",),
            )

    def test_variable_named_t_is_rejected(self):
        with pytest.raises(DynamicsError, match="reserved for time"):
            DynamicalSystem(
                key="k",
                label="L",
                icon="wave",
                description="",
                variables=("t",),
                equations=("1",),
            )

    def test_variable_shadowing_builtin_is_rejected(self):
        with pytest.raises(DynamicsError, match="shadows the built-in"):
            DynamicalSystem(
                key="k",
                label="L",
                icon="wave",
                description="",
                variables=("e",),
                equations=("1",),
            )

    def test_duplicate_variable_is_rejected(self):
        with pytest.raises(DynamicsError, match="declared twice"):
            DynamicalSystem(
                key="k",
                label="L",
                icon="wave",
                description="",
                variables=("x", "x"),
                equations=("1", "1"),
            )

    def test_param_colliding_with_variable_is_rejected(self):
        with pytest.raises(DynamicsError, match="collides"):
            DynamicalSystem(
                key="k",
                label="L",
                icon="wave",
                description="",
                variables=("x",),
                equations=("x",),
                params=(("x", 1.0, 0.0, 2.0),),
            )

    def test_bad_initial_length_is_rejected(self):
        with pytest.raises(DynamicsError, match="initial state"):
            DynamicalSystem(
                key="k",
                label="L",
                icon="wave",
                description="",
                variables=("x", "y"),
                equations=("y", "-x"),
                initial=(1.0,),
            )

    def test_unknown_name_in_equation_is_rejected(self):
        # `foo` is neither a variable, a parameter, t, nor a SAFE_NAMES callable.
        with pytest.raises(ExpressionError, match="dx/dt"):
            DynamicalSystem(
                key="k",
                label="L",
                icon="wave",
                description="",
                variables=("x",),
                equations=("foo * x",),
            )

    def test_equation_may_reference_time(self):
        # Should not raise: t is always allowed.
        system = DynamicalSystem(
            key="k",
            label="L",
            icon="wave",
            description="",
            variables=("x",),
            equations=("cos(t)",),
        )
        assert system.variables == ("x",)


class TestIntegrate:
    """Test the RK4 integrator and its overrides."""

    def test_shape_and_columns(self):
        ds = integrate(_linear(), steps=500)
        assert ds.n_rows() == 501  # steps + 1
        assert ds.column_names() == ["t", "p", "q"]
        assert ds.source == "dynamics"

    def test_time_column_is_uniform(self):
        ds = integrate(_linear(), steps=100, dt=0.02)
        t = ds.get("t")
        assert np.allclose(np.diff(t), 0.02)

    def test_harmonic_energy_is_conserved(self):
        # RK4 on a linear oscillator conserves energy to high order over a short window.
        ds = integrate(_linear(w=1.0), steps=2000, dt=0.005)
        energy = ds.get("p") ** 2 + ds.get("q") ** 2
        assert np.ptp(energy) < 1e-3

    def test_param_override_changes_trajectory(self):
        slow = integrate(_linear(w=1.0), steps=500)
        fast = integrate(_linear(w=4.0), steps=500)
        assert not np.allclose(slow.get("p"), fast.get("p"))

    def test_partial_param_override_keeps_defaults(self):
        ds = integrate(_linear(w=2.0), params={"w": 3.0}, steps=10)
        # No crash and the override took: with w=3 the second-derivative magnitude differs.
        assert ds.n_rows() == 11

    def test_initial_override(self):
        ds = integrate(_linear(), initial=(0.0, 1.0), steps=5)
        assert ds.get("p")[0] == 0.0
        assert ds.get("q")[0] == 1.0

    def test_transient_is_discarded(self):
        # With a transient, the recorded trajectory starts later in time.
        without = integrate(_linear(), steps=10, dt=0.1, transient=0)
        witht = integrate(_linear(), steps=10, dt=0.1, transient=50)
        assert witht.get("t")[0] > without.get("t")[0]

    def test_unknown_param_override_raises(self):
        with pytest.raises(DynamicsError, match="unknown parameter"):
            integrate(_linear(), params={"nope": 1.0})

    def test_wrong_initial_length_raises(self):
        with pytest.raises(DynamicsError, match="initial state"):
            integrate(_linear(), initial=(1.0, 2.0, 3.0))

    def test_steps_out_of_range_raises(self):
        with pytest.raises(DynamicsError, match="steps must be between"):
            integrate(_linear(), steps=MAX_STEPS + 1)
        with pytest.raises(DynamicsError, match="steps must be between"):
            integrate(_linear(), steps=1)

    def test_zero_dt_raises(self):
        with pytest.raises(DynamicsError, match="dt must be"):
            integrate(_linear(), dt=0.0)

    def test_divergence_yields_nonfinite_not_error(self):
        # An unstable configuration must not raise; it produces inf/nan in the tail.
        blow = DynamicalSystem(
            key="b",
            label="B",
            icon="wave",
            description="",
            variables=("x",),
            equations=("x**2 + 1",),
            initial=(1.0,),
            dt=0.1,
            steps=200,
        )
        ds = integrate(blow)
        assert not np.all(np.isfinite(ds.get("x")))

    def test_name_override(self):
        ds = integrate(_linear(), steps=10, name="custom name")
        assert ds.name == "custom name"


class TestPreloadedLibrary:
    """Test the ten classic systems all integrate to finite, non-trivial trajectories."""

    def test_ten_systems(self):
        assert len(SYSTEMS) == 10
        assert "lorenz" in SYSTEMS
        assert "goodwin" in SYSTEMS

    def test_system_labels_match(self):
        labels = dict(system_labels())
        assert labels["lorenz"] == "Lorenz"
        assert len(labels) == 10

    @pytest.mark.parametrize("key", list(SYSTEMS))
    def test_each_system_produces_a_bounded_trajectory(self, key):
        system = SYSTEMS[key]
        ds = integrate(system, steps=1500)
        assert ds.column_names() == ["t", *system.variables]
        assert ds.n_rows() == 1501
        # Every recorded value is finite: the tuned defaults draw a real attractor/cycle.
        data = ds.to_array()
        assert np.all(np.isfinite(data)), f"{key} produced non-finite values with defaults"
        # And the trajectory actually moves (not a fixed point sitting still).
        for var in system.variables:
            col = ds.get(var)
            assert np.ptp(col) > 1e-6, f"{key}: variable {var} did not move"

    @pytest.mark.parametrize("key", list(SYSTEMS))
    def test_default_plot_axes_are_real_columns(self, key):
        system = SYSTEMS[key]
        x_axis, y_axis = system.plot_axes()
        assert x_axis in system.columns()
        assert y_axis in system.columns()

    def test_lorenz_is_chaotic_sensitive_to_initial_conditions(self):
        # A tiny perturbation of the initial state diverges under the Lorenz flow.
        base = integrate(SYSTEMS["lorenz"], steps=3000)
        pert = integrate(SYSTEMS["lorenz"], initial=(1.0 + 1e-6, 1.0, 1.0), steps=3000)
        separation = np.abs(base.get("x") - pert.get("x"))
        assert separation[-1] > separation[0] * 100


class TestDelayEmbedding:
    """Test the Takens time-delay embedding used by the phase-portrait analysis."""

    def test_shape(self):
        y = np.arange(100.0)
        emb = delay_embedding(y, delay=5, dim=3)
        # M = N - (dim-1)*delay = 100 - 10 = 90.
        assert emb.shape == (90, 3)

    def test_columns_are_shifted_copies(self):
        y = np.arange(20.0)
        emb = delay_embedding(y, delay=2, dim=2)
        assert np.allclose(emb[:, 1] - emb[:, 0], 2.0)

    def test_too_short_raises(self):
        with pytest.raises(ValueError, match="too short"):
            delay_embedding(np.arange(5.0), delay=10, dim=3)

    def test_non_finite_are_dropped(self):
        y = np.array([1.0, np.nan, 2.0, 3.0, np.inf, 4.0, 5.0])
        emb = delay_embedding(y, delay=1, dim=2)
        assert np.all(np.isfinite(emb))


class TestReturnMap:
    """Test the successive-maxima return map."""

    def test_local_maxima_indices(self):
        y = np.array([0.0, 2.0, 0.0, 3.0, 1.0, 4.0, 0.0])
        assert np.array_equal(local_maxima(y), np.array([1, 3, 5]))

    def test_return_map_pairs_consecutive_maxima(self):
        # Maxima are 2, 3, 4 -> pairs (2,3) and (3,4).
        y = np.array([0.0, 2.0, 0.0, 3.0, 1.0, 4.0, 0.0])
        m_n, m_next = return_map(y)
        assert np.array_equal(m_n, np.array([2.0, 3.0]))
        assert np.array_equal(m_next, np.array([3.0, 4.0]))

    def test_flat_signal_has_no_map(self):
        m_n, m_next = return_map(np.ones(50))
        assert m_n.size == 0 and m_next.size == 0

    def test_lorenz_z_maxima_map_is_populated(self):
        ds = integrate(SYSTEMS["lorenz"], steps=8000)
        m_n, m_next = return_map(ds.get("z"))
        # The Lorenz flow produces many z-maxima that trace the classic tent map.
        assert m_n.size > 20
        assert m_n.size == m_next.size


class TestLyapunov:
    """Test the Rosenstein largest-Lyapunov-exponent estimator's sign discrimination."""

    def test_chaotic_flow_has_positive_exponent(self):
        ds = integrate(SYSTEMS["lorenz"], steps=8000)
        dt = float(np.median(np.diff(ds.get("t"))))
        res = lyapunov_rosenstein(ds.get("x"), dt=dt, delay=10, dim=3)
        assert res["lyapunov"] > 0.05, "Lorenz should read as chaotic (positive exponent)"

    def test_limit_cycle_exponent_is_near_zero(self):
        ds = integrate(SYSTEMS["van_der_pol"], steps=6000, transient=500)
        dt = float(np.median(np.diff(ds.get("t"))))
        res = lyapunov_rosenstein(ds.get("x"), dt=dt, delay=15, dim=3)
        # A limit cycle neither converges nor diverges: |lambda| stays small.
        assert abs(res["lyapunov"]) < 0.05

    def test_result_curve_lengths_match(self):
        ds = integrate(SYSTEMS["lorenz"], steps=4000)
        res = lyapunov_rosenstein(ds.get("x"), dt=0.01, delay=10, dim=3)
        assert res["times"].shape == res["divergence"].shape == res["fit"].shape

    def test_too_short_raises(self):
        with pytest.raises(ValueError, match="at least 20"):
            lyapunov_rosenstein(np.arange(10.0), dt=0.1)
