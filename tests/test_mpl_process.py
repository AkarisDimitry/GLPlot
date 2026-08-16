"""Test the out-of-process matplotlib viewer in glplot.utils.mpl_process.

Covers snapshot payload serialization, the child entry point, viewer-process
bookkeeping, and the routing of GPULinePlot.transfer_to_matplotlib_default
(the 'M' key) without requiring OpenGL, GPU, or a window.

Regression: the default 'M' action used to draw the figure in-process, where
glfw.poll_events() destroys it on macOS. It must now hand the snapshot to a
separate process, while registered programmatic targets stay in-process.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

import glplot.pyplot as gplt
from glplot.engine import GPULinePlot
from glplot.utils import mpl_process
from glplot.utils.mpl_bridge import GLPlotSnapshot


@pytest.fixture(autouse=True)
def clean_state():
    """Reset pyplot global state around every test."""
    gplt._cleanup_pyplot_state()
    yield
    gplt._cleanup_pyplot_state()


def make_snapshot() -> GLPlotSnapshot:
    """Build a small, recognisable snapshot: red left half, green right half."""
    rgba = np.zeros((8, 16, 4), dtype=np.uint8)
    rgba[:, :8] = (255, 0, 0, 255)
    rgba[:, 8:] = (0, 255, 0, 255)
    return GLPlotSnapshot(
        rgba=rgba,
        extent=(-1.0, 3.0, 0.0, 2.0),
        xlim=(-1.0, 3.0),
        ylim=(0.0, 2.0),
        width_px=16,
        height_px=8,
        transparent=False,
    )


class TestPayload:
    """Test snapshot serialization to and from the transfer file."""

    def test_round_trip_preserves_snapshot(self, tmp_path):
        """Every snapshot field must survive the write/load round trip."""
        snap = make_snapshot()
        path = str(tmp_path / "p.npz")
        mpl_process.write_payload(snap, path)
        back, _labels = mpl_process.load_payload(path)

        assert np.array_equal(back.rgba, snap.rgba)
        assert back.extent == snap.extent
        assert back.xlim == snap.xlim
        assert back.ylim == snap.ylim
        assert back.width_px == snap.width_px
        assert back.height_px == snap.height_px
        assert back.transparent is snap.transparent

    def test_round_trip_preserves_labels(self, tmp_path):
        """Axis labels and title must survive the round trip."""
        path = str(tmp_path / "p.npz")
        mpl_process.write_payload(make_snapshot(), path, xlabel="Time (s)", ylabel="V", title="T")
        _snap, labels = mpl_process.load_payload(path)
        assert labels == {"xlabel": "Time (s)", "ylabel": "V", "title": "T"}

    def test_absent_labels_load_as_none(self, tmp_path):
        """Unset labels must come back as None, not empty strings."""
        path = str(tmp_path / "p.npz")
        mpl_process.write_payload(make_snapshot(), path)
        _snap, labels = mpl_process.load_payload(path)
        assert labels == {"xlabel": None, "ylabel": None, "title": None}

    def test_payload_loads_without_pickle(self, tmp_path):
        """The payload must contain only arrays, so untrusted pickle is never needed."""
        path = str(tmp_path / "p.npz")
        mpl_process.write_payload(make_snapshot(), path, title="T")
        with np.load(path, allow_pickle=False) as data:
            assert "rgba" in data.files


class TestChildEntryPoint:
    """Test the child-process main(), driven in-process against the Agg backend."""

    def test_main_renders_snapshot_and_labels(self, tmp_path, monkeypatch):
        """main() must render the transferred array with its extent and labels."""
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        monkeypatch.setenv("MPLBACKEND", "Agg")
        snap = make_snapshot()
        path = str(tmp_path / "p.npz")
        mpl_process.write_payload(snap, path, xlabel="X", ylabel="Y", title="T")

        plt.close("all")
        assert mpl_process.main([path]) == 0

        ax = plt.gcf().axes[0]
        image = ax.images[0]
        assert np.array_equal(np.asarray(image.get_array()), snap.rgba)
        assert tuple(image.get_extent()) == snap.extent
        assert ax.get_xlabel() == "X"
        assert ax.get_ylabel() == "Y"
        assert ax.get_title() == "T"
        plt.close("all")

    def test_main_unlinks_payload(self, tmp_path, monkeypatch):
        """The child owns the temp payload and must delete it."""
        pytest.importorskip("matplotlib")
        monkeypatch.setenv("MPLBACKEND", "Agg")
        path = str(tmp_path / "p.npz")
        mpl_process.write_payload(make_snapshot(), path)
        mpl_process.main([path])
        assert not os.path.exists(path)

    def test_main_rejects_bad_arguments(self):
        """Wrong argument count must exit non-zero rather than raise."""
        assert mpl_process.main([]) == 2
        assert mpl_process.main(["a", "b"]) == 2

    def test_main_survives_unreadable_payload(self, tmp_path):
        """A corrupt payload must be reported, not raise, and must be cleaned up."""
        path = tmp_path / "bad.npz"
        path.write_bytes(b"not an npz")
        assert mpl_process.main([str(path)]) == 1
        assert not path.exists()


class TestLaunch:
    """Test viewer-process spawning and bookkeeping."""

    def test_launch_is_non_blocking_and_reaps(self, monkeypatch):
        """Launch must return a live handle immediately and reap the child later."""
        pytest.importorskip("matplotlib")
        # Agg makes plt.show() a no-op, so the child renders and exits at once.
        monkeypatch.setenv("MPLBACKEND", "Agg")
        proc = mpl_process.launch_snapshot_viewer(make_snapshot(), title="T")
        assert proc is not None
        assert proc.wait(timeout=120) == 0
        assert mpl_process.active_viewers() == 0

    def test_launch_spawns_a_fresh_detached_interpreter(self, monkeypatch):
        """Forking a process that holds a GL context is unsafe on macOS.

        The viewer must be a brand-new interpreter running this module's entry
        point, detached from the parent's session so it outlives the parent.
        """
        import subprocess
        import sys

        monkeypatch.setattr(mpl_process, "matplotlib_available", lambda: True)
        recorded = {}

        class FakeProc:
            pid = 1234

            def poll(self):
                return 0

        def fake_popen(cmd, **kwargs):
            recorded["cmd"] = cmd
            recorded["kwargs"] = kwargs
            return FakeProc()

        monkeypatch.setattr(subprocess, "Popen", fake_popen)
        assert mpl_process.launch_snapshot_viewer(make_snapshot()) is not None

        cmd = recorded["cmd"]
        assert cmd[0] == sys.executable  # a fresh interpreter, i.e. exec not fork
        assert cmd[1:3] == ["-m", "glplot.utils.mpl_process"]
        assert cmd[3].endswith(".npz")
        if os.name != "nt":
            assert recorded["kwargs"]["start_new_session"] is True
        # The child must be able to import glplot even without an installed copy.
        assert "PYTHONPATH" in recorded["kwargs"]["env"]

    def test_degrades_without_matplotlib(self, monkeypatch):
        """With matplotlib absent, launch must warn and return None, never raise."""
        monkeypatch.setattr(mpl_process, "matplotlib_available", lambda: False)
        with pytest.warns(RuntimeWarning):
            assert mpl_process.launch_snapshot_viewer(make_snapshot()) is None

    def test_failed_spawn_returns_none_and_cleans_up(self, monkeypatch):
        """A child that cannot start must not raise and must leave no temp payload."""
        import subprocess
        import tempfile

        monkeypatch.setattr(mpl_process, "matplotlib_available", lambda: True)

        def boom(*args, **kwargs):
            raise OSError("cannot spawn")

        monkeypatch.setattr(subprocess, "Popen", boom)
        before = set(os.listdir(tempfile.gettempdir()))
        with pytest.warns(RuntimeWarning):
            assert mpl_process.launch_snapshot_viewer(make_snapshot()) is None
        leaked = {
            f
            for f in set(os.listdir(tempfile.gettempdir())) - before
            if f.startswith("glplot_snapshot_")
        }
        assert leaked == set()


class TestTransferRouting:
    """Test which path the 'M' key takes. GPULinePlot builds with no window."""

    def build_plot(self, snap):
        """Construct a windowless engine whose snapshot capture needs no GL."""
        plot = GPULinePlot()
        plot.capture_snapshot = lambda **kwargs: snap
        return plot

    def test_default_action_uses_a_separate_process(self, monkeypatch):
        """Regression: with no target registered, M must hand off to a child process.

        Drawing in-process is what glfw.poll_events() destroys on macOS.
        """
        snap = make_snapshot()
        plot = self.build_plot(snap)
        plot.xlabel = "Time (s)"
        plot.ylabel = "Volts"
        plot.title = "My Plot"

        calls = []
        monkeypatch.setattr(
            mpl_process, "launch_snapshot_viewer", lambda s, **kw: calls.append((s, kw))
        )
        plot.transfer_to_matplotlib_default()

        assert len(calls) == 1
        sent, kwargs = calls[0]
        assert sent is snap
        assert kwargs == {"xlabel": "Time (s)", "ylabel": "Volts", "title": "My Plot"}

    def test_default_action_without_labels(self, monkeypatch):
        """A plot that never set xlabel/ylabel must transfer without raising."""
        plot = self.build_plot(make_snapshot())
        calls = []
        monkeypatch.setattr(
            mpl_process, "launch_snapshot_viewer", lambda s, **kw: calls.append((s, kw))
        )
        plot.transfer_to_matplotlib_default()
        assert calls[0][1]["xlabel"] is None
        assert calls[0][1]["ylabel"] is None

    def test_callback_target_stays_in_process(self, monkeypatch):
        """A registered callback must receive the snapshot and spawn nothing."""
        snap = make_snapshot()
        plot = self.build_plot(snap)
        received = []
        plot.set_matplotlib_transfer_target(callback=received.append)

        calls = []
        monkeypatch.setattr(mpl_process, "launch_snapshot_viewer", lambda s, **kw: calls.append(s))
        plot.transfer_to_matplotlib_default()

        assert received == [snap]
        assert calls == []

    def test_ax_target_stays_in_process(self, monkeypatch):
        """A registered axis must be drawn into directly, spawning nothing."""
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plot = self.build_plot(make_snapshot())
        fig, ax = plt.subplots()
        plot.set_matplotlib_transfer_target(ax=ax)

        calls = []
        monkeypatch.setattr(mpl_process, "launch_snapshot_viewer", lambda s, **kw: calls.append(s))
        plot.transfer_to_matplotlib_default()

        assert len(ax.images) == 1
        assert calls == []
        plt.close("all")
