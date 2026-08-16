"""Suite-wide fixtures.

One job, in two halves: make the headless imgui harnesses independent of anything outside
the test that created them. Dear ImGui keeps two pieces of state that quietly break this,
and both produce failures that point at innocent code.

**The ini file.** ImGui persists per-window position, size and *collapsed* state to
``imgui.ini`` in the working directory, and loads it when a context is created. Run any GUI
example, collapse a panel, and from then on every test that drives that panel fails --
``imgui.begin`` returns ``expanded=False`` and silently skips every widget inside, so the
panel's own hit tests find nothing. Nothing in the diff is wrong; the failure lives in an
untracked file nobody thinks to look at. Every context created under pytest therefore gets
``io.ini_file_name = ""``, which disables the load and the save.

**Leaked contexts.** A harness that calls ``create_context()`` without ever destroying it
leaves the old context alive, and a live one is not inert: a context whose last frame had a
window under the mouse makes the *next* context report ``want_capture_mouse`` True on its
first frame. Destroying it makes that go away. So contexts created during a test are
tracked and any survivor is destroyed at teardown.

Both are done by wrapping the constructor rather than fixing each of the thirteen harness
fixtures, so a fourteenth cannot forget.
"""

from __future__ import annotations

import pytest

try:  # pragma: no cover - GL-less systems skip the GUI tests anyway
    import imgui as _imgui
except Exception:  # pragma: no cover
    _imgui = None


@pytest.fixture(autouse=True)
def imgui_context_hygiene(monkeypatch):
    """No ini file, and no context outliving the test that made it.

    Autouse and function-scoped: it is set up before the test's own fixtures and so torn
    down after them, which means a harness fixture that *does* destroy its context has
    already run by the time the sweep below looks for survivors. Tracking is by identity
    (``create_context`` returns the very object ``get_current_context`` does), so a context
    destroyed properly is off the list and never destroyed twice.
    """
    if _imgui is None:  # pragma: no cover
        yield
        return

    real_create = _imgui.create_context
    real_destroy = _imgui.destroy_context
    live = []

    def create_context(*args, **kwargs):
        ctx = real_create(*args, **kwargs)
        live.append(ctx)
        try:
            _imgui.get_io().ini_file_name = ""
        except Exception:  # pragma: no cover - defensive, never seen
            pass
        return ctx

    def destroy_context(ctx=None, *args, **kwargs):
        target = ctx if ctx is not None else _imgui.get_current_context()
        for i, known in enumerate(live):
            if known is target:
                del live[i]
                break
        return real_destroy(ctx, *args, **kwargs)

    monkeypatch.setattr(_imgui, "create_context", create_context)
    monkeypatch.setattr(_imgui, "destroy_context", destroy_context)
    try:
        yield
    finally:
        while live:
            try:
                real_destroy(live.pop())
            except Exception:  # pragma: no cover - defensive
                pass
