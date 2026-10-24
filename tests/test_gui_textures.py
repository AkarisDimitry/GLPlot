"""The texture cache and the window-aware thumbnail holder.

No test here makes a GL call: the cache takes its upload/delete functions as arguments precisely
so that its logic -- lazy builds, a per-frame budget, remembered failures, replacement, and the
difference between *releasing* textures and *forgetting* ids from a dead context -- can be pinned
with fakes. (Live GL calls are kept out of pytest on purpose: they pass alone and fail
non-deterministically in the full suite.)
"""

from __future__ import annotations

import numpy as np
import pytest

from glplot.gui.textures import Texture, TextureCache


class FakeGL:
    """Records what the cache asked of 'GL' and hands out sequential ids."""

    def __init__(self, fail_on=()):
        self.uploads = []
        self.deleted = []
        self.fail_on = set(fail_on)
        self._next = 100

    def upload(self, rgba):
        if len(self.uploads) in self.fail_on:
            self.uploads.append(None)
            return None
        self._next += 1
        self.uploads.append(rgba.shape)
        return self._next

    def delete(self, texture_id):
        self.deleted.append(texture_id)


def pixels(h=4, w=6, channels=4, value=7):
    return np.full((h, w, channels), value, dtype=np.uint8)


@pytest.fixture
def gl():
    return FakeGL()


@pytest.fixture
def cache(gl):
    return TextureCache(gl.upload, gl.delete, budget=2)


class TestLazyBuild:
    def test_a_texture_is_built_once_from_the_loader(self, cache, gl):
        calls = []

        def loader():
            calls.append(1)
            return pixels()

        first = cache.get("a", loader)
        again = cache.get("a", loader)
        assert first == again == Texture(101, 6, 4)
        assert len(calls) == 1 and len(gl.uploads) == 1

    def test_peek_never_builds(self, cache, gl):
        assert cache.peek("a") is None
        assert gl.uploads == []

    def test_rgb_is_given_an_opaque_alpha_channel(self, cache, gl):
        cache.put("rgb", pixels(channels=3))
        assert gl.uploads == [(4, 6, 4)]

    @pytest.mark.parametrize(
        "bad", [np.zeros((4, 6)), np.zeros((4, 6, 2), dtype=np.uint8), np.zeros(5)]
    )
    def test_a_wrong_shape_is_refused_and_remembered(self, cache, gl, bad):
        assert cache.put("bad", bad) is None
        assert gl.uploads == []

    def test_a_loader_that_finds_nothing_is_not_asked_again(self, cache):
        calls = []

        def loader():
            calls.append(1)
            return None

        for _ in range(5):
            assert cache.get("missing", loader) is None
        assert len(calls) == 1

    def test_a_failed_upload_is_remembered_too(self):
        gl = FakeGL(fail_on={0})
        cache = TextureCache(gl.upload, gl.delete)
        assert cache.get("a", pixels) is None
        assert cache.get("a", pixels) is None
        assert len(gl.uploads) == 1


class TestBudget:
    def test_only_a_few_textures_are_built_per_frame(self, cache, gl):
        got = [cache.get(i, pixels) for i in range(5)]
        assert [t is not None for t in got] == [True, True, False, False, False]

    def test_the_next_frame_builds_the_rest_and_keeps_the_first(self, cache, gl):
        for i in range(5):
            cache.get(i, pixels)
        cache.begin_frame()
        for i in range(5):
            cache.get(i, pixels)
        assert len(cache) == 4 and len(gl.uploads) == 4

    def test_being_over_budget_is_not_a_failure(self, cache):
        for i in range(2):
            cache.get(i, pixels)
        assert cache.get("late", pixels) is None
        cache.begin_frame()
        assert cache.get("late", pixels) is not None

    def test_already_cached_textures_cost_no_budget(self, cache):
        cache.get("a", pixels)
        cache.begin_frame()
        for _ in range(10):
            assert cache.get("a", pixels) is not None
        assert cache.get("b", pixels) is not None and cache.get("c", pixels) is not None


class TestReplacementAndRelease:
    def test_putting_again_deletes_the_old_texture(self, cache, gl):
        first = cache.put("a", pixels())
        second = cache.put("a", pixels(h=8))
        assert gl.deleted == [first.id]
        assert cache.peek("a") == second and second.height == 8

    def test_a_failed_replacement_keeps_the_old_texture(self):
        gl = FakeGL(fail_on={1})
        cache = TextureCache(gl.upload, gl.delete)
        first = cache.put("a", pixels())
        assert cache.put("a", pixels()) is None
        assert cache.peek("a") == first and gl.deleted == []

    def test_drop_deletes(self, cache, gl):
        texture = cache.put("a", pixels())
        cache.drop("a")
        cache.drop("never-there")
        assert gl.deleted == [texture.id] and cache.peek("a") is None

    def test_release_deletes_everything_and_clears_failures(self, cache, gl):
        cache.put("a", pixels())
        cache.put("b", pixels())
        cache.get("gone", lambda: None)
        cache.release()
        assert sorted(gl.deleted) == [101, 102] and len(cache) == 0
        cache.begin_frame()
        assert cache.get("gone", pixels) is not None  # the failure was forgotten

    def test_forget_drops_the_ids_without_touching_gl(self, cache, gl):
        """The ids belong to a context that no longer exists: deleting them is itself an error."""
        cache.put("a", pixels())
        cache.forget()
        assert gl.deleted == [] and len(cache) == 0


class TestDisabled:
    def test_a_cache_with_no_uploader_never_loads_anything(self):
        cache = TextureCache()
        calls = []
        assert cache.enabled is False
        assert cache.get("a", lambda: calls.append(1) or pixels()) is None
        assert cache.put("a", pixels()) is None
        assert calls == []

    def test_a_cache_with_an_uploader_is_enabled(self, cache):
        assert cache.enabled is True
