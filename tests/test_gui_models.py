"""Test the GUI trained-model registry in glplot.gui.models.

Pure logic: no OpenGL context, no window and no imgui are created here. Mirrors
tests/test_gui_datasets.py's TestDataStore, since ModelStore is deliberately the same
shape as DataStore.
"""

from __future__ import annotations

from glplot.gui.models import ModelStore, TrainedModel


def _model(name: str = "m") -> TrainedModel:
    """Build a minimal TrainedModel for reuse across tests."""
    return TrainedModel(
        name=name,
        technique="fit",
        created=0.0,
        source_label="ds.y (x, y)",
        input_columns=("x",),
    )


class TestModelStore:
    """Test the ModelStore registry."""

    def test_add_assigns_a_unique_name_on_collision(self):
        """Test that a colliding model name is suffixed on add."""
        store = ModelStore()
        store.add(_model("m"))
        second = store.add(_model("m"))
        assert second.name == "m (2)"
        assert store.names() == ["m", "m (2)"]

    def test_remove(self):
        """Test that remove() unregisters the model and reports True."""
        store = ModelStore()
        model = store.add(_model("m"))
        assert store.remove(model) is True
        assert len(store) == 0
        assert store.get("m") is None

    def test_remove_unregistered_returns_false(self):
        """Test that removing a model that was never added returns False."""
        assert ModelStore().remove(_model()) is False

    def test_get_by_name(self):
        """Test that get() returns the registered model by name, or None."""
        store = ModelStore()
        model = store.add(_model("mine"))
        assert store.get("mine") is model
        assert store.get("nope") is None

    def test_names_lists_in_registration_order(self):
        """Test that names() and iteration both preserve insertion order."""
        store = ModelStore()
        store.add(_model("a"))
        store.add(_model("b"))
        assert store.names() == ["a", "b"]
        assert [model.name for model in store] == ["a", "b"]

    def test_add_does_not_rename_an_already_registered_model(self):
        """Test that re-adding an already-registered model is a no-op (no rename)."""
        store = ModelStore()
        model = store.add(_model("m"))
        store.add(model)
        assert len(store) == 1
        assert model.name == "m"

    def test_len_and_iter(self):
        """Test that __len__ and __iter__ reflect the registered models."""
        store = ModelStore()
        assert len(store) == 0
        assert list(store) == []
        store.add(_model("a"))
        store.add(_model("b"))
        assert len(store) == 2
        assert [model.name for model in store] == ["a", "b"]

    def test_unique_name(self):
        """Test that unique_name() returns the base when free, else a suffix."""
        store = ModelStore()
        assert store.unique_name("m") == "m"
        store.add(_model("m"))
        assert store.unique_name("m") == "m (2)"
        store.add(_model("m"))
        assert store.unique_name("m") == "m (3)"
