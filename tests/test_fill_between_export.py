"""A translucent ``fill_between`` exports as a flat tint, not a field of vertical seams.

The export strokes every triangle of a band; with the edge at the face's own alpha the
strokes of neighbouring triangles overlap along the shared sides. The default edge is now the
face colour with no opacity, and an edge the caller asked for still wins.
"""

import numpy as np
import pytest

import glplot.pyplot as gplt
from glplot.utils.preview import _build_preview_figure


@pytest.fixture(autouse=True)
def clean_state():
    gplt._cleanup_pyplot_state()
    yield
    gplt._cleanup_pyplot_state()


X = np.linspace(0.0, 1.0, 30)


@pytest.mark.parametrize("func", ["fill_between", "fill_betweenx"])
def test_default_edge_has_no_opacity_of_its_own(func):
    layer = getattr(gplt, func)(X, X, X + 1.0, alpha=0.3, color="tab:blue")
    assert layer.style.edge_color[3] == 0.0
    assert layer.style.face_color[3] == pytest.approx(0.3)


def test_the_exported_collection_has_an_unstroked_edge():
    gplt.fill_between(X, X, X + 1.0, alpha=0.3, color="tab:blue")
    ax = _build_preview_figure(gplt.get_engine()).axes[0]
    (collection,) = ax.collections
    assert np.all(collection.get_edgecolor()[:, 3] == 0.0)
    assert np.all(collection.get_facecolor()[:, 3] == pytest.approx(0.3))


def test_an_explicit_edge_still_wins():
    layer = gplt.fill_between(X, X, X + 1.0, alpha=0.3, edgecolor="red", linewidth=2)
    assert layer.style.edge_color[3] > 0.0
    assert layer.metadata.get("edge_as_outline") is True
