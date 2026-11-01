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


def test_an_opaque_fill_has_no_seam_where_its_triangles_meet():
    # antialiasing leaves a hairline gap along each shared side; on a dark page it shows as a
    # diagonal across every quad unless the triangles are stroked in their own colour
    import io

    from PIL import Image

    from glplot.utils.preview import render_preview_bytes

    gplt.figure(figsize=(4, 4))
    gplt.plot_style("3b1b")
    gplt.fill_between([0.0, 2.0], [0.0, 0.0], [2.0, 2.0], color="#236B8E")
    gplt.xlim(-0.5, 2.5)
    gplt.ylim(-0.5, 2.5)
    gplt.axis("off")
    pixels = np.asarray(Image.open(io.BytesIO(render_preview_bytes(gplt.get_engine(), 1.0))))
    near = np.abs(pixels[..., :3].astype(int) - [35, 107, 142]).sum(axis=-1) <= 40
    rows, cols = np.nonzero(near)
    assert rows.size, "the fill was drawn"
    # the fill's own box, minus its antialiased rim: every pixel inside must be the fill
    inside = near[rows.min() + 3 : rows.max() - 2, cols.min() + 3 : cols.max() - 2]
    assert inside.size > 1000 and inside.all(), f"{int((~inside).sum())} pixels show the page"
