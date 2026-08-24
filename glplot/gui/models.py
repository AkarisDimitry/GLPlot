"""Trained models: the session-scoped, cross-panel-visible output of "Save as model" in
Math Lab's Fit/Cluster/PCA/UMAP tabs.

Pure logic -- numpy and stdlib only, no imgui, no engine imports (CONTRACT 5.1 rule 7),
mirroring :mod:`glplot.gui.datasets`. :class:`ModelStore` is deliberately the same shape
as :class:`~glplot.gui.datasets.DataStore`: a :class:`TrainedModel` is meant to be as
cross-panel-visible a citizen as a ``DataSet`` is, even though only Math Lab reads or
writes this registry today.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


def _next_available(base: str, taken: Sequence[str]) -> str:
    """Return ``base`` if free, else ``base (2)``, ``base (3)``, ...

    A faithful copy of :func:`glplot.gui.datasets._next_available` -- that helper is
    private to its module, and this file has exactly the same one-liner need, so it is
    replicated rather than reached into ``datasets.py`` for.
    """
    if base not in taken:
        return base
    index = 2
    while f"{base} ({index})" in taken:
        index += 1
    return f"{base} ({index})"


@dataclass(eq=False)
class TrainedModel:
    """A saved, transferable model: the persisted output of training a Fit/Cluster/PCA/UMAP
    result in Math Lab. Session-scoped, in-memory only (no serialization) -- it survives tab
    switches and source changes because it holds no view into the live source: everything it
    needs was copied at Save time.

    ``eq=False``: a generated ``__eq__`` would compare ndarrays and raise.
    """

    name: str
    technique: str  # "fit" | "cluster" | "pca" | "umap" -- the _TABS key trained from
    created: float  # time.time() at Save; display only
    source_label: str  # e.g. dataset/layer name + column names -- provenance, display only
    input_columns: Tuple[str, ...]  # columns/roles a target dataset must supply to reuse this model
    train_metrics: List[Tuple[str, str]] = field(
        default_factory=list
    )  # same shape as _Result.stats
    val_metrics: List[Tuple[str, str]] = field(default_factory=list)
    test_metrics: List[Tuple[str, str]] = field(default_factory=list)
    subsample_note: Optional[str] = None
    preview: Optional[Any] = (
        None  # the exact _Result Save captured -- thumbnail source, never recomputed
    )

    # -- Fit --
    fit_model_key: str = ""
    fit_expr: str = ""
    fit_n_peaks: int = 0
    fit_popt: Optional[np.ndarray] = None
    fit_pcov: Optional[np.ndarray] = None

    # -- Cluster --
    cluster_method: str = ""
    cluster_centroid_x: Optional[np.ndarray] = None
    cluster_centroid_y: Optional[np.ndarray] = None

    # -- PCA --
    pca_columns: Tuple[str, ...] = ()
    pca_scale: str = "zscore"
    pca_mean_: Optional[np.ndarray] = None
    pca_components_: Optional[np.ndarray] = None
    pca_scale_stats: Optional[Dict[str, Any]] = None

    # -- UMAP --
    umap_columns: Tuple[str, ...] = ()
    umap_scale: str = "zscore"
    umap_scale_stats: Optional[Dict[str, Any]] = None
    umap_reducer: Optional[Any] = None  # a fitted umap.UMAP; None means transfer is unavailable


class ModelStore:
    """The workspace's session-scoped registry of trained models -- deliberately the same shape
    as DataStore (glplot/gui/datasets.py): a TrainedModel is meant to be as cross-panel-visible a
    citizen as a DataSet. Only Math Lab reads/writes it today; nothing here is Math-Lab-specific.
    """

    def __init__(self) -> None:
        self.models: List[TrainedModel] = []

    def __len__(self) -> int:
        return len(self.models)

    def __iter__(self):
        return iter(self.models)

    def names(self) -> List[str]:
        """The model names, in order."""
        return [model.name for model in self.models]

    def unique_name(self, base: str) -> str:
        """Return ``base`` if free, else ``base (2)``, ``base (3)``, ..."""
        return _next_available(base, self.names())

    def add(self, model: TrainedModel) -> TrainedModel:
        """Register ``model``, renaming it in place if its name collides. Returns ``model``.

        Re-adding an already-registered model is a no-op (it is not renamed).
        """
        if any(existing is model for existing in self.models):
            return model
        model.name = self.unique_name(model.name)
        self.models.append(model)
        return model

    def remove(self, model: TrainedModel) -> bool:
        """Unregister ``model``. Returns False if it was not registered."""
        for i, existing in enumerate(self.models):
            if existing is model:
                del self.models[i]
                return True
        return False

    def get(self, name: str) -> Optional[TrainedModel]:
        """Return the model named ``name``, or None."""
        for model in self.models:
            if model.name == name:
                return model
        return None
