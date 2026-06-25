"""Functional archetypes via clustering on style skills (not traditional positions)."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from nba_draft.fit.types import SKILL_DIMS, Player


def skills_matrix(players: list[Player]) -> NDArray[np.float64]:
    """Stack players' skill vectors into a (n_players, n_skills) matrix."""
    if not players:
        return np.empty((0, len(SKILL_DIMS)))
    return np.vstack([p.skill_vector() for p in players])


class ArchetypeModel:
    """KMeans over standardized style skills, assigning each player a functional archetype.

    Descriptive (not predictive), so it is fit on whatever player pool is available. Cluster
    labels are arbitrary integers unless named via :meth:`name_clusters`.
    """

    def __init__(self, n_archetypes: int = 5, *, seed: int = 42) -> None:
        self.n_archetypes = n_archetypes
        self.seed = seed
        self._kmeans: Any | None = None
        self._mean: NDArray[np.float64] | None = None
        self._std: NDArray[np.float64] | None = None

    def _standardize(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        assert self._mean is not None and self._std is not None
        return (x - self._mean) / self._std

    def fit(self, players: list[Player]) -> ArchetypeModel:
        from sklearn.cluster import KMeans

        x = skills_matrix(players)
        if x.shape[0] < self.n_archetypes:
            raise ValueError("Need at least n_archetypes players to fit.")
        self._mean = x.mean(axis=0)
        self._std = np.where(x.std(axis=0) == 0, 1.0, x.std(axis=0))
        self._kmeans = KMeans(n_clusters=self.n_archetypes, random_state=self.seed, n_init=10)
        self._kmeans.fit(self._standardize(x))
        return self

    def predict(self, players: list[Player]) -> NDArray[np.int64]:
        if self._kmeans is None:
            raise RuntimeError("ArchetypeModel must be fit before predict().")
        labels = self._kmeans.predict(self._standardize(skills_matrix(players)))
        return np.asarray(labels, dtype=np.int64)
