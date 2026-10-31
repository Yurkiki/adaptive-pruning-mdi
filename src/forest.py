"""The three forests and the bootstrap bookkeeping the importance measures need."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble._forest import _generate_sample_indices

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]


@dataclass(frozen=True)
class ForestConfig:
    n_trees: int = 500
    max_features: float = 1.0 / 3.0
    min_samples_leaf_grid: tuple[int, ...] = (1, 2, 3, 5, 8, 13, 20, 30, 50, 80)
    ccp_fractions: tuple[float, ...] = (0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1)
    n_jobs: int = -1

    def __post_init__(self) -> None:
        if 1 not in self.min_samples_leaf_grid:
            raise ValueError("min_samples_leaf grid must contain the unpruned case, 1")
        if 0.0 not in self.ccp_fractions:
            raise ValueError("ccp grid must contain the unpruned case, 0.0")


@dataclass(frozen=True)
class Fitted:
    name: str
    forest: RandomForestRegressor
    selected_param: float
    oob_mse: float
    inbag_counts: IntArray = field(repr=False)

    @property
    def oob_mask(self) -> BoolArray:
        return np.asarray(self.inbag_counts == 0, dtype=np.bool_)

    @property
    def mean_leaves(self) -> float:
        counts = [int((e.tree_.children_left == -1).sum()) for e in self.forest.estimators_]
        return float(np.mean(counts))

    @property
    def mean_depth(self) -> float:
        return float(np.mean([e.get_depth() for e in self.forest.estimators_]))


def inbag_counts(forest: RandomForestRegressor, n: int) -> IntArray:
    """How many times each training row entered each tree's bootstrap sample."""
    rows = [
        np.bincount(_generate_sample_indices(est.random_state, n, n), minlength=n)
        for est in forest.estimators_
    ]
    return np.asarray(rows, dtype=np.int64)


def oob_predictions(
    forest: RandomForestRegressor, x: FloatArray, counts: IntArray
) -> tuple[FloatArray, BoolArray]:
    """Average prediction over the trees for which each row was out of bag."""
    per_tree = np.stack([est.predict(x) for est in forest.estimators_])
    weight = (counts == 0).astype(np.float64)
    denom = weight.sum(axis=0)
    covered = denom > 0.0
    predictions = np.zeros(x.shape[0], dtype=np.float64)
    predictions[covered] = (per_tree * weight).sum(axis=0)[covered] / denom[covered]
    return predictions, covered


def oob_mse(
    forest: RandomForestRegressor, x: FloatArray, y: FloatArray, counts: IntArray
) -> float:
    predictions, covered = oob_predictions(forest, x, counts)
    return float(np.mean((y[covered] - predictions[covered]) ** 2))


def _fit(
    x: FloatArray,
    y: FloatArray,
    config: ForestConfig,
    seed: int,
    min_samples_leaf: int = 1,
    ccp_alpha: float = 0.0,
) -> RandomForestRegressor:
    forest = RandomForestRegressor(
        n_estimators=config.n_trees,
        max_features=config.max_features,
        min_samples_leaf=min_samples_leaf,
        ccp_alpha=ccp_alpha,
        bootstrap=True,
        random_state=seed,
        n_jobs=config.n_jobs,
    )
    forest.fit(x, y)
    return forest


def build_full(
    x: FloatArray, y: FloatArray, config: ForestConfig, seed: int
) -> Fitted:
    forest = _fit(x, y, config, seed)
    counts = inbag_counts(forest, x.shape[0])
    return Fitted("full", forest, 1.0, oob_mse(forest, x, y, counts), counts)


def build_global(
    x: FloatArray, y: FloatArray, config: ForestConfig, seed: int
) -> Fitted:
    """Uniform depth control: one min_samples_leaf for the whole forest, tuned on OOB."""
    cap = max(1, x.shape[0] // 10)
    grid = tuple(v for v in config.min_samples_leaf_grid if v <= cap) or (1,)
    best: Fitted | None = None
    for leaf_size in grid:
        forest = _fit(x, y, config, seed, min_samples_leaf=leaf_size)
        counts = inbag_counts(forest, x.shape[0])
        error = oob_mse(forest, x, y, counts)
        if best is None or error < best.oob_mse:
            best = Fitted("global", forest, float(leaf_size), error, counts)
    assert best is not None
    return best


def build_pruned(
    x: FloatArray, y: FloatArray, config: ForestConfig, seed: int
) -> Fitted:
    """Adaptive pruning by cost complexity, tuned on OOB.

    Stands in for alpha-trimming while that is unimplemented; the grid is expressed
    as a fraction of Var(y) because ccp_alpha is measured in impurity units.
    """
    scale = float(np.var(y))
    best: Fitted | None = None
    for fraction in config.ccp_fractions:
        alpha = fraction * scale
        forest = _fit(x, y, config, seed, ccp_alpha=alpha)
        counts = inbag_counts(forest, x.shape[0])
        error = oob_mse(forest, x, y, counts)
        if best is None or error < best.oob_mse:
            best = Fitted("pruned", forest, alpha, error, counts)
    assert best is not None
    return best


BUILDERS = (build_full, build_global, build_pruned)
