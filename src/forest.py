"""The three forests and the bootstrap bookkeeping the importance measures need."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import numpy.typing as npt
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble._forest import _generate_sample_indices

import prune
from mdi import TreeView

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]


@dataclass(frozen=True)
class ForestConfig:
    n_trees: int = 500
    max_features: float = 1.0 / 3.0
    min_samples_leaf_grid: tuple[int, ...] = (1, 2, 3, 5, 8, 13, 20, 30, 50, 80)
    ccp_fractions: tuple[float, ...] = (0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1)
    alpha_grid: tuple[float, ...] = (
        0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0
    )
    alpha_min_samples_leaf: int = 5
    n_jobs: int = -1

    def __post_init__(self) -> None:
        if 1 not in self.min_samples_leaf_grid:
            raise ValueError("min_samples_leaf grid must contain the unpruned case, 1")
        if 0.0 not in self.ccp_fractions:
            raise ValueError("ccp grid must contain the unpruned case, 0.0")
        if 0.0 not in self.alpha_grid:
            raise ValueError("alpha grid must contain the unpruned case, 0.0")


@dataclass(frozen=True)
class Fitted:
    name: str
    forest: prune.PrunedForest
    selected_param: float
    oob_mse: float
    inbag_counts: IntArray = field(repr=False)

    @property
    def oob_mask(self) -> BoolArray:
        return np.asarray(self.inbag_counts == 0, dtype=np.bool_)

    @property
    def mean_leaves(self) -> float:
        return self.forest.mean_leaves

    @property
    def mean_depth(self) -> float:
        return self.forest.mean_depth


def ensemble(model: RandomForestRegressor) -> prune.PrunedForest:
    """Wrap a scikit-learn forest so it looks like an alpha-trimmed one."""
    return prune.PrunedForest(
        tuple(TreeView.of(tree) for tree in model.estimators_), alpha=0.0
    )


def inbag_counts(forest: RandomForestRegressor, n: int) -> IntArray:
    """How many times each training row entered each tree's bootstrap sample."""
    rows = [
        np.bincount(_generate_sample_indices(est.random_state, n, n), minlength=n)
        for est in forest.estimators_
    ]
    return np.asarray(rows, dtype=np.int64)


def oob_predictions(
    forest: prune.PrunedForest, x: FloatArray, counts: IntArray
) -> tuple[FloatArray, BoolArray]:
    """Average prediction over the trees for which each row was out of bag."""
    per_tree = forest.predict_per_tree(x)
    weight = (counts == 0).astype(np.float64)
    denom = weight.sum(axis=0)
    covered = denom > 0.0
    predictions = np.zeros(x.shape[0], dtype=np.float64)
    predictions[covered] = (per_tree * weight).sum(axis=0)[covered] / denom[covered]
    return predictions, covered


def oob_mse(
    forest: prune.PrunedForest, x: FloatArray, y: FloatArray, counts: IntArray
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
    model = _fit(x, y, config, seed)
    counts = inbag_counts(model, x.shape[0])
    wrapped = ensemble(model)
    return Fitted("full", wrapped, 1.0, oob_mse(wrapped, x, y, counts), counts)


def build_global(
    x: FloatArray, y: FloatArray, config: ForestConfig, seed: int
) -> Fitted:
    """Uniform depth control: one min_samples_leaf for the whole forest, tuned on OOB."""
    cap = max(1, x.shape[0] // 10)
    grid = tuple(v for v in config.min_samples_leaf_grid if v <= cap) or (1,)
    best: Fitted | None = None
    for leaf_size in grid:
        model = _fit(x, y, config, seed, min_samples_leaf=leaf_size)
        counts = inbag_counts(model, x.shape[0])
        wrapped = ensemble(model)
        error = oob_mse(wrapped, x, y, counts)
        if best is None or error < best.oob_mse:
            best = Fitted("global", wrapped, float(leaf_size), error, counts)
    assert best is not None
    return best


@dataclass(frozen=True)
class PruningPath:
    """Every pruning level, fitted once.

    ccp_alpha prunes a tree that was already grown, so the whole path costs one
    sweep and any selection rule can be applied to it afterwards: by OOB error
    for the honest arm, by importance error for the oracle arm.
    """

    fits: tuple[Fitted, ...]

    @property
    def alphas(self) -> tuple[float, ...]:
        return tuple(fit.selected_param for fit in self.fits)


def pruning_path(
    x: FloatArray, y: FloatArray, config: ForestConfig, seed: int
) -> PruningPath:
    """Adaptive pruning by cost complexity, across the whole grid.

    Stands in for alpha-trimming while that is unimplemented. The grid is a
    fraction of Var(y) because ccp_alpha is measured in impurity units.
    """
    scale = float(np.var(y))
    fits: list[Fitted] = []
    for fraction in config.ccp_fractions:
        alpha = fraction * scale
        model = _fit(x, y, config, seed, ccp_alpha=alpha)
        counts = inbag_counts(model, x.shape[0])
        wrapped = ensemble(model)
        fits.append(
            Fitted("ccp", wrapped, alpha, oob_mse(wrapped, x, y, counts), counts)
        )
    return PruningPath(tuple(fits))


def select_by_oob(path: PruningPath, name: str = "ccp") -> Fitted:
    return replace(min(path.fits, key=lambda fit: fit.oob_mse), name=name)


def build_pruned(
    x: FloatArray, y: FloatArray, config: ForestConfig, seed: int
) -> Fitted:
    return select_by_oob(pruning_path(x, y, config, seed))


def alpha_path(
    x: FloatArray, y: FloatArray, config: ForestConfig, seed: int
) -> PruningPath:
    """Alpha-trimming across the whole alpha grid, from a single forest fit.

    The base forest uses min_samples_leaf = 5, the node size the paper treats as a
    fully grown forest. Growing to purity instead would leave every leaf with zero
    variance, which makes the stump model's log-likelihood diverge and no merge can
    ever fire -- alpha-trimming needs an estimable sigma^2 in the leaves.
    """
    model = _fit(x, y, config, seed, min_samples_leaf=config.alpha_min_samples_leaf)
    counts = inbag_counts(model, x.shape[0])
    views = tuple(TreeView.of(tree) for tree in model.estimators_)
    fits = []
    for alpha in config.alpha_grid:
        trimmed = prune.trim_forest(views, alpha)
        fits.append(
            Fitted("alpha", trimmed, alpha, oob_mse(trimmed, x, y, counts), counts)
        )
    return PruningPath(tuple(fits))


def build_alpha(
    x: FloatArray, y: FloatArray, config: ForestConfig, seed: int
) -> Fitted:
    return select_by_oob(alpha_path(x, y, config, seed), name="alpha")


BUILDERS = (build_full, build_global, build_pruned, build_alpha)
