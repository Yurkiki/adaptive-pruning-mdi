"""The models -- one regression tree by default, a forest as the planned extension.

Every arm is a *family*: a grid of pruning levels and a way to grow one model per
level. The level is chosen by K-fold cross-validated prediction error on the
training split. A single tree has no bootstrap and so no out-of-bag points; CV
is the classical CART answer, and it keeps the validation split untouched for
the importance measures.

`kind="forest"` grows a random forest instead and goes through exactly the same
families and the same CV, so the extension needs no other change. (OOB tuning
would be cheaper for forests; it is left out until forests are the focus.)
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

import numpy as np
import numpy.typing as npt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor

import prune
from tree import Model, TreeView

FloatArray = npt.NDArray[np.float64]

KINDS = ("tree", "forest")


@dataclass(frozen=True)
class ModelConfig:
    kind: str = "tree"
    n_trees: int = 500
    max_features: float | None = None
    cv_folds: int = 5
    min_samples_leaf_grid: tuple[int, ...] = (1, 2, 3, 5, 8, 13, 20, 30, 50, 80)
    ccp_fractions: tuple[float, ...] = (0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1)
    alpha_grid: tuple[float, ...] = (
        0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0
    )
    alpha_min_samples_leaf: int = 5
    n_jobs: int = -1

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {self.kind!r}")
        if self.cv_folds < 2:
            raise ValueError("cv_folds must be at least 2")
        if 1 not in self.min_samples_leaf_grid:
            raise ValueError("min_samples_leaf grid must contain the unpruned case, 1")
        if 0.0 not in self.ccp_fractions:
            raise ValueError("ccp grid must contain the unpruned case, 0.0")
        if 0.0 not in self.alpha_grid:
            raise ValueError("alpha grid must contain the unpruned case, 0.0")

    @property
    def features_per_split(self) -> float:
        """All features for a CART tree, a third for a regression forest."""
        if self.max_features is not None:
            return self.max_features
        return 1.0 if self.kind == "tree" else 1.0 / 3.0


@dataclass(frozen=True)
class Fitted:
    name: str
    model: Model
    selected_param: float
    cv_mse: float

    @property
    def mean_leaves(self) -> float:
        return self.model.mean_leaves

    @property
    def mean_depth(self) -> float:
        return self.model.mean_depth


def grow(
    x: FloatArray,
    y: FloatArray,
    config: ModelConfig,
    seed: int,
    min_samples_leaf: int = 1,
    ccp_alpha: float = 0.0,
) -> Model:
    """One unpruned-by-default model of the configured kind."""
    if config.kind == "tree":
        single = DecisionTreeRegressor(
            max_features=config.features_per_split,
            min_samples_leaf=min_samples_leaf,
            ccp_alpha=ccp_alpha,
            random_state=seed,
        ).fit(x, y)
        return Model((TreeView.of(single),))
    forest = RandomForestRegressor(
        n_estimators=config.n_trees,
        max_features=config.features_per_split,
        min_samples_leaf=min_samples_leaf,
        ccp_alpha=ccp_alpha,
        bootstrap=True,
        random_state=seed,
        n_jobs=config.n_jobs,
    ).fit(x, y)
    return Model(tuple(TreeView.of(t) for t in forest.estimators_))


Levels = Callable[[ModelConfig, int], tuple[float, ...]]
Grower = Callable[
    [FloatArray, FloatArray, ModelConfig, int, tuple[float, ...], float], tuple[Model, ...]
]


@dataclass(frozen=True)
class Family:
    """A pruning method: its grid of levels and how to grow one model per level."""

    name: str
    levels: Levels
    grow: Grower


def _full_levels(config: ModelConfig, n: int) -> tuple[float, ...]:
    return (1.0,)


def _full_grow(
    x: FloatArray, y: FloatArray, config: ModelConfig, seed: int,
    levels: tuple[float, ...], scale: float
) -> tuple[Model, ...]:
    return (grow(x, y, config, seed),)


def _global_levels(config: ModelConfig, n: int) -> tuple[float, ...]:
    cap = max(1, n // 10)
    return tuple(float(v) for v in config.min_samples_leaf_grid if v <= cap) or (1.0,)


def _global_grow(
    x: FloatArray, y: FloatArray, config: ModelConfig, seed: int,
    levels: tuple[float, ...], scale: float
) -> tuple[Model, ...]:
    return tuple(grow(x, y, config, seed, min_samples_leaf=int(v)) for v in levels)


def _ccp_levels(config: ModelConfig, n: int) -> tuple[float, ...]:
    return config.ccp_fractions


def _ccp_grow(
    x: FloatArray, y: FloatArray, config: ModelConfig, seed: int,
    levels: tuple[float, ...], scale: float
) -> tuple[Model, ...]:
    # ccp_alpha is in impurity units, so the grid is a fraction of Var(y). The
    # scale comes from the whole training split, so a level means the same absolute
    # alpha inside cross-validation as in the model finally fitted.
    return tuple(grow(x, y, config, seed, ccp_alpha=f * scale) for f in levels)


def _alpha_levels(config: ModelConfig, n: int) -> tuple[float, ...]:
    return config.alpha_grid


def _alpha_grow(
    x: FloatArray, y: FloatArray, config: ModelConfig, seed: int,
    levels: tuple[float, ...], scale: float
) -> tuple[Model, ...]:
    """Alpha-trimming across the grid, from a single fit.

    The base uses min_samples_leaf = 5, the node size [SHL24] treat as fully grown.
    Growing to purity would leave every leaf with zero variance, which makes the
    stump log-likelihood diverge so no merge can ever fire.
    """
    base = grow(x, y, config, seed, min_samples_leaf=config.alpha_min_samples_leaf)
    return tuple(prune.trim(base, a) for a in levels)


FULL = Family("full", _full_levels, _full_grow)
GLOBAL = Family("global", _global_levels, _global_grow)
CCP = Family("ccp", _ccp_levels, _ccp_grow)
ALPHA = Family("alpha", _alpha_levels, _alpha_grow)

FAMILIES: dict[str, Family] = {f.name: f for f in (FULL, GLOBAL, CCP, ALPHA)}


@dataclass(frozen=True)
class PruningPath:
    """Every pruning level, grown on the whole training split, with its CV error.

    Any selection rule can be applied afterwards: by CV error for the honest arm,
    by importance error for the oracle arm.
    """

    fits: tuple[Fitted, ...]

    @property
    def levels(self) -> tuple[float, ...]:
        return tuple(fit.selected_param for fit in self.fits)


def cv_errors(
    family: Family,
    levels: tuple[float, ...],
    x: FloatArray,
    y: FloatArray,
    config: ModelConfig,
    seed: int,
    scale: float,
) -> FloatArray:
    """K-fold cross-validated MSE of every level, on the training split only."""
    folds = KFold(n_splits=config.cv_folds, shuffle=True, random_state=seed)
    squared = np.zeros(len(levels), dtype=np.float64)
    for inner, held in folds.split(x):
        models = family.grow(x[inner], y[inner], config, seed, levels, scale)
        for k, model in enumerate(models):
            squared[k] += float(((model.predict(x[held]) - y[held]) ** 2).sum())
    return squared / x.shape[0]


def pruning_path(
    family: Family, x: FloatArray, y: FloatArray, config: ModelConfig, seed: int
) -> PruningPath:
    levels = family.levels(config, x.shape[0])
    scale = float(np.var(y))
    models = family.grow(x, y, config, seed, levels, scale)
    errors = cv_errors(family, levels, x, y, config, seed, scale)
    return PruningPath(
        tuple(
            Fitted(family.name, model, level, float(error))
            for model, level, error in zip(models, levels, errors, strict=True)
        )
    )


def select_by_cv(path: PruningPath, name: str | None = None) -> Fitted:
    best = min(path.fits, key=lambda fit: fit.cv_mse)
    return best if name is None else replace(best, name=name)


def build(
    family: Family, x: FloatArray, y: FloatArray, config: ModelConfig, seed: int
) -> Fitted:
    return select_by_cv(pruning_path(family, x, y, config, seed))
