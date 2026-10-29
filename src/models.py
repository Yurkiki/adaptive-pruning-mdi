"""The forest and the pruning families.

Every arm is a *family*: a grid of pruning levels and a way to grow one forest
per level. The level is picked by out-of-bag prediction error inside the
training split, so the validation split stays reserved for the measures.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

import numpy as np
import numpy.typing as npt
from sklearn.ensemble import RandomForestRegressor

from tree import Model, TreeView

FloatArray = npt.NDArray[np.float64]
IndexArray = npt.NDArray[np.intp]


@dataclass(frozen=True)
class ModelConfig:
    n_trees: int = 500
    max_features: float = 1.0 / 3.0
    min_samples_leaf_grid: tuple[int, ...] = (1, 2, 3, 5, 8, 13, 20, 30, 50, 80)
    n_jobs: int = -1

    def __post_init__(self) -> None:
        if 1 not in self.min_samples_leaf_grid:
            raise ValueError("min_samples_leaf grid must contain the unpruned case, 1")


@dataclass(frozen=True)
class Forest:
    """A fitted forest, with the rows each tree was fitted on kept alongside it.

    The in-bag rows are what lets a *pruned* forest still be scored out of bag:
    forest.oob_prediction_ is fixed at fit time and knows nothing about trimming.
    """

    model: Model
    in_bag: tuple[IndexArray, ...]


@dataclass(frozen=True)
class Fitted:
    name: str
    model: Model
    selected_param: float
    oob_mse: float

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
) -> Forest:
    """One forest, grown to purity unless told otherwise."""
    forest = RandomForestRegressor(
        n_estimators=config.n_trees,
        max_features=config.max_features,
        min_samples_leaf=min_samples_leaf,
        ccp_alpha=ccp_alpha,
        bootstrap=True,
        random_state=seed,
        n_jobs=config.n_jobs,
    ).fit(x, y)
    return Forest(
        Model(tuple(TreeView.of(t) for t in forest.estimators_)),
        tuple(np.asarray(rows, dtype=np.intp) for rows in forest.estimators_samples_),
    )


def oob_mse(forest: Forest, x: FloatArray, y: FloatArray) -> float:
    """Average every tree over the rows it never saw, then score that prediction."""
    total = np.zeros(x.shape[0], dtype=np.float64)
    count = np.zeros(x.shape[0], dtype=np.float64)
    for tree, in_bag in zip(forest.model.trees, forest.in_bag, strict=True):
        held = np.ones(x.shape[0], dtype=bool)
        held[in_bag] = False
        total[held] += tree.predict(x[held])
        count[held] += 1.0
    seen = count > 0
    return float(np.mean((total[seen] / count[seen] - y[seen]) ** 2))


Levels = Callable[[ModelConfig, int], tuple[float, ...]]
Grower = Callable[[FloatArray, FloatArray, ModelConfig, int, tuple[float, ...]], tuple[Forest, ...]]


@dataclass(frozen=True)
class Family:
    """A pruning method: its grid of levels and how to grow one forest per level."""

    name: str
    levels: Levels
    grow: Grower


def _full_levels(config: ModelConfig, n: int) -> tuple[float, ...]:
    return (1.0,)


def _full_grow(
    x: FloatArray, y: FloatArray, config: ModelConfig, seed: int, levels: tuple[float, ...]
) -> tuple[Forest, ...]:
    return (grow(x, y, config, seed),)


def _global_levels(config: ModelConfig, n: int) -> tuple[float, ...]:
    return tuple(float(v) for v in config.min_samples_leaf_grid)


def _global_grow(
    x: FloatArray, y: FloatArray, config: ModelConfig, seed: int, levels: tuple[float, ...]
) -> tuple[Forest, ...]:
    return tuple(grow(x, y, config, seed, min_samples_leaf=int(v)) for v in levels)


FULL = Family("full", _full_levels, _full_grow)
GLOBAL = Family("global", _global_levels, _global_grow)

FAMILIES: dict[str, Family] = {f.name: f for f in (FULL, GLOBAL)}


@dataclass(frozen=True)
class PruningPath:
    """Every pruning level, grown on the whole training split, with its OOB error.

    Any selection rule can be applied afterwards: by OOB error for the honest
    arm, by importance error for the oracle arm.
    """

    fits: tuple[Fitted, ...]

    @property
    def levels(self) -> tuple[float, ...]:
        return tuple(fit.selected_param for fit in self.fits)


def pruning_path(
    family: Family, x: FloatArray, y: FloatArray, config: ModelConfig, seed: int
) -> PruningPath:
    levels = family.levels(config, x.shape[0])
    forests = family.grow(x, y, config, seed, levels)
    return PruningPath(
        tuple(
            Fitted(family.name, forest.model, level, oob_mse(forest, x, y))
            for forest, level in zip(forests, levels, strict=True)
        )
    )


def select_by_oob(path: PruningPath, name: str | None = None) -> Fitted:
    best = min(path.fits, key=lambda fit: fit.oob_mse)
    return best if name is None else replace(best, name=name)


def build(
    family: Family, x: FloatArray, y: FloatArray, config: ModelConfig, seed: int
) -> Fitted:
    return select_by_oob(pruning_path(family, x, y, config, seed))
