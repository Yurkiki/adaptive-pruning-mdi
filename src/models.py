"""Growing the forests, and the grid of depths we will prune back to."""

from __future__ import annotations

from dataclasses import dataclass
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
