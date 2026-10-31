"""The three importance measures.

mdi_train    classic MDI, measured on the data that chose the splits
mdi_oob      the same quantity re-evaluated off-sample          [Li et al. 2019]
ufi          in-bag impurity drop plus the off-sample drop      [Zhou & Hooker 2021]

ufi carries twice the scale of the other two on signal features, so callers
compare `ufi_half` against the truth. See SPEC 5.3.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np
import numpy.typing as npt
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
IndexArray = npt.NDArray[np.intp]

Transition = tuple[IndexArray, IndexArray, IndexArray, IndexArray]


def node_means(tree: DecisionTreeRegressor) -> FloatArray:
    """In-bag mean of y at every node, bootstrap weights included."""
    return np.asarray(tree.tree_.value[:, 0, 0], dtype=np.float64)


def mdi_train(tree: DecisionTreeRegressor, n_features: int) -> FloatArray:
    """Unnormalised classic MDI. Equals compute_feature_importances(normalize=False)."""
    inner = tree.tree_
    nodes = np.flatnonzero(inner.feature >= 0)
    left = inner.children_left[nodes]
    right = inner.children_right[nodes]
    weight = inner.weighted_n_node_samples
    drop = (
        weight[nodes] * inner.impurity[nodes]
        - weight[left] * inner.impurity[left]
        - weight[right] * inner.impurity[right]
    )
    out = np.zeros(n_features, dtype=np.float64)
    np.add.at(out, inner.feature[nodes], drop)
    return np.asarray(out / weight[0], dtype=np.float64)


def _descend(tree: DecisionTreeRegressor, x: FloatArray) -> Iterator[Transition]:
    """Walk every row down the tree one level at a time.

    Yields (rows, split_feature, parent_node, child_node) for the rows still moving.
    """
    inner = tree.tree_
    node = np.zeros(x.shape[0], dtype=np.intp)
    while True:
        rows = np.flatnonzero(inner.feature[node] >= 0)
        if rows.size == 0:
            return
        parent = node[rows]
        feature = inner.feature[parent]
        goes_left = x[rows, feature] <= inner.threshold[parent]
        child = np.where(
            goes_left, inner.children_left[parent], inner.children_right[parent]
        ).astype(np.intp)
        yield rows, feature, parent, child
        node[rows] = child


def mdi_oob(
    tree: DecisionTreeRegressor,
    x: FloatArray,
    y: FloatArray,
    n_features: int,
    weights: FloatArray | None = None,
) -> FloatArray:
    """Mean of (mu_child - mu_parent) * y over the evaluation rows reaching each split.

    Evaluated on the in-bag rows with their bootstrap counts as `weights`, this
    reproduces mdi_train exactly.
    """
    row_weight = np.ones(x.shape[0]) if weights is None else weights.astype(np.float64)
    mean = node_means(tree)
    out = np.zeros(n_features, dtype=np.float64)
    for rows, feature, parent, child in _descend(tree, x):
        contribution = row_weight[rows] * (mean[child] - mean[parent]) * y[rows]
        np.add.at(out, feature, contribution)
    return np.asarray(out / row_weight.sum(), dtype=np.float64)


def off_sample_drop(
    tree: DecisionTreeRegressor, x: FloatArray, y: FloatArray, n_features: int
) -> FloatArray:
    """The impurity drop measured on held-out rows against the in-bag node means.

    Algebraically identical to omega'_t H'(t) - omega'_L H'(L) - omega'_R H'(R),
    since (y - mu_parent)^2 - (y - mu_child)^2 factors into the expression below.
    """
    mean = node_means(tree)
    out = np.zeros(n_features, dtype=np.float64)
    for rows, feature, parent, child in _descend(tree, x):
        gap = mean[child] - mean[parent]
        contribution = gap * (2.0 * y[rows] - mean[parent] - mean[child])
        np.add.at(out, feature, contribution)
    return np.asarray(out / x.shape[0], dtype=np.float64)


def ufi(
    tree: DecisionTreeRegressor, x: FloatArray, y: FloatArray, n_features: int
) -> FloatArray:
    return np.asarray(
        mdi_train(tree, n_features) + off_sample_drop(tree, x, y, n_features),
        dtype=np.float64,
    )


def forest_mdi_train(forest: RandomForestRegressor, n_features: int) -> FloatArray:
    per_tree = [mdi_train(tree, n_features) for tree in forest.estimators_]
    return np.asarray(np.mean(per_tree, axis=0), dtype=np.float64)


def forest_mdi_oob(
    forest: RandomForestRegressor, x: FloatArray, y: FloatArray, n_features: int
) -> FloatArray:
    per_tree = [mdi_oob(tree, x, y, n_features) for tree in forest.estimators_]
    return np.asarray(np.mean(per_tree, axis=0), dtype=np.float64)


def forest_ufi_half(
    forest: RandomForestRegressor, x: FloatArray, y: FloatArray, n_features: int
) -> FloatArray:
    per_tree = [ufi(tree, x, y, n_features) for tree in forest.estimators_]
    return np.asarray(np.mean(per_tree, axis=0) / 2.0, dtype=np.float64)


MEASURES = ("mdi_train", "mdi_oob", "ufi_half")
