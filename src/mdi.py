"""The three importance measures.

mdi_train    classic MDI, measured on the data that chose the splits
mdi_oob      the same quantity re-evaluated off-sample          [Li et al. 2019]
ufi          in-bag impurity drop plus the off-sample drop      [Zhou & Hooker 2021]

ufi carries twice the scale of the other two on signal features, so callers
compare `ufi_half` against the truth. See PROTOCOL, "The measures".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Protocol

import numpy as np
import numpy.typing as npt
from sklearn.tree import DecisionTreeRegressor

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
IndexArray = npt.NDArray[np.intp]

Transition = tuple[IndexArray, IndexArray, IndexArray, IndexArray]


@dataclass(frozen=True)
class TreeView:
    """The tree arrays, detached from scikit-learn so that pruning can rewrite them.

    Pruning marks a node terminal by setting its feature to -2 and its children to
    -1. The orphaned subtree stays in the arrays but is unreachable, so nothing
    needs renumbering and every routine here keeps working unchanged.
    """

    children_left: IntArray
    children_right: IntArray
    feature: IntArray
    threshold: FloatArray
    value: FloatArray
    weighted_n_node_samples: FloatArray
    impurity: FloatArray

    @classmethod
    def of(cls, tree: DecisionTreeRegressor) -> TreeView:
        inner = tree.tree_
        return cls(
            children_left=np.array(inner.children_left, dtype=np.int64),
            children_right=np.array(inner.children_right, dtype=np.int64),
            feature=np.array(inner.feature, dtype=np.int64),
            threshold=np.array(inner.threshold, dtype=np.float64),
            value=np.array(inner.value[:, 0, 0], dtype=np.float64),
            weighted_n_node_samples=np.array(
                inner.weighted_n_node_samples, dtype=np.float64
            ),
            impurity=np.array(inner.impurity, dtype=np.float64),
        )

    @property
    def n_leaves(self) -> int:
        return int(self.reachable_leaves().sum())

    def reachable_leaves(self) -> npt.NDArray[np.bool_]:
        terminal = np.zeros(self.feature.size, dtype=bool)
        stack = [0]
        while stack:
            node = stack.pop()
            if self.feature[node] < 0:
                terminal[node] = True
            else:
                stack.extend((self.children_left[node], self.children_right[node]))
        return terminal

    def depth(self) -> int:
        deepest = 0
        stack = [(0, 0)]
        while stack:
            node, level = stack.pop()
            if self.feature[node] < 0:
                deepest = max(deepest, level)
            else:
                stack.append((int(self.children_left[node]), level + 1))
                stack.append((int(self.children_right[node]), level + 1))
        return deepest

    def predict(self, x: FloatArray) -> FloatArray:
        node = np.zeros(x.shape[0], dtype=np.intp)
        for rows, _, _, child in _descend(self, x):
            node[rows] = child
        return self.value[node]


class ForestLike(Protocol):
    @property
    def trees(self) -> tuple[TreeView, ...]: ...

    def predict(self, x: FloatArray) -> FloatArray: ...


def node_means(tree: TreeView) -> FloatArray:
    """In-bag mean of y at every node, bootstrap weights included."""
    return tree.value


def mdi_train(tree: TreeView, n_features: int) -> FloatArray:
    """Unnormalised classic MDI. Equals compute_feature_importances(normalize=False)."""
    inner = tree
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


def _descend(tree: TreeView, x: FloatArray) -> Iterator[Transition]:
    """Walk every row down the tree one level at a time.

    Yields (rows, split_feature, parent_node, child_node) for the rows still moving.
    """
    inner = tree
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
    tree: TreeView,
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
    tree: TreeView, x: FloatArray, y: FloatArray, n_features: int
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


def forest_mdi_train(forest: ForestLike, n_features: int) -> FloatArray:
    per_tree = [mdi_train(tree, n_features) for tree in forest.trees]
    return np.asarray(np.mean(per_tree, axis=0), dtype=np.float64)


def forest_mdi_oob(
    forest: ForestLike, x: FloatArray, y: FloatArray, n_features: int
) -> FloatArray:
    per_tree = [mdi_oob(tree, x, y, n_features) for tree in forest.trees]
    return np.asarray(np.mean(per_tree, axis=0), dtype=np.float64)


def forest_ufi_half(
    forest: ForestLike, x: FloatArray, y: FloatArray, n_features: int
) -> FloatArray:
    per_tree = [ufi(tree, x, y, n_features) for tree in forest.trees]
    return np.asarray(np.mean(per_tree, axis=0) / 2.0, dtype=np.float64)


MEASURES = ("mdi_train", "mdi_oob", "ufi_half")
