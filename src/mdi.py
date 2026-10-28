"""The importance measures.

mdi_train    classic MDI, measured on the data that chose the splits
mdi_oob      the same quantity re-evaluated off-sample          [Li et al. 2019]

Each is defined on one tree.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from tree import TreeView, descend

FloatArray = npt.NDArray[np.float64]


def node_means(tree: TreeView) -> FloatArray:
    """Training mean of y at every node (bootstrap weights included for a forest)."""
    return tree.value


def mdi_train(tree: TreeView, n_features: int) -> FloatArray:
    """Unnormalised classic MDI. Equals compute_feature_importances(normalize=False)."""
    nodes = np.flatnonzero(tree.feature >= 0)
    left = tree.children_left[nodes]
    right = tree.children_right[nodes]
    weight = tree.weighted_n_node_samples
    drop = (
        weight[nodes] * tree.impurity[nodes]
        - weight[left] * tree.impurity[left]
        - weight[right] * tree.impurity[right]
    )
    out = np.zeros(n_features, dtype=np.float64)
    np.add.at(out, tree.feature[nodes], drop)
    return np.asarray(out / weight[0], dtype=np.float64)


def mdi_oob(
    tree: TreeView,
    x: FloatArray,
    y: FloatArray,
    n_features: int,
    weights: FloatArray | None = None,
) -> FloatArray:
    """Mean of (mu_child - mu_parent) * y over the evaluation rows reaching each split.

    Evaluated on the training rows (with their bootstrap counts as `weights` for a
    forest tree), this reproduces mdi_train exactly.
    """
    row_weight = np.ones(x.shape[0]) if weights is None else weights.astype(np.float64)
    mean = node_means(tree)
    out = np.zeros(n_features, dtype=np.float64)
    for rows, feature, parent, child in descend(tree, x):
        contribution = row_weight[rows] * (mean[child] - mean[parent]) * y[rows]
        np.add.at(out, feature, contribution)
    return np.asarray(out / row_weight.sum(), dtype=np.float64)
