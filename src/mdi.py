"""The classic mean decrease in impurity, measured on the training data.

Defined on one tree: the weighted impurity drop at every split on feature j,
summed over those splits and divided by the weight at the root. Unnormalised, so
it can be compared with Var(f_j(X_j)) instead of summing to one.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from tree import TreeView

FloatArray = npt.NDArray[np.float64]


def node_means(tree: TreeView) -> FloatArray:
    """Training mean of y at every node, bootstrap weights included."""
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
