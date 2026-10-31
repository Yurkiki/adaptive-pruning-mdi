"""The three importance measures.

mdi_train    classic MDI, measured on the data that chose the splits
mdi_oob      the same quantity re-evaluated off-sample          [Li et al. 2019]
ufi          in-bag impurity drop plus the off-sample drop      [Zhou & Hooker 2021]

Each is defined on one tree; `model_*` averages over the trees of a model, so a
forest and a lone tree go through the same code.

ufi carries twice the scale of the other two on signal features, so callers
compare `ufi_half` against the truth. See PROTOCOL, "The measures".
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from tree import Model, TreeView, descend

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


def off_sample_drop(
    tree: TreeView, x: FloatArray, y: FloatArray, n_features: int
) -> FloatArray:
    """The impurity drop measured on held-out rows against the training node means.

    Algebraically identical to omega'_t H'(t) - omega'_L H'(L) - omega'_R H'(R),
    since (y - mu_parent)^2 - (y - mu_child)^2 factors into the expression below.
    """
    mean = node_means(tree)
    out = np.zeros(n_features, dtype=np.float64)
    for rows, feature, parent, child in descend(tree, x):
        gap = mean[child] - mean[parent]
        contribution = gap * (2.0 * y[rows] - mean[parent] - mean[child])
        np.add.at(out, feature, contribution)
    return np.asarray(out / x.shape[0], dtype=np.float64)


def ufi(tree: TreeView, x: FloatArray, y: FloatArray, n_features: int) -> FloatArray:
    return np.asarray(
        mdi_train(tree, n_features) + off_sample_drop(tree, x, y, n_features),
        dtype=np.float64,
    )


def model_mdi_train(model: Model, n_features: int) -> FloatArray:
    per_tree = [mdi_train(tree, n_features) for tree in model.trees]
    return np.asarray(np.mean(per_tree, axis=0), dtype=np.float64)


def model_mdi_oob(
    model: Model, x: FloatArray, y: FloatArray, n_features: int
) -> FloatArray:
    per_tree = [mdi_oob(tree, x, y, n_features) for tree in model.trees]
    return np.asarray(np.mean(per_tree, axis=0), dtype=np.float64)


def model_ufi_half(
    model: Model, x: FloatArray, y: FloatArray, n_features: int
) -> FloatArray:
    per_tree = [ufi(tree, x, y, n_features) for tree in model.trees]
    return np.asarray(np.mean(per_tree, axis=0) / 2.0, dtype=np.float64)


MEASURES = ("mdi_train", "mdi_oob", "ufi_half")


def all_measures(
    model: Model, x_val: FloatArray, y_val: FloatArray, n_features: int
) -> dict[str, FloatArray]:
    return {
        "mdi_train": model_mdi_train(model, n_features),
        "mdi_oob": model_mdi_oob(model, x_val, y_val, n_features),
        "ufi_half": model_ufi_half(model, x_val, y_val, n_features),
    }
