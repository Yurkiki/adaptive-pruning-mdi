"""Tree arrays detached from scikit-learn, and a model as a tuple of them.

A single tree is a model of one tree; a forest is a model of many. Every measure
is defined per tree and averaged over the model, so nothing downstream needs to
know which of the two it holds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

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
        for rows, _, _, child in descend(self, x):
            node[rows] = child
        return self.value[node]


def descend(tree: TreeView, x: FloatArray) -> Iterator[Transition]:
    """Walk every row down the tree one level at a time.

    Yields (rows, split_feature, parent_node, child_node) for the rows still moving.
    """
    node = np.zeros(x.shape[0], dtype=np.intp)
    while True:
        rows = np.flatnonzero(tree.feature[node] >= 0)
        if rows.size == 0:
            return
        parent = node[rows]
        feature = tree.feature[parent]
        goes_left = x[rows, feature] <= tree.threshold[parent]
        child = np.where(
            goes_left, tree.children_left[parent], tree.children_right[parent]
        ).astype(np.intp)
        yield rows, feature, parent, child
        node[rows] = child


@dataclass(frozen=True)
class Model:
    """One tree or a forest: predictions and measures average over `trees`."""

    trees: tuple[TreeView, ...]

    @property
    def n_trees(self) -> int:
        return len(self.trees)

    def predict(self, x: FloatArray) -> FloatArray:
        return np.asarray(
            np.mean([tree.predict(x) for tree in self.trees], axis=0), dtype=np.float64
        )

    @property
    def mean_leaves(self) -> float:
        return float(np.mean([tree.n_leaves for tree in self.trees]))

    @property
    def mean_depth(self) -> float:
        return float(np.mean([tree.depth() for tree in self.trees]))
