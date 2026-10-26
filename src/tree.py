"""The tree arrays, pulled out of scikit-learn so that pruning can rewrite them."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from sklearn.tree import DecisionTreeRegressor

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
IndexArray = npt.NDArray[np.intp]


@dataclass(frozen=True)
class TreeView:
    """The tree arrays, detached from scikit-learn so that pruning can rewrite them."""

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
        return int((self.feature < 0).sum())

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
