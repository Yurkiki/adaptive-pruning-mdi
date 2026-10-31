"""Alpha-trimming: accumulated information pruning [Surjanovic, Henrey & Loughin 2024].

Algorithm 1 of the paper, transcribed from section 3.1. A fully grown tree is
traversed from the bottom up, in reverse construction order. At each internal
node the single-mean model for that node is weighed against the models held by
its two children, and the children are merged into the node when

    I_N + alpha * P_0,|N|  <=  I_NL + I_NR + alpha * P_1,|N|            (5)

with the modified BIC penalties of section 3.3, P_0,n = 2 log n for the root
model (mu, sigma^2) and P_1,n = 5 log n for the stump model (mu1, mu2, sigma^2,
split point). If the split survives, the node carries the information beneath it
upward as I_N <- I_NL + I_NR + alpha * (P_1 - P_0), which is what makes the rule
*accumulated*: a split kept here can still be dissolved higher up along with its
whole branch.

alpha = 0 performs no pruning and returns the standard random forest; alpha = 1
prunes each tree to be a good standalone predictor. Pruning is post hoc, so the
whole alpha grid costs one forest fit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from mdi import TreeView

FloatArray = npt.NDArray[np.float64]

VARIANCE_FLOOR = 1e-12


def _penalties(count: float) -> tuple[float, float]:
    """P_0,n = 2 log n and P_1,n = 5 log n (section 3.3)."""
    log_n = float(np.log(max(count, 2.0)))
    return 2.0 * log_n, 5.0 * log_n


def _information(count: float, sum_squares: float, variance: float) -> float:
    """|N| log(2 pi sigma^2) + (1 / sigma^2) * sum of squared errors.

    Equation (4) for a parent, where sigma^2 is that node's own variance and the
    trailing term collapses to |N|; and the child formula, where sigma^2 is shared
    between the two children so the term does not collapse.
    """
    safe = max(variance, VARIANCE_FLOOR)
    return count * float(np.log(2.0 * np.pi * safe)) + sum_squares / safe


def alpha_trim(tree: TreeView, alpha: float) -> TreeView:
    """Return a pruned copy of `tree`. alpha = 0 returns it unchanged."""
    if alpha == 0.0:
        return tree

    children_left = tree.children_left.copy()
    children_right = tree.children_right.copy()
    feature = tree.feature.copy()
    weight = tree.weighted_n_node_samples
    node_sse = weight * tree.impurity

    n_nodes = feature.size
    information = np.zeros(n_nodes, dtype=np.float64)
    subtree_sse = node_sse.copy()
    is_terminal = tree.feature < 0

    # reverse construction order: scikit-learn numbers a node before its subtree,
    # so descending index order visits every child before its parent
    for node in range(n_nodes - 1, -1, -1):
        if tree.feature[node] < 0:
            continue
        left = int(children_left[node])
        right = int(children_right[node])
        count = float(weight[node])

        parent_information = _information(count, count * tree.impurity[node],
                                          float(tree.impurity[node]))

        if is_terminal[left] and is_terminal[right]:
            shared = (node_sse[left] + node_sse[right]) / count
            left_information = _information(float(weight[left]), float(node_sse[left]), shared)
            right_information = _information(float(weight[right]), float(node_sse[right]), shared)
        elif is_terminal[left]:
            shared = (node_sse[left] + subtree_sse[right]) / count
            left_information = _information(float(weight[left]), float(node_sse[left]), shared)
            right_information = information[right]
        elif is_terminal[right]:
            shared = (subtree_sse[left] + node_sse[right]) / count
            right_information = _information(float(weight[right]), float(node_sse[right]), shared)
            left_information = information[left]
        else:
            left_information = information[left]
            right_information = information[right]

        root_penalty, stump_penalty = _penalties(count)
        merge = (
            parent_information + alpha * root_penalty
            <= left_information + right_information + alpha * stump_penalty
        )

        if merge:
            children_left[node] = -1
            children_right[node] = -1
            feature[node] = -2
            is_terminal[node] = True
            subtree_sse[node] = node_sse[node]
        else:
            information[node] = (
                left_information
                + right_information
                + alpha * (stump_penalty - root_penalty)
            )
            subtree_sse[node] = subtree_sse[left] + subtree_sse[right]

    # A split kept at one level can still be dissolved by a merge higher up, which
    # strands its descendants: unreachable, but still carrying feature >= 0. Anything
    # that sums over internal nodes rather than walking from the root would count
    # them, so detach them explicitly.
    reachable = np.zeros(n_nodes, dtype=bool)
    stack = [0]
    while stack:
        node = stack.pop()
        reachable[node] = True
        if feature[node] >= 0:
            stack.append(int(children_left[node]))
            stack.append(int(children_right[node]))
    orphaned = ~reachable
    feature = feature.copy()
    feature[orphaned] = -2
    children_left[orphaned] = -1
    children_right[orphaned] = -1

    return TreeView(
        children_left=children_left,
        children_right=children_right,
        feature=feature,
        threshold=tree.threshold,
        value=tree.value,
        weighted_n_node_samples=weight,
        impurity=tree.impurity,
    )


@dataclass(frozen=True)
class PrunedForest:
    """A forest of alpha-trimmed trees, exposing what the measures and OOB need."""

    trees: tuple[TreeView, ...]
    alpha: float

    def predict(self, x: FloatArray) -> FloatArray:
        return np.asarray(
            np.mean([tree.predict(x) for tree in self.trees], axis=0), dtype=np.float64
        )

    def predict_per_tree(self, x: FloatArray) -> FloatArray:
        return np.asarray([tree.predict(x) for tree in self.trees], dtype=np.float64)

    @property
    def mean_leaves(self) -> float:
        return float(np.mean([tree.n_leaves for tree in self.trees]))

    @property
    def mean_depth(self) -> float:
        return float(np.mean([tree.depth() for tree in self.trees]))


def trim_forest(trees: tuple[TreeView, ...], alpha: float) -> PrunedForest:
    return PrunedForest(tuple(alpha_trim(tree, alpha) for tree in trees), alpha)


ALPHA_GRID: tuple[float, ...] = tuple(round(0.1 * k, 1) for k in range(31))
