"""Tests for the ground truth and the importance formulas."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.tree import DecisionTreeRegressor

import mdi
import simulate
from tree import TreeView, descend

SCENARIO = simulate.scenario_s1()
FEATURES = SCENARIO.n_features


@pytest.fixture(scope="module")
def data() -> tuple[np.ndarray, np.ndarray]:
    return simulate.sample(SCENARIO, 400, np.random.default_rng(3))


@pytest.fixture(scope="module")
def full_tree(data) -> tuple[TreeView, DecisionTreeRegressor]:
    """The same unpruned tree, as our view and as the scikit-learn object."""
    x, y = data
    reference = DecisionTreeRegressor(max_features=1.0, random_state=0).fit(x, y)
    return TreeView.of(reference), reference


# --- ground truth ----------------------------------------------------------


@pytest.mark.parametrize("component", simulate.SIGNAL_LIBRARY, ids=lambda c: c.name)
def test_analytic_variance_matches_monte_carlo(component: simulate.Component) -> None:
    estimate, stderr = simulate.monte_carlo_variance(
        component, 1_000_000, np.random.default_rng(0)
    )
    assert abs(estimate - component.variance) < 4.0 * stderr


@pytest.mark.parametrize("component", simulate.SIGNAL_LIBRARY, ids=lambda c: c.name)
def test_components_are_centred(component: simulate.Component) -> None:
    values = component.func(np.random.default_rng(1).random(1_000_000))
    assert abs(values.mean()) < 4.0 * values.std() / np.sqrt(values.size)


def test_variance_decomposition() -> None:
    _, y = simulate.sample(SCENARIO, 1_000_000, np.random.default_rng(2))
    expected = SCENARIO.true_importance.sum() + SCENARIO.sigma**2
    assert float(y.var()) == pytest.approx(expected, rel=0.01)


# --- model construction ----------------------------------------------------


def test_treeview_predict_matches_sklearn(data, full_tree) -> None:
    x, _ = data
    view, reference = full_tree
    assert view.predict(x) == pytest.approx(reference.predict(x), abs=1e-12)


# --- the importance measures ----------------------------------------------


def test_mdi_train_matches_sklearn(full_tree) -> None:
    view, reference = full_tree
    expected = reference.tree_.compute_feature_importances(normalize=False)
    assert mdi.mdi_train(view, FEATURES) == pytest.approx(expected, abs=1e-12)


def test_full_tree_mdi_sums_to_training_variance(data, full_tree) -> None:
    """The deep-tree bias of Scornet 2020: the measure absorbs the noise variance."""
    _, y = data
    view, _ = full_tree
    total = float(mdi.mdi_train(view, FEATURES).sum())
    assert total == pytest.approx(float(np.var(y)), rel=1e-9)


def test_descent_reaches_the_same_leaves_as_sklearn(data, full_tree) -> None:
    """Our own routing must agree with tree.apply, or every off-sample sum is wrong."""
    x, _ = data
    view, reference = full_tree
    node = np.zeros(x.shape[0], dtype=np.intp)
    for rows, _, _, child in descend(view, x):
        node[rows] = child
    assert (node == reference.apply(x)).all()
