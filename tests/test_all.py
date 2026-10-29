"""Tests for the ground truth and the importance formulas."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.tree import DecisionTreeRegressor

import mdi
import models
import simulate
from tree import TreeView, descend

SCENARIO = simulate.scenario_s1()
FEATURES = SCENARIO.n_features
FOREST = models.ModelConfig(n_trees=25, n_jobs=1)


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


def test_zero_pruning_reproduces_full_tree(data) -> None:
    x, y = data
    full = models.grow(x, y, FOREST, 0).model
    unpruned = models.grow(x, y, FOREST, 0, ccp_alpha=0.0).model
    assert full.mean_leaves == unpruned.mean_leaves


# --- cross-validated tuning ------------------------------------------------


def test_selection_takes_the_lowest_oob_error(data) -> None:
    x, y = data
    path = models.pruning_path(models.GLOBAL, x, y, FOREST, 0)
    chosen = models.select_by_oob(path)
    assert chosen.oob_mse == min(fit.oob_mse for fit in path.fits)


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


def test_training_identity(data, full_tree) -> None:
    """mdi_oob evaluated on the training rows equals mdi_train, exactly."""
    x, y = data
    view, _ = full_tree
    assert mdi.mdi_oob(view, x, y, FEATURES) == pytest.approx(
        mdi.mdi_train(view, FEATURES), abs=1e-10
    )


def test_off_sample_measure_is_not_shift_invariant(data, full_tree) -> None:
    """Li et al. 2019: sum f_T,k over the training rows is zero, off-sample it is not.

    Shifting y by c moves the measure by exactly c times that sum, which is why the
    off-sample measures need y centred and the in-sample one does not.
    """
    x, y = data
    view, _ = full_tree
    ones = np.ones_like(y)
    assert mdi.mdi_oob(view, x, ones, FEATURES) == pytest.approx(
        np.zeros(FEATURES), abs=1e-12
    )
    held = simulate.sample(SCENARIO, 100, np.random.default_rng(11))[0]
    off_sample = mdi.mdi_oob(view, held, np.ones(100), FEATURES)
    assert np.abs(off_sample).max() > 1e-6


def test_model_measure_averages_its_trees(data) -> None:
    x, y = data
    config = models.ModelConfig(n_trees=5, n_jobs=1)
    model = models.grow(x, y, config, 0).model
    per_tree = [mdi.mdi_train(t, FEATURES) for t in model.trees]
    assert mdi.model_mdi_train(model, FEATURES) == pytest.approx(
        np.mean(per_tree, axis=0), abs=1e-12
    )
