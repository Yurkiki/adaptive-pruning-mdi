"""The tests that guard the ground truth and the importance formulas.

test_inbag_identity is the critical one: it validates the node bookkeeping, the
in-bag means and the row routing in a single exact assertion.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

import analyse
import experiment
import forest
import mdi
import simulate

SCENARIO = simulate.scenario_s1()
FEATURES = SCENARIO.n_features


@pytest.fixture(scope="module")
def fitted() -> tuple[forest.Fitted, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(3)
    x, y = simulate.sample(SCENARIO, 400, rng)
    config = forest.ForestConfig(n_trees=30, n_jobs=1)
    return forest.build_full(x, y, config, 0), x, y


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


def test_rescaled_component_hits_target_variance() -> None:
    scaled = simulate.rescaled(simulate.SINE, 0.004)
    estimate, stderr = simulate.monte_carlo_variance(
        scaled, 1_000_000, np.random.default_rng(4)
    )
    assert scaled.variance == 0.004
    assert abs(estimate - 0.004) < 4.0 * stderr


# --- bootstrap bookkeeping -------------------------------------------------


def test_inbag_counts_sum_to_n(fitted) -> None:
    fit, x, _ = fitted
    assert (fit.inbag_counts.sum(axis=1) == x.shape[0]).all()


def test_oob_mask_agrees_with_counts(fitted) -> None:
    fit, _, _ = fitted
    assert (fit.oob_mask == (fit.inbag_counts == 0)).all()


def test_oob_fraction_is_one_over_e(fitted) -> None:
    fit, _, _ = fitted
    assert float(fit.oob_mask.mean()) == pytest.approx(0.368, abs=0.02)


def test_oob_mse_matches_sklearn(fitted) -> None:
    _, x, y = fitted
    reference = RandomForestRegressor(
        n_estimators=30, max_features=1.0 / 3.0, bootstrap=True,
        random_state=0, oob_score=True, n_jobs=1,
    ).fit(x, y)
    counts = forest.inbag_counts(reference, x.shape[0])
    expected = float(np.mean((y - reference.oob_prediction_) ** 2))
    assert forest.oob_mse(reference, x, y, counts) == pytest.approx(expected, abs=1e-9)


# --- forest construction ---------------------------------------------------


def test_zero_pruning_reproduces_full_forest(fitted) -> None:
    _, x, y = fitted
    config = forest.ForestConfig(n_trees=30, n_jobs=1)
    full = forest._fit(x, y, config, 0)
    unpruned = forest._fit(x, y, config, 0, ccp_alpha=0.0)
    assert [t.tree_.node_count for t in full.estimators_] == [
        t.tree_.node_count for t in unpruned.estimators_
    ]


def test_pruning_actually_prunes(fitted) -> None:
    _, x, y = fitted
    config = forest.ForestConfig(n_trees=30, n_jobs=1)
    full = forest._fit(x, y, config, 0)
    pruned = forest._fit(x, y, config, 0, ccp_alpha=0.05 * float(np.var(y)))
    assert sum(t.tree_.node_count for t in pruned.estimators_) < sum(
        t.tree_.node_count for t in full.estimators_
    )


def test_grids_must_contain_the_unpruned_case() -> None:
    with pytest.raises(ValueError):
        forest.ForestConfig(min_samples_leaf_grid=(2, 5))
    with pytest.raises(ValueError):
        forest.ForestConfig(ccp_fractions=(1e-3, 1e-2))


# --- the importance measures ----------------------------------------------


def test_mdi_train_matches_sklearn(fitted) -> None:
    fit, _, _ = fitted
    for tree in fit.forest.estimators_:
        expected = tree.tree_.compute_feature_importances(normalize=False)
        assert mdi.mdi_train(tree, FEATURES) == pytest.approx(expected, abs=1e-12)


def test_full_tree_mdi_sums_to_inbag_variance(fitted) -> None:
    """The deep-tree bias of Scornet 2020: the measure absorbs the noise variance."""
    fit, _, _ = fitted
    for tree in fit.forest.estimators_:
        total = float(mdi.mdi_train(tree, FEATURES).sum())
        assert total == pytest.approx(float(tree.tree_.impurity[0]), rel=1e-9)


def test_descent_reaches_the_same_leaves_as_sklearn(fitted) -> None:
    """Our own routing must agree with tree.apply, or every off-sample sum is wrong."""
    fit, x, _ = fitted
    for tree in fit.forest.estimators_[:5]:
        node = np.zeros(x.shape[0], dtype=np.intp)
        for rows, _, _, child in mdi._descend(tree, x):
            node[rows] = child
        assert (node == tree.apply(x)).all()


def test_inbag_identity(fitted) -> None:
    """mdi_oob on the in-bag rows, weighted by bootstrap counts, equals mdi_train."""
    fit, x, y = fitted
    for tree, counts in zip(fit.forest.estimators_, fit.inbag_counts, strict=True):
        weights = counts.astype(np.float64)
        off_sample = mdi.mdi_oob(tree, x, y, FEATURES, weights=weights)
        assert off_sample == pytest.approx(mdi.mdi_train(tree, FEATURES), abs=1e-10)


def test_ufi_carries_twice_the_scale_of_mdi_oob() -> None:
    """SPEC 5.3: raw UFI runs at 2x on signal features, so we halve it."""
    scenario = simulate.scenario_s1(snr=50.0)
    rng = np.random.default_rng(7)
    x, y = simulate.sample(scenario, 4000, rng)
    x_val, y_val = simulate.sample(scenario, 4000, rng)
    centre = float(y.mean())
    fit = forest.build_full(
        x, y - centre, forest.ForestConfig(n_trees=100, n_jobs=1), 0
    )
    signal = ~scenario.is_noise
    oob = mdi.forest_mdi_oob(fit.forest, x_val, y_val - centre, FEATURES)
    half = mdi.forest_ufi_half(fit.forest, x_val, y_val - centre, FEATURES)
    assert half[signal] == pytest.approx(oob[signal], rel=0.1)


def test_out_of_sample_measures_give_noise_features_nothing() -> None:
    scenario = simulate.scenario_s1(snr=4.0)
    rng = np.random.default_rng(8)
    x, y = simulate.sample(scenario, 2000, rng)
    x_val, y_val = simulate.sample(scenario, 2000, rng)
    centre = float(y.mean())
    fit = forest.build_full(
        x, y - centre, forest.ForestConfig(n_trees=100, n_jobs=1), 0
    )
    noise = scenario.is_noise
    explainable = float(scenario.true_importance.sum())
    for measure in (mdi.forest_mdi_oob, mdi.forest_ufi_half):
        values = measure(fit.forest, x_val, y_val - centre, FEATURES)
        assert abs(float(values[noise].sum())) < 0.05 * explainable
    in_bag = mdi.forest_mdi_train(fit.forest, FEATURES)
    assert float(in_bag[noise].sum()) > 0.05 * explainable


def test_off_sample_measure_is_not_shift_invariant(fitted) -> None:
    """Li et al. 2019: sum f_T,k over the in-bag rows is zero, off-sample it is not.

    Shifting y by c moves the measure by exactly c times that sum, which is why the
    off-sample measures need y centred and the in-bag one does not.
    """
    fit, x, y = fitted
    ones = np.ones_like(y)
    for tree, counts in zip(fit.forest.estimators_, fit.inbag_counts, strict=True):
        in_bag = mdi.mdi_oob(tree, x, ones, FEATURES, weights=counts.astype(np.float64))
        assert in_bag == pytest.approx(np.zeros(FEATURES), abs=1e-12)

    off_sample = mdi.forest_mdi_oob(fit.forest, x[:100], ones[:100], FEATURES)
    assert np.abs(off_sample).max() > 1e-6


# --- metrics ---------------------------------------------------------------


def test_mse_decomposes_into_bias_squared_plus_variance() -> None:
    rng = np.random.default_rng(5)
    frame = pd.DataFrame(
        {
            "scenario": "S1", "n": 100, "forest": "full", "measure": "mdi_oob",
            "rep": np.repeat(np.arange(20), 3),
            "feature": np.tile(np.arange(3), 20),
            "value": rng.normal(size=60),
            "true_importance": np.tile([0.5, 0.2, 0.0], 20),
            "is_noise": np.tile([False, False, True], 20),
        }
    )
    table = analyse.per_feature(frame)
    assert table["mse"].to_numpy() == pytest.approx(
        (table["bias"] ** 2 + table["variance"]).to_numpy(), abs=1e-12
    )


def test_auc_endpoints() -> None:
    noise = np.array([False, False, True, True])
    assert analyse.auc_signal_vs_noise(np.array([3.0, 4.0, 1.0, 2.0]), noise) == 1.0
    assert analyse.auc_signal_vs_noise(np.array([1.0, 2.0, 3.0, 4.0]), noise) == 0.0
    assert analyse.auc_signal_vs_noise(np.ones(4), noise) == 0.5


def test_holm_is_monotone_and_at_least_raw() -> None:
    raw = np.array([0.001, 0.02, 0.3, 0.04])
    adjusted = analyse.holm(raw)
    assert (adjusted >= raw).all()
    order = np.argsort(raw)
    assert (np.diff(adjusted[order]) >= -1e-12).all()


# --- end to end ------------------------------------------------------------


def test_run_one_returns_one_row_per_arm_and_feature() -> None:
    config = experiment.RunConfig(
        n_reps=1, forest=forest.ForestConfig(n_trees=10, n_jobs=1), n_test=200
    )
    rows = experiment.run_one(SCENARIO, 120, 0, config)
    expected = len(forest.BUILDERS) * len(mdi.MEASURES) * FEATURES
    assert len(rows) == expected
    assert set(rows[0]) == set(experiment.COLUMNS)
    assert not any(np.isnan(float(r["value"])) for r in rows)


def test_runs_are_reproducible() -> None:
    config = experiment.RunConfig(
        n_reps=1, forest=forest.ForestConfig(n_trees=10, n_jobs=1), n_test=200
    )
    first = experiment.run_one(SCENARIO, 120, 0, config)
    second = experiment.run_one(SCENARIO, 120, 0, config)
    assert [r["value"] for r in first] == [r["value"] for r in second]
