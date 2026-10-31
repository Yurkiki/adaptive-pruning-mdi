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
import prune
import simulate

SCENARIO = simulate.scenario_s1()
FEATURES = SCENARIO.n_features


@pytest.fixture(scope="module")
def fitted() -> tuple[forest.Fitted, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(3)
    x, y = simulate.sample(SCENARIO, 400, rng)
    config = forest.ForestConfig(n_trees=30, n_jobs=1)
    return forest.build_full(x, y, config, 0), x, y


@pytest.fixture(scope="module")
def sklearn_model(fitted) -> object:
    """The same forest as `fitted`, still as scikit-learn objects."""
    _, x, y = fitted
    return forest._fit(x, y, forest.ForestConfig(n_trees=30, n_jobs=1), 0)


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
    wrapped = forest.ensemble(reference)
    assert forest.oob_mse(wrapped, x, y, counts) == pytest.approx(expected, abs=1e-9)


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


def test_mdi_train_matches_sklearn(fitted, sklearn_model) -> None:
    fit, _, _ = fitted
    for view, tree in zip(fit.forest.trees, sklearn_model.estimators_, strict=True):
        expected = tree.tree_.compute_feature_importances(normalize=False)
        assert mdi.mdi_train(view, FEATURES) == pytest.approx(expected, abs=1e-12)


def test_full_tree_mdi_sums_to_inbag_variance(fitted) -> None:
    """The deep-tree bias of Scornet 2020: the measure absorbs the noise variance."""
    fit, _, _ = fitted
    for view in fit.forest.trees:
        total = float(mdi.mdi_train(view, FEATURES).sum())
        assert total == pytest.approx(float(view.impurity[0]), rel=1e-9)


def test_descent_reaches_the_same_leaves_as_sklearn(fitted, sklearn_model) -> None:
    """Our own routing must agree with tree.apply, or every off-sample sum is wrong."""
    fit, x, _ = fitted
    for view, tree in zip(fit.forest.trees[:5], sklearn_model.estimators_[:5], strict=True):
        node = np.zeros(x.shape[0], dtype=np.intp)
        for rows, _, _, child in mdi._descend(view, x):
            node[rows] = child
        assert (node == tree.apply(x)).all()


def test_inbag_identity(fitted) -> None:
    """mdi_oob on the in-bag rows, weighted by bootstrap counts, equals mdi_train."""
    fit, x, y = fitted
    for view, counts in zip(fit.forest.trees, fit.inbag_counts, strict=True):
        weights = counts.astype(np.float64)
        off_sample = mdi.mdi_oob(view, x, y, FEATURES, weights=weights)
        assert off_sample == pytest.approx(mdi.mdi_train(view, FEATURES), abs=1e-10)


def test_ufi_carries_twice_the_scale_of_mdi_oob() -> None:
    """Raw UFI runs at 2x on signal features, so we halve it. See PROTOCOL."""
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
    for view, counts in zip(fit.forest.trees, fit.inbag_counts, strict=True):
        in_bag = mdi.mdi_oob(view, x, ones, FEATURES, weights=counts.astype(np.float64))
        assert in_bag == pytest.approx(np.zeros(FEATURES), abs=1e-12)

    off_sample = mdi.forest_mdi_oob(fit.forest, x[:100], ones[:100], FEATURES)
    assert np.abs(off_sample).max() > 1e-6


# --- alpha-trimming -------------------------------------------------------


@pytest.fixture(scope="module")
def trimmable() -> tuple[tuple[mdi.TreeView, ...], np.ndarray, np.ndarray]:
    """Trees grown at the node size the alpha-trimming paper assumes."""
    rng = np.random.default_rng(12)
    x, y = simulate.sample(SCENARIO, 400, rng)
    config = forest.ForestConfig(n_trees=20, n_jobs=1)
    model = forest._fit(x, y, config, 0, min_samples_leaf=config.alpha_min_samples_leaf)
    return tuple(mdi.TreeView.of(t) for t in model.estimators_), x, y


def test_alpha_zero_is_the_unpruned_tree(trimmable) -> None:
    views, _, _ = trimmable
    for view in views:
        assert prune.alpha_trim(view, 0.0) is view


def test_pruning_is_monotone_in_alpha(trimmable) -> None:
    views, _, _ = trimmable
    sizes = [prune.trim_forest(views, a).mean_leaves
             for a in (0.0, 0.5, 1.0, 2.0, 3.0)]
    assert sizes == sorted(sizes, reverse=True)
    assert sizes[0] > sizes[-1]


def test_alpha_trimming_reaches_small_trees(trimmable) -> None:
    """Cost-complexity never got below ~0.76 of full; alpha-trimming must do better."""
    views, _, _ = trimmable
    full = prune.trim_forest(views, 0.0).mean_leaves
    assert prune.trim_forest(views, 3.0).mean_leaves < 0.25 * full


def test_pruned_trees_stay_well_formed(trimmable) -> None:
    views, x, _ = trimmable
    for view in views[:5]:
        pruned = prune.alpha_trim(view, 1.0)
        internal = pruned.feature >= 0
        assert (pruned.children_left[internal] >= 0).all()
        assert (pruned.children_right[internal] >= 0).all()
        terminal = pruned.feature < 0
        assert (pruned.children_left[terminal] == -1).all()
        assert np.isfinite(pruned.predict(x)).all()


def test_pruning_leaves_no_orphaned_internal_nodes(trimmable) -> None:
    """A split kept low down can be dissolved by a merge higher up.

    Its descendants are then unreachable but would still carry feature >= 0, and
    mdi_train sums over internal nodes rather than walking from the root, so it
    would count them. Caught in the wild: 19 stranded nodes in one tree.
    """
    views, _, _ = trimmable
    for view in views:
        for alpha in (0.25, 0.5, 1.0, 2.0, 3.0):
            pruned = prune.alpha_trim(view, alpha)
            reachable = np.zeros(pruned.feature.size, dtype=bool)
            stack = [0]
            while stack:
                node = stack.pop()
                reachable[node] = True
                if pruned.feature[node] >= 0:
                    stack.append(int(pruned.children_left[node]))
                    stack.append(int(pruned.children_right[node]))
            assert not ((pruned.feature >= 0) & ~reachable).any()


def test_pruned_mdi_train_only_counts_reachable_splits(trimmable) -> None:
    """mdi_train on a pruned tree must match a walk from the root."""
    views, _, _ = trimmable
    for view in views[:6]:
        pruned = prune.alpha_trim(view, 1.5)
        counted = np.flatnonzero(pruned.feature >= 0)
        walked: list[int] = []
        stack = [0]
        while stack:
            node = stack.pop()
            if pruned.feature[node] >= 0:
                walked.append(node)
                stack.append(int(pruned.children_left[node]))
                stack.append(int(pruned.children_right[node]))
        assert sorted(counted.tolist()) == sorted(walked)


def test_treeview_predict_matches_sklearn(trimmable) -> None:
    views, x, y = trimmable
    config = forest.ForestConfig(n_trees=20, n_jobs=1)
    model = forest._fit(x, y, config, 0, min_samples_leaf=config.alpha_min_samples_leaf)
    assert prune.trim_forest(views, 0.0).predict(x) == pytest.approx(
        model.predict(x), abs=1e-12
    )


def test_pure_leaves_make_alpha_trimming_inert() -> None:
    """Documents why the base forest is not grown to purity.

    With zero within-leaf variance the stump log-likelihood diverges, the merge
    test can never fire, and alpha-trimming silently does nothing.
    """
    rng = np.random.default_rng(13)
    x, y = simulate.sample(SCENARIO, 300, rng)
    config = forest.ForestConfig(n_trees=10, n_jobs=1)
    pure = forest._fit(x, y, config, 0, min_samples_leaf=1)
    views = tuple(mdi.TreeView.of(t) for t in pure.estimators_)
    assert prune.trim_forest(views, 3.0).mean_leaves == prune.trim_forest(views, 0.0).mean_leaves


def test_alpha_path_covers_the_grid_from_one_fit(trimmable) -> None:
    _, x, y = trimmable
    config = forest.ForestConfig(n_trees=10, n_jobs=1)
    path = forest.alpha_path(x, y, config, 0)
    assert path.alphas == config.alpha_grid
    assert all(fit.name == "alpha" for fit in path.fits)
    assert path.fits[0].selected_param == 0.0


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


def test_run_one_emits_every_arm_and_the_whole_pruning_path() -> None:
    config = experiment.RunConfig(
        n_reps=1, forest=forest.ForestConfig(n_trees=10, n_jobs=1), n_test=200
    )
    rows, path_rows = experiment.run_one(SCENARIO, 120, 0, config)

    split_forests = {"full", "global", "ccp", "alpha0", "alpha", "oracle"}
    expected = (len(split_forests) * len(mdi.MEASURES) + 1) * FEATURES
    assert len(rows) == expected
    assert {r["forest"] for r in rows} == split_forests | {"standard"}
    assert set(rows[0]) == set(experiment.COLUMNS)
    assert not any(np.isnan(float(r["value"])) for r in rows)

    levels = len(config.forest.ccp_fractions) + len(config.forest.alpha_grid)
    assert len(path_rows) == levels * len(mdi.MEASURES)
    assert set(experiment.PATH_COLUMNS) == set(path_rows[0])
    assert {r["method"] for r in path_rows} == {"ccp", "alpha"}


def test_oracle_never_loses_to_the_oob_pruning_level() -> None:
    """The oracle picks on importance error, so by construction it cannot be worse."""
    config = experiment.RunConfig(
        n_reps=1, forest=forest.ForestConfig(n_trees=20, n_jobs=1), n_test=200
    )
    rows, _ = experiment.run_one(SCENARIO, 200, 0, config)
    frame = pd.DataFrame(rows)
    truth = SCENARIO.true_importance

    def error(name: str) -> float:
        values = (
            frame[(frame["forest"] == name) & (frame["measure"] == experiment.ORACLE_MEASURE)]
            .sort_values("feature")["value"]
            .to_numpy(dtype=np.float64)
        )
        return float(((values - truth) ** 2).sum())

    assert error("oracle") <= error("alpha") + 1e-12


def test_standard_arm_uses_every_point() -> None:
    config = experiment.RunConfig(
        n_reps=1, forest=forest.ForestConfig(n_trees=10, n_jobs=1), n_test=200
    )
    rows, _ = experiment.run_one(SCENARIO, 300, 0, config)
    frame = pd.DataFrame(rows)
    standard = frame[frame["forest"] == "standard"]["mean_leaves"].iloc[0]
    split = frame[frame["forest"] == "full"]["mean_leaves"].iloc[0]
    assert standard > split
    assert set(frame[frame["forest"] == "standard"]["measure"]) == {"mdi_train"}


def test_cost_complexity_path_is_monotone() -> None:
    rng = np.random.default_rng(9)
    x, y = simulate.sample(SCENARIO, 300, rng)
    config = forest.ForestConfig(n_trees=10, n_jobs=1)
    path = forest.pruning_path(x, y, config, 0)
    sizes = [fit.mean_leaves for fit in path.fits]
    assert sizes == sorted(sizes, reverse=True)
    assert path.fits[0].selected_param == 0.0


def test_runs_are_reproducible() -> None:
    config = experiment.RunConfig(
        n_reps=1, forest=forest.ForestConfig(n_trees=10, n_jobs=1), n_test=200
    )
    first, _ = experiment.run_one(SCENARIO, 120, 0, config)
    second, _ = experiment.run_one(SCENARIO, 120, 0, config)
    assert [r["value"] for r in first] == [r["value"] for r in second]


def test_summary_reports_standard_errors_and_relative_loss() -> None:
    config = experiment.RunConfig(
        n_reps=1, forest=forest.ForestConfig(n_trees=10, n_jobs=1), n_test=200
    )
    rows = [r for rep in range(3) for r in experiment.run_one(SCENARIO, 120, rep, config)[0]]
    table = analyse.summary(pd.DataFrame(rows))
    for column in ("total_mse_se", "noise_mass_se", "relative_loss", "excess_risk"):
        assert column in table
    assert (table["total_mse_se"] >= 0).all()
