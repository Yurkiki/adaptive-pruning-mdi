"""The tests that guard the ground truth and the importance formulas.

test_training_identity is the critical one: it validates the node bookkeeping,
the training means and the row routing in a single exact assertion.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor

import analyse
import experiment
import mdi
import models
import prune
import simulate
from tree import Model, TreeView, descend

SCENARIO = simulate.scenario_s1()
FEATURES = SCENARIO.n_features
TREE = models.ModelConfig()


@pytest.fixture(scope="module")
def data() -> tuple[np.ndarray, np.ndarray]:
    return simulate.sample(SCENARIO, 400, np.random.default_rng(3))


@pytest.fixture(scope="module")
def full_tree(data) -> tuple[TreeView, DecisionTreeRegressor]:
    """The same unpruned tree, as our view and as the scikit-learn object."""
    x, y = data
    model = models.grow(x, y, TREE, 0)
    reference = DecisionTreeRegressor(max_features=1.0, random_state=0).fit(x, y)
    return model.trees[0], reference


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


# --- model construction ----------------------------------------------------


def test_default_model_is_one_cart_tree(data) -> None:
    x, y = data
    model = models.grow(x, y, TREE, 0)
    assert model.n_trees == 1
    assert TREE.features_per_split == 1.0


def test_forest_kind_grows_many_trees(data) -> None:
    x, y = data
    config = models.ModelConfig(kind="forest", n_trees=7, n_jobs=1)
    model = models.grow(x, y, config, 0)
    assert model.n_trees == 7
    assert config.features_per_split == pytest.approx(1.0 / 3.0)
    assert mdi.model_mdi_train(model, FEATURES).shape == (FEATURES,)


def test_unknown_kind_is_rejected() -> None:
    with pytest.raises(ValueError):
        models.ModelConfig(kind="boosting")


def test_treeview_predict_matches_sklearn(data, full_tree) -> None:
    x, _ = data
    view, reference = full_tree
    assert view.predict(x) == pytest.approx(reference.predict(x), abs=1e-12)


def test_zero_pruning_reproduces_full_tree(data) -> None:
    x, y = data
    full = models.grow(x, y, TREE, 0)
    unpruned = models.grow(x, y, TREE, 0, ccp_alpha=0.0)
    assert full.mean_leaves == unpruned.mean_leaves


def test_cost_complexity_path_is_monotone(data) -> None:
    x, y = data
    path = models.pruning_path(models.CCP, x, y, TREE, 0)
    sizes = [fit.mean_leaves for fit in path.fits]
    assert sizes == sorted(sizes, reverse=True)
    assert sizes[0] > sizes[-1]
    assert path.levels[0] == 0.0


def test_grids_must_contain_the_unpruned_case() -> None:
    with pytest.raises(ValueError):
        models.ModelConfig(min_samples_leaf_grid=(2, 5))
    with pytest.raises(ValueError):
        models.ModelConfig(ccp_fractions=(1e-3, 1e-2))
    with pytest.raises(ValueError):
        models.ModelConfig(alpha_grid=(0.5, 1.0))


# --- cross-validated tuning ------------------------------------------------


def test_cv_error_matches_a_manual_kfold(data) -> None:
    x, y = data
    expected = 0.0
    folds = KFold(n_splits=TREE.cv_folds, shuffle=True, random_state=0)
    for inner, held in folds.split(x):
        fitted = DecisionTreeRegressor(max_features=1.0, random_state=0)
        fitted.fit(x[inner], y[inner])
        expected += float(((fitted.predict(x[held]) - y[held]) ** 2).sum())
    errors = models.cv_errors(models.FULL, (1.0,), x, y, TREE, 0, float(np.var(y)))
    assert errors[0] == pytest.approx(expected / x.shape[0], rel=1e-12)


def test_ccp_level_means_the_same_alpha_in_cv_and_in_the_fit(data) -> None:
    """The ccp grid is a fraction of Var(y), which differs fold to fold.

    Taking the scale from the whole training split keeps a level pinned to one
    absolute alpha; taking it per fold would cross-validate models other than the
    ones actually fitted.
    """
    x, y = data
    levels = (0.0, 1e-3, 1e-2)
    scale = float(np.var(y))
    folds = KFold(n_splits=TREE.cv_folds, shuffle=True, random_state=0)
    fold_scales = [float(np.var(y[inner])) for inner, _ in folds.split(x)]
    assert max(fold_scales) - min(fold_scales) > 1e-6 * scale  # the folds do differ

    grown = models._ccp_grow(x, y, TREE, 0, levels, scale)
    direct = tuple(
        models.grow(x, y, TREE, 0, ccp_alpha=level * scale) for level in levels
    )
    for a, b in zip(grown, direct, strict=True):
        assert a.mean_leaves == b.mean_leaves


def test_alpha0_is_found_by_value_not_by_position(data) -> None:
    """alpha0 is the unpruned end of the path, and the baseline for every
    pruning comparison. Reading it off fits[0] would silently hand back a pruned
    model if the grid were reordered."""
    x, y = data
    shuffled = replace(TREE, alpha_grid=(1.0, 0.0, 2.0))
    path = models.pruning_path(models.ALPHA, x, y, shuffled, 0)

    assert path.fits[0].selected_param == 1.0  # position 0 is NOT alpha = 0
    unpruned = models.select_level(path, 0.0, "alpha0")
    assert unpruned.selected_param == 0.0
    assert unpruned.mean_leaves > path.fits[0].mean_leaves
    assert unpruned.name == "alpha0"

    with pytest.raises(ValueError):
        models.select_level(path, 99.0)


def test_selection_takes_the_lowest_cv_error(data) -> None:
    x, y = data
    path = models.pruning_path(models.GLOBAL, x, y, TREE, 0)
    chosen = models.select_by_cv(path)
    assert chosen.cv_mse == min(fit.cv_mse for fit in path.fits)


def test_pruning_beats_the_full_tree_in_cv(data) -> None:
    """A lone fully grown tree overfits; some pruning level must predict better."""
    x, y = data
    full = models.build(models.FULL, x, y, TREE, 0)
    for family in (models.GLOBAL, models.CCP, models.ALPHA):
        assert models.build(family, x, y, TREE, 0).cv_mse < full.cv_mse


def test_global_grid_is_capped_by_sample_size() -> None:
    assert max(models.GLOBAL.levels(TREE, 120)) <= 12


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


def test_training_identity_holds_after_pruning(data) -> None:
    x, y = data
    base = models.grow(x, y, TREE, 0, min_samples_leaf=TREE.alpha_min_samples_leaf)
    view = prune.trim(base, 1.0).trees[0]
    assert mdi.mdi_oob(view, x, y, FEATURES) == pytest.approx(
        mdi.mdi_train(view, FEATURES), abs=1e-10
    )


def test_ufi_carries_twice_the_scale_of_mdi_oob() -> None:
    """Raw UFI runs at 2x on signal features, so we halve it. See PROTOCOL.

    A single tree is noisier than the forest this was first measured on (2.5%
    agreement there), hence the looser tolerance; the factor 2 is what matters.
    """
    scenario = simulate.scenario_s1(snr=50.0)
    rng = np.random.default_rng(7)
    x, y = simulate.sample(scenario, 4000, rng)
    x_val, y_val = simulate.sample(scenario, 4000, rng)
    centre = float(y.mean())
    model = models.grow(x, y - centre, TREE, 0, min_samples_leaf=5)
    signal = ~scenario.is_noise
    oob = mdi.model_mdi_oob(model, x_val, y_val - centre, FEATURES)
    half = mdi.model_ufi_half(model, x_val, y_val - centre, FEATURES)
    ratio = 2.0 * half[signal] / oob[signal]
    assert (ratio > 1.7).all()
    assert half[signal] == pytest.approx(oob[signal], rel=0.15)


def test_out_of_sample_measures_give_noise_features_nothing() -> None:
    scenario = simulate.scenario_s1(snr=4.0)
    rng = np.random.default_rng(8)
    x, y = simulate.sample(scenario, 2000, rng)
    x_val, y_val = simulate.sample(scenario, 2000, rng)
    centre = float(y.mean())
    model = models.grow(x, y - centre, TREE, 0)
    noise = scenario.is_noise
    explainable = float(scenario.true_importance.sum())
    for measure in (mdi.model_mdi_oob, mdi.model_ufi_half):
        values = measure(model, x_val, y_val - centre, FEATURES)
        assert abs(float(values[noise].sum())) < 0.05 * explainable
    in_bag = mdi.model_mdi_train(model, FEATURES)
    assert float(in_bag[noise].sum()) > 0.05 * explainable


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
    config = models.ModelConfig(kind="forest", n_trees=5, n_jobs=1)
    model = models.grow(x, y, config, 0)
    per_tree = [mdi.mdi_train(t, FEATURES) for t in model.trees]
    assert mdi.model_mdi_train(model, FEATURES) == pytest.approx(
        np.mean(per_tree, axis=0), abs=1e-12
    )


# --- alpha-trimming -------------------------------------------------------


@pytest.fixture(scope="module")
def trimmable(data) -> Model:
    """A tree grown at the node size the alpha-trimming paper assumes."""
    x, y = data
    return models.grow(x, y, TREE, 0, min_samples_leaf=TREE.alpha_min_samples_leaf)


def test_alpha_zero_is_the_unpruned_tree(trimmable) -> None:
    view = trimmable.trees[0]
    assert prune.alpha_trim(view, 0.0) is view


def test_pruning_is_monotone_in_alpha(trimmable) -> None:
    sizes = [prune.trim(trimmable, a).mean_leaves for a in (0.0, 0.5, 1.0, 2.0, 3.0)]
    assert sizes == sorted(sizes, reverse=True)
    assert sizes[0] > sizes[-1]


def test_alpha_trimming_reaches_small_trees(trimmable) -> None:
    full = prune.trim(trimmable, 0.0).mean_leaves
    assert prune.trim(trimmable, 3.0).mean_leaves < 0.25 * full


def test_pruned_trees_stay_well_formed(data, trimmable) -> None:
    x, _ = data
    pruned = prune.alpha_trim(trimmable.trees[0], 1.0)
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
    would count them.
    """
    for alpha in (0.25, 0.5, 1.0, 2.0, 3.0):
        pruned = prune.alpha_trim(trimmable.trees[0], alpha)
        reachable = np.zeros(pruned.feature.size, dtype=bool)
        stack = [0]
        while stack:
            node = stack.pop()
            reachable[node] = True
            if pruned.feature[node] >= 0:
                stack.append(int(pruned.children_left[node]))
                stack.append(int(pruned.children_right[node]))
        assert not ((pruned.feature >= 0) & ~reachable).any()


def test_pure_leaves_make_alpha_trimming_inert(data) -> None:
    """Documents why the base tree is not grown to purity.

    With zero within-leaf variance the stump log-likelihood diverges, the merge
    test can never fire, and alpha-trimming silently does nothing.
    """
    x, y = data
    pure = models.grow(x, y, TREE, 0, min_samples_leaf=1)
    assert prune.trim(pure, 3.0).mean_leaves == pure.mean_leaves


def test_alpha_path_covers_the_grid_from_one_fit(data) -> None:
    x, y = data
    path = models.pruning_path(models.ALPHA, x, y, TREE, 0)
    assert path.levels == TREE.alpha_grid
    assert all(fit.name == "alpha" for fit in path.fits)


# --- metrics ---------------------------------------------------------------


def test_mse_decomposes_into_bias_squared_plus_variance() -> None:
    rng = np.random.default_rng(5)
    frame = pd.DataFrame(
        {
            "scenario": "S1", "n": 100, "arm": "full", "measure": "mdi_oob",
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


def test_scored_flags_a_model_too_small_to_rank() -> None:
    """A heavily pruned tree leaves most features at exactly zero.

    Noise mass then reads as a flawless 0 and AUC is carried by ties, so the
    summary has to say how many features the model actually scored.
    """
    rows = []
    for rep in range(3):
        for feature in range(10):
            # only the first two features are ever split on
            value = 0.4 if feature < 2 else 0.0
            rows.append(
                {
                    "scenario": "S1", "n": 100, "arm": "tiny", "measure": "mdi_train",
                    "rep": rep, "feature": feature, "value": value,
                    "true_importance": 0.5 if feature < 5 else 0.0,
                    "is_noise": feature >= 5, "sigma_sq": 0.25,
                    "test_mse": 1.0, "mean_leaves": 3.0,
                }
            )
    table = analyse.summary(pd.DataFrame(rows))
    assert float(table["noise_mass"].iloc[0]) == 0.0  # looks perfect
    assert float(table["scored"].iloc[0]) == pytest.approx(0.2)  # but is not


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

SMALL_RUN = experiment.RunConfig(n_reps=1, n_test=200)


def test_run_one_emits_every_arm_and_the_whole_pruning_path() -> None:
    rows, path_rows = experiment.run_one(SCENARIO, 150, 0, SMALL_RUN)

    split_arms = {"full", "global", "ccp", "alpha0", "alpha", "oracle"}
    expected = (len(split_arms) * len(mdi.MEASURES) + 1) * FEATURES
    assert len(rows) == expected
    assert {r["arm"] for r in rows} == split_arms | {"standard"}
    assert {r["kind"] for r in rows} == {"tree"}
    assert set(rows[0]) == set(experiment.COLUMNS)
    assert not any(np.isnan(float(r["value"])) for r in rows)

    levels = len(TREE.ccp_fractions) + len(TREE.alpha_grid)
    assert len(path_rows) == levels * len(mdi.MEASURES)
    assert set(experiment.PATH_COLUMNS) == set(path_rows[0])
    assert {r["method"] for r in path_rows} == {"ccp", "alpha"}


def test_oracle_never_loses_to_the_cv_pruning_level() -> None:
    """The oracle picks on importance error, so by construction it cannot be worse."""
    rows, _ = experiment.run_one(SCENARIO, 200, 0, SMALL_RUN)
    frame = pd.DataFrame(rows)
    truth = SCENARIO.true_importance

    def error(arm: str) -> float:
        values = (
            frame[(frame["arm"] == arm) & (frame["measure"] == experiment.ORACLE_MEASURE)]
            .sort_values("feature")["value"]
            .to_numpy(dtype=np.float64)
        )
        return float(((values - truth) ** 2).sum())

    assert error("oracle") <= error("alpha") + 1e-12


def test_standard_arm_uses_every_point() -> None:
    rows, _ = experiment.run_one(SCENARIO, 300, 0, SMALL_RUN)
    frame = pd.DataFrame(rows)
    standard = frame[frame["arm"] == "standard"]["mean_leaves"].iloc[0]
    split = frame[frame["arm"] == "full"]["mean_leaves"].iloc[0]
    assert standard > split
    assert set(frame[frame["arm"] == "standard"]["measure"]) == {"mdi_train"}


def test_forest_kind_runs_end_to_end() -> None:
    config = experiment.RunConfig(
        n_reps=1, n_test=200,
        model=models.ModelConfig(kind="forest", n_trees=5, n_jobs=1, cv_folds=2),
    )
    rows, _ = experiment.run_one(SCENARIO, 120, 0, config)
    assert {r["kind"] for r in rows} == {"forest"}


def test_runs_are_reproducible() -> None:
    first, _ = experiment.run_one(SCENARIO, 120, 0, SMALL_RUN)
    second, _ = experiment.run_one(SCENARIO, 120, 0, SMALL_RUN)
    assert [r["value"] for r in first] == [r["value"] for r in second]


def test_summary_reports_standard_errors_and_relative_loss() -> None:
    rows = [r for rep in range(3) for r in experiment.run_one(SCENARIO, 120, rep, SMALL_RUN)[0]]
    table = analyse.summary(pd.DataFrame(rows))
    for column in ("total_mse_se", "noise_mass_se", "relative_loss", "excess_risk"):
        assert column in table
    assert (table["total_mse_se"] >= 0).all()
