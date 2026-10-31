"""Run the experiment and write two long CSVs.

results.csv  one row per (n, rep, forest, measure, feature)
path.csv     one row per (n, rep, alpha, measure) -- importance error against the
             amount of pruning, which is what shows whether OOB picks a useful
             pruning level at all

The split-sample forests are all built on the same training split, so comparing
measures within a forest isolates the evaluation data and comparing forests
within a measure isolates the pruning. The standard forest is built on every
point instead, because that is what practitioners actually do.

alpha0 is the alpha arm at alpha = 0: same base forest, no pruning. Comparing
alpha against alpha0 isolates the pruning alone, because the alpha arm's base
uses min_samples_leaf = 5 and so is not the same object as the full arm.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd

import forest
import mdi
import simulate

FloatArray = npt.NDArray[np.float64]

COLUMNS = (
    "scenario", "n", "rep", "forest", "measure", "feature", "value",
    "true_importance", "is_noise", "sigma_sq", "oob_mse", "test_mse",
    "selected_param", "mean_leaves", "mean_depth",
)

PATH_COLUMNS = (
    "scenario", "n", "rep", "method", "level", "measure",
    "importance_error", "oob_mse", "test_mse", "mean_leaves",
)

ORACLE_MEASURE = "ufi_half"


@dataclass(frozen=True)
class RunConfig:
    scenario: str = "S1"
    snr: float = 4.0
    sizes: tuple[int, ...] = (250, 1000)
    n_reps: int = 20
    val_fraction: float = 1.0 / 3.0
    n_test: int = 10_000
    forest: forest.ForestConfig = forest.ForestConfig()


def measures_of(
    fit: forest.Fitted, x_val: FloatArray, y_val: FloatArray, n_features: int
) -> dict[str, FloatArray]:
    return {
        "mdi_train": mdi.forest_mdi_train(fit.forest, n_features),
        "mdi_oob": mdi.forest_mdi_oob(fit.forest, x_val, y_val, n_features),
        "ufi_half": mdi.forest_ufi_half(fit.forest, x_val, y_val, n_features),
    }


def importance_error(estimate: FloatArray, truth: FloatArray) -> float:
    return float(((estimate - truth) ** 2).sum())


def select_oracle(
    path: forest.PruningPath,
    x_val: FloatArray,
    y_val: FloatArray,
    truth: FloatArray,
) -> forest.Fitted:
    """Pick the pruning level that minimises importance error rather than OOB error.

    Only available in simulation. It bounds what pruning could deliver, so that a
    null result for the honest arm can be read as "pruning does not help" or
    "OOB chose the wrong amount" rather than being ambiguous between the two.
    """

    def error(fit: forest.Fitted) -> float:
        estimate = measures_of(fit, x_val, y_val, truth.size)[ORACLE_MEASURE]
        return importance_error(estimate, truth)

    return replace(min(path.fits, key=error), name="oracle")


def run_one(
    scenario: simulate.Scenario, n: int, rep: int, config: RunConfig
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rng = np.random.default_rng(rep)
    n_features = scenario.n_features
    truth = scenario.true_importance
    sigma_sq = scenario.sigma**2

    x, y = simulate.sample(scenario, n, rng)
    x_test, y_test_raw = simulate.sample(scenario, config.n_test, rng)

    n_val = int(round(n * config.val_fraction))
    order = rng.permutation(n)
    val, train = order[:n_val], order[n_val:]

    centre = float(y[train].mean())
    x_train, y_train = x[train], y[train] - centre
    x_val, y_val = x[val], y[val] - centre
    y_test = y_test_raw - centre

    ccp = forest.pruning_path(x_train, y_train, config.forest, rep)
    alpha = forest.alpha_path(x_train, y_train, config.forest, rep)
    fits = {
        "full": forest.build_full(x_train, y_train, config.forest, rep),
        "global": forest.build_global(x_train, y_train, config.forest, rep),
        "ccp": forest.select_by_oob(ccp, name="ccp"),
        "alpha0": replace(alpha.fits[0], name="alpha0"),
        "alpha": forest.select_by_oob(alpha, name="alpha"),
        "oracle": select_oracle(alpha, x_val, y_val, truth),
    }
    standard = forest.build_full(x, y - float(y.mean()), config.forest, rep)

    rows: list[dict[str, object]] = []
    shared = {
        "scenario": scenario.name, "n": n, "rep": rep, "sigma_sq": sigma_sq,
    }

    def emit(name: str, fit: forest.Fitted, values: dict[str, FloatArray]) -> None:
        test_mse = float(np.mean((fit.forest.predict(x_test) - y_test) ** 2))
        for measure, estimate in values.items():
            for j in range(n_features):
                rows.append(
                    shared
                    | {
                        "forest": name, "measure": measure, "feature": j,
                        "value": float(estimate[j]),
                        "true_importance": float(truth[j]),
                        "is_noise": bool(scenario.is_noise[j]),
                        "oob_mse": fit.oob_mse, "test_mse": test_mse,
                        "selected_param": fit.selected_param,
                        "mean_leaves": fit.mean_leaves,
                        "mean_depth": fit.mean_depth,
                    }
                )

    for name, fit in fits.items():
        emit(name, fit, measures_of(fit, x_val, y_val, n_features))

    standard_test = float(
        np.mean((standard.forest.predict(x_test) - (y_test_raw - float(y.mean()))) ** 2)
    )
    for j in range(n_features):
        rows.append(
            shared
            | {
                "forest": "standard", "measure": "mdi_train", "feature": j,
                "value": float(mdi.forest_mdi_train(standard.forest, n_features)[j]),
                "true_importance": float(truth[j]),
                "is_noise": bool(scenario.is_noise[j]),
                "oob_mse": standard.oob_mse, "test_mse": standard_test,
                "selected_param": standard.selected_param,
                "mean_leaves": standard.mean_leaves,
                "mean_depth": standard.mean_depth,
            }
        )

    path_rows: list[dict[str, object]] = []
    paths: tuple[tuple[str, tuple[float, ...], forest.PruningPath], ...] = (
        ("ccp", config.forest.ccp_fractions, ccp),
        ("alpha", config.forest.alpha_grid, alpha),
    )
    for method, levels, walked in paths:
        for level, fit in zip(levels, walked.fits, strict=True):
            values = measures_of(fit, x_val, y_val, n_features)
            test_mse = float(np.mean((fit.forest.predict(x_test) - y_test) ** 2))
            for measure, estimate in values.items():
                path_rows.append(
                    {
                        "scenario": scenario.name, "n": n, "rep": rep,
                        "method": method, "level": level,
                        "measure": measure,
                        "importance_error": importance_error(estimate, truth),
                        "oob_mse": fit.oob_mse, "test_mse": test_mse,
                        "mean_leaves": fit.mean_leaves,
                    }
                )

    return rows, path_rows


def run(config: RunConfig, out: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    scenario = simulate.SCENARIOS[config.scenario](snr=config.snr)
    rows: list[dict[str, object]] = []
    path_rows: list[dict[str, object]] = []
    for n in config.sizes:
        start = time.perf_counter()
        for rep in range(config.n_reps):
            new_rows, new_path = run_one(scenario, n, rep, config)
            rows.extend(new_rows)
            path_rows.extend(new_path)
        print(
            f"  {config.scenario} n={n:<5d} {config.n_reps} reps in "
            f"{time.perf_counter() - start:6.1f}s"
        )

    frame = pd.DataFrame(rows, columns=list(COLUMNS))
    path_frame = pd.DataFrame(path_rows, columns=list(PATH_COLUMNS))
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    path_out = out.with_name(out.stem + "_path.csv")
    path_frame.to_csv(path_out, index=False)
    print(f"  wrote {len(frame)} rows to {out} and {len(path_frame)} to {path_out}")
    return frame, path_frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="S1", choices=sorted(simulate.SCENARIOS))
    parser.add_argument("--snr", type=float, default=4.0)
    parser.add_argument("--sizes", type=int, nargs="+", default=[250, 1000])
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--trees", type=int, default=500)
    parser.add_argument("--val-fraction", type=float, default=1.0 / 3.0)
    parser.add_argument("--out", type=Path, default=Path("results/results.csv"))
    args = parser.parse_args()

    run(
        RunConfig(
            scenario=args.scenario,
            snr=args.snr,
            sizes=tuple(args.sizes),
            n_reps=args.reps,
            val_fraction=args.val_fraction,
            forest=replace(forest.ForestConfig(), n_trees=args.trees),
        ),
        args.out,
    )


if __name__ == "__main__":
    main()
