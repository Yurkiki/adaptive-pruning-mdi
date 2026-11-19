"""Run the experiment and write two long CSVs.

results.csv  one row per (n, rep, arm, measure, feature)
path.csv     one row per (n, rep, level, measure) -- importance error against the
             amount of pruning, which is what shows whether CV picks a useful
             pruning level at all

The model is a single regression tree by default (`--kind tree`); `--kind forest`
runs the identical protocol on random forests.

Every split-sample arm is built on the same training split, so comparing
measures within an arm isolates the evaluation data and comparing arms within a
measure isolates the pruning. The standard arm is built on every point instead,
because that is what practitioners actually do.

alpha0 is the alpha arm at alpha = 0: same base model, no pruning. Comparing
alpha against alpha0 isolates the pruning alone, because the alpha arm's base
uses min_samples_leaf = 5 and so is not the same object as the full arm.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd

import mdi
import models
import simulate

FloatArray = npt.NDArray[np.float64]

COLUMNS = (
    "scenario", "kind", "n", "rep", "arm", "measure", "feature", "value",
    "true_importance", "is_noise", "sigma_sq", "cv_mse", "test_mse",
    "selected_param", "mean_leaves", "mean_depth",
)

PATH_COLUMNS = (
    "scenario", "kind", "n", "rep", "method", "level", "measure",
    "importance_error", "cv_mse", "test_mse", "mean_leaves",
)

# One oracle per measure. An oracle chosen on ufi_half bounds what pruning could
# deliver *for ufi_half*; read at another measure it is just some other level, so
# the "pruning does not help" claim needs its own oracle per measure. The
# mdi_train oracle is the one that makes [S20]'s remedy comparable to [ZH21]'s:
# best-possible pruning of the classic MDI, against the held-out measure unpruned.
ORACLE_MEASURES = ("ufi_half", "mdi_oob", "mdi_train")
ORACLE_ARMS = {
    "ufi_half": "oracle",
    "mdi_oob": "oracle_oob",
    "mdi_train": "oracle_train",
}


@dataclass(frozen=True)
class RunConfig:
    scenario: str = "S1"
    snr: float = 4.0
    sizes: tuple[int, ...] = (250, 1000)
    n_reps: int = 20
    val_fraction: float = 1.0 / 3.0
    n_test: int = 10_000
    model: models.ModelConfig = field(default_factory=models.ModelConfig)


def importance_error(estimate: FloatArray, truth: FloatArray) -> float:
    return float(((estimate - truth) ** 2).sum())


def select_oracle(
    path: models.PruningPath,
    x_val: FloatArray,
    y_val: FloatArray,
    truth: FloatArray,
    measure: str = "ufi_half",
) -> models.Fitted:
    """Pick the pruning level that minimises `measure`'s importance error.

    Only available in simulation. It bounds what pruning could deliver, so that a
    null result for the honest arm can be read as "pruning does not help" or
    "CV chose the wrong amount" rather than being ambiguous between the two. The
    bound is measure-specific, hence one oracle per measure.

    It selects on `x_val, y_val`, the same rows the reported value is later
    computed on, so its number is **optimistic** -- an upper bound, not an
    achievable score. Measured on S1/S4 at n = 1200 over 100 repetitions, the
    optimism is 3% to 11%, and it flips the sign: this oracle appears to beat
    alpha = 0 by 1-11%, while an oracle that selects on a separate held-out split
    *loses* to alpha = 0 by 2-4% and wins in only 26-44 reps of 100. Read a gain
    here as "pruning cannot do better than this", never as "pruning delivers
    this".
    """

    def error(fit: models.Fitted) -> float:
        estimate = mdi.all_measures(fit.model, x_val, y_val, truth.size)[measure]
        return importance_error(estimate, truth)

    return replace(min(path.fits, key=error), name=ORACLE_ARMS[measure])


def run_one(
    scenario: simulate.Scenario, n: int, rep: int, config: RunConfig
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rng = np.random.default_rng(rep)
    n_features = scenario.n_features
    truth = scenario.true_importance
    sigma_sq = scenario.sigma**2
    setup = config.model

    x, y = simulate.sample(scenario, n, rng)
    x_test, y_test_raw = simulate.sample(scenario, config.n_test, rng)

    n_val = int(round(n * config.val_fraction))
    order = rng.permutation(n)
    val, train = order[:n_val], order[n_val:]

    centre = float(y[train].mean())
    x_train, y_train = x[train], y[train] - centre
    x_val, y_val = x[val], y[val] - centre
    y_test = y_test_raw - centre

    ccp = models.pruning_path(models.CCP, x_train, y_train, setup, rep)
    alpha = models.pruning_path(models.ALPHA, x_train, y_train, setup, rep)
    fits = {
        "full": models.build(models.FULL, x_train, y_train, setup, rep),
        "global": models.build(models.GLOBAL, x_train, y_train, setup, rep),
        "ccp": models.select_by_cv(ccp),
        "alpha0": models.select_level(alpha, 0.0, "alpha0"),
        "alpha": models.select_by_cv(alpha),
        **{
            ORACLE_ARMS[m]: select_oracle(alpha, x_val, y_val, truth, m)
            for m in ORACLE_MEASURES
        },
    }
    overall = float(y.mean())
    standard = models.build(models.FULL, x, y - overall, setup, rep)

    rows: list[dict[str, object]] = []
    shared = {
        "scenario": scenario.name, "kind": setup.kind, "n": n, "rep": rep,
        "sigma_sq": sigma_sq,
    }

    def emit(
        arm: str, fit: models.Fitted, values: dict[str, FloatArray], test_mse: float
    ) -> None:
        for measure, estimate in values.items():
            for j in range(n_features):
                rows.append(
                    shared
                    | {
                        "arm": arm, "measure": measure, "feature": j,
                        "value": float(estimate[j]),
                        "true_importance": float(truth[j]),
                        "is_noise": bool(scenario.is_noise[j]),
                        "cv_mse": fit.cv_mse, "test_mse": test_mse,
                        "selected_param": fit.selected_param,
                        "mean_leaves": fit.mean_leaves,
                        "mean_depth": fit.mean_depth,
                    }
                )

    for arm, fit in fits.items():
        test_mse = float(np.mean((fit.model.predict(x_test) - y_test) ** 2))
        emit(arm, fit, mdi.all_measures(fit.model, x_val, y_val, n_features), test_mse)

    standard_test = float(
        np.mean((standard.model.predict(x_test) - (y_test_raw - overall)) ** 2)
    )
    emit(
        "standard", standard,
        {"mdi_train": mdi.model_mdi_train(standard.model, n_features)},
        standard_test,
    )

    path_rows: list[dict[str, object]] = []
    for method, walked in (("ccp", ccp), ("alpha", alpha)):
        for fit in walked.fits:
            values = mdi.all_measures(fit.model, x_val, y_val, n_features)
            test_mse = float(np.mean((fit.model.predict(x_test) - y_test) ** 2))
            for measure, estimate in values.items():
                path_rows.append(
                    {
                        "scenario": scenario.name, "kind": setup.kind, "n": n,
                        "rep": rep, "method": method, "level": fit.selected_param,
                        "measure": measure,
                        "importance_error": importance_error(estimate, truth),
                        "cv_mse": fit.cv_mse, "test_mse": test_mse,
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
            f"  {config.scenario} [{config.model.kind}] n={n:<5d} {config.n_reps} reps "
            f"in {time.perf_counter() - start:6.1f}s"
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
    parser.add_argument("--kind", default="tree", choices=models.KINDS)
    parser.add_argument("--snr", type=float, default=4.0)
    parser.add_argument("--sizes", type=int, nargs="+", default=[250, 1000])
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--trees", type=int, default=500, help="forest only")
    parser.add_argument("--folds", type=int, default=5)
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
            model=models.ModelConfig(
                kind=args.kind, n_trees=args.trees, cv_folds=args.folds
            ),
        ),
        args.out,
    )


if __name__ == "__main__":
    main()
