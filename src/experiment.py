"""Run the experiment and write one long CSV.

All three forests are built on the same training split, so comparing measures
within a forest isolates the evaluation data, and comparing forests within a
measure isolates the pruning.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

import forest
import mdi
import simulate

COLUMNS = (
    "scenario",
    "n",
    "rep",
    "forest",
    "measure",
    "feature",
    "value",
    "true_importance",
    "is_noise",
    "oob_mse",
    "test_mse",
    "selected_param",
    "mean_leaves",
    "mean_depth",
)


@dataclass(frozen=True)
class RunConfig:
    scenario: str = "S1"
    sizes: tuple[int, ...] = (250, 1000)
    n_reps: int = 20
    val_fraction: float = 1.0 / 3.0
    n_test: int = 10_000
    forest: forest.ForestConfig = forest.ForestConfig()


def run_one(scenario: simulate.Scenario, n: int, rep: int, config: RunConfig) -> list[dict[str, object]]:
    rng = np.random.default_rng(rep)
    n_features = scenario.n_features

    x, y = simulate.sample(scenario, n, rng)
    x_test, y_test = simulate.sample(scenario, config.n_test, rng)

    n_val = int(round(n * config.val_fraction))
    order = rng.permutation(n)
    val, train = order[:n_val], order[n_val:]

    centre = float(y[train].mean())
    x_train, y_train = x[train], y[train] - centre
    x_val, y_val = x[val], y[val] - centre
    y_test = y_test - centre

    rows: list[dict[str, object]] = []
    for build in forest.BUILDERS:
        fit = build(x_train, y_train, config.forest, rep)
        test_mse = float(np.mean((fit.forest.predict(x_test) - y_test) ** 2))
        values = {
            "mdi_train": mdi.forest_mdi_train(fit.forest, n_features),
            "mdi_oob": mdi.forest_mdi_oob(fit.forest, x_val, y_val, n_features),
            "ufi_half": mdi.forest_ufi_half(fit.forest, x_val, y_val, n_features),
        }
        for measure, estimate in values.items():
            for j in range(n_features):
                rows.append(
                    {
                        "scenario": scenario.name,
                        "n": n,
                        "rep": rep,
                        "forest": fit.name,
                        "measure": measure,
                        "feature": j,
                        "value": float(estimate[j]),
                        "true_importance": float(scenario.true_importance[j]),
                        "is_noise": bool(scenario.is_noise[j]),
                        "oob_mse": fit.oob_mse,
                        "test_mse": test_mse,
                        "selected_param": fit.selected_param,
                        "mean_leaves": fit.mean_leaves,
                        "mean_depth": fit.mean_depth,
                    }
                )
    return rows


def run(config: RunConfig, out: Path) -> pd.DataFrame:
    scenario = simulate.SCENARIOS[config.scenario]()
    rows: list[dict[str, object]] = []
    for n in config.sizes:
        start = time.perf_counter()
        for rep in range(config.n_reps):
            rows.extend(run_one(scenario, n, rep, config))
        elapsed = time.perf_counter() - start
        print(f"  {config.scenario} n={n:<5d} {config.n_reps} reps in {elapsed:6.1f}s")
    frame = pd.DataFrame(rows, columns=list(COLUMNS))
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    print(f"  wrote {len(frame)} rows to {out}")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="S1", choices=sorted(simulate.SCENARIOS))
    parser.add_argument("--sizes", type=int, nargs="+", default=[250, 1000])
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--trees", type=int, default=500)
    parser.add_argument("--out", type=Path, default=Path("results/results.csv"))
    args = parser.parse_args()

    config = RunConfig(
        scenario=args.scenario,
        sizes=tuple(args.sizes),
        n_reps=args.reps,
        forest=replace(forest.ForestConfig(), n_trees=args.trees),
    )
    run(config, args.out)


if __name__ == "__main__":
    main()
