"""Real datasets, where the true importance is unknown.

Scornet asked for real data as well as simulations. With no ground truth we
cannot measure bias, so we manufacture a partial one: append columns that are
known to be irrelevant and see how much importance each measure hands them.
A measure that gives the injected columns ~0 is behaving; one that does not is
exhibiting the bias on data we did not generate.

Three kinds of junk are injected, because they fail in different ways:
permuted copies of real columns keep the marginal distribution and so are the
hardest case; continuous noise is the easy case; categorical noise with many
levels is the classic cardinality trap for in-bag MDI.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.datasets import fetch_california_housing, load_diabetes

import analyse
import forest
import mdi

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]

COLUMNS = (
    "dataset", "rep", "forest", "measure", "feature", "name", "value",
    "is_injected", "oob_mse", "test_mse", "selected_param", "mean_leaves",
)


def load(name: str, rng: np.random.Generator, max_rows: int = 3000) -> tuple[FloatArray, FloatArray, list[str]]:
    if name == "diabetes":
        data = load_diabetes()
    elif name == "california":
        data = fetch_california_housing()
    else:
        raise ValueError(f"unknown dataset {name}")
    x = np.asarray(data.data, dtype=np.float64)
    y = np.asarray(data.target, dtype=np.float64)
    names = list(data.feature_names)
    if x.shape[0] > max_rows:
        keep = rng.choice(x.shape[0], size=max_rows, replace=False)
        x, y = x[keep], y[keep]
    return x, y, names


DATASETS = ("diabetes", "california")


def inject_noise(
    x: FloatArray,
    names: list[str],
    rng: np.random.Generator,
    n_permuted: int = 3,
    n_continuous: int = 2,
    categorical_levels: tuple[int, ...] = (2, 10, 50),
) -> tuple[FloatArray, list[str], BoolArray]:
    """Append columns that cannot carry signal, and say which they are."""
    columns = [x]
    new_names: list[str] = []

    chosen = rng.choice(x.shape[1], size=min(n_permuted, x.shape[1]), replace=False)
    for index in chosen:
        columns.append(rng.permutation(x[:, index]).reshape(-1, 1))
        new_names.append(f"perm[{names[index]}]")

    for k in range(n_continuous):
        columns.append(rng.normal(size=(x.shape[0], 1)))
        new_names.append(f"noise_cont_{k}")

    for levels in categorical_levels:
        columns.append(rng.integers(0, levels, size=(x.shape[0], 1)).astype(np.float64))
        new_names.append(f"noise_cat_{levels}")

    augmented = np.hstack(columns)
    injected = np.zeros(augmented.shape[1], dtype=bool)
    injected[x.shape[1]:] = True
    return augmented, names + new_names, injected


def run_one(
    dataset: str, rep: int, config: forest.ForestConfig, val_fraction: float = 1.0 / 3.0
) -> list[dict[str, object]]:
    rng = np.random.default_rng(rep)
    x, y, names = load(dataset, rng)
    x, names, injected = inject_noise(x, names, rng)
    n_features = x.shape[1]

    order = rng.permutation(x.shape[0])
    n_test = x.shape[0] // 5
    test, rest = order[:n_test], order[n_test:]
    n_val = int(round(len(rest) * val_fraction))
    val, train = rest[:n_val], rest[n_val:]

    centre = float(y[train].mean())
    x_train, y_train = x[train], y[train] - centre
    x_val, y_val = x[val], y[val] - centre
    x_test, y_test = x[test], y[test] - centre

    fits = {
        "full": forest.build_full(x_train, y_train, config, rep),
        "global": forest.build_global(x_train, y_train, config, rep),
        "ccp": forest.build_pruned(x_train, y_train, config, rep),
        "alpha": forest.build_alpha(x_train, y_train, config, rep),
    }
    rows: list[dict[str, object]] = []
    for name, fit in fits.items():
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
                        "dataset": dataset, "rep": rep, "forest": name,
                        "measure": measure, "feature": j, "name": names[j],
                        "value": float(estimate[j]),
                        "is_injected": bool(injected[j]),
                        "oob_mse": fit.oob_mse, "test_mse": test_mse,
                        "selected_param": fit.selected_param,
                        "mean_leaves": fit.mean_leaves,
                    }
                )
    return rows


def summarise(frame: pd.DataFrame) -> pd.DataFrame:
    """Noise mass and separation, per arm. No ground truth, so no bias or MSE."""
    records: list[dict[str, object]] = []
    for key, group in frame.groupby(["dataset", "forest", "measure", "rep"], sort=False):
        ordered = group.sort_values("feature")
        values = ordered["value"].to_numpy(dtype=np.float64)
        injected = ordered["is_injected"].to_numpy(dtype=bool)
        total = float(np.abs(values).sum())
        records.append(
            dict(zip(["dataset", "forest", "measure", "rep"], key))
            | {
                "injected_mass": float(values[injected].sum()),
                "injected_share": float(np.abs(values[injected]).sum() / total)
                if total > 0 else float("nan"),
                "auc": analyse.auc_signal_vs_noise(values, injected),
                "test_mse": float(group["test_mse"].iloc[0]),
                "mean_leaves": float(group["mean_leaves"].iloc[0]),
            }
        )
    per_rep = pd.DataFrame(records)
    keys = ["dataset", "forest", "measure"]
    columns = ["injected_mass", "injected_share", "auc", "test_mse", "mean_leaves"]
    grouped = per_rep.groupby(keys, as_index=False)
    out = grouped.agg(**{c: (c, "mean") for c in columns})
    errors = grouped.agg(
        **{f"{c}_se": (c, lambda s: float(s.std(ddof=1) / np.sqrt(s.size)))
           for c in ("injected_mass", "injected_share", "auc")}
    )
    return out.merge(errors, on=keys)


def stability(frame: pd.DataFrame) -> pd.DataFrame:
    """Mean pairwise Kendall tau between repetitions -- does the ranking hold still?"""
    from scipy import stats

    records: list[dict[str, object]] = []
    for key, group in frame.groupby(["dataset", "forest", "measure"], sort=False):
        wide = group.pivot_table(index="feature", columns="rep", values="value")
        taus = [
            float(stats.kendalltau(wide.iloc[:, a], wide.iloc[:, b]).statistic)
            for a in range(wide.shape[1])
            for b in range(a + 1, wide.shape[1])
        ]
        records.append(
            dict(zip(["dataset", "forest", "measure"], key))
            | {"rank_stability": float(np.mean(taus)) if taus else float("nan")}
        )
    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--trees", type=int, default=300)
    parser.add_argument("--out", type=Path, default=Path("results/real.csv"))
    args = parser.parse_args()

    config = replace(forest.ForestConfig(), n_trees=args.trees)
    rows: list[dict[str, object]] = []
    for dataset in args.datasets:
        for rep in range(args.reps):
            rows.extend(run_one(dataset, rep, config))
        print(f"  {dataset}: {args.reps} reps done")

    frame = pd.DataFrame(rows, columns=list(COLUMNS))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)

    table = summarise(frame).merge(stability(frame), on=["dataset", "forest", "measure"])
    table["forest"] = pd.Categorical(table["forest"], analyse.FOREST_ORDER, ordered=True)
    table["measure"] = pd.Categorical(table["measure"], analyse.MEASURE_ORDER, ordered=True)
    for dataset, block in table.groupby("dataset", observed=True):
        print(f"\n=== {dataset} " + "=" * 62)
        print(
            block.sort_values(["forest", "measure"])
            .drop(columns="dataset")
            .to_string(index=False, float_format=lambda v: f"{v:8.4f}")
        )
    print("\n  injected_share: fraction of total importance given to known-junk columns")
    print("  auc: separation of real features from injected ones (1.0 is perfect)")


if __name__ == "__main__":
    main()
