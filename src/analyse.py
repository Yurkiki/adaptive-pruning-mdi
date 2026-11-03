"""Bias, variance and the summary metrics, from the results CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import stats

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]

GROUP = ["scenario", "n", "arm", "measure"]

MEASURE_ORDER = ["mdi_train", "mdi_oob", "ufi_half"]
ARM_ORDER = ["full", "global", "ccp"]


# --- metrics ---------------------------------------------------------------


def auc_signal_vs_noise(values: FloatArray, is_noise: BoolArray) -> float:
    """Probability a signal feature outranks a noise feature, ties counting a half."""
    signal, noise = values[~is_noise], values[is_noise]
    if signal.size == 0 or noise.size == 0:
        return float("nan")
    ranks = stats.rankdata(np.concatenate([signal, noise]))
    rank_sum = float(ranks[: signal.size].sum())
    return (rank_sum - signal.size * (signal.size + 1) / 2) / (signal.size * noise.size)


def per_feature(frame: pd.DataFrame) -> pd.DataFrame:
    """Bias, variance and MSE per feature. Variance uses 1/R so MSE = bias^2 + var."""
    work = frame.assign(sq_err=(frame["value"] - frame["true_importance"]) ** 2)
    keys = GROUP + ["feature"]
    out = work.groupby(keys, as_index=False).agg(
        mean_value=("value", "mean"),
        true_importance=("true_importance", "first"),
        is_noise=("is_noise", "first"),
        mse=("sq_err", "mean"),
        reps=("value", "size"),
    )
    variance = work.groupby(keys)["value"].var(ddof=0).rename("variance").reset_index()
    out = out.merge(variance, on=keys)
    out["bias"] = out["mean_value"] - out["true_importance"]
    return out


def per_rep(frame: pd.DataFrame) -> pd.DataFrame:
    """Every summary statistic, computed per repetition so it carries a standard error."""
    records: list[dict[str, object]] = []
    for key, group in frame.groupby(GROUP + ["rep"], sort=False):
        ordered = group.sort_values("feature")
        values = ordered["value"].to_numpy(dtype=np.float64)
        truth = ordered["true_importance"].to_numpy(dtype=np.float64)
        noise = ordered["is_noise"].to_numpy(dtype=bool)
        explainable = float(truth.sum())
        tau = (
            float(stats.kendalltau(values[~noise], truth[~noise]).statistic)
            if (~noise).sum() > 1
            else float("nan")
        )
        records.append(
            dict(zip(GROUP + ["rep"], key))
            | {
                "loss": float(((values - truth) ** 2).sum()),
                "relative_loss": float(((values - truth) ** 2).sum() / (truth**2).sum()),
                "noise_mass": float(values[noise].sum()),
                "noise_mass_abs": float(np.abs(values[noise]).sum()),
                "signal_mass": float(values[~noise].sum()),
                "explainable": explainable,
                "auc": auc_signal_vs_noise(values, noise),
                "tau": tau,
                "excess_risk": float(
                    group["test_mse"].iloc[0] - group["sigma_sq"].iloc[0]
                ),
                "mean_leaves": float(group["mean_leaves"].iloc[0]),
            }
        )
    return pd.DataFrame(records)


def _mean_and_se(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    grouped = frame.groupby(GROUP, as_index=False)
    out = grouped.agg(**{c: (c, "mean") for c in columns})
    errors = grouped.agg(
        **{f"{c}_se": (c, lambda s: float(s.std(ddof=1) / np.sqrt(s.size))) for c in columns}
    )
    return out.merge(errors, on=GROUP)


def summary(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per arm, with Monte Carlo standard errors on every mean."""
    columns = [
        "loss", "relative_loss", "noise_mass", "noise_mass_abs", "signal_mass",
        "auc", "tau", "excess_risk", "mean_leaves",
    ]
    table = _mean_and_se(per_rep(frame), columns)
    table = table.rename(columns={"loss": "total_mse", "loss_se": "total_mse_se"})
    table["explainable"] = (
        frame.groupby(GROUP)["true_importance"].sum().values
        / frame.groupby(GROUP)["rep"].nunique().values
    )
    table["arm"] = pd.Categorical(table["arm"], ARM_ORDER, ordered=True)
    table["measure"] = pd.Categorical(table["measure"], MEASURE_ORDER, ordered=True)
    return table.sort_values(["scenario", "n", "arm", "measure"]).reset_index(drop=True)


# --- paired comparison -----------------------------------------------------


# --- text report -----------------------------------------------------------


def report(frame: pd.DataFrame) -> None:
    table = summary(frame)
    show = [
        "arm", "measure", "total_mse", "total_mse_se", "relative_loss",
        "noise_mass", "noise_mass_se", "signal_mass", "explainable",
        "auc", "tau", "excess_risk", "mean_leaves",
    ]
    for (scenario, n), block in table.groupby(["scenario", "n"], observed=True):
        print(f"\n=== {scenario}  n = {n} " + "=" * 60)
        print(block[show].to_string(index=False, float_format=lambda v: f"{v:8.4f}"))

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/results.csv"))
    args = parser.parse_args()

    report(pd.read_csv(args.results))


if __name__ == "__main__":
    main()
