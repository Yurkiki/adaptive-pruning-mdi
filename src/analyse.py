"""Metrics, paired comparisons and plots, all derived from the results CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import stats

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]

GROUP = ["scenario", "n", "forest", "measure"]


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
    variance = (
        work.groupby(keys)["value"].var(ddof=0).rename("variance").reset_index()
    )
    out = out.merge(variance, on=keys)
    out["bias"] = out["mean_value"] - out["true_importance"]
    return out


def per_rep(frame: pd.DataFrame) -> pd.DataFrame:
    """Loss, AUC and Kendall tau for each individual repetition."""
    records: list[dict[str, object]] = []
    for key, group in frame.groupby(GROUP + ["rep"]):
        ordered = group.sort_values("feature")
        values = ordered["value"].to_numpy(dtype=np.float64)
        truth = ordered["true_importance"].to_numpy(dtype=np.float64)
        noise = ordered["is_noise"].to_numpy(dtype=bool)
        tau = (
            float(stats.kendalltau(values[~noise], truth[~noise]).statistic)
            if (~noise).sum() > 1
            else float("nan")
        )
        records.append(
            dict(zip(GROUP + ["rep"], key))
            | {
                "loss": float(((values - truth) ** 2).sum()),
                "auc": auc_signal_vs_noise(values, noise),
                "tau": tau,
            }
        )
    return pd.DataFrame(records)


def summary(frame: pd.DataFrame) -> pd.DataFrame:
    features = per_feature(frame)
    reps = per_rep(frame)

    signal = features[~features["is_noise"]]
    noise = features[features["is_noise"]]

    table = features.groupby(GROUP, as_index=False).agg(total_mse=("mse", "sum"))
    table = table.merge(
        noise.groupby(GROUP, as_index=False).agg(
            noise_mass=("mean_value", "sum"),
            noise_mass_abs=("mean_value", lambda s: float(np.abs(s).sum())),
        ),
        on=GROUP,
        how="left",
    )
    table = table.merge(
        signal.groupby(GROUP, as_index=False).agg(
            signal_mass=("mean_value", "sum"), explainable=("true_importance", "sum")
        ),
        on=GROUP,
        how="left",
    )
    table = table.merge(
        reps.groupby(GROUP, as_index=False).agg(auc=("auc", "mean"), tau=("tau", "mean")),
        on=GROUP,
    )
    table = table.merge(
        frame.groupby(GROUP, as_index=False).agg(
            test_mse=("test_mse", "mean"),
            mean_leaves=("mean_leaves", "mean"),
            selected_param=("selected_param", "mean"),
        ),
        on=GROUP,
    )
    return table.sort_values(GROUP).reset_index(drop=True)


def compare(
    reps: pd.DataFrame,
    left: tuple[str, str],
    right: tuple[str, str],
    column: str = "loss",
    n_boot: int = 10_000,
    seed: int = 0,
) -> dict[str, float | str]:
    """Paired comparison of two (forest, measure) arms on the same repetitions."""

    def series(arm: tuple[str, str]) -> pd.Series:
        rows = reps[(reps["forest"] == arm[0]) & (reps["measure"] == arm[1])]
        return rows.set_index("rep")[column].sort_index()

    a, b = series(left), series(right)
    shared = a.index.intersection(b.index)
    diff = (a.loc[shared] - b.loc[shared]).to_numpy(dtype=np.float64)

    rng = np.random.default_rng(seed)
    draws = rng.choice(diff, size=(n_boot, diff.size), replace=True).mean(axis=1)
    low, high = np.percentile(draws, [2.5, 97.5])

    nonzero = diff[diff != 0.0]
    p_value = (
        float(stats.wilcoxon(nonzero).pvalue) if nonzero.size > 0 else float("nan")
    )
    return {
        "left": f"{left[0]}/{left[1]}",
        "right": f"{right[0]}/{right[1]}",
        "mean_diff": float(diff.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "left_wins": float((diff < 0).mean()),
        "p_value": p_value,
        "n_pairs": float(diff.size),
    }


def holm(p_values: FloatArray) -> FloatArray:
    order = np.argsort(p_values)
    m = p_values.size
    adjusted = np.empty(m, dtype=np.float64)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (m - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


HEADLINE: tuple[tuple[tuple[str, str], tuple[str, str], str], ...] = (
    (("pruned", "mdi_oob"), ("global", "mdi_oob"), "Q2 adaptive vs global (novelty)"),
    (("pruned", "mdi_oob"), ("full", "mdi_oob"), "Q1 pruning, held out"),
    (("full", "mdi_oob"), ("full", "mdi_train"), "Q3 held out alone"),
    (("pruned", "mdi_oob"), ("full", "mdi_train"), "combined vs standard practice"),
    (("pruned", "mdi_train"), ("full", "mdi_train"), "Q1 pruning, in sample"),
)


def report(frame: pd.DataFrame) -> None:
    table = summary(frame)
    show = [
        "forest", "measure", "total_mse", "noise_mass", "signal_mass",
        "explainable", "auc", "tau", "test_mse", "mean_leaves",
    ]
    for (scenario, n), block in table.groupby(["scenario", "n"]):
        print(f"\n=== {scenario}  n = {n} " + "=" * 52)
        print(block[show].to_string(index=False, float_format=lambda v: f"{v:9.4f}"))

    reps = per_rep(frame)
    for (scenario, n), block in reps.groupby(["scenario", "n"]):
        print(f"\n=== {scenario}  n = {n}  paired comparisons " + "=" * 30)
        results = [compare(block, a, b) | {"question": q} for a, b, q in HEADLINE]
        out = pd.DataFrame(results)
        out["p_holm"] = holm(out["p_value"].to_numpy(dtype=np.float64))
        print(
            out[["question", "mean_diff", "ci_low", "ci_high", "left_wins", "p_holm"]]
            .to_string(index=False, float_format=lambda v: f"{v:9.4f}")
        )
        print("  mean_diff < 0 means the left arm has the lower loss (is better)")


def plots(frame: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    features = per_feature(frame)

    for (scenario, n), block in features.groupby(["scenario", "n"]):
        arms = block.groupby(["forest", "measure"])
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        for (forest_name, measure), group in arms:
            group = group.sort_values("feature")
            label = f"{forest_name}/{measure}"
            axes[0].plot(group["feature"], group["bias"], marker="o", label=label)
            axes[1].plot(group["feature"], group["variance"], marker="o", label=label)
        axes[0].axhline(0.0, color="black", lw=0.8)
        axes[0].set_title(f"bias by feature ({scenario}, n={n})")
        axes[1].set_title(f"variance by feature ({scenario}, n={n})")
        for ax in axes:
            ax.set_xlabel("feature")
            ax.legend(fontsize=7)
        fig.tight_layout()
        path = out_dir / f"bias_variance_{scenario}_n{n}.png"
        fig.savefig(path, dpi=130)
        plt.close(fig)
        print(f"  wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/results.csv"))
    parser.add_argument("--figures", type=Path, default=Path("results/figures"))
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    frame = pd.read_csv(args.results)
    report(frame)
    if not args.no_plots:
        plots(frame, args.figures)


if __name__ == "__main__":
    main()
