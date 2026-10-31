"""Metrics, paired comparisons and figures, all derived from the results CSVs."""

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

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#e6e5e1"
SERIES = {"mdi_train": "#2a78d6", "mdi_oob": "#eb6834", "ufi_half": "#1baf7a"}
ACCENT = "#4a3aa7"

MEASURE_ORDER = ["mdi_train", "mdi_oob", "ufi_half"]
FOREST_ORDER = ["standard", "full", "global", "ccp", "alpha", "oracle"]


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
    table["forest"] = pd.Categorical(table["forest"], FOREST_ORDER, ordered=True)
    table["measure"] = pd.Categorical(table["measure"], MEASURE_ORDER, ordered=True)
    return table.sort_values(["scenario", "n", "forest", "measure"]).reset_index(drop=True)


# --- paired comparison -----------------------------------------------------


def compare(
    reps: pd.DataFrame,
    left: tuple[str, str],
    right: tuple[str, str],
    column: str = "loss",
    n_boot: int = 10_000,
    seed: int = 0,
) -> dict[str, float | str]:
    def series(arm: tuple[str, str]) -> pd.Series:
        rows = reps[(reps["forest"] == arm[0]) & (reps["measure"] == arm[1])]
        return rows.set_index("rep")[column].sort_index()

    a, b = series(left), series(right)
    shared = a.index.intersection(b.index)
    diff = (a.loc[shared] - b.loc[shared]).to_numpy(dtype=np.float64)
    if diff.size == 0:
        return {"left": f"{left[0]}/{left[1]}", "right": f"{right[0]}/{right[1]}",
                "mean_diff": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "left_wins": float("nan"),
                "median_diff": float("nan"), "p_value": float("nan"), "n_pairs": 0.0}

    rng = np.random.default_rng(seed)
    draws = rng.choice(diff, size=(n_boot, diff.size), replace=True).mean(axis=1)
    low, high = np.percentile(draws, [2.5, 97.5])
    nonzero = diff[diff != 0.0]
    p_value = float(stats.wilcoxon(nonzero).pvalue) if nonzero.size > 0 else float("nan")
    return {
        "left": f"{left[0]}/{left[1]}",
        "right": f"{right[0]}/{right[1]}",
        "mean_diff": float(diff.mean()),
        "median_diff": float(np.median(diff)),
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


Comparison = tuple[tuple[str, str], tuple[str, str], str]

HEADLINE: tuple[Comparison, ...] = (
    (("alpha", "ufi_half"), ("global", "ufi_half"), "alpha-trimming vs global pruning"),
    (("alpha", "ufi_half"), ("full", "ufi_half"), "alpha-trimming, measured held out"),
    (("alpha", "ufi_half"), ("ccp", "ufi_half"), "alpha-trimming vs cost-complexity"),
    (("alpha", "mdi_train"), ("full", "mdi_train"), "alpha-trimming, measured in sample"),
    (("full", "ufi_half"), ("full", "mdi_train"), "held out alone  [UFI]"),
    (("full", "mdi_oob"), ("full", "mdi_train"), "held out alone  [MDI-oob]"),
    (("full", "ufi_half"), ("full", "mdi_oob"), "UFI vs MDI-oob, same forest"),
    (("alpha", "ufi_half"), ("standard", "mdi_train"), "new method vs standard practice"),
    (("oracle", "ufi_half"), ("alpha", "ufi_half"), "oracle vs OOB alpha"),
)


def comparison_table(reps: pd.DataFrame) -> pd.DataFrame:
    rows = [compare(reps, a, b) | {"comparison": label} for a, b, label in HEADLINE]
    out = pd.DataFrame(rows)
    usable = out["p_value"].notna()
    out["p_holm"] = np.nan
    out.loc[usable, "p_holm"] = holm(out.loc[usable, "p_value"].to_numpy(dtype=np.float64))
    return out


# --- text report -----------------------------------------------------------


def report(frame: pd.DataFrame) -> None:
    table = summary(frame)
    show = [
        "forest", "measure", "total_mse", "total_mse_se", "relative_loss",
        "noise_mass", "noise_mass_se", "signal_mass", "explainable",
        "auc", "tau", "excess_risk", "mean_leaves",
    ]
    for (scenario, n), block in table.groupby(["scenario", "n"], observed=True):
        print(f"\n=== {scenario}  n = {n} " + "=" * 60)
        print(block[show].to_string(index=False, float_format=lambda v: f"{v:8.4f}"))

    reps = per_rep(frame)
    for (scenario, n), block in reps.groupby(["scenario", "n"], sort=False):
        print(f"\n=== {scenario}  n = {n}  paired comparisons " + "=" * 32)
        out = comparison_table(block)
        print(
            out[["comparison", "mean_diff", "median_diff", "ci_low", "ci_high",
                 "left_wins", "p_holm"]]
            .to_string(index=False, float_format=lambda v: f"{v:8.4f}")
        )
    print("\n  mean_diff < 0 means the left arm has the lower loss (is better)")
    print("  median_diff disagreeing in sign with mean_diff means outlier-driven")


# --- figures ---------------------------------------------------------------


def _style(ax: object) -> None:
    import matplotlib.axes

    assert isinstance(ax, matplotlib.axes.Axes)
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6, linestyle="-", zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK_SOFT, labelsize=8, length=3, width=0.8)
    ax.title.set_color(INK)
    ax.xaxis.label.set_color(INK_SOFT)
    ax.yaxis.label.set_color(INK_SOFT)


def plots(frame: pd.DataFrame, path_frame: pd.DataFrame | None, out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.figure
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    table = summary(frame)
    features = per_feature(frame)
    reps = per_rep(frame)
    scenario = str(frame["scenario"].iloc[0])
    sizes = sorted(frame["n"].unique())
    written: list[Path] = []

    def finish(fig: object, name: str) -> None:
        assert isinstance(fig, matplotlib.figure.Figure)
        fig.patch.set_facecolor(SURFACE)
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
        target = out_dir / f"{scenario}_{name}.png"
        fig.savefig(target, dpi=150, facecolor=SURFACE)
        plt.close(fig)
        written.append(target)

    # 1. noise mass -- the bug and the fix, in one picture
    fig, axes = plt.subplots(1, len(sizes), figsize=(5.2 * len(sizes), 4.2), squeeze=False)
    for ax, n in zip(axes[0], sizes):
        block = table[(table["n"] == n) & (table["forest"] == "full")]
        block = block.sort_values("measure")
        labels = block["measure"].astype(str).tolist()
        heights = block["noise_mass"].to_numpy(dtype=np.float64)
        errors = block["noise_mass_se"].to_numpy(dtype=np.float64)
        colours = [SERIES[m] for m in labels]
        bars = ax.bar(labels, heights, color=colours, width=0.55, zorder=3)
        ax.errorbar(labels, heights, yerr=errors, fmt="none", ecolor=INK_SOFT,
                    elinewidth=1.0, capsize=4, zorder=4)
        ax.axhline(0.0, color=INK, linewidth=1.0, zorder=2)
        for bar, value in zip(bars, heights):
            ax.annotate(f"{value:.3f}", (bar.get_x() + bar.get_width() / 2, value),
                        textcoords="offset points", xytext=(0, 6 if value >= 0 else -14),
                        ha="center", fontsize=8, color=INK)
        _style(ax)
        ax.set_title(f"n = {n}", fontsize=10)
        ax.set_ylabel("summed importance of the 5 noise features")
    fig.suptitle(
        f"{scenario}: importance handed to features that have none (truth = 0)",
        fontsize=11, color=INK,
    )
    finish(fig, "1_noise_mass")

    # 2. recovery of the true importance, per feature
    fig, axes = plt.subplots(1, len(sizes), figsize=(5.2 * len(sizes), 4.4), squeeze=False)
    for ax, n in zip(axes[0], sizes):
        block = features[(features["n"] == n) & (features["forest"] == "full")]
        top = float(block["true_importance"].max()) * 1.15
        ax.plot([0, top], [0, top], color=INK_SOFT, linewidth=1.0, zorder=2)
        ax.annotate("perfect recovery", (top * 0.72, top * 0.76), fontsize=8,
                    color=INK_SOFT, rotation=38)
        for measure in MEASURE_ORDER:
            rows = block[block["measure"] == measure]
            ax.plot(rows["true_importance"], rows["mean_value"], linestyle="none",
                    marker="o", markersize=8, color=SERIES[measure], label=measure,
                    markeredgecolor=SURFACE, markeredgewidth=1.6, zorder=3)
        _style(ax)
        ax.set_title(f"n = {n}", fontsize=10)
        ax.set_xlabel("true importance  $I_j$")
        ax.set_ylabel("estimated importance")
        ax.legend(frameon=False, fontsize=8, labelcolor=INK_SOFT)
    fig.suptitle(f"{scenario}: recovery of the true importance", fontsize=11,
                 color=INK)
    finish(fig, "2_recovery")

    # 3. is the error bias or variance?
    fig, axes = plt.subplots(1, len(sizes), figsize=(5.2 * len(sizes), 4.2), squeeze=False)
    for ax, n in zip(axes[0], sizes):
        block = features[(features["n"] == n) & (features["forest"] == "full")]
        agg = block.groupby("measure", observed=True).apply(
            lambda g: pd.Series(
                {"bias2": float((g["bias"] ** 2).sum()), "variance": float(g["variance"].sum())}
            ),
            include_groups=False,
        ).reindex(MEASURE_ORDER)
        ax.bar(agg.index.astype(str), agg["bias2"], color=SERIES["mdi_train"],
               width=0.55, label="bias$^2$", zorder=3)
        ax.bar(agg.index.astype(str), agg["variance"], bottom=agg["bias2"],
               color=SERIES["ufi_half"], width=0.55, label="variance",
               edgecolor=SURFACE, linewidth=2.0, zorder=3)
        _style(ax)
        ax.set_title(f"n = {n}", fontsize=10)
        ax.set_ylabel("total MSE against $I_j$")
        ax.legend(frameon=False, fontsize=8, labelcolor=INK_SOFT)
    fig.suptitle(f"{scenario}: the error is almost all bias, not variance",
                 fontsize=11, color=INK)
    finish(fig, "3_bias_vs_variance")

    # 4. paired comparisons
    fig, axes = plt.subplots(1, len(sizes), figsize=(6.6 * len(sizes), 4.6), squeeze=False)
    for ax, n in zip(axes[0], sizes):
        out = comparison_table(reps[reps["n"] == n]).iloc[::-1]
        positions = np.arange(len(out))
        significant = out["p_holm"].to_numpy(dtype=np.float64) < 0.05
        colours = [ACCENT if s else INK_SOFT for s in significant]
        for pos, row, colour in zip(positions, out.itertuples(), colours):
            ax.plot([row.ci_low, row.ci_high], [pos, pos], color=colour, linewidth=2.0,
                    solid_capstyle="round", zorder=3)
            ax.plot(row.mean_diff, pos, marker="o", markersize=8, color=colour,
                    markeredgecolor=SURFACE, markeredgewidth=1.4, zorder=4)
        ax.axvline(0.0, color=INK, linewidth=1.0, zorder=2)
        ax.set_yticks(positions)
        ax.set_yticklabels(out["comparison"], fontsize=8)
        _style(ax)
        ax.set_title(f"n = {n}   (filled = significant after Holm)", fontsize=10)
        ax.set_xlabel("mean paired difference in loss   (negative = left arm better)")
    fig.suptitle(f"{scenario}: paired comparisons with 95% bootstrap intervals",
                 fontsize=11, color=INK)
    finish(fig, "4_comparisons")

    # 5. per-feature bias
    fig, axes = plt.subplots(1, len(sizes), figsize=(5.2 * len(sizes), 4.2), squeeze=False)
    for ax, n in zip(axes[0], sizes):
        block = features[(features["n"] == n) & (features["forest"] == "full")]
        for measure in MEASURE_ORDER:
            rows = block[block["measure"] == measure].sort_values("feature")
            ax.plot(rows["feature"], rows["bias"], marker="o", markersize=6,
                    linewidth=1.6, color=SERIES[measure], label=measure,
                    markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=3)
        ax.axhline(0.0, color=INK, linewidth=1.0, zorder=2)
        n_signal = int((~block["is_noise"]).sum() / block["measure"].nunique())
        ax.axvspan(n_signal - 0.5, block["feature"].max() + 0.5, color=GRID, alpha=0.6,
                   zorder=1)
        ax.annotate("noise features", (n_signal + 0.1, ax.get_ylim()[1] * 0.85),
                    fontsize=8, color=INK_SOFT)
        _style(ax)
        ax.set_title(f"n = {n}", fontsize=10)
        ax.set_xlabel("feature")
        ax.set_ylabel("bias")
        ax.legend(frameon=False, fontsize=8, labelcolor=INK_SOFT)
    fig.suptitle(f"{scenario}: where each measure goes wrong", fontsize=11, color=INK)
    finish(fig, "5_bias_by_feature")

    # 6. does any pruning level help? the whole path, not just the chosen point
    if path_frame is not None and len(path_frame) > 0:
        methods = [m for m in ("alpha", "ccp") if m in set(path_frame["method"])]
        fig, axes = plt.subplots(len(methods), len(sizes),
                                 figsize=(5.6 * len(sizes), 4.2 * len(methods)),
                                 squeeze=False)
        for row, method in enumerate(methods):
            for column, n in enumerate(sizes):
                ax = axes[row][column]
                block = path_frame[
                    (path_frame["n"] == n) & (path_frame["method"] == method)
                ]
                levels = sorted(block["level"].unique())
                position = {value: index for index, value in enumerate(levels)}
                for measure in MEASURE_ORDER:
                    rows = (
                        block[block["measure"] == measure]
                        .groupby("level", as_index=False)["importance_error"]
                        .mean()
                        .sort_values("level")
                    )
                    ax.plot([position[v] for v in rows["level"]],
                            rows["importance_error"], marker="o", markersize=6,
                            linewidth=1.8, color=SERIES[measure], label=measure,
                            markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=3)
                picked = (
                    block[block["measure"] == MEASURE_ORDER[0]]
                    .loc[lambda d: d.groupby("rep")["oob_mse"].idxmin()]["level"]
                )
                chosen = float(np.median([position[v] for v in picked]))
                ax.axvline(chosen, color=ACCENT, linewidth=1.6, zorder=2)
                ax.annotate("OOB picks here", (chosen, 0.03),
                            xycoords=("data", "axes fraction"), fontsize=8,
                            color=ACCENT, ha="center",
                            bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 2})
                sizes_at = (
                    block.groupby("level")["mean_leaves"].mean().reindex(levels)
                )
                ax.set_xticks(range(len(levels)))
                ax.set_xticklabels(
                    [f"{v:g}\n{int(round(s))} lv" for v, s in zip(levels, sizes_at)],
                    fontsize=7,
                )
                _style(ax)
                label = "alpha-trimming" if method == "alpha" else "cost-complexity"
                ax.set_title(f"{label},  n = {n}", fontsize=10)
                ax.set_xlabel("alpha" if method == "alpha" else "ccp_alpha / Var(y)")
                ax.set_ylabel("importance error  $\\sum_j (\\hat{M}_j - I_j)^2$")
                if row == 0 and column == 0:
                    ax.legend(frameon=False, fontsize=8, labelcolor=INK_SOFT)
        fig.suptitle(f"{scenario}: importance error against how hard we prune",
                     fontsize=11, color=INK)
        finish(fig, "6_pruning_path")

    for target in written:
        print(f"  wrote {target}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/results.csv"))
    parser.add_argument("--path", type=Path, default=None)
    parser.add_argument("--figures", type=Path, default=Path("results/figures"))
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    frame = pd.read_csv(args.results)
    report(frame)
    if not args.no_plots:
        candidate = args.path or args.results.with_name(args.results.stem + "_path.csv")
        path_frame = pd.read_csv(candidate) if candidate.exists() else None
        plots(frame, path_frame, args.figures)


if __name__ == "__main__":
    main()
