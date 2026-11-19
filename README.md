# Adaptive Pruning and Out-of-Sample MDI

Does a variable importance measured on **adaptively pruned** trees and
evaluated on **held-out data** beat the standard MDI?

The study runs on **single regression trees** first; random forests are a
switch away (`--kind forest`) for the later extension.

**Start with [PROTOCOL.md](PROTOCOL.md)** — the problem, the aim, the measures and
what is left to run.

## Setup

Python 3.10 or later.

```sh
pip install -r requirements.txt
python -m pytest tests/ -q
```

## Run

```sh
python src/experiment.py --scenario S1 --sizes 250 1000 --reps 20 --out results/s1.csv
python src/analyse.py --results results/s1.csv
python src/realdata.py --reps 10

# the same protocol on forests, later
python src/experiment.py --kind forest --trees 300 --out results/s1_forest.csv
```

Figures land in `results/figures/`; `results/s1_path.csv` holds the importance
error at every pruning level.

## What is where

| File | Contents |
|---|---|
| `src/simulate.py` | additive generators — S1, S4 and the five additive Duroux & Scornet models; true importance `I_j = Var(f_j(X_j))` known analytically |
| `src/tree.py` | tree arrays detached from scikit-learn; `Model` = one tree or a forest |
| `src/models.py` | the pruning families (full, global depth control, cost-complexity, alpha-trimming), tuned by K-fold CV |
| `src/mdi.py` | the three measures: `mdi_train`, `mdi_oob`, `ufi_half` |
| `src/prune.py` | alpha-trimming (Algorithm 1 of Surjanovic, Henrey & Loughin 2024) |
| `src/realdata.py` | real datasets with injected known-junk columns (paused: simulated data first) |
| `src/experiment.py` | run loop; writes the results CSV and the pruning-path CSV |
| `src/analyse.py` | bias/variance/MSE, noise mass, AUC, Kendall tau, paired Wilcoxon |
| `tests/test_all.py` | the tests, including the exact training-data identity |

All split-sample arms are built on the same training split, so comparing measures
within an arm isolates the evaluation data, and comparing arms within a
measure isolates the pruning. Pruning levels are chosen by 5-fold CV inside the
training split; the validation split is used only for the importance measures.

## Two things to know before reading results

`ufi` carries **twice** the scale of the other measures on signal features, so
everything downstream uses `ufi_half`. Confirmed by
`test_ufi_carries_twice_the_scale_of_mdi_oob`.

The off-sample measures are **not** invariant to shifting `y`, so `y` is centred
by the training mean. Confirmed by `test_off_sample_measure_is_not_shift_invariant`.
