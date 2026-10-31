# Adaptive Pruning and Out-of-Sample MDI

Does a variable importance measured on **adaptively pruned** random forests and
evaluated on **held-out data** beat the standard MDI?

**Start with [PROTOCOL.md](PROTOCOL.md)** — the problem, the aim, the measures and
what is left to build.

## Setup

```sh
pip install -r requirements.txt
python -m pytest tests/ -q
```

## Run

```sh
python src/experiment.py --scenario S1 --sizes 250 1000 --reps 20 --out results/s1.csv
python src/analyse.py --results results/s1.csv
```

Figures land in `results/figures/`; `results/s1_path.csv` holds the importance
error at every pruning level.

## What is where

| File | Contents |
|---|---|
| `src/simulate.py` | additive data generator; true importance `I_j = Var(f_j(X_j))` known analytically |
| `src/forest.py` | the three forests (full, global depth control, adaptively pruned) and bootstrap bookkeeping |
| `src/mdi.py` | the three measures: `mdi_train`, `mdi_oob`, `ufi_half` |
| `src/experiment.py` | run loop, writes one long CSV |
| `src/analyse.py` | bias/variance/MSE, noise mass, AUC, Kendall tau, paired Wilcoxon |
| `tests/test_all.py` | the tests, including the exact in-bag identity |

All three forests are built on the same training split, so comparing measures
within a forest isolates the evaluation data, and comparing forests within a
measure isolates the pruning.

## Two things to know before reading results

`ufi` carries **twice** the scale of the other measures on signal features, so
everything downstream uses `ufi_half`. Confirmed by
`test_ufi_carries_twice_the_scale_of_mdi_oob`.

The off-sample measures are **not** invariant to shifting `y`, so `y` is centred
by the training mean. Confirmed by `test_off_sample_measure_is_not_shift_invariant`.
