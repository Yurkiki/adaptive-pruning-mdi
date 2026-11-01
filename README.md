# Adaptive Pruning and Out-of-Sample MDI

Does a variable importance measured on **adaptively pruned** forests and
evaluated on **held-out data** beat the standard MDI?

**Start with [SPEC.md](SPEC.md)** — the problem, the aim, the measures and what is
left to run.

## Setup

Python 3.10 or later.

```sh
pip install -r requirements.txt
python -m pytest tests/ -q
```

## Run

```sh
python src/experiment.py --sizes 250 1000 --reps 20 --out results/s1.csv
python src/analyse.py --results results/s1.csv
```

Figures land in `results/figures/`.
