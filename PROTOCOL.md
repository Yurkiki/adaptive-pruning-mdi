# PROTOCOL — what to build, in order

Companion to [SPEC.md](SPEC.md). SPEC is *what the study is*; this is *what to type*.

**5 files, ~500 lines, one CSV.** If it starts feeling like a framework, you've gone wrong.

---

## Step 0 — See the bug first

Before any structure. Run this:

```python
import numpy as np
from sklearn.ensemble import RandomForestRegressor

rng = np.random.default_rng(0)
X = rng.normal(size=(100, 5))     # 5 pure-noise features
y = rng.normal(size=100)          # y unrelated to all of them

rf = RandomForestRegressor(n_estimators=500, random_state=0).fit(X, y)

total = np.mean([t.tree_.compute_feature_importances(normalize=False).sum()
                 for t in rf.estimators_])
print(rf.feature_importances_)          # truth is 0,0,0,0,0 — it won't be
print(total, "vs Var(y) =", y.var())    # these will match
```

Nothing in `X` relates to `y`, yet every feature gets positive importance and the total lands on `Var(y)`. **That is the bug the project exists to fix.** Everything below is machinery for measuring it properly.

---

## The whole codebase

```
simulate.py      ~80 lines    fake data where we know the true answer
forest.py        ~80 lines    build the 4 forests, bootstrap bookkeeping
mdi.py          ~120 lines    the three importance measures
experiment.py   ~100 lines    run loop -> results.csv
analyse.py       ~80 lines    metrics, tests, plots
tests/test_all.py ~80 lines   the 6 tests that matter
```

**Data format: one CSV, long.** One row per `(n, rep, arm, feature)`:

```
n, rep, arm, feature, value, I_j, is_noise, oob_mse, test_mse, selected_param, mean_leaves
```

Run-level columns repeat on every row. That's redundant and completely fine at ~4000 rows — it buys you never writing a join. Open it in a text editor when something looks wrong.

**No config files.** Scenarios are a dict in `simulate.py`; run settings are function arguments.

**Seeds:** `rng = np.random.default_rng(rep)` for data, `random_state=rep` for forests. Pass them explicitly. Never call `np.random.*` directly.

---

## Stage 1 — Data with known answers (`simulate.py`)

| # | Do | Done when |
|---|---|---|
| 1 | Five component functions on `[0,1]`, each **centered** (`E[f]=0`) with its variance worked out by hand: linear, sine, quadratic, step, bump. A dict of `{name: (func, variance)}` is enough — no classes needed | each callable on a vector |
| 2 | Scenario **S1**: `d=10`, features 0–4 get functions, 5–9 get `f=0`. Returns the function list and the vector `I_j` | `I_j` has exactly 5 nonzero entries |
| 3 | `sample(n, rng)` → `(X, y)` with `X ~ U[0,1]^10`, `y = Σf_j(x_j) + N(0,σ²)`, `σ = sqrt(sum(I)/SNR)` | shapes right |
| 4 | **TEST ①** analytic `I_j` vs Monte Carlo at `N=10^6` | agree within `4·stderr` |
| 5 | **TEST ②** at `n=10^6`: `Var(y) ≈ sum(I_j) + σ²` | agrees to MC error |

> Tests ① and ② are the foundation. If your ground truth is wrong, every number in the project is wrong and nothing downstream will tell you.

---

## Stage 2 — The four forests (`forest.py`)

The four arms from SPEC §11:

| Arm | What |
|---|---|
| **A** | Full forest, MDI in-bag — what everyone does now |
| **B** | Full forest, MDI on validation — isolates the held-out effect |
| **C** | `min_samples_leaf` tuned by OOB, MDI on validation — shallower but *not* adaptive |
| **D** | `ccp_alpha` tuned by OOB, MDI on validation — adaptive (alpha-trimming's stand-in) |

| # | Do | Done when |
|---|---|---|
| 6 | `build_full(X, y, seed)` → `RandomForestRegressor(min_samples_leaf=1, max_features=1/3, random_state=seed)` | fits |
| 7 | Bootstrap bookkeeping — the fiddly bit, worth getting exactly right: <br>`from sklearn.ensemble._forest import _generate_sample_indices`<br>`idx = _generate_sample_indices(tree.random_state, n, n)`<br>`counts = np.bincount(idx, minlength=n)` → in-bag counts; OOB is `counts == 0` | returns `(B, n)` arrays |
| 8 | **TEST ③** each row of counts sums to `n`; OOB fraction ≈ 0.368; `oob_mask == (counts == 0)` | exact |
| 9 | `oob_mse(forest, X, y)` from your own masks. Then `build_global` and `build_pruned`: loop a grid, keep the argmin | selected value is in the grid |
| 10 | **TEST ④** `ccp_alpha=0` gives the same `node_count` as Full at the same seed; and assert `0 ∈ ccp_grid`, `1 ∈ msl_grid` | exact |

> Step 10's grid assertion means tuning can never do worse than Full. Two lines, saves a confusing afternoon.

---

## Stage 3 — The three measures (`mdi.py`)

| # | Do | Done when |
|---|---|---|
| 11 | `mdi_train(forest, counts)` — unnormalized, averaged over trees. **Never `feature_importances_`** (it's normalized to sum to 1) | matches `tree_.compute_feature_importances(normalize=False)` |
| 12 | **TEST ⑤** on a Full forest, `sum(mdi_train) ≈ Var(y)` | this is Step 0's bug, now formal |
| 13 | `mdi_oob(tree, X_eval, y_eval)` — the core formula, ~10 lines (sketch below) | returns a length-`d` vector |
| 14 | **TEST ⑥ ★ the most important test in the project** — `mdi_oob` evaluated on the **in-bag** points (with bootstrap weights) reproduces `mdi_train` **exactly** | agrees to ~1e-9 |
| 15 | `ufi(tree, X_eval, y_eval)` = in-bag impurity drop + the same drop measured on eval points against **in-bag** means | returns a length-`d` vector |
| 16 | **TEST ⑦** at large `n`, high SNR: `ufi_j / I_j → 2` while `mdi_oob_j / I_j → 1`. Then **use `ufi/2` everywhere** | ratios as stated |
| 17 | Centre `y` by the **training** mean before any out-of-sample measure. Never clip negatives — noise features must be free to come out negative | review |

The core formula, concretely:

```python
def mdi_oob(tree, X_eval, y_eval, d):
    t = tree.tree_
    path = tree.decision_path(X_eval)          # sparse (n_eval, n_nodes)
    mu = t.value[:, 0, 0]                      # in-bag node means
    out = np.zeros(d)
    for node in np.where(t.feature >= 0)[0]:   # internal nodes only
        rows = path[:, node].nonzero()[0]      # eval points reaching it
        if len(rows) == 0:
            continue
        L, R = t.children_left[node], t.children_right[node]
        went_left = np.asarray(path[rows, L].todense()).ravel() > 0
        mu_child = np.where(went_left, mu[L], mu[R])
        out[t.feature[node]] += np.sum((mu_child - mu[node]) * y_eval[rows])
    return out / len(X_eval)
```

> Test ⑥ is non-negotiable. It proves your node bookkeeping, your in-bag means and your routing are all correct in one shot. If it fails, **stop** — every later number is meaningless. If it passes, the hard part is behind you.

---

## Stage 4 — Run it (`experiment.py`)

| # | Do | Done when |
|---|---|---|
| 18 | `run_one(n, rep)`: sample `D` (size `n`), split 2/3 train + 1/3 validation, build the four arms, compute their measures, return rows | `4 × d` rows |
| 19 | Loop reps, concat, write `results.csv` | file exists |
| 20 | Smoke run: `n=250`, `R=3`, `B=100` | minutes, no NaNs |
| 21 | First real run: `n ∈ {250, 1000}`, `R=20`, `B=500` | the four-arm table |

> **Stop at 21 and read the numbers.** Noise mass should be ≈0 for B/C/D and clearly positive for A. If it isn't, the bug is in Stage 3 — more scenarios will not reveal it.

---

## Stage 5 — Analyse (`analyse.py`)

All from `results.csv` with `groupby`. No new abstractions.

| # | Do | Done when |
|---|---|---|
| 22 | Per feature: `bias = mean − I_j`, `var` (with **1/R**), `mse`. **TEST ⑧**: `mse == bias² + var` exactly — only true with the 1/R convention | exact |
| 23 | Noise mass (signed **and** absolute), signal mass vs `sum(I_j)` | A overshoots by ≈σ² |
| 24 | AUC signal-vs-noise (ties = 0.5), Kendall τ among signal features | sanity-checked on a handmade case |
| 25 | Per-rep loss `L_r = Σ_j (M̂ − I_j)²`; **D vs C** paired: mean difference, 95% CI from `scipy.stats.bootstrap`, `scipy.stats.wilcoxon`, win rate | the headline number |
| 26 | Summary table + two plots: bias–variance per feature, and MDI error vs pruning amount | — |

**D vs C is the whole project.** Is *adaptive* pruning better than merely *shallower* trees? D≈C is a perfectly good answer; it just has to be a measured one.

---

## Stage 6 — Only after Stage 5 works

| # | Do | Why |
|---|---|---|
| 27 | **Oracle arm**: pick the pruning amount that minimises `Σ(M̂ − I_j)²` instead of OOB error. ~15 lines once Step 18 exists | Without it, "pruning doesn't help" and "the tuning picked the wrong amount" are indistinguishable (SPEC §1.1) |
| 28 | **Scenario S4**: `I_j` across orders of magnitude, mixing smooth with step/localized functions | Where *locally* adaptive pruning should actually win — your best shot at a positive result |
| 29 | **Real data**: California Housing + Diabetes, with `k` known-junk columns appended (permuted copies + i.i.d. noise). Measure noise mass and AUC real-vs-injected | Scornet asked for real data; no ground truth, so injected junk is the substitute |
| 30 | **Alpha-trimming proper** ([SHL24] §4) replacing `ccp_alpha` in arm D | The paper's actual method. Re-read §4 by eye first — SPEC §5.4 was transcribed from an automated PDF read. Test `α=0` leaves the tree untouched, and reproduce one of their benchmarks before trusting it |

---

## The 6 tests that matter

Everything else is optional. These are not:

| | Test | Catches |
|---|---|---|
| ① | analytic `I_j` vs Monte Carlo | wrong ground truth |
| ② | `Var(y) = ΣI_j + σ²` | wrong data generator |
| ③ | bootstrap masks consistent | off-by-one in-bag/OOB — silent and fatal |
| ⑤ | Full forest `sum(mdi_train) = Var(y)` | wrong MDI; also confirms you reproduced the known bug |
| ⑥ | **`mdi_oob` in-bag ≡ `mdi_train`** | everything in Stage 3 at once |
| ⑦ | `ufi/2 ≈ I_j` | a factor-2 error that would corrupt every bias number |

---

## Deliberately left out

Not oversights — add any of them only when you feel the actual pain:

- **Parquet, separate metadata table** → one CSV is inspectable and fast enough at this size
- **YAML config, dataclasses** → function arguments; two users don't need a config format
- **Resumable runs** → add if a full run starts crashing midway
- **`joblib` parallelism** → add when runtime annoys you, not before. Keep seeds per-rep so it stays reproducible
- **CI-coverage calibration test** → `scipy` is already tested
- **Pinned versions** → a `requirements.txt` with names is fine
- **Protocol A (OOB instead of validation)**, MDI-oob *and* UFI everywhere, scenarios S2/S3/S5, correlated and interaction scenarios (S6/S7), the leakage check, `R=100` → all in SPEC, all off the critical path

---

## One blocking question

Does *"jeu de données de validation"* mean a literal train/validation split, or out-of-bag points? It changes Steps 18 and the arm definitions. **Ask Scornet before Step 18** — Steps 1–17 are unaffected, so there's no reason to wait to start.
