# PROTOCOL

## The problem

A random forest reports how important each variable is: when a split is made on
feature `j`, the spread of `y` inside the node drops, and MDI adds up that drop
over every split on `j`. Read plainly, it says *how much uncertainty about y did
this variable remove*.

It is wrong in two separate ways.

**Bug A — deep trees invent importance.** Grow a tree until every leaf holds one
point and the spread inside a leaf is zero, so the drops add up to the *entire*
variance of `y`. But `Var(y) = signal + noise`, so the measure has just shared the
**noise** variance out among the features as if it were importance. Take 100 points
of pure noise `y` and 5 pure-noise features — nothing related to anything — and MDI
still hands each feature ≈0.2. Scornet proves the limit: `Σ_j MDI → Var(m(X)) + σ²`
when the right answer is `Var(m(X))`. In our own runs, `mdi_train` gives the five
noise features a combined **0.37** when the truth is 0.

**Bug B — grading on the questions you studied.** The tree searched every candidate
split and kept the one that looked best *on the training data*. MDI then measures
how good that split was on **the same data**. The score is inflated by construction.

## The aim

Scornet's brief, 2026-10-24:

> « évaluer la pertinence d'un MDI construit sur des arbres (forêts) **moins
> profonds** (par exemple utilisant la dernière méthode), calculé sur un **jeu de
> données de validation**, à la fois sur jeux de données **réelles et simulées**. »

Each clause fixes one bug:

| Bug | Fix | Source |
|---|---|---|
| A, deep trees absorb noise | grow shallower trees, **adaptively** | [SHL24] alpha-trimming |
| B, same-data evaluation | measure the drop on **held-out** data | [ZH21] UFI |

Both fixes exist separately. **The contribution is combining them**, and *évaluer
la pertinence* means the answer is allowed to come back negative.

## Why this is not obvious

Scornet's Theorem 3 holds **as k→∞**: for a *theoretical* tree, deeper is *better* —
the population MDI converges to `Var(f_j(X_j))`. The bias is a finite-sample effect.
So pruning cuts both ways:

- it removes the noise absorption (good), but
- it moves you to a shallower tree that has not resolved the signal yet, so the
  measure **undershoots** (bad).

We have now measured both halves. Pruning **helps** the in-bag measure and **hurts**
the held-out one, consistently across scenarios and sample sizes. The two fixes
partly work against each other.

Worse, alpha is tuned by OOB **prediction** error while we care about **importance**
error, and nothing says the same amount of pruning is best for both. That is why
there is an oracle arm: without it, "pruning does not help importance" and "the
tuning picked the wrong amount" are indistinguishable.

## References

| Tag | Paper | Used for |
|---|---|---|
| [S20] | Scornet, *Trees, forests, and impurity-based variable importance in regression*, [arXiv:2001.04295](https://arxiv.org/abs/2001.04295), AIHP 59(1) 21–52, 2023 | why deep trees are wrong; the target `Var(f_j(X_j))` |
| [ZH21] | Zhou & Hooker, *Unbiased Measurement of Feature Importance in Tree-Based Methods*, [arXiv:1903.05179](https://arxiv.org/abs/1903.05179), TKDD 15(2), 2021 | **UFI**, the measure in the brief |
| [SHL24] | Surjanovic, Henrey & Loughin, *Alpha-Trimming*, [arXiv:2408.07151](https://arxiv.org/abs/2408.07151) | the adaptive pruning method |
| [L19] | Li, Wang, Basu, Kumbier & Yu, *A Debiased MDI Feature Importance Measure*, [arXiv:1906.10845](https://arxiv.org/abs/1906.10845) | **MDI-oob**, the competing held-out measure |

> arXiv:1903.05179 is **Zhou & Hooker**, not Li et al. — the two are often confused
> because both attack training-sample bias. The brief points at UFI.

## The measures

`mdi_train` — classic MDI, unnormalised. Never `feature_importances_`, which is
normalised to sum to 1.

`mdi_oob` [L19] — the same quantity re-evaluated off-sample:
mean over held-out rows of `(mu_child − mu_parent) · y`, with every `mu` taken from
the **in-bag** data.

`ufi_half` [ZH21] — in-bag impurity drop **plus** the drop measured on held-out rows
against the in-bag means.

> **UFI runs at twice the scale.** For a real split both terms converge to the same
> true impurity drop, so `UFI → 2·I_j` on signal features while noise still cancels
> to 0. [ZH21] only ever rank with it, so the constant never mattered to them; we
> compare numerically against `I_j`, so **it is halved everywhere**. Measured: raw
> `ufi/I_j` ≈ 1.3–1.9, and `ufi_half` then tracks `mdi_oob` to within 2.5%.

Two shared conventions: `y` is centred by the **training** mean, because the
held-out measures are not shift-invariant (in-bag `Σ f_T,k = 0`, off-sample it is
not); and **negative values are never clipped** — noise features must be free to
come out negative.

## The arms

Five forests. `full`, `global`, `pruned` and `oracle` are built on the same
training split, so comparing measures within a forest isolates the evaluation
data and comparing forests within a measure isolates the pruning. `standard` is
built on every point, because that is what practitioners actually do.

| Arm | Forest | Role |
|---|---|---|
| `standard` | full, all `n` points | what everyone does now — the thing to beat |
| `full` | full, training split | isolates the held-out effect |
| `global` | `min_samples_leaf` by OOB | shallower but **not** adaptive |
| `pruned` | `ccp_alpha` by OOB | adaptive — alpha-trimming's stand-in |
| `oracle` | `ccp_alpha` by importance error | simulation only; bounds what pruning could deliver |

**`pruned` vs `global` is the whole project.** Is *adaptive* pruning better than
merely *shallower* trees?

`ccp_alpha` stands in for alpha-trimming because the brief says *par exemple*: it
is adaptive, it is in scikit-learn today, and it let the pipeline reach real
numbers without a from-scratch implementation first.

## Status

Built and tested: the simulator with analytic `I_j` (S1, S4), the five arms, the
three measures, the full pruning path, metrics with Monte Carlo standard errors,
paired Wilcoxon with bootstrap intervals and Holm, and six figures. 35 tests,
`mypy --strict` clean.

Remaining, in order:

| # | Do | Why |
|---|---|---|
| 1 | **Force a pruning range that actually prunes.** OOB picks no pruning in 40–60% of reps and tree size only falls to 76–82% of full, so `pruned` and `global` are near-copies of `full` and the headline question has nothing to bite on. Read the answer off the pruning-path figure rather than the OOB-selected point | otherwise the novelty claim is untestable |
| 2 | **Validation-fraction sweep** — `--val-fraction 0.2 0.33 0.5` | at `n=250` the held-out set is 83 rows against ~106 leaves; untested |
| 3 | **Real data**: California Housing + Diabetes with `k` known-junk columns appended (permuted copies, i.i.d. continuous, categorical). Measure noise mass and AUC real-vs-injected | the brief asks for real data; no ground truth, so injected junk is the substitute |
| 4 | **Alpha-trimming proper** ([SHL24] §4) replacing `ccp_alpha` | the paper's actual method. Criterion: merge when `I_N + α·P_0 ≤ I_L + I_R + α·P_1`, with `I_N` an accumulated Gaussian log-likelihood, `P_0 = 2log n`, `P_1 = 5log n`, bottom-up, α on `{0, 0.1, …, 3.0}`, α=0 being the standard forest. Re-read §4 by eye first — this was transcribed from an automated PDF read. No public code; the authors work in Julia |

## The tests that matter

Six of the 35 carry the weight:

| Test | Catches |
|---|---|
| analytic `I_j` vs Monte Carlo | wrong ground truth |
| `Var(y) = ΣI_j + σ²` | wrong data generator |
| bootstrap masks consistent | off-by-one in-bag/OOB — silent and fatal |
| full forest `Σ mdi_train = Var(y)` | wrong MDI; also confirms we reproduced Bug A |
| **`mdi_oob` weighted in-bag ≡ `mdi_train`** | all of the node bookkeeping, means and routing at once. Exact, to 4e-16 |
| `ufi_half ≈ mdi_oob` | the factor-2 error that would corrupt every bias number |

## Reading the results

`results/figures/` holds six per scenario. The two that carry the argument:

- **`1_noise_mass`** — importance handed to features that have none. The bug and
  the fix in one picture.
- **`6_pruning_path`** — importance error against how hard we prune, with the
  OOB-selected level marked. This is the depth tension made visible.

Total MSE against `I_j` is **not** a sufficient headline. Each measure is nearly
unbiased exactly where the other fails — `mdi_train` on signal, `mdi_oob` on noise —
so total MSE can favour the biased measure purely through which error happens to be
squared larger. Read noise mass and AUC alongside it. `median_diff` disagreeing in
sign with `mean_diff` means the result is outlier-driven.

## Open questions for Scornet

1. **"Jeu de données de validation"** — a literal train/validation split, or
   out-of-bag points? We implemented the split, which also removes any reuse
   concern. OOB would be more sample-efficient.
2. **UFI or MDI-oob?** The brief links Zhou & Hooker. We compute both; UFI is
   currently the better performer.
3. **Is the oracle arm in scope** — "does alpha-trimming as published help", or
   "is there *any* pruning level that helps"? We assume both.
4. **Classification?** All three papers cover it; this is regression-only.

## Deliberately left out

Add only when the pain is real: resumable runs, `joblib` parallelism, Parquet,
config files, pinned versions, the leakage check (moot — alpha is chosen on OOB of
the training split, the measure on the validation split, so there is no reuse),
Protocol-A OOB-based measures, scenarios with correlated or interacting inputs
(no ground truth there, so `Var(f_j(X_j))` stops being the target at all).
