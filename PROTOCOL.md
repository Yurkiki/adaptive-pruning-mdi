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

Both adaptive methods are present: `ccp` is scikit-learn cost-complexity pruning,
`alpha` is **alpha-trimming itself** ([SHL24] Algorithm 1), implemented in
`src/prune.py` from the paper's section 3.1.

> **Alpha-trimming needs leaves with variance in them.** Its merge test weighs the
> single-mean model against the two-mean model, and a leaf grown to purity has
> `sigma^2 = 0`, which sends the stump log-likelihood to infinity: the test can
> never fire and pruning silently does nothing. The base forest for the `alpha`
> arm therefore uses `min_samples_leaf = 5`, the node size [SHL24] treat as a
> fully grown forest. So `alpha = 0` is "the standard random forest" in *their*
> sense, which is not the same object as our `full` arm. A test pins this down:
> `test_pure_leaves_make_alpha_trimming_inert`.

This matters because it is what made the pruning question answerable at all.
Cost-complexity tuned by OOB only ever shrank trees to 76-82% of full, so `ccp`
and `global` were near-copies of `full`. Alpha-trimming spans the real range —
41 leaves at `alpha = 0`, 20 at `alpha = 1`, 6 at `alpha = 2`.

## Status

Every clause of the brief now has code behind it:

| His words | Where |
|---|---|
| arbres moins profonds, de manière adaptative | `src/prune.py` — alpha-trimming, plus cost-complexity as a second adaptive method |
| calculé sur un jeu de données de validation | `src/experiment.py` — literal train/validation split |
| jeux de données simulées | `src/simulate.py` — S1, S4, with analytic `I_j` |
| jeux de données réelles | `src/realdata.py` — diabetes, California Housing, with injected junk columns |
| évaluer la pertinence | `src/analyse.py` — metrics, paired tests, six figures |

42 tests, `mypy --strict` clean.

Remaining, in order:

| # | Do | Why |
|---|---|---|
| 1 | **Validation-fraction sweep** — `--val-fraction 0.2 0.33 0.5` | at `n=250` the held-out set is 83 rows against ~106 leaves; untested |
| 2 | **Low-SNR scenario.** [SHL24] say pruning pays off most when the signal-to-noise ratio is low, and every run so far is at SNR = 4 | the condition their method is designed for is the one we have not tried |
| 3 | **The full alpha grid** `{0, 0.1, ..., 3.0}` for final numbers | runs use a 9-point grid for speed; the paper's is 31 |
| 4 | **Reproduce one [SHL24] benchmark** | independent check that our transcription of Algorithm 1 is faithful |

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
