# SPEC

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
still hands each feature ≈0.2. For a tree grown to purity, `Σ_j MDI` equals the
empirical `Var(y)`, which tends to `Var(m(X)) + σ²` when the right answer is
`Var(m(X))` [S20].

**Bug B — grading on the questions you studied.** The tree searched every candidate
split and kept the one that looked best *on the training data*. MDI then measures
how good that split was on **the same data**. The score is inflated by construction.

## The aim

Scornet's brief (email, October 2026):

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

Scornet's Theorem 3 [S20] holds **as the depth k→∞** (and, for the empirical
tree, as n→∞): deeper is *better*, and MDI converges to `Var(f_j(X_j))`. The bias
is a finite-sample effect.
So pruning cuts both ways:

- it removes the noise absorption (good), but
- it moves you to a shallower tree that has not resolved the signal yet, so the
  measure **undershoots** (bad).

## References

| Tag | Paper | Used for |
|---|---|---|
| [S20] | Scornet, *Trees, forests, and impurity-based variable importance in regression*, [arXiv:2001.04295](https://arxiv.org/abs/2001.04295), AIHP 59(1) 21–52, 2023 | why deep trees are wrong; the target `Var(f_j(X_j))` |
| [ZH21] | Zhou & Hooker, *Unbiased Measurement of Feature Importance in Tree-Based Methods*, [arXiv:1903.05179](https://arxiv.org/abs/1903.05179), ACM TKDD 15(2), 2021 | **UFI** ("unbiased feature importance"), the measure in the brief |
| [SHL24] | Surjanovic, Henrey & Loughin, *Alpha-Trimming: Locally Adaptive Tree Pruning for Random Forests*, [arXiv:2408.07151](https://arxiv.org/abs/2408.07151), 2024 | the adaptive pruning method |
| [L19] | Li, Wang, Basu, Kumbier & Yu, *A Debiased MDI Feature Importance Measure for Random Forests*, [arXiv:1906.10845](https://arxiv.org/abs/1906.10845), NeurIPS 2019 | **MDI-oob**, the competing held-out measure |

> arXiv:1903.05179 is **Zhou & Hooker**, not Li et al. — the two are often confused
> because both attack training-sample bias. The brief points at UFI.

## The measures

`mdi_train` — classic MDI, unnormalised. Never `feature_importances_`, which is
normalised to sum to 1.

`mdi_oob` [L19] — the same quantity re-evaluated off-sample:
mean over held-out rows of `(mu_child − mu_parent) · y`, with every `mu` taken from
the **training** (in-bag) data. [L19] define it on the forest's out-of-bag points; here it is evaluated on the
**validation split** instead, which is the same formula on different held-out
data rather than their estimator as published.

`ufi` [ZH21] — training impurity drop **plus** the same drop measured on
held-out rows against the training means.

Two shared conventions: `y` is centred by the **training** mean, because the
held-out measures are not shift-invariant (in-bag `Σ f_T,k = 0`, off-sample it is
not); and **negative values are never clipped** — noise features must be free to
come out negative.

## The arms

Five arms. All except `standard` are built on the same training split, so
comparing measures within an arm isolates the evaluation data and comparing arms
within a measure isolates the pruning. `standard` is built on every point,
because that is what practitioners actually do.

| Arm | Model | Role |
|---|---|---|
| `standard` | grown to purity on all `n` points, `mdi_train` only | what everyone does now: the thing to beat |
| `full` | grown to purity, training split | isolates the held-out effect |
| `global` | `min_samples_leaf` chosen out of bag | shallower but **not** adaptive |
| `ccp` | cost-complexity `ccp_alpha` chosen out of bag | a second adaptive method |
| `alpha` | alpha-trimming, alpha chosen out of bag | **the method under study** |

**`alpha` vs `global` is the whole project.** Is *adaptive* pruning better than
merely *shallower* trees?

`alpha` is **alpha-trimming itself** ([SHL24] Algorithm 1), implemented in
`src/prune.py` from the paper's section 3.1.


## To do

- [x] simulated data with a known true `I_j`
- [x] MDI on the training data, to reproduce bug A
- [x] the same drops measured on held-out data
- [ ] alpha-trimming, from the paper's Algorithm 1
- [ ] the arms, the metrics, the paired tests
- [ ] real data with injected junk columns
- [ ] a low-SNR scenario, which is where [SHL24] expect pruning to pay
- [ ] reproduce one [SHL24] benchmark as a check on our transcription
