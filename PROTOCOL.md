# PROTOCOL

## The problem

A regression tree (and, averaged over its trees, a random forest) reports how
important each variable is: when a split is made on feature `j`, the spread of
`y` inside the node drops, and MDI adds up that drop over every split on `j`. Read plainly, it says *how much uncertainty about y did
this variable remove*.

It is wrong in two separate ways.

**Bug A — deep trees invent importance.** Grow a tree until every leaf holds one
point and the spread inside a leaf is zero, so the drops add up to the *entire*
variance of `y`. But `Var(y) = signal + noise`, so the measure has just shared the
**noise** variance out among the features as if it were importance. Take 100 points
of pure noise `y` and 5 pure-noise features — nothing related to anything — and MDI
still hands each feature ≈0.2. For a tree grown to purity, `Σ_j MDI` equals the
empirical `Var(y)`, which tends to `Var(m(X)) + σ²` when the right answer is
`Var(m(X))` [S20]. In our forest runs, `mdi_train` gave the five noise features a
combined **0.37** when the truth is 0; a first single-tree run (`n = 1000`) gives 0.18.

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

## Scope: single trees first, forests later

The study is run on **one regression tree** first (`--kind tree`, the default),
and only then, potentially, extended to random forests (`--kind forest`).

- A single tree is where the two biases are cleanest. There is no averaging to
  hide them, and alpha = 1 is exactly the regime [SHL24] describe as "a good
  standalone predictor".
- The code treats a model as a tuple of trees. A tree is a model of one tree,
  every measure is computed per tree and then averaged, so the forest extension
  goes through the same code with no change.
- **A single tree has no bootstrap, so it has no OOB points.** Every pruning
  level (`min_samples_leaf`, `ccp_alpha`, alpha) is therefore chosen by **5-fold
  cross-validation on the training split**, the classical CART answer. The
  validation split stays reserved for the importance measures, so no point is
  used twice. Forests use the same CV for now; OOB tuning would be cheaper and
  is left for when forests become the focus.
- The CART tree considers every feature at each split (`max_features = 1.0`).
  The forest uses a third of them.

## Why this is not obvious

Scornet's Theorem 3 [S20] holds **as the depth k→∞** (and, for the empirical
tree, as n→∞): deeper is *better*, and MDI converges to `Var(f_j(X_j))`. The bias
is a finite-sample effect.
So pruning cuts both ways:

- it removes the noise absorption (good), but
- it moves you to a shallower tree that has not resolved the signal yet, so the
  measure **undershoots** (bad).

On forests we measured both halves: pruning **helped** the in-bag measure and
**hurt** the held-out one, consistently across scenarios and sample sizes, so the
two fixes partly worked against each other. Whether this holds for single trees is
the first thing to establish (Status, item 0).

Worse, alpha is tuned by cross-validated **prediction** error while we care about **importance**
error, and nothing says the same amount of pruning is best for both. That is why
there is an oracle arm: without it, "pruning does not help importance" and "the
tuning picked the wrong amount" are indistinguishable.

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
the **training** (in-bag) data. [L19] define it on a forest's out-of-bag points; a
single tree has none, so here it is evaluated on the **validation split**. It is
Li et al.'s formula on different held-out data, not their estimator as published.

`ufi_half` [ZH21] — training impurity drop **plus** the drop measured on held-out
rows against the training means, halved (below).

> **UFI runs at twice the scale.** For a real split both terms converge to the same
> true impurity drop, so `UFI → 2·I_j` on signal features while noise still cancels
> to 0. [ZH21] only ever rank with it, so the constant never mattered to them; we
> compare numerically against `I_j`, so **it is halved everywhere**. Measured on
> forests: raw `ufi/I_j` ≈ 1.3–1.9, and `ufi_half` then tracks `mdi_oob` to within
> 2.5%. A single tree is noisier: within about 11%.

Two shared conventions: `y` is centred by the **training** mean, because the
held-out measures are not shift-invariant (in-bag `Σ f_T,k = 0`, off-sample it is
not); and **negative values are never clipped** — noise features must be free to
come out negative.

## The arms

Seven arms. All except `standard` are built on the same training split, so
comparing measures within an arm isolates the evaluation data and comparing arms
within a measure isolates the pruning. `standard` is built on every point,
because that is what practitioners actually do.

| Arm | Model (a tree by default) | Role |
|---|---|---|
| `standard` | grown to purity, all `n` points, `mdi_train` only | what everyone does now: the thing to beat |
| `full` | grown to purity, training split | isolates the held-out effect |
| `global` | `min_samples_leaf` chosen by CV | shallower but **not** adaptive |
| `ccp` | cost-complexity `ccp_alpha` chosen by CV | a second adaptive method |
| `alpha0` | `min_samples_leaf = 5` base, no trimming | the clean baseline for `alpha` |
| `alpha` | alpha-trimming, alpha chosen by CV | **the method under study** |
| `oracle` | alpha chosen by `ufi_half` importance error | simulation only; bounds what pruning could deliver |
| `oracle_oob` | alpha chosen by `mdi_oob` importance error | the same bound for the validation-only measure; the two oracles land on different levels in roughly a third of reps, so one cannot stand in for the other |

**`alpha` vs `global` is the whole project.** Is *adaptive* pruning better than
merely *shallower* trees? `alpha` vs `alpha0` isolates the pruning alone.

`alpha` is **alpha-trimming itself** ([SHL24] Algorithm 1), implemented in
`src/prune.py` from the paper's section 3.1.

> **Alpha-trimming needs leaves with variance in them.** Its merge test weighs the
> single-mean model against the two-mean model, and a leaf grown to purity has
> `sigma^2 = 0`, which sends the stump log-likelihood to infinity: the test can
> never fire and pruning silently does nothing. The base tree for the `alpha`
> arm therefore uses `min_samples_leaf = 5`, the node size [SHL24] treat as
> fully grown. So `alpha = 0` is "unpruned" in *their* sense, which is not the
> same object as our `full` arm; hence `alpha0`. A test pins this down:
> `test_pure_leaves_make_alpha_trimming_inert`.

This mattered for forests: there, cost-complexity tuned by OOB only ever shrank
trees to 76-82% of full, so `ccp` and `global` were near-copies of `full`, while
alpha-trimming spanned the real range (41 leaves at `alpha = 0`, 20 at `alpha = 1`,
6 at `alpha = 2`). A single tree is pruned much harder by CV: in a first run at
`n = 1000`, from 667 leaves (`full`) to about 30 (`global`, `alpha`) and 12 (`ccp`).

## Status

Every clause of the brief now has code behind it:

| His words | Where |
|---|---|
| arbres moins profonds, de manière adaptative | `src/prune.py` — alpha-trimming, plus cost-complexity as a second adaptive method |
| calculé sur un jeu de données de validation | `src/experiment.py` — literal train/validation split; pruning tuned by CV inside the training split |
| jeux de données simulées | `src/simulate.py` — S1, S4, with analytic `I_j` |
| jeux de données réelles | `src/realdata.py` — built, but **paused on Scornet's instruction** (see below) |
| évaluer la pertinence | `src/analyse.py` — metrics, paired tests, six figures |

48 tests. Model code: `src/tree.py` (tree arrays, `Model`), `src/models.py` (families, CV tuning).

Remaining, in order:

| # | Do | Why |
|---|---|---|
| 0 | **Re-run every finding below on single trees** (S1, S4, several `n` and SNR) | the numbers below were measured on forests; finding 6 has now been redone, see below |
| 1 | **Validation-fraction sweep** — `--val-fraction 0.2 0.33 0.5` | at `n=250` the held-out set is 83 rows against 167 leaves in the full tree; untested |
| 2 | **Low-SNR scenario.** [SHL24] say pruning pays off most when the signal-to-noise ratio is low, and every run so far is at SNR = 4 | the condition their method is designed for is the one we have not tried |
| 3 | **A finer alpha grid** `{0, 0.1, ..., 3.0}` for final numbers (`ModelConfig.alpha_grid`) | runs use an 11-point grid for speed |
| 4 | **Reproduce one [SHL24] benchmark** | independent check that our transcription of Algorithm 1 is faithful |
| 5 | **Run the Duroux--Scornet models** (below) beside S1 and S4 | scenarios Scornet pointed us at; wider `d`, sparser signal |

## Simulated data only, for now

Scornet, on the scope:

> « données simulées pour le moment : comme la véritable importance sur les jeux
> de données réelles est inconnue, il sera plus difficile de commenter les
> résultats. »

`src/realdata.py` works and is kept, but it is **out of scope until the simulated
picture is settled**. Our own measurements back the instruction: on a heavily
pruned single tree the injected-share metric reads a flawless 0.00% simply
because 16 of 18 features sit at exactly zero — `scored` catches it, but it means
real data cannot adjudicate anything yet.

### The Duroux & Scornet models

He also pointed at the eight regression models of
[Duroux & Scornet (2016)](https://hal.science/hal-01287521), section 4, page 9.
Covariates are uniform on `[0,1]^d`, and the models are written in
`Xt = 2(X - 0.5)`, uniform on `[-1,1]`.

**Only five are usable here.** Models 2, 5 and 7 contain products and
multi-variable indicators, so `Var(f_j(X_j))` is the population MDI of no tree
([S20] Lemma 2) and there is no ground truth to measure bias against. They are
rejected explicitly rather than silently.

| tag | `d` | `n` (paper) | signal features | `sum I_j` | the paper's own SNR |
|---|---|---|---|---|---|
| `DS1` | 50 | 800 | 2 | 0.129 | noiseless |
| `DS3` | 100 | 600 | 4 | 1.449 | 5.8 |
| `DS4` | 100 | 600 | 4 | 9.920 | 39.7 |
| `DS6` | 30 | 500 | 10 | 2.500 | 26.5 |
| `DS8` | 1000 | 500 | 4 (at indices 0, 2, 4, 5) | 3.196 | noiseless |

Each contribution is centred and its variance obtained by quadrature, then
checked against Monte Carlo in the tests. Two notes on faithfulness:

- **Noise comes from the SNR here**, which departs from the paper twice. Models 1
  and 8 carry no noise term at all, and a noiseless response leaves no `sigma^2`
  for a deep tree to absorb — the entire bias under study. Model 6's noise is
  `-1{N(0,1) > 1.25}`, a Bernoulli of variance ≈ 0.094, not a Gaussian; only its
  variance matters for the absorption mechanism, but the distribution is not the
  paper's. The column above records what the paper's own noise corresponds to.
- **`N(0, 0.5)`** is read as a standard deviation, the R convention, since the
  paper's experiments were run in R. If it means a variance, the SNR column
  halves. Worth one line of confirmation.

```sh
python src/experiment.py --scenario DS3 --sizes 600 --reps 20 --out results/ds3.csv
```

## What we found: forests, earlier code

> These results were measured on **random forests tuned by OOB**, before the
> switch to single trees. They are kept as the reference for the forest
> extension; none of them has been re-measured on trees yet (Status, item 0).

Ordered by how much I would stake on it.

**1. The held-out fix works.** Importance handed to features whose truth is
exactly zero, `full` forest:

| | S1 | S4 |
|---|---|---|
| `mdi_train` | 0.368 | 0.797 |
| `ufi_half` | −0.015 | −0.024 |

Roughly a 75x reduction, stable at every sample size. This is half the brief and
it is not in doubt.

**2. UFI carries twice the scale of MDI-oob** on signal features. Not stated in
[ZH21] because they only ever rank with it. Unhalved, every bias number doubles.

**3. Alpha-trimming is inert on trees grown to purity.** Zero within-leaf variance
sends the stump log-likelihood to infinity, the merge test never fires, and
pruning silently does nothing -- no error, no warning. Not in the paper. This is
worth telling Scornet.

**4. Scornet's deep-tree bias, reproduced exactly.** Full-tree MDI sums to the
in-bag `Var(y)` to 1e-9, and the held-out measures recover only 54% of the signal
at `n_train` = 167, rising to 84% at 4000 -- measured with a 100k validation set,
so it is the forest's own limit, not estimation noise.

**5. Pruning is two-sided.** It *helps* the in-bag measure and *hurts* the
held-out one, consistently and significantly. In S4 at `n=1000`, pruning improved
`mdi_train` in 8 reps of 8 (best alpha 0.5-1.0) while making `ufi_half` worse in
8 of 8. The two fixes partly work against each other: the held-out measure has no
noise inflation left for pruning to remove, so pruning only costs it resolution.

**6. No pruning level helps the held-out MDI -- at SNR 4.** Across alpha from 0 to
3 (67.6 down to 4.5 leaves) the error rises monotonically, and the *oracle*, free
to choose alpha using the true `I_j`, picked alpha = 0 in 16 repetitions of 16.
That is not a weak null: the arm that exists to separate "pruning does not help"
from "OOB chose badly" answers the first.

> **Redone on a single tree, and the mechanism is not identical.** `S1`, `n = 1000`,
> SNR 4, 20 reps, median importance error: the oracle picks `alpha > 0` in 16 reps
> of 20 for `ufi_half` and 14 of 20 for `mdi_oob` — it does *not* refuse to prune
> the way it does on forests. But the gain it buys is **0.2%** (`ufi_half`:
> 0.0142 to 0.0142) and **2.6%** (`mdi_oob`: 0.0186 to 0.0181): the best available
> pruning is worth nothing. Meanwhile the level CV actually selects **doubles** the
> error (+131% and +96%). The conclusion survives on a stronger footing — it is not
> that pruning cannot help, it is that the most it could help is negligible while
> the practical selector actively hurts.

> **The caveat that bounds this.** Alpha-trimming did not improve *prediction*
> either in that regime (test MSE also minimised at alpha = 0), and [SHL24]
> Proposition 3.3 says the right amount of pruning is a function of the
> signal-to-noise ratio, with low SNR favouring alpha >= 1. So the method was
> being tested outside the conditions it was designed for. The SNR sweep is what
> settles whether finding 6 generalises.

**7. UFI beats MDI-oob** on the same forest, significantly (85% of reps).

### Mistakes found along the way

Both were caught by checking rather than by a test failing, which is the
uncomfortable part.

- **Stranded nodes.** A split kept low in the tree can be dissolved by a merge
  higher up; its descendants became unreachable but kept `feature >= 0`, and
  `mdi_train` sums over internal nodes instead of walking from the root. Worst
  case 19 stranded nodes in one tree. It corrupted `mdi_train` and `ufi_half` on
  the alpha arm. `mdi_oob` was immune because it descends from the root. Two
  regression tests now cover it.
- **The alpha arm's base is not the full arm.** It uses `min_samples_leaf = 5`,
  so `alpha` vs `full` confounds pruning with base depth. The `alpha0` arm --
  same base, no pruning -- exists so the pruning comparison is clean.

## The tests that matter

Six of the 48 carry the weight:

| Test | Catches |
|---|---|
| analytic `I_j` vs Monte Carlo | wrong ground truth |
| `Var(y) = ΣI_j + σ²` | wrong data generator |
| CV error equals a hand-written K-fold | tuning on the wrong rows — silent and fatal |
| full tree `Σ mdi_train = Var(y)` | wrong MDI; also confirms we reproduced Bug A |
| **`mdi_oob` on the training rows ≡ `mdi_train`** | all of the node bookkeeping, means and routing at once. Exact, to 4e-16 |
| `ufi_half ≈ mdi_oob` | the factor-2 error that would corrupt every bias number |

## Reading the results

`results/figures/` holds six per scenario. The two that carry the argument:

- **`1_noise_mass`** — importance handed to features that have none. The bug and
  the fix in one picture.
- **`6_pruning_path`** — importance error against how hard we prune, with the
  CV-selected level marked. This is the depth tension made visible.

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
5. **Trees first**: is single-tree pruning tuned by 5-fold CV the right
   reference before moving to forests?

## Deliberately left out

Add only when the pain is real: resumable runs, `joblib` parallelism, Parquet,
config files, pinned versions, the leakage check (moot — alpha is chosen by CV inside
the training split, the measure on the validation split, so there is no reuse),
Protocol-A OOB-based measures, scenarios with correlated or interacting inputs
(no ground truth there, so `Var(f_j(X_j))` stops being the target at all).
