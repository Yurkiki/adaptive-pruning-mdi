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

## Papers

- [S20] Scornet, *Trees, forests, and impurity-based variable importance in
  regression*, arXiv:2001.04295
- [ZH21] Zhou & Hooker, *Unbiased Measurement of Feature Importance in
  Tree-Based Methods*, arXiv:1903.05179
- [SHL24] Surjanovic, Henrey & Loughin, *Alpha-Trimming: Locally Adaptive Tree
  Pruning for Random Forests*, arXiv:2408.07151
- [L19] Li, Wang, Basu, Kumbier & Yu, *A Debiased MDI Feature Importance Measure
  for Random Forests*, arXiv:1906.10845


## To do

- [ ] simulated data with a known true `I_j`
- [ ] MDI on the training data, to reproduce bug A
- [ ] the same drops measured on held-out data
- [ ] alpha-trimming, from the paper's Algorithm 1
- [ ] the arms, the metrics, the paired tests
- [ ] real data with injected junk columns
- [ ] a low-SNR scenario, which is where [SHL24] expect pruning to pay
- [ ] reproduce one [SHL24] benchmark as a check on our transcription
