# Adaptive Pruning and Out-of-Sample MDI — Project Specification

*Authors: Seth, Louei. Supervisor: Erwan Scornet.*

## 0. The brief

Scornet's statement of the aim (email, 2026-10-24):

> « le but du projet serait d'évaluer la pertinence d'un MDI construit sur des arbres (forêts) **moins profonds** (par exemple utilisant la dernière méthode), calculé sur un **jeu de données de validation**, à la fois sur jeux de données **réelles et simulées**. »

Three requirements, which this protocol is organised around:

1. **Less deep trees**, obtained *adaptively* — alpha-trimming [SHL24] is the suggested method.
2. MDI computed on a **validation set** — a held-out sample, not the training data.
3. Both **simulated and real** datasets.

The reference papers he gave, and the role of each:

| Ref | Paper | Role |
|---|---|---|
| [S20] | Scornet, *Trees, forests, and impurity-based variable importance*, [arXiv:2001.04295](https://arxiv.org/abs/2001.04295) | MDI bias when trees are **too deep**; defines the population MDI and its additive-model limit |
| [ZH21] | Zhou & Hooker, *Unbiased Measurement of Feature Importance in Tree-Based Methods*, [arXiv:1903.05179](https://arxiv.org/abs/1903.05179), [TKDD 15(2), 2021](https://dl.acm.org/doi/10.1145/3429445) | MDI bias when importance is computed **on the training sample**; defines **UFI** |
| [SHL24] | Surjanovic, Henrey & Loughin, *Alpha-trimming: locally adaptive tree pruning for random forests*, [arXiv:2408.07151](https://arxiv.org/abs/2408.07151) | The **adaptive pruning** method; α tuned by OOB error |
| [L19] | Li, Wang, Basu, Kumbier & Yu, *A Debiased MDI Feature Importance Measure for Random Forests*, [arXiv:1906.10845](https://arxiv.org/abs/1906.10845) | **MDI-oob**, the other out-of-sample MDI. Not in the brief; included as the natural competitor to UFI |

> **Note on the second link.** arXiv:1903.05179 is Zhou & Hooker (UFI), *not* Li et al. The two papers are often confused because both attack training-sample bias. The measure the brief points at is therefore **UFI**; MDI-oob is a closely related but distinct estimator, and we treat the two as **co-primary** (§5.2–5.3).

---

## 1. Goal

Evaluate whether an MDI computed on **adaptively pruned** random-forest trees (alpha-trimming) and evaluated on **held-out data** estimates variable importance more accurately than existing variants, on simulated data (where the truth is known) and on real data.

### 1.1 The depth tension — why this project is not obvious

This is the conceptual core of the project and must be stated carefully, because the phrase "MDI is biased when trees are too deep" compresses two different things.

[S20] Theorem 3: assuming Model 3 (additive, independent inputs),

> for all γ>0, there exists K such that for all k>K, for all j, |MDI⋆_{𝒯⋆_k}(X^(j)) − 𝕍[m_j(X^(j))]| ≤ γ

So for the **theoretical (population) tree, deeper is better**: the population MDI converges to `Var(m_j(X_j))` only in the deep limit. A *shallow* theoretical tree does **not** have population MDI equal to `Var(m_j(X_j))`.

The bias [S20] warns about is a **finite-sample** effect. For empirical fully-grown trees,

> lim_{n→∞} Σ_j MDI_{𝒯_n}(X^(j)) = 𝕍[m(X)] + σ²

the measure absorbs the noise variance σ², and the paper concludes *"MDI must not be computed using fully grown trees."*

Pruning therefore cuts **both ways**:

- it reduces the finite-sample noise-absorption bias (**upward** bias removed), but
- it moves us to a shallower theoretical tree, whose population MDI falls short of `Var(m_j(X_j))` (**downward** approximation bias introduced).

There is consequently an interior optimal depth, and **two separate questions** that must not be conflated:

- **(a)** Does pruning *at the right amount* improve MDI accuracy?
- **(b)** Does pruning *at the α chosen by OOB prediction error* improve MDI accuracy?

α in [SHL24] is tuned to minimise **prediction** error. Nothing guarantees that this α also minimises **MDI** error. We therefore include an oracle arm (⑩, §3.4) that picks α to minimise MDI error directly, available in simulation only. Without it, a null result on the main claim is uninterpretable: we could not tell "pruning does not help MDI" from "OOB-tuned pruning picks the wrong α for MDI."

### 1.2 Research questions

- **Q1.** Does adaptive pruning reduce MDI error? (⑨ vs ⑦ on validation; ⑤ vs ① in-sample)
- **Q2.** Does adaptive pruning beat **global** depth control? (⑨ vs ⑧; ⑥ vs ④) — **the novelty claim.**
- **Q3.** Does out-of-sample evaluation help at fixed tree size? (② vs ①, ④ vs ③, ⑥ vs ⑤)
- **Q4.** Validation set vs OOB, at equal total sample budget? (⑨ vs ⑥)
- **Q5.** Is any MDI gain obtained without hurting predictive accuracy? (test MSE)
- **Q6.** Is OOB-tuned α the right α *for importance*? (⑨ vs ⑩) — see §1.1.

---

## 2. Data-generating model (simulations)

$$
Y = \sum_{j=1}^{d} f_j(X_j) + \varepsilon,\qquad \varepsilon \sim \mathcal{N}(0,\sigma^2),\ \varepsilon \perp X.
$$

- Each $f_j$ is **centered** ($\mathbb{E}[f_j(X_j)] = 0$), so $\mathbb{E}[Y]=0$. This matters for the out-of-sample formulas (§5.2).
- Noise features have $f_j \equiv 0$.
- **True importance:** $I_j = \mathrm{Var}(f_j(X_j))$, analytic where possible, else Monte Carlo on $10^6$ points (report the MC standard error). Noise features have $I_j = 0$.
- **Explainable variance:** $\sum_j I_j = \mathrm{Var}(Y) - \sigma^2$. Report $\mathrm{SNR} = \sum_j I_j / \sigma^2$; set $\sigma$ through the SNR so scenarios are comparable.

> **Validity of the target.** $I_j = \mathrm{Var}(f_j(X_j))$ is the population MDI only for **additive models with independent inputs**, and only in the **deep-tree limit** ([S20] Thm 3; their Model 3 is stated for $X \sim \mathcal{U}([0,1]^d)$, so independence *and* uniformity are the literal assumptions — our S1 complies). Under correlation or interactions MDI has no single well-defined target: [S20] Lemma 2 (interactions) and Lemma 4 (dependence) exhibit two theoretical trees giving *different* MDI to the same variable. Those scenarios need a different evaluation (§8.2).

---

## 3. Experimental procedure (one scenario, one $n$)

### 3.1 Two protocols, one sample budget

The brief asks for MDI on a **validation set**; the literature ([ZH21], [L19]) uses **out-of-bag** points. These are different sample-splitting strategies, and we run both at **equal total budget $n$** so the comparison is fair.

| | Protocol A (OOB) | Protocol B (validation) |
|---|---|---|
| Forest built on | all $n$ points | $n_{\text{tr}} = \lceil 2n/3 \rceil$ |
| α / `min_samples_leaf` tuned on | OOB of those $n$ | OOB within $D_{\text{tr}}$ |
| MDI evaluated on | OOB points (per tree) | $D_{\text{val}}$, $n_{\text{val}} = n - n_{\text{tr}}$ (all trees use all of it) |
| Trade-off | no data wasted; OOB reused for tuning *and* MDI | clean separation; weaker forest |

Protocol B is the brief's literal reading and has **no reuse problem by construction** — $D_{\text{val}}$ never informs tuning. Protocol A is sample-efficient but reuses OOB twice, which §6 tests. Q4 is exactly the question of which trade is better.

The 2/3–1/3 split is a default; include $n_{\text{val}}/n \in \{1/5, 1/3, 1/2\}$ as a sensitivity check, since the split ratio trades forest quality against MDI variance and the optimum is not known a priori.

### 3.2 Data generation (each repetition $r = 1,\dots,R$)

- $D$: size $n$. **The only data the methods may use.** Protocol B splits it into $D_{\text{tr}} \cup D_{\text{val}}$.
- $D_{\text{test}}$: independent, $10^4$ points, used **only** for prediction error, never for MDI.
- $D_{\text{fresh}}$: independent, size $n$, used **only** for the §6 leakage diagnostic.

Seeds: data seed $= r$; forest seed shared by all forests within a repetition.

### 3.3 Build the three forests

Within each protocol, all three forests use the **same $B$, the same `max_features`, the same bootstrap samples and the same seed**.

| Forest | Construction |
|---|---|
| **Full** | Grown until leaves are pure (`min_samples_leaf = 1`) |
| **Global** | Same, but `min_samples_leaf` chosen from a grid by **OOB** MSE |
| **Alpha** | The **Full trees themselves**, pruned by alpha-trimming (§5.4). A single α for the whole forest, chosen from the grid by **OOB** MSE |

- Alpha prunes the exact Full trees — [SHL24] confirm α *"can be adjusted without refitting the trees"* — so **Alpha vs Full isolates the effect of pruning.**
- Global trees are regrown with a different stopping rule, so once structures diverge the random feature subsets do too; Global vs Full shares only the bootstrap samples. Expected, but note it when interpreting.
- `max_features` identical across forests. Default $1/3$ (regression), with $1.0$ as sensitivity check.
- **Implementation (scikit-learn):** bootstrap is applied as sample weights (counts), so `min_samples_leaf` counts **distinct** samples, not bootstrap-weighted ones. Recover the in-bag/OOB split from the *same* bootstrap indices for every forest.

### 3.4 The measures

Each out-of-sample cell is computed with **both** estimators (MDI-oob and UFI, §5.2–5.3), reported side by side.

**Protocol A:**

| | MDI-train (in-bag) | MDI-oos (OOB) |
|---|---|---|
| **Full** | ① classic MDI | ② existing [ZH21]/[L19] |
| **Global** | ③ | ④ existing |
| **Alpha** | ⑤ **new** | ⑥ **new** |

**Protocol B:**

| | MDI-oos (validation) |
|---|---|
| **Full** | ⑦ |
| **Global** | ⑧ |
| **Alpha** | ⑨ **the brief's method — primary claim** |

**Diagnostic arm:**

- **⑩ Alpha-oracle:** Protocol B, α chosen to minimise $\sum_j(\widehat M_j - I_j)^2$ instead of OOB MSE. **Simulation only**, not a competitor — it upper-bounds what pruning could deliver and answers Q6.

Also record per forest: OOB MSE, test MSE on $D_{\text{test}}$, selected α or `min_samples_leaf`, mean leaves and mean depth per tree.

### 3.5 Then
Compare to the truth (§7), decide (§7.3), repeat across scenarios and $n$ (§8).

---

## 4. Hyper-parameters (defaults)

| Parameter | Default |
|---|---|
| Number of trees $B$ | 500 |
| `max_features` | $1/3$ (sensitivity: $1.0$) |
| `min_samples_leaf` grid (Global) | $\{1, 2, 3, 5, 8, 13, 20, 30, 50, 80\}$, capped at $n/10$ |
| **α grid (Alpha)** | **$\{0, 0.1, 0.2, \dots, 3.0\}$ — 31 linearly spaced values, exactly the grid of [SHL24]** |
| Validation fraction (Protocol B) | $1/3$ (sensitivity: $1/5$, $1/2$) |
| Sample sizes $n$ | $\{100, 250, 500, 1000, 2500, 5000\}$ |
| Repetitions $R$ | 100 |
| Test size | $10^4$ |
| MC size for $I_j$ | $10^6$ |

On the α grid: [SHL24] report *"an optimal value of α is often between 0 and 3"*, and that *"in some cases with a considerably low SNR it is favourable to try values of α ≥ 1"* — so do **not** truncate the grid below 3. α = 0 is the standard random forest, α = 1 prunes each tree to be a good standalone predictor. If the selected α hits the upper endpoint 3.0 in more than ~5% of repetitions (plausible in the low-SNR arms of S2), extend the grid to 5.0 and report that you did.

Both grids include the Full forest as a special case (`min_samples_leaf = 1`; α = 0), so tuning can never do worse than Full in OOB error.

---

## 5. Definitions of the importance measures

All measures **unnormalized**. For a tree $T$ with in-bag sample $D_b$ and evaluation sample $D_e$ (OOB points in Protocol A, $D_{\text{val}}$ in Protocol B): for node $t$ with children $t_L,t_R$ splitting on feature $v(t)$, write $\mu_t$ for the **in-bag** mean of $Y$ in $t$.

### 5.1 MDI-train (unnormalized)
$$
\mathrm{MDI}^{\text{train}}_j(T) = \sum_{t:\,v(t)=j} \frac{N_b(t)}{N_b}\Big[\mathrm{imp}_b(t) - \tfrac{N_b(t_L)}{N_b(t)}\mathrm{imp}_b(t_L) - \tfrac{N_b(t_R)}{N_b(t)}\mathrm{imp}_b(t_R)\Big],
$$
with $\mathrm{imp}_b$ the in-bag variance (bootstrap-weighted) and $N_b(\cdot)$ in-bag counts. Average over trees. In scikit-learn: `tree_.compute_feature_importances(normalize=False)`. **Do not use `feature_importances_`** — it is normalized to sum to 1.

Sanity property: for Full trees, $\sum_j \mathrm{MDI}^{\text{train}}_j \approx \widehat{\mathrm{Var}}(Y)$, including σ². This is precisely the deep-tree bias of [S20] (§1.1).

### 5.2 MDI-oob [L19] — co-primary
[L19] Proposition 1 rewrites MDI as $\frac{1}{|\mathcal{D}(T)|}\sum_{i\in\mathcal{D}(T)} f_{T,k}(\mathbf{x}_i)\,y_i$ with
$$
f_{T,k}(X) = \sum_{t:\,v(t)=k}\big\{\mu(t_L)\mathbb{1}(X\in R_{t_L}) + \mu(t_R)\mathbb{1}(X\in R_{t_R}) - \mu(t)\mathbb{1}(X\in R_t)\big\},
$$
and evaluates it off-sample. Since a point in $R_t$ falls in exactly one child, this reduces to
$$
\mathrm{MDI}\text{-}\mathrm{oob}_j(T) = \frac{1}{|D_e|}\sum_{t:\,v(t)=j}\ \sum_{i \in D_e(t)} \big(\mu_{t(i)} - \mu_t\big)\, y_i,
$$
where $t(i)\in\{t_L,t_R\}$ is the child containing $i$ and all $\mu$ come from **in-bag** data. Average over trees. Scale: comparable to $I_j$ directly (no constant).

### 5.3 UFI [ZH21] — co-primary, the measure in the brief
$$
\mathrm{UFI}_j(T) = \sum_{t:\,v(t)=j}\big(\Delta(t) + \Delta'(t)\big),
$$
with $\Delta$ the ordinary in-bag impurity decrease (§5.1 bracket, weighted) and $\Delta'$ the same decrease evaluated on $D_e$ **against the in-bag means**:
$$
\Delta'(t) = \omega'_t H'(t) - \omega'_{t_L}H'(t_L) - \omega'_{t_R}H'(t_R),\qquad H'(t) = \frac{1}{|D_e(t)|}\sum_{i\in D_e(t)}(y_i - \mu_t)^2,
$$
$\omega'$ being evaluation-sample proportions. For a noise feature $\Delta>0$ (spurious in-bag gain) and $\Delta'<0$ (the overfit child means *increase* held-out error) and they cancel: [ZH21] Theorem 4 — *"if y is independent of X_j in every hyper-rectangle subset of the feature space, then we always have 𝔼 VI^{T,R}_j = 0."*

> **⚠ Scale factor 2.** For a genuine split both $\Delta$ and $\Delta'$ converge to the same true impurity decrease $D$, so $\mathrm{UFI}_j \to 2\,\mathrm{MDI}^\star_j \to 2\,I_j$ for signal features, while noise features still give 0. [ZH21] only ever use UFI for **ranking**, where a constant is invisible, so the paper never states this. **We compare numerically against $I_j$, so UFI must be halved.** Define $\widetilde{\mathrm{UFI}} = \tfrac12\mathrm{UFI}$ and use it everywhere bias/MSE against $I_j$ is reported. This factor is *derived, not quoted* — it is a required unit test (§10).

### 5.4 Alpha-trimming [SHL24]
Grow each tree fully, then traverse nodes **bottom-up** (reverse construction order). Merge the children of node $N$ — i.e. prune back to $N$ — when
$$
I_N + \alpha P_{0,|N|} \ \le\ I_{N_L} + I_{N_R} + \alpha P_{1,|N|},
$$
where $I_N$ is the Gaussian-deviance information of the single-mean model in $N$ ($|N|\log(2\pi\hat\sigma^2_N) + |N|$, $\hat\sigma^2_N = \mathrm{RSS}_N/|N|$), $P_{0,n} = 2\log n$ penalises the one-mean model and $P_{1,n} = 5\log n$ the stump (two means, variance, split point). If the children are *not* merged, $I_N$ is updated to the accumulated child information before continuing upward. α = 0 recovers the standard random forest. Selection: $\alpha^\star \leftarrow \arg\min_{\alpha\in\mathcal{A}} \mathrm{OOB}(\cdot)$, no cross-validation needed, and **no refitting** — the trees are grown once and re-pruned per α from stored sums.

> **⚠ Transcribed from [SHL24] Eq. (14) and surrounding text.** Re-read their §4 before implementing and confirm the exact form of $I_N$, the penalty constants, and whether the accumulation step replaces or adds. There is **no scikit-learn implementation and no public code we could find** (the authors appear to work in Julia), so this is a from-scratch component — budget for it (§10) and validate it by reproducing one [SHL24] benchmark before using it for MDI.

### 5.5 Conventions shared by all out-of-sample measures
- **Centering.** [L19] note $\sum_{i\in\mathcal{D}(T)} f_{T,k}(\mathbf{x}_i)=0$, so MDI-oob is shift-invariant **in-bag but not off-sample**. Keep $Y$ centered (true by construction in simulation; on real data subtract the *training* mean) and fix this convention for all methods.
- **Never clip negative values.** Noise features must be free to come out negative; clipping reintroduces the bias we are removing.
- A node reached by no evaluation point contributes 0.
- **In-bag identity test.** Evaluating §5.2 on the in-bag points (with bootstrap weights) must reproduce §5.1 exactly, since $\sum_{i\in t}(\mu_{t(i)}-\mu_t)y_i = N_b(t)\,\Delta\mathrm{imp}(t)$.

---

## 6. Leakage check (Protocol A only)

In Protocol A the OOB points are used **twice** for Global and Alpha: to select the hyper-parameter, then to compute MDI. Protocol B is immune by construction, which is one of its selling points.

- **⑥-fresh / ④-fresh:** α (resp. `min_samples_leaf`) selected on OOB as usual, but MDI computed on $D_{\text{fresh}}$.
- Compare ⑥ vs ⑥-fresh and ④ vs ④-fresh with §7.3. Both are needed: ④ reuses OOB the same way, so checking only ⑥ would make the Q2 comparison unfair.
- These use $2n$ points, so they are **diagnostics only**, never competitors in the main ranking.

---

## 7. Evaluation

### 7.1 Per-feature metrics (method $m$, feature $j$, estimates $\widehat M^{(r)}_{m,j}$)

| Metric | Formula |
|---|---|
| Bias | $\bar M_{m,j} - I_j$ |
| Variance | $\frac1R\sum_r (\widehat M^{(r)}_{m,j} - \bar M_{m,j})^2$ |
| MSE | $\frac1R\sum_r (\widehat M^{(r)}_{m,j} - I_j)^2$ |

With the $1/R$ variance, $\mathrm{MSE} = \mathrm{Bias}^2 + \mathrm{Var}$ holds exactly. Report Monte Carlo standard errors. Use $\widetilde{\mathrm{UFI}}$ (§5.3), not raw UFI.

> **Caveat on bias.** Neither [ZH21] nor [L19] proves that its measure *targets* $I_j$. [ZH21] Thm 4 gives $\mathbb{E}=0$ for **noise** features only; [L19] bound the bias of *classic* MDI and justify MDI-oob empirically, with **no unbiasedness theorem**. So "MDI-oos is unbiased for $I_j$" is a hypothesis this study tests, not an assumption it rests on. §7.2 item 3 is the direct check.

### 7.2 Summaries across features

1. **Total MSE** $\sum_j \mathrm{MSE}_{m,j}$, plus relative $\sum_j \mathrm{MSE}_{m,j}/\sum_j I_j^2$.
2. **Noise mass:** $\sum_{j\in\text{noise}}\widehat M_{m,j}$ (signed, should be ≈0) and $\sum_{j\in\text{noise}}|\widehat M_{m,j}|$.
3. **Signal mass:** $\sum_{j\in\text{signal}}\widehat M_{m,j}$ vs $\sum_j I_j$ — should match the **explainable variance**, not $\mathrm{Var}(Y)$. This is also the scale check that validates the §5.3 factor of 2.
4. **Signal-vs-noise AUC** per repetition (ties ½), averaged over $R$.
5. **Kendall's τ** between $\widehat M_{m,\cdot}$ and $I_\cdot$ on signal features (matters in S4), averaged over $R$.
6. **Prediction:** test MSE and excess risk $\text{test MSE} - \sigma^2$ per forest.
7. **Tree size:** mean leaves and depth, to confirm Alpha really does prune and to relate pruning amount to MDI gain.

### 7.3 Paired comparisons

All methods see the same datasets, so compare **per repetition**.

- Per-repetition loss $L^{(r)}_m = \sum_j (\widehat M^{(r)}_{m,j} - I_j)^2$; repeat with AUC and τ.
- For each pair: $\Delta^{(r)} = L^{(r)}_a - L^{(r)}_b$, reported as mean with 95% CI (paired bootstrap or $t$), win proportion, and a **Wilcoxon signed-rank** test.
- **Multiplicity:** Holm within each scenario; treat CIs as the primary output rather than p-values.

| Comparison | Question |
|---|---|
| ⑨ vs ⑦ | Q1: pruning, on validation |
| ⑤ vs ① | Q1: pruning, in-sample |
| **⑨ vs ⑧** | **Q2: adaptive vs global (primary novelty claim)** |
| ⑥ vs ④ | Q2 under Protocol A |
| ② vs ①, ④ vs ③, ⑥ vs ⑤ | Q3: out-of-sample effect at fixed tree size |
| ⑨ vs ⑥ | Q4: validation vs OOB at equal budget |
| ⑨ vs ① | Combined effect vs the standard practice |
| test MSE: Alpha vs Full, Alpha vs Global | Q5: no accuracy loss |
| **⑨ vs ⑩** | **Q6: is OOB-tuned α the right α for MDI? (§1.1)** |
| MDI-oob vs $\widetilde{\mathrm{UFI}}$ within each cell | Which out-of-sample estimator is better |

---

## 8. Scenarios

### 8.1 Additive, independent inputs (truth $I_j$ well defined)

| ID | Description | Purpose |
|---|---|---|
| S1 | $d=10$, 5 signal + 5 noise, $X\sim U[0,1]^d$, smooth $f_j$ | Baseline; matches [S20] Model 3 literally |
| S2 | S1 at SNR ∈ {0.25, 1, 4, 16} | Noise level; low SNR is where [SHL24] expect large α |
| S3 | **Mixed feature types:** binary, categorical ($k$ levels), continuous noise features | Classic MDI cardinality bias |
| S4 | **Heterogeneous signal:** $I_j$ over orders of magnitude; some $f_j$ smooth, some step-like or localized | Where *locally* adaptive pruning should help most — the key scenario for Q2 |
| S5 | Larger $d$ (e.g. 50, mostly noise) | Sparsity |

### 8.2 Beyond additive-independent
Here $\mathrm{Var}(f_j(X_j))$ is not the MDI target [S20], so **bias and MSE against $I_j$ cannot be the main metrics.**

| ID | Description | Primary metrics |
|---|---|---|
| S6 | Additive $f$, **correlated** Gaussian-copula inputs (ρ ∈ {0.3,0.6,0.9}), incl. noise features correlated with signal | AUC, noise mass. Secondary: compare to $\mathrm{Var}(f_j(X_j))$ and first-order Sobol $\mathrm{Var}(\mathbb{E}[Y\mid X_j])$, labelled **reference values, not ground truth** |
| S7 | **Interactions** ($X_1X_2$, $\mathbb{1}\{X_3>.5\}X_4$) | AUC, noise mass, τ vs total Sobol indices (reference only) |

---

## 9. Real data

No ground truth, so:

1. **Injected noise features:** append $k$ known-irrelevant columns (permutations of real columns, plus i.i.d. continuous and categorical noise). Metrics: noise mass, AUC real vs injected [L19].
2. **Semi-synthetic:** keep real $X$ (real correlation structure), simulate $Y=\sum_j f_j(X_j)+\varepsilon$ on a chosen feature subset. Target is reference-only (§8.2).
3. **Stability:** over $R$ subsamples, report each MDI's variance and rank stability (mean pairwise Kendall τ).
4. **Prediction:** held-out or CV MSE for Full, Global, Alpha.

Datasets: California Housing, Diabetes, Abalone, Ames Housing, Bike Sharing, plus datasets from [SHL24] for comparability — they benchmark on **46** sets (42 from Chipman et al. 2010, plus *constant*, *elbow*, *logistic*, *sine*), several of which are standard regression benchmarks.

---

## 10. Deliverables

- **Code:** generators; the three forests; **alpha-trimming from scratch** (§5.4 — the largest single implementation item, no public code available); both out-of-sample estimators; both protocols; the oracle arm; metrics; paired tests. Seeds and config files per run.
- **Unit tests:**
  1. §5.5 in-bag identity (MDI-oob in-bag ≡ classic MDI).
  2. $I_j$ analytic vs Monte Carlo.
  3. α = 0 reproduces the Full forest **exactly**.
  4. **UFI scale:** on a high-SNR additive simulation, $\mathrm{UFI}_j/I_j \to 2$ while $\mathrm{MDI}\text{-}\mathrm{oob}_j/I_j \to 1$, confirming §5.3.
  5. Noise features: signed noise mass of each out-of-sample measure is statistically indistinguishable from 0.
  6. Alpha-trimming reproduces one [SHL24] benchmark result.
  7. Full-forest in-bag MDI sums to $\widehat{\mathrm{Var}}(Y)$ (the [S20] deep-tree signature).
- **Results:** per scenario × $n$, tables of total MSE / AUC / noise mass / test MSE for ①–⑩; bias–variance plots per feature; metric curves vs $n$ and SNR; **MDI error as a function of α** (the §1.1 tension made visible, with the OOB-selected and oracle α marked); a final **win/loss map** with paired-CI significance.
- **Report** answering Q1–Q6 and the §6 leakage check.

---

## 11. Open questions for Erwan

1. **"Jeu de données de validation" — literal or OOB?** We implement both (§3.1) and compare them (Q4), with Protocol B as the primary since it matches the wording. Confirm this is the intent, and whether a 1/3 validation fraction is sensible.
2. **UFI or MDI-oob?** The linked paper is Zhou & Hooker (UFI); Li et al.'s MDI-oob is the other natural candidate. We treat them as co-primary. Is there a preference?
3. **Q6 / the oracle arm.** Tuning α by OOB *prediction* error need not optimise *importance* (§1.1). Is the intended scope "does alpha-trimming as published help MDI" (⑨ only) or "is there a pruning level that helps MDI" (⑨ and ⑩)? We assume both.
4. **Classification?** All three papers cover it; this protocol is regression-only. Confirm that is in scope.

---

## 12. Changes from our earlier draft

| # | Was | Now | Reason |
|---|---|---|---|
| 1 | arXiv:1903.05179 cited as Li et al. / MDI-oob | It is **Zhou & Hooker, UFI**; Li et al. is arXiv:1906.10845 | Factual citation error; verified against both arXiv pages and TKDD |
| 2 | MDI-oob primary, UFI an appendix robustness check | **Co-primary**, UFI defined explicitly in §5.3 | The brief links the UFI paper; and UFI has the stronger theory ([ZH21] Thm 4 vs no theorem in [L19]) |
| 3 | α grid "20 log-spaced values" | $\{0,0.1,\dots,3.0\}$, 31 **linear** values | That is the actual grid in [SHL24] |
| 4 | Alpha-trimming left unspecified | Criterion, penalties and algorithm written out (§5.4), with implementation risk flagged | It must be built from scratch; no public code |
| 5 | Validation data only a leakage diagnostic (⑥-fresh) | **Protocol B is a first-class arm** (⑦⑧⑨), ⑨ is the primary claim | The brief says MDI computed on a validation set |
| 6 | "MDI is biased when trees are too deep" | §1.1 separates population (deeper = better) from finite-sample (deep = noise absorption), and notes pruning's **downward** approximation bias | [S20] Thm 3 holds as $k\to\infty$; the naive reading inverts the theory |
| 7 | α tuned by OOB MSE, full stop | Added **oracle-α arm ⑩** and Q6 | Prediction-optimal α ≠ importance-optimal α; without this a null Q2 is uninterpretable |
| 8 | UFI and MDI-oob implicitly same scale | **UFI halved** (§5.3) before any comparison to $I_j$ | Derived: $\mathrm{UFI}\to 2I_j$ for signal, 0 for noise. Unit test 4 |
| 9 | Bias vs $I_j$ assumed meaningful for MDI-oos | §7.1 caveat: that neither paper proves | Honest framing; it is a hypothesis under test |
| 10 | $I_j$ target stated for "additive + independent" | Added the **deep-tree limit** and [S20]'s uniform-$X$ assumption | Thm 3 is asymptotic in depth |
| 11 | Leakage check for ⑥ only | Also ④; and noted Protocol B is immune by construction | Fairness of the Q2 comparison |
| 12 | — | Added tree-size reporting (§7.2.7) and MDI-error-vs-α curves (§10) | Makes the §1.1 mechanism directly observable |
