# Adaptive Pruning and Out-of-Sample MDI — Project Specification

## Summary

MDI (Mean Decrease in Impurity), the usual random-forest variable importance, is biased in two ways. It overestimates importance when trees are grown too deep, and it does so again when it is computed on the same data used to build the trees. This project tests whether **pruning the trees adaptively** (alpha-trimming) and **computing MDI on out-of-bag points** reduce that bias, separately and together.

**Plan.** Three forests are built on identical bootstrap samples:
- **Full:** fully grown trees.
- **Global:** a single minimum leaf size for all trees, chosen by OOB error.
- **Alpha:** the Full trees pruned by alpha-trimming, with α chosen by OOB error.

Each forest gets two MDIs, one on in-bag data and one on out-of-bag data, giving six measures. In simulations the true importances are known ($I_j = \mathrm{Var}(f_j(X_j))$ for additive models with independent inputs). Each measure is scored on bias, variance, MSE, the importance given to noise features, and ranking quality, over 100 repetitions per scenario and sample size. Methods are compared in pairs on the same datasets, with Wilcoxon tests and confidence intervals. The main claim to test is whether the new pruned out-of-sample MDI (Alpha forest, out-of-bag) beats the existing methods, especially global depth control. Two further checks:
- forest prediction error, to make sure pruning does not cost accuracy;
- a fresh-data control, to make sure reusing the OOB points does not bias the result.

Real datasets complete the study, using injected noise features, semi-synthetic targets and stability analysis.

## 1. Goal

Evaluate whether a **Mean Decrease in Impurity (MDI)** computed on **adaptively pruned random-forest trees** (alpha-trimming), and **evaluated on held-out data** rather than on the training sample, gives a more accurate variable importance than existing variants. The evaluation uses both **simulated** data (where the true importances are known) and **real** data.

### Background (references given by the supervisor, E. Scornet)

| Ref | Paper | Role in the project |
|---|---|---|
| [S20] | Scornet, *Trees, forests, and impurity-based variable importance* (arXiv:2001.04295) | MDI bias when trees are **too deep**; for additive models with independent inputs, the population MDI of feature $j$ equals $\mathrm{Var}(f_j(X_j))$ |
| [L19] | Li, Wang, Basu, Kumbier, Yu, *A debiased MDI feature importance measure for random forests* (arXiv:1903.05179) | MDI bias when importance is computed **on the training sample**; MDI-oob |
| [SHL24] | Surjanovic, Henrey, Loughin, *Alpha-trimming: locally adaptive tree pruning for random forests* (arXiv:2408.07151) | Adaptive pruning that **improves forest accuracy**; α tuned by OOB error |
| [ZH21] | Zhou & Hooker, *Unbiased measurement of feature importance in tree-based methods* (TKDD 2021) | UFI, an alternative out-of-sample MDI |

### Research questions

- **Q1.** Does adaptive pruning by itself reduce MDI error? (⑤ vs ①)
- **Q2.** Does adaptive pruning beat global depth control? (⑤ vs ③, ⑥ vs ④). This is the novelty claim.
- **Q3.** Do pruning and out-of-sample computation add up? (⑥ vs ②, ⑥ vs ⑤)
- **Q4.** Is any MDI gain obtained without hurting predictive accuracy?

---

## 2. Data-generating model (simulations)

$$
Y = \sum_{j=1}^{d} f_j(X_j) + \varepsilon,\qquad \varepsilon \sim \mathcal{N}(0,\sigma^2),\ \varepsilon \perp X.
$$

- Each $f_j$ is **centered** ($\mathbb{E}[f_j(X_j)] = 0$), so $\mathbb{E}[Y]=0$. This matters for the out-of-sample formula (§5.2).
- Noise features have $f_j \equiv 0$.
- **True importance:** $I_j = \mathrm{Var}(f_j(X_j))$, computed analytically when possible, otherwise by Monte Carlo on $10^6$ points (report the MC standard error). Noise features have $I_j = 0$.
- **Explainable variance:** $\sum_j I_j = \mathrm{Var}(Y) - \sigma^2$. Report the signal-to-noise ratio $\mathrm{SNR} = \sum_j I_j / \sigma^2$ for each scenario. Set $\sigma$ through the SNR so that scenarios can be compared.

> **Validity of the target.** $I_j = \mathrm{Var}(f_j(X_j))$ is the population MDI **only for additive models with independent inputs** [S20]. Under correlation or interactions, MDI has no single, well-defined population target, so those scenarios need a different evaluation (§8.2).

---

## 3. Experimental procedure (one scenario, one $n$)

### Step 1: Fix the scenario
Choose $d$, the distribution of $X$, the functions $f_j$, $\sigma$ (via SNR), the sample size $n$, and the number of repetitions $R$ (default $R = 100$). Compute $I_j$ once.

### Step 2: Generate data (each repetition $r = 1,\dots,R$)
- $D$: training set of size $n$. **This is the only data the methods may use.**
- $D_{\text{test}}$: independent set (e.g. $10^4$ points), used **only** to measure prediction error, never for MDI.
- $D_{\text{fresh}}$: independent set of size $n$, used **only** for the leakage check (§6).

Seeds: data seed $= r$; forest seed shared by all forests within a repetition.

### Step 3: Build the three forests on $D$

All three use the **same number of trees $B$, the same `max_features`, the same bootstrap samples and the same random seed**.

| Forest | Construction |
|---|---|
| **Full** | Grown until leaves are pure (`min_samples_leaf = 1`) |
| **Global** | Same as Full, but `min_samples_leaf` chosen from a grid by **OOB** MSE |
| **Alpha** | The **Full trees themselves**, each pruned by alpha-trimming [SHL24]. A single α for the whole forest, chosen from a grid by **OOB** MSE |

- Because Alpha is obtained by pruning the exact Full trees, **Alpha vs Full isolates the effect of pruning.**
- Global trees are grown again with a different stopping rule. Once the structures differ, the random feature subsets differ too, so Global vs Full shares only the bootstrap samples. This is expected; just be aware of it.
- When `max_features < d`, use the same value for all forests. Default: `max_features = 1/3` (regression), with `max_features = 1.0` as a sensitivity check.
- Implementation note (scikit-learn): bootstrap is implemented through sample weights (counts), so `min_samples_leaf` counts distinct samples. Make sure the in-bag/OOB split is recovered exactly from the same bootstrap indices for every forest.

### Step 4: Compute the 6 MDIs (always **unnormalized**)

| | MDI-train (in-bag) | MDI-oos (OOB) |
|---|---|---|
| **Full** | ① classic MDI | ② existing [L19]/[ZH21] |
| **Global** | ③ | ④ existing [L19] |
| **Alpha** | ⑤ **new** | ⑥ **new** |

Also record, for each forest: OOB MSE, test MSE on $D_{\text{test}}$, selected hyper-parameter (`min_samples_leaf` or α), and the mean number of leaves and depth per tree.

### Step 5: Compare each MDI to the truth (after the $R$ repetitions)
See §7.

### Step 6: Decide which method is better
See §7.3.

### Step 7: Repeat across scenarios and $n$
See §8.

---

## 4. Hyper-parameters (defaults)

| Parameter | Default |
|---|---|
| Number of trees $B$ | 500 |
| `max_features` | $1/3$ (sensitivity: $1.0$) |
| `min_samples_leaf` grid (Global) | $\{1, 2, 3, 5, 8, 13, 20, 30, 50, 80\}$, capped at $n/10$ |
| α grid (Alpha) | grid from [SHL24], e.g. 20 log-spaced values including "no pruning" |
| Sample sizes $n$ | $\{100, 250, 500, 1000, 2500, 5000\}$ |
| Repetitions $R$ | 100 |
| Test size | $10^4$ |
| MC size for $I_j$ | $10^6$ |

The grids must include the Full forest as a special case (`min_samples_leaf = 1`, no pruning), so that tuning can never do worse than Full in OOB error.

---

## 5. Definitions of the importance measures

Notation for a tree $T$ with in-bag sample $D_b$ and out-of-bag sample $D_o$. For a node $t$ with children $t_L, t_R$ splitting on feature $v(t)$, write $\mu_t$ for the **in-bag** mean of $Y$ in $t$.

### 5.1 MDI-train (unnormalized)
$$
\mathrm{MDI}^{\text{train}}_j(T) = \sum_{t:\,v(t)=j} \frac{N_b(t)}{N_b}\Big[\mathrm{imp}_b(t) - \tfrac{N_b(t_L)}{N_b(t)}\mathrm{imp}_b(t_L) - \tfrac{N_b(t_R)}{N_b(t)}\mathrm{imp}_b(t_R)\Big],
$$
where $\mathrm{imp}_b$ is the in-bag variance (weighted by bootstrap counts) and $N_b(\cdot)$ the in-bag counts. The forest MDI is the average over trees. In scikit-learn this is `tree_.compute_feature_importances(normalize=False)`. **Do not use `feature_importances_`**, which is normalized to sum to 1.

Sanity property: for Full trees (pure leaves), $\sum_j \mathrm{MDI}^{\text{train}}_j \approx \widehat{\mathrm{Var}}(Y)$, i.e. it includes the noise $\sigma^2$. This is the deep-tree bias of [S20].

### 5.2 MDI-oos (unnormalized)
**Primary choice: MDI-oob [L19].** For regression,
$$
\mathrm{MDI}^{\text{oos}}_j(T) = \frac{1}{|D_o|}\sum_{t:\,v(t)=j}\ \sum_{i \in D_o(t)} \big(\mu_{t(i)} - \mu_t\big)\, y_i,
$$
where $t(i) \in \{t_L, t_R\}$ is the child that OOB point $i$ falls into, and the node means $\mu$ come from the **in-bag** data. Then average over trees.

- **Robustness variant:** UFI [ZH21]. Report it in an appendix, or as a secondary row if it differs materially.
- **Implementation check (unit test):** evaluating the same formula on the **in-bag** points (with bootstrap weights) must reproduce §5.1 exactly, because in-bag $\sum_{i\in t}(\mu_{t(i)}-\mu_t)y_i = N_b(t)\cdot\Delta\mathrm{imp}(t)$.
- On OOB data the formula is **not** invariant to shifting $Y$. Keep $Y$ centered (true by construction in simulations; on real data, center by the training mean) and keep this convention fixed for all methods.
- **Do not clip negative values.** Noise features should give estimates centered around 0, sometimes negative. Clipping would reintroduce bias.
- A tree with no OOB points reaching a node contributes 0 at that node.

---

## 6. Leakage check (OOB reuse)

For Global and Alpha, the OOB points are used **twice**: once to select the hyper-parameter (`min_samples_leaf` or α), then again to compute MDI-oos. To test whether this biases the results:

- **⑥-fresh:** Alpha forest (α selected by OOB as usual), but MDI-oos computed on $D_{\text{fresh}}$ instead of the OOB points (every tree uses all of $D_{\text{fresh}}$).
- **④-fresh:** same for the Global forest. The issue applies to ④ too, so check both to keep the Q2 comparison fair.

Compare ⑥ vs ⑥-fresh and ④ vs ④-fresh with the paired analysis of §7.3. If they agree, the reuse is harmless. If not, report the gap as a finding.

Note: ⑥-fresh uses $2n$ points, so it is **only a diagnostic**, never a competitor in the main ranking (the main comparison keeps all methods on the same $n$ points).

---

## 7. Evaluation

### 7.1 Per-feature metrics (method $m$, feature $j$, estimates $\widehat{M}^{(r)}_{m,j}$, $r=1..R$)

| Metric | Formula | Meaning |
|---|---|---|
| Bias | $\bar M_{m,j} - I_j$ | Systematic over/under-estimation |
| Variance | $\frac{1}{R}\sum_r (\widehat M^{(r)}_{m,j} - \bar M_{m,j})^2$ | Stability across datasets |
| MSE | $\frac{1}{R}\sum_r (\widehat M^{(r)}_{m,j} - I_j)^2$ | Overall accuracy |

With the $1/R$ variance, $\mathrm{MSE} = \mathrm{Bias}^2 + \mathrm{Var}$ holds exactly. Report Monte Carlo standard errors.

### 7.2 Summaries across features (one number per method)

1. **Total MSE** $= \sum_j \mathrm{MSE}_{m,j}$. Also report a **relative** version $\sum_j \mathrm{MSE}_{m,j} / \sum_j I_j^2$ so that scenarios can be compared.
2. **Noise mass:** $\sum_{j \in \text{noise}} \widehat M_{m,j}$ (signed; should be ≈ 0) and $\sum_{j\in\text{noise}} |\widehat M_{m,j}|$.
3. **Signal mass:** $\sum_{j \in \text{signal}} \widehat M_{m,j}$ vs $\sum_j I_j$. It should match the **explainable variance**, not $\mathrm{Var}(Y)$.
4. **Ranking, signal vs noise:** per repetition, AUC of "signal features ranked above noise features" (ties count ½), averaged over $R$.
5. **Ranking among signal features:** per repetition, Kendall's τ between $\widehat M_{m,\cdot}$ and $I_\cdot$ on the signal features (matters in the heterogeneous-signal scenarios), averaged over $R$.
6. **Prediction:** test MSE and excess risk $\text{test MSE} - \sigma^2$ for each forest.

### 7.3 Paired comparisons

All methods run on the same datasets, so compare them **per repetition**.

- Per-repetition loss: $L^{(r)}_m = \sum_j (\widehat M^{(r)}_{m,j} - I_j)^2$ (its mean over $r$ is the total MSE). Repeat the analysis with the AUC and with τ.
- For each pair $(a,b)$: differences $\Delta^{(r)} = L^{(r)}_a - L^{(r)}_b$. Report the mean with a 95% CI (paired bootstrap or $t$), the proportion of repetitions where $a$ wins, and a **Wilcoxon signed-rank** test.
- **Multiplicity:** many pairs × scenarios × $n$. Apply a Holm correction within each scenario, and treat the CIs as the main output rather than p-values.

| Comparison | Question |
|---|---|
| ⑤ vs ① | Q1: pruning alone (in-sample) |
| ② vs ①, ④ vs ③, ⑥ vs ⑤ | Out-of-sample computation alone, at fixed tree size |
| ⑤ vs ③, ⑥ vs ④ | **Q2: adaptive vs global depth control (novelty)** |
| ⑥ vs ② | Q3: pruning on top of out-of-sample |
| ⑥ vs ① | Combined effect |
| test MSE: Alpha vs Full, Alpha vs Global | Q4: no loss in accuracy |

---

## 8. Scenarios

### 8.1 Additive, independent inputs (truth $I_j$ well defined)

| ID | Description | Purpose |
|---|---|---|
| S1 | $d=10$, 5 signal + 5 noise, $X\sim U[0,1]^d$, smooth $f_j$ (linear, sine, quadratic…) | Baseline |
| S2 | S1 at several SNRs (e.g. 0.25, 1, 4, 16) | Effect of noise level |
| S3 | **Mixed feature types:** noise features that are binary, categorical with $k$ levels, and continuous | Classic MDI bias toward high-cardinality features |
| S4 | **Heterogeneous signal:** $I_j$ spanning orders of magnitude; some $f_j$ smooth, some step-like or localized | Where *locally* adaptive pruning should help most |
| S5 | Larger $d$ (e.g. 50, mostly noise) | Sparsity |

### 8.2 Beyond the additive-independent case
Here $\mathrm{Var}(f_j(X_j))$ is no longer the MDI target [S20], so **bias and MSE against $I_j$ cannot be the main metrics.**

| ID | Description | Primary metrics |
|---|---|---|
| S6 | Additive $f$, **correlated** Gaussian-copula inputs (ρ ∈ {0.3, 0.6, 0.9}), including noise features correlated with signal | Signal-vs-noise AUC, noise mass. Secondary: compare to $\mathrm{Var}(f_j(X_j))$ and to first-order Sobol $\mathrm{Var}(\mathbb{E}[Y\mid X_j])$, clearly labeled as reference values rather than ground truth |
| S7 | **Interactions** (e.g. $X_1X_2$, $\mathbb{1}\{X_3>.5\}\cdot X_4$) | AUC, noise mass, Kendall τ vs total Sobol indices (reference only) |

---

## 9. Real data

The supervisor asked for real data as well. With no ground truth, use these protocols:

1. **Injected noise features:** append $k$ known-irrelevant columns to $X$ (random permutations of existing columns, plus i.i.d. continuous and categorical noise). Metrics: noise mass, AUC of real features vs injected noise [L19].
2. **Semi-synthetic:** keep the real $X$ (real correlation structure), simulate $Y = \sum_j f_j(X_j) + \varepsilon$ on a chosen subset of features. The target is reference-only, as in §8.2.
3. **Stability:** over $R$ subsamples (or bootstrap resamples) of the dataset, report the variance of each MDI and the rank stability (mean pairwise Kendall τ between repetitions).
4. **Prediction:** cross-validated or held-out MSE of Full, Global and Alpha.

Candidate datasets (regression, moderate size, mixed types): California Housing, Diabetes, Abalone, Ames Housing, Bike Sharing, and the datasets used in [SHL24] for comparability.

---

## 10. Deliverables

- Code: data generators, the three forests, the 6 MDIs (+ UFI variant, + fresh-data versions), metrics, paired tests. Seeds and configuration files for every run.
- Unit tests: §5.2 in-bag identity; $I_j$ (analytic vs Monte Carlo); Alpha with α = 0 reproduces Full exactly.
- Results:
  - per scenario × $n$: table of total MSE / AUC / noise mass / test MSE for ①–⑥;
  - bias–variance plots per feature;
  - curves of the metrics as functions of $n$ and of SNR;
  - a final **win/loss map** (method × scenario × $n$) with paired-CI significance.
- Short report answering Q1–Q4 and the leakage check (§6).

---

## 11. Changes from the original plan

| # | Original plan | Change | Reason |
|---|---|---|---|
| 1 | Simulations only | Added real-data protocol (§9) | Explicitly asked by the supervisor |
| 2 | $I_j = \mathrm{Var}(f_j)$ used for every scenario | Valid only in the additive and independent case; correlated and interaction scenarios use ranking metrics plus reference targets (§8.2) | [S20]: MDI has no intrinsic target otherwise |
| 3 | Leakage check only for ⑥ | Also run it for ④ | ④ reuses OOB points in the same way. Without it, ⑥ vs ④ (the novelty claim) could be unfair |
| 4 | "UFI or MDI-oob" | MDI-oob as primary, UFI as robustness check; exact formula and centering convention fixed | The two differ; the scale and sign conventions must be fixed to compare with $I_j$ |
| 5 | MSE = bias² + variance | Variance with $1/R$, or compute MSE directly | Otherwise the decomposition is not exact |
| 6 | Noise mass "≈ 0" | Report signed **and** absolute; never clip negative values | Out-of-sample MDIs are unbiased only if negative values are kept |
| 7 | AUC only | Added Kendall τ among signal features | AUC ignores the ordering among signal features (heterogeneous scenario) |
| 8 | Paired Wilcoxon | Added CIs, win proportions, Holm correction | Many comparisons |
| 9 | Missing comparisons | Added ② vs ①, ④ vs ③, ⑥ vs ① | Separates the out-of-sample effect from the pruning effect |
| 10 | Unnormalized MDI stated | Explicitly forbid sklearn `feature_importances_` (normalized) | A common pitfall |
| 11 | Feature types | Added mixed-cardinality noise features (S3) | The classic MDI bias case |
