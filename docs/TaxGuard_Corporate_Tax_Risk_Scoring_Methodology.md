# TaxGuard: Corporate Tax Return Risk Scoring — Methodology, Calculations and References

**Title:** TaxGuard: An AI-Based Anomaly Detection Framework for Corporate Tax Return Risk Scoring at ZIMRA

**Theme:** Area 1 — Data, Automation and Intelligent Systems in the 4IR Era

**Category:** Prototype Demonstration

**Authors:** Edith Muyambiri 078685719 & Andile Bhebhe 0777303324

**Institution context:** Zimbabwe Revenue Authority (ZIMRA)

**Dataset:** `corporate_tax_risk_dataset.csv` — 1,900 corporate filings

**Report date:** June 2026

---

## ABSTRACT

Corporate tax non-compliance poses a significant threat to Zimbabwe’s fiscal sustainability. However, the Zimbabwe Revenue Authority (ZIMRA) currently relies heavily on manual audit selection processes that are resource-intensive, inconsistent, and unable to scale with the growing volume and complexity of corporate tax filings. The absence of a data-driven risk prioritization mechanism results in inefficient audit allocation, where high-risk fraudulent submissions may evade detection while compliant firms are subjected to disproportionate scrutiny. This paper proposes TaxGuard, a hybrid machine learning framework that integrates both unsupervised and supervised techniques to detect anomalies in corporate tax returns and assign probabilistic risk scores to each filing. The system ingests historical tax submissions, audited financial statements and transactional data, applying Isolation Forest and Autoencoder neural networks for anomaly detection. These are complemented by a Gradient Boosted classifier trained on confirmed fraud cases to enhance predictive precision over time. Key detection features include deviations in income-to-expense ratios, inter-period financial inconsistencies, sector-based outliers, and irregular value-added tax (VAT) to turnover relationships. Each tax return is assigned a risk score supported by SHAP-based explainability, enabling auditors to identify the specific financial indicators contributing to a flagged case. This improves transparency, reduces subjective bias and enhances audit prioritization. Furthermore, the system incorporates continuous learning by retraining on outcomes from completed audits, thereby progressively improving detection accuracy. The study demonstrates how TaxGuard advances intelligent fiscal governance in emerging economies by strengthening revenue assurance and audit efficiency. It aligns with Zimbabwe’s National AI Strategy (2026–2030) and National Development Strategy 2 objectives, particularly in enhancing domestic revenue mobilization and promoting accountability through data-driven decision-making.

**Keywords:** tax compliance; anomaly detection; gradient boosting; SHAP; audit prioritization; ZIMRA; Zimbabwe; machine learning; fiscal governance

**Prototype metrics (this repository):** ROC-AUC 0.9948, PR-AUC 0.9920, Brier 0.0244, F1 0.9576 — see Section 5.0.

---

*This document supplements the abstract with reproducible calculations (Section 4.0), written in the equation style of the credit-scoring reference papers in `docs/`.*

---

## Table of contents

1. [Introduction](#10-introduction)
2. [Problem statement](#20-problem-statement)
3. [Related scoring paradigms](#30-related-scoring-paradigms)
4. [Materials and methods](#40-materials-and-methods)
5. [Results and discussion](#50-results-and-discussion)
6. [Policy implications for ZIMRA](#60-policy-implications-for-zimra)
7. [Implementation and reproducibility](#70-implementation-and-reproducibility)
8. [Conclusion](#80-conclusion)
9. [References](#90-references)

---

## 1.0 Introduction

### 1.1 Tax risk and fiscal governance

> **Practitioner level.** Tax risk is the possibility that a filed return understates liability, overstates deductions, or uses aggressive structures (offshore, transfer pricing) so that **effective tax paid** diverges from **statutory obligation**. Poor audit targeting wastes auditor time on compliant firms and leaves high-risk firms unexamined.

> **Expert level.** Let y_i ∈ {0, 1} indicate whether filing i is labelled **High** risk (primary deployment target). The revenue authority seeks a score s_i ∈ [0, 1] that ranks filings so that, for fixed audit capacity k, the top-k ranked cases maximise expected audit yield subject to fairness and explainability constraints [1–3].

Corporate income tax underpins Zimbabwe’s **National Development Strategy 2** and **National AI Strategy (2026–2030)** objectives for accountable, data-driven public administration [4,5]. In emerging economies, revenue authorities face a dual pressure: taxpayers expect equitable treatment, while treasuries require predictable inflows to fund infrastructure, health, education, and social protection. Corporate income tax is particularly difficult to administer because liabilities depend on accounting profit, transfer pricing, offshore structures, and discretionary planning positions that may only become visible after a field audit.

Historically, audit case selection blended rule-based filters (sector, refund claims, payment delays) with examiner judgement. That approach worked when filing volumes were moderate, but it scales poorly. Examiners cannot review every return; they must rank cases. Without a quantitative ranker, ranking reverts to heuristics such as “large revenue” or “recent adverse event,” which the prototype data show to be weaker than statutory–effective tax rate gaps. TaxGuard is designed as a **decision-support layer**: it does not replace statutory powers or professional judgement; it orders the queue and documents why a return surfaced.

### 1.2 Audit selection as a scoring problem

Analogous to **credit scoring**—where banks estimate probability of default (PD) from applicant characteristics [6,7]—ZIMRA can treat **audit prioritisation** as predicting elevated non-compliance or elevated **Tax_Risk_Label**:

| Credit scoring | TaxGuard audit scoring |
|----------------|------------------------|
| Loan default / good–bad | High tax risk vs other tiers |
| Applicant financials | Return lines (revenue, ETR, offshore) |
| Credit bureau history | Audit outcomes, fines, planning scores |
| PD ∈ (0,1) | Calibrated risk score ∈ (0,1) |
| Reject / approve | Ranked audit queue |

### 1.3 Objectives of this study

1. Profile data quality and target distributions suitable for AMS integration.
2. Engineer **domain features** aligned with international audit risk practice (OECD BEPS indicators, control environment) [8].
3. Rank predictors by **univariate AUC**, correlation structure, and **composite policy flags**.
4. Train and evaluate a **hybrid ML pipeline** with leakage-safe **5-fold stratified OOF** validation [9].
5. Provide **SHAP-based explanations** for individual filings.
6. Benchmark against ZIMRA’s existing **`Audit_Likelihood`** heuristic.

### 1.4 Research design and document structure

This paper follows the structure used in Zimbabwean credit-scoring research [12] and linear probability optimisation studies [11]: introduction and problem framing, related models, materials and methods (with numbered equations), results, policy implications, and references. **Section 4.0** is the technical core: every formula maps to executable code. **Section 5.0** interprets outputs against ZIMRA operational goals. Readers may approach the text at three levels: (i) abstract and Section 6.0 for policy stakeholders; (ii) tables and figures for analysts; (iii) equations [1]–[30] for data scientists integrating TaxGuard into an Audit Management System (AMS).

---

## 2.0 Problem statement

Manual audit selection at ZIMRA exhibits:

- **Revenue-size bias** — large turnover does not imply high statutory–effective gap.
- **Recency bias** — prior **Clean** outcomes mask new planning or deviation signals.
- **Isolated review** — offshore exposure and weak controls are not scored multiplicatively.
- **Scale limits** — filing growth outpaces uniform human review.

**Proposed solution (TaxGuard).** A scoring function p_hat_i = f(x_i), where x_i combines raw return fields and engineered features, f is a stacked ensemble, and SHAP values explain which indicators drive p_hat_i [10].

### 2.1 Operational consequences of manual selection

When high-risk returns are under-selected, revenue is lost and compliant competitors face an uneven market. When low-risk returns are over-audited, firms bear unnecessary compliance costs and ZIMRA diverts senior examiner time away from productive cases. In credit markets, similar mis-ranking is associated with “reject inference” and sample bias [13]; in tax administration, the analogue is **over-reliance on enforcement history**—firms with no prior fines may escape review even when current-year indicators are extreme. The prototype quantifies one such cohort: top-decile tax rate deviation with zero fine history shows a **91.4%** high-risk rate versus a **33.4%** population baseline (Table 4).

### 2.2 Requirements derived from the abstract

The abstract specifies: (a) hybrid unsupervised + supervised detection; (b) probabilistic scores; (c) SHAP explainability; (d) continuous learning from completed audits; (e) detection of income-to-expense anomalies, inter-period inconsistencies, sector outliers, and VAT-to-turnover irregularities. Items (a)–(d) are implemented in this repository; (e) is partially implemented (profit margin, tax gaps) with **sector codes** and **VAT ratios** documented as AMS integration extensions.

---

## 3.0 Related scoring paradigms

*Parallels to Sections III–IV in linear probability and credit ML literature [11,12].*

### 3.1 Baseline: linear scorecard (ZIMRA `Audit_Likelihood`)

The legacy heuristic resembles a **linear probability model** (LPM) [11]:

$$
s_i^{\mathrm{LPM}} = \beta_0 + \boldsymbol{\beta}^{\top} \mathbf{x}_i^{\mathrm{raw}}
$$

where $\mathbf{x}_i^{\mathrm{raw}}$ contains manually weighted criteria. LPM scores are not strictly calibrated probabilities and may lie outside $[0,1]$ without a link function [13]. TaxGuard replaces this with a **calibrated** ensemble output $\hat{p}_i \in (0,1)$.

### 3.2 Logistic link and maximum likelihood

For binary high-risk label $y_i \in \{0,1\}$, the **logit model** [12] defines

$$
\mathbb{P}(y_i = 1 \mid \mathbf{x}_i) = \sigma(\boldsymbol{\omega}^{\top} \mathbf{x}_i + b), \qquad \sigma(z) = \frac{1}{1 + e^{-z}}
$$

Given $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^{N}$, the **binary cross-entropy** (negative log-likelihood) is

$$
\mathcal{L}_{\mathrm{BCE}}(\boldsymbol{\omega}, b) = -\frac{1}{N} \sum_{i=1}^{N} \Big[ y_i \log p_i + (1 - y_i) \log (1 - p_i) \Big], \quad p_i = \sigma(\boldsymbol{\omega}^{\top} \mathbf{x}_i + b)
$$

TaxGuard’s **stacking meta-learner** (Section 4.5) minimises $\mathcal{L}_{\mathrm{BCE}}$ on OOF base-model outputs $\mathbf{z}_i$ rather than on raw $\mathbf{x}_i$ alone.

### 3.3 Supervised learner: histogram-based gradient boosting

The abstract’s **Gradient Boosted classifier** is implemented as **HistGradientBoostingClassifier** [15, 18]. At iteration $t$, a regression tree $h_t$ is fitted to pseudo-responses (negative gradient of $\mathcal{L}_{\mathrm{BCE}}$), and the additive model updates

$$
F_T(\mathbf{x}) = F_{T-1}(\mathbf{x}) + \eta \, h_t(\mathbf{x}), \qquad p_i \approx \sigma(F_T(\mathbf{x}_i))
$$

with learning rate $\eta = 0.05$, depth limit 8, and $L_2$ regularisation $\lambda = 1$ (prototype hyperparameters). This captures non-linear interactions such as $\text{Planning\_Deviation\_Interaction}$.

### 3.4 Unsupervised anomaly learners

**Isolation Forest** [16] assigns an anomaly score from expected path length in random trees (implementation: `sklearn.ensemble.IsolationForest`).

**Autoencoder** [17]: with standardised input $\tilde{\mathbf{x}}_i = (\mathbf{x}_i - \boldsymbol{\mu}) / \boldsymbol{\sigma}$,

$$
\mathcal{L}_{\mathrm{AE}} = \frac{1}{N} \sum_{i=1}^{N} \left\| \tilde{\mathbf{x}}_i - g_{\theta}(f_{\theta}(\tilde{\mathbf{x}}_i)) \right\|_2^2
$$

Anomaly score $a_i^{\mathrm{AE}} = -\|\tilde{\mathbf{x}}_i - \hat{\tilde{\mathbf{x}}}_i\|_2^2$ (higher ⇒ more anomalous). Both $a_i^{\mathrm{IF}}$ and $a_i^{\mathrm{AE}}$ enter the stack as features for $\hat{p}_i$.

### 3.5 Comparative lessons from credit scoring in Zimbabwe

Nyoni and Matshisela [12] compared Lasso, Random Forest, SVM, and logit models on German Credit data using **10-fold cross-validation** and **AUROC**, reporting best performance near **0.80**. Nyathi et al. [11] optimised a **linear probability** scorecard for banking, emphasising interpretable weights and reduced rejection of creditworthy applicants without history. TaxGuard transfers three lessons: (1) **never evaluate on the same data used to train** (OOF folds); (2) **report AUROC and calibration**, not accuracy alone; (3) **combine interpretable features with ML** so policy officers and data scientists share a common feature language. TaxGuard differs in label definition (tax risk tier vs loan default) and in combining **unsupervised anomaly scores** with boosting, which is uncommon in classic credit papers but justified when fraud patterns are heterogeneous.

---

## 4.0 Materials and methods

### 4.0.1 Notation

| Symbol | Dimension / domain | Description |
|--------|-------------------|-------------|
| $N$ | scalar | Number of corporate filings ($N = 1900$) |
| $\mathbf{x}_i$ | $\mathbb{R}^d$ | Engineered feature vector for filing $i$ |
| $y_i$ | $\{0,1\}$ | Primary label: $\mathbb{1}[\text{Tax\_Risk\_Label}_i = \text{High}]$ |
| $y_i^{(\mathrm{nc})}$ | $\{0,1\}$ | Secondary label: non-compliant audit outcome |
| $\hat{p}_i$ | $(0,1)$ | Calibrated TaxGuard risk score (ensemble output) |
| $\mathbf{z}_i$ | $\mathbb{R}^3$ | Stack of OOF base scores (IF, AE, HGB) |
| $\sigma(\cdot)$ | — | Logistic sigmoid |
| $K$ | scalar | Stratified CV folds ($K = 5$) |
| $\tau^{\star}$ | scalar | Operating threshold on $\hat{p}_i$ (F1-optimal OOF) |

### 4.1 Theoretical framework

```
┌──────────────────┐     ┌─────────────────────┐     ┌────────────────────┐
│ Historical       │     │ Feature engineering │     │ Hybrid ML + SHAP   │
│ filings + audit  │ ──► │ + policy composites │ ──► │ OOF risk score     │
│ outcomes         │     │                     │     │ + explanations     │
└──────────────────┘     └─────────────────────┘     └────────────────────┘
         │                          │                            │
         └──────────────────────────┴────────────────────────────┘
                    Continuous learning on new audit closures
```

*Adapted from the credit-scoring pipeline in [11, Figure 1]: past debtor history → model → new customer prediction.*

**Continuous learning loop (deployment target).** After each audit cycle, outcomes labelled **Qualified** or **Adverse** act as confirmed non-compliance signals for retraining, consistent with the abstract’s “confirmed fraud cases” wording. **Clean** outcomes suppress false positives. Retraining frequency (e.g. quarterly) is an operational choice; the prototype exports `taxguard_ensemble.pkl` and metrics JSON so ZIMRA IT can schedule batch retraining without notebook execution.

**Data inputs.** The abstract references historical tax submissions, audited financial statements, and transactional data. The prototype CSV consolidates return-level fields available for analytics; full AMS integration would join VAT returns, payroll, customs, and third-party data feeds. Feature engineering code is modular so new columns (e.g. `VAT_to_Turnover`) can be added without rewriting the ensemble.

### 4.2 Dataset description

**Source file:** `corporate_tax_risk_dataset.csv`

| Quality metric | Value |
|----------------|-------|
| Records (N) | 1,900 |
| Raw features | 15 |
| Engineered features | 16 (29 total columns after engineering) |
| Missing values | 0 |
| Duplicate `Company_ID` | 0 |

**Table 1 — Raw variables**

| Variable | Type | Description |
|----------|------|-------------|
| `Company_ID` | ID | Unique entity |
| `Revenue_Million` | Continuous | Annual revenue (USD millions) |
| `Profit_Before_Tax_Million` | Continuous | Pre-tax profit |
| `Statutory_Tax_Rate` | Continuous | Applicable statutory rate (%) |
| `Effective_Tax_Rate` | Continuous | Reported effective rate (%) |
| `Offshore_Transactions_Million` | Continuous | Offshore transaction value |
| `Ownership_Concentration_Percent` | Continuous | Ownership concentration |
| `Internal_Control_Score` | Ordinal 1–5 | Control quality (higher = stronger) |
| `Audit_Likelihood` | Continuous | Existing ZIMRA heuristic ∈ [0,1] |
| `Audit_Outcome` | Categorical | Clean / Qualified / Adverse |
| `History_Fines_Million` | Continuous | Prior penalties |
| `Offshore_Subsidiaries` | Count | Offshore entities |
| `Aggressive_Tax_Planning_Score` | Continuous | Planning aggressiveness index |
| `Tax_Rate_Deviation` | Continuous | ETR vs statutory gap (%) |
| `Tax_Risk_Label` | Categorical | Low / Medium / High (target tier) |

**Table 2 — Target distributions**

| Target | Category | Count | Share |
|--------|----------|-------|-------|
| `Tax_Risk_Label` | Low | 634 | 33.4% |
| | Medium | 632 | 33.3% |
| | High | 634 | 33.4% |
| `Audit_Outcome` | Clean | 1,319 | 69.4% |
| | Qualified | 479 | 25.2% |
| | Adverse | 102 | 5.4% |

**Problem formulation.** Let $\mathcal{D} = \{(\mathbf{x}_i, y_i, y_i^{(\mathrm{nc})})\}_{i=1}^{N}$ denote the corporate filing dataset after feature engineering $\phi: \mathbb{R}^{p_{\mathrm{raw}}} \to \mathbb{R}^{d}$. The **primary learning task** is to learn $f: \mathbb{R}^{d} \to (0,1)$ minimising expected BCE on $y_i$ while producing interpretable attributions for auditors.

$$
y_i = \mathbb{1}\big[\text{Tax\_Risk\_Label}_i = \text{High}\big], \qquad
y_i^{(\mathrm{nc})} = \mathbb{1}\big[\text{Audit\_Outcome}_i \in \{\text{Qualified}, \text{Adverse}\}\big]
$$

Empirical prevalence: $\bar{y} = \frac{1}{N}\sum_i y_i = 0.3337$; $\bar{y}^{(\mathrm{nc})} = 0.3058$.

**Mapping to the abstract:** “Confirmed fraud cases” in deployment correspond to audit closures labelled **Adverse** or **Qualified**; these labels drive **continuous learning** retraining (Section 4.1). The prototype’s primary classifier target is **High** tax risk tier for audit queue ranking.

**Abstract detection features vs this dataset:**

| Abstract feature | Prototype variable / formula |
|------------------|------------------------------|
| Income-to-expense deviation | `Profit_Margin` [6] (profit / revenue) |
| Inter-period financial inconsistency | `Tax_Rate_Deviation`, `ETR_to_STR_Ratio` [9–10] |
| Sector-based outliers | Future work (sector codes not in CSV) |
| Irregular VAT-to-turnover | Future work (VAT not in CSV; noted in abstract) |

**Data governance.** Zero missing values and zero duplicate IDs support straight-through ingestion into a warehouse mart. Balanced **High / Medium / Low** tiers (≈33% each) simplify threshold tuning in the prototype; live ZIMRA populations may be imbalanced, requiring **PR-AUC** and cost-sensitive thresholds (Section 4.6). **Audit_Outcome** imbalance (69% Clean) mirrors realistic post-audit distributions and motivates a **secondary** non-compliance model (AUROC ≈ 0.51) separate from the primary queue ranker (AUROC ≈ 0.99).

---

### 4.3 Feature engineering map $\phi(\cdot)$

Raw return attributes $\mathbf{r}_i$ are transformed into $\mathbf{x}_i = \phi(\mathbf{r}_i)$ via `engineer_taxguard_features` (deterministic, no learnable parameters). Define stabilised scalars $R_i^{\ast} = \max(R_i, 10^{-6})$, $S_i^{\ast} = \max(\mathrm{STR}_i, 1)$.

**Profitability (income-to-expense proxy):**

$$
\text{Profit\_Margin}_i = \frac{P_i}{R_i^{\ast}}
$$

**Offshore / BEPS-style exposure:**

$$
\text{Offshore\_Intensity}_i = \frac{O_i^{\mathrm{off}}}{R_i^{\ast}}, \qquad
\text{Offshore\_per\_Sub}_i = \frac{O_i^{\mathrm{off}}}{n_i^{\mathrm{off}} + 1}
$$

where $O_i^{\mathrm{off}}$ = `Offshore_Transactions_Million`, $n_i^{\mathrm{off}}$ = `Offshore_Subsidiaries`.

**Statutory–effective tax gap:**

$$
\text{ETR/STR}_i = \frac{\mathrm{ETR}_i}{S_i^{\ast}}, \qquad
\text{Gap}_i = \left| \mathrm{DEV}_i \right|
$$

with $\mathrm{DEV}_i$ = `Tax_Rate_Deviation` (percentage points, supplied in data).

**Tax liability and underpayment:**

$$
\text{Tax\_Gap}_i = P_i \cdot \frac{\mathrm{STR}_i}{100} \left(1 - \frac{\mathrm{ETR}_i}{\mathrm{STR}_i}\right)
$$

$$
L_i^{\mathrm{impl}} = P_i \frac{\mathrm{ETR}_i}{100}, \quad L_i^{\mathrm{exp}} = P_i \frac{\mathrm{STR}_i}{100}, \quad
\text{Underpay\_Ratio}_i = \frac{L_i^{\mathrm{exp}} - L_i^{\mathrm{impl}}}{\max(L_i^{\mathrm{exp}}, 10^{-6})}
$$

**Governance:**

$$
\text{Control\_Risk}_i = 6 - c_i, \qquad \text{Fine\_Intensity}_i = \frac{F_i}{R_i^{\ast}}
$$

where $c_i$ = `Internal_Control_Score`, $F_i$ = `History_Fines_Million`.

**Interaction tensor (selected bilinear terms):**

$$
\psi_{1,i} = \pi_i \cdot \mathrm{DEV}_i \qquad \text{(planning} \times \text{ deviation)}
$$

$$
\psi_{2,i} = \omega_i \cdot \mathrm{Offshore\_Intensity}_i \qquad \text{(ownership} \times \text{ offshore)}
$$

$$
\psi_{3,i} = \alpha_i \cdot \lambda_i^{\mathrm{audit}} \qquad \text{(audit synergy)}
$$

with $\pi_i$ = `Aggressive_Tax_Planning_Score`, $\omega_i$ = `Ownership_Concentration_Percent`, $\lambda_i^{\mathrm{audit}}$ = `Audit_Likelihood`.

**Policy indicators** (cohort rules for Section 4.4.4): $M_{1,i} = \mathbb{1}[\text{Profit\_Margin}_i < Q_{0.25} \land R_i > \mathrm{median}(R)]$; $M_{2,i} = \mathbb{1}[\text{Offshore\_Intensity}_i > Q_{0.75} \land c_i \leq 2]$.

**Worked example (C0004):** $R \approx 1456.9$, $P \approx 229.6$, $\mathrm{STR}=20\%$, $\mathrm{ETR}\approx 7.69\%$, $\mathrm{DEV}\approx 12.31$ pp ⇒ $\mathrm{ETR}/\mathrm{STR} \approx 0.38$, indicating large statutory–effective separation on substantial profit.

---

### 4.4 Exploratory and univariate analysis

#### 4.4.1 Univariate ROC-AUC

For univariate score $z_i$ and label $y_i$, at threshold $t$ define $\hat{y}_i(t) = \mathbb{1}[z_i \geq t]$. With TP, FP, TN, FN induced by $t$,

$$
\mathrm{TPR}(t) = \frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FN}}, \qquad
\mathrm{FPR}(t) = \frac{\mathrm{FP}}{\mathrm{FP}+\mathrm{TN}}
$$

$$
\mathrm{AUROC}(z) = \int_0^1 \mathrm{TPR}\big(\mathrm{FPR}^{-1}(u)\big)\, du
$$

Estimated via the trapezoidal rule (`sklearn.metrics.roc_auc_score`). AUROC $= 0.5$ is chance; values $> 0.8$ are strong in credit-scoring benchmarks [12].

**Table 3 — Top univariate AUC for High risk** *(from `experiment_summary.json`)*

| Feature | AUC-ROC (High) | Interpretation |
|---------|----------------|----------------|
| Tax_Rate_Deviation | **0.9165** | Very strong |
| Statutory_Effective_Gap | 0.9165 | Alias of absolute deviation |
| Planning_Deviation_Interaction | 0.8661 | Strong composite |
| Tax_Underpayment_Ratio | 0.7800 | Strong |
| Tax_Gap_Million | 0.7453 | Moderate–strong |
| Audit_Planning_Synergy | 0.7105 | Moderate |
| **Audit_Likelihood (baseline)** | **0.6965** | Moderate — below top engineered |

**Discussion.** Effective tax rate alone shows weak High-risk AUROC (≈0.22) because direction matters: both unusually high and low ETR relative to peers can signal risk in different ways. Deviation and gap features encode **distance from statutory obligation**, which aligns with examiner intuition once made explicit. Planning interactions (AUROC ≈0.87) show that **aggressive planning amplifies deviation signal**, supporting composite rules in Table 4.

#### 4.4.2 Chi-square independence test

For contingency table $O_{rc}$ (risk tier $r$, outcome $c$) with expected $E_{rc} = \frac{(\sum_r O_{rc})(\sum_c O_{rc})}{N}$ under independence:

$$
\chi^2 = \sum_{r,c} \frac{(O_{rc} - E_{rc})^2}{E_{rc}}, \quad \mathrm{df} = (R-1)(C-1)
$$

**Result:** $\chi^2 = 6.13$, $\mathrm{df} = 4$, $p = 0.190$.

#### 4.4.3 Mann–Whitney U (High vs Low tier)

Non-parametric test on feature X between **High** and **Low** risk tiers (`01_taxguard_eda.ipynb`, §12) — confirms separation without assuming normality.

#### 4.4.4 Composite cohort uplift

For binary policy mask $M_i \in \{0,1\}$:

$$
\rho_M = \frac{\sum_{i: M_i=1} y_i}{\sum_i M_i}, \qquad
\mathrm{Uplift}_M = \frac{\rho_M}{\bar{y}}
$$

**Table 4 — Overlooked indicators**

| Indicator | Flagged (n) | High-risk rate | Uplift × |
|-----------|---------------|----------------|----------|
| Top-decile tax deviation + zero fines | 116 | 0.914 | **2.74** |
| High planning + clean audit history | 368 | 0.467 | 1.40 |
| High offshore + weak controls | 196 | 0.383 | 1.15 |
| Concentrated ownership + ≥3 offshore subs | 99 | 0.354 | 1.06 |
| Low profit margin + high revenue | 180 | 0.328 | 0.98 |

**Policy reading.** The **2.74×** uplift for first-time aggressors is the strongest operational finding: ZIMRA should not treat “no fines” as “no risk.” The **1.40×** uplift for high planning with clean history challenges recency bias. Revenue-only proxies (0.98×) should be deprioritised relative to rate-gap metrics—consistent with the abstract’s emphasis on inter-period and structural inconsistency rather than turnover alone.

---

### 4.5 Hybrid machine learning pipeline

#### 4.5.1 Stratified K-fold out-of-fold (OOF) protocol

Partition indices into folds $\mathcal{F}_1, \ldots, \mathcal{F}_K$ ($K=5$) preserving class ratio of $y_i$. For fold $k$, train base models on $\mathcal{D} \setminus \mathcal{F}_k$ and predict on $\mathcal{F}_k$ only:

$$
\hat{z}_i^{(k)} = \big( a_i^{\mathrm{IF}},\, a_i^{\mathrm{AE}},\, p_i^{\mathrm{HGB}} \big)^{\top}, \quad i \in \mathcal{F}_k
$$

Concatenating folds yields OOF matrix $\mathbf{Z} \in \mathbb{R}^{N \times 3}$ with no label leakage from the same fold [9].

**Algorithm 1 — TaxGuard OOF training and scoring**

```
Input: Feature matrix X ∈ R^{N×d}, labels y ∈ {0,1}^N, folds F_1…F_K
Output: OOF scores p_hat ∈ (0,1)^N, threshold τ*

1. Initialise vectors z_IF, z_AE, p_HGB ∈ R^N to zero
2. For k = 1 … K:
3.     Train IsolationForest, Autoencoder, HGB on {i : i ∉ F_k}
4.     For i ∈ F_k:
5.         z_IF[i] ← IF.score_samples(x_i)
6.         z_AE[i] ← −MSE(autoencoder, x_i)
7.         p_HGB[i] ← HGB.predict_proba(x_i)[1]
8. Stack Z = [z_IF, z_AE, p_HGB]
9. For k = 1 … K:   // meta-learner OOF
10.    Fit logistic regression on rows i ∉ F_k; predict p_hat[i] for i ∈ F_k
11. τ* ← argmax_t F1(y, 1[p_hat ≥ t])
12. Return p_hat, τ*
```

#### 4.5.2 Base model specifications

| Module | Class | Key hyperparameters | OOF AUROC |
|--------|-------|---------------------|-----------|
| Isolation Forest | `IsolationForest` | `n_estimators=300`, `contamination=0.33` | 0.3157 |
| Autoencoder | MLP $(48,24,48)$ | `early_stopping=True`, MSE loss (Eq. 5) | 0.4240 |
| Gradient boost | `HistGradientBoostingClassifier` | `max_depth=8`, `η=0.05`, `T=600`, `λ=1` | **0.9964** |

#### 4.5.3 Stacking meta-learner

Let $\mathbf{z}_i$ denote the OOF stack for filing $i$. The **level-2** model is logistic regression:

$$
\hat{p}_i = \sigma(\mathbf{w}^{\top} \mathbf{z}_i + b)
$$

with $(\mathbf{w}, b)$ fitted by minimising Eq. (3) on OOF pairs $(\mathbf{z}_i, y_i)$ using the same $K$-fold scheme.

#### 4.5.4 Evaluation metrics (OOF test statistics)

| Metric | Definition | TaxGuard value |
|--------|------------|----------------|
| $\mathrm{AUROC}(\hat{p}, y)$ | Eq. (17) on OOF $\hat{p}_i$ | **0.9948** |
| $\mathrm{AP}(\hat{p}, y)$ | $\int \mathrm{Precision}(r)\, dr$ (PR-AUC) | **0.9920** |
| $\mathrm{Brier}(\hat{p}, y)$ | $\frac{1}{N}\sum_i (y_i - \hat{p}_i)^2$ | **0.0244** |
| $F_1(\tau)$ | $\frac{2PR}{P+R}$ at threshold $\tau$ | **0.9576** at $\tau^{\star}$ |
| $\tau^{\star}$ | $\arg\max_{\tau} F_1(\tau)$ | **0.7811** |
| Baseline AUROC | $\mathrm{AUROC}(\lambda^{\mathrm{audit}}, y)$ | 0.6965 |
| Gini | $2 \cdot \mathrm{AUROC} - 1$ [12] | **0.9896** |

**Hard classifier:**

$$
\hat{y}_i = \mathbb{1}[\hat{p}_i \geq \tau^{\star}]
$$

**Secondary task:** $\mathrm{AUROC}(\hat{p}^{(\mathrm{nc})}, y^{(\mathrm{nc})}) = 0.5091$ on OOF stack trained for $y_i^{(\mathrm{nc})}$ — supports using audit outcomes for **retraining**, not primary queue ranking.

**Table 6 — OOF classification report @ τ* = 0.7811**

| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| Not High Risk | 0.9727 | 0.9858 | 0.9792 | 1,266 |
| High Risk | 0.9708 | 0.9448 | 0.9576 | 634 |
| **Accuracy** | | | **0.9721** | 1,900 |

#### 4.5.7 SHAP explainability

For additively explained model output $f(\mathbf{x}_i)$, SHAP values $\phi_{ij}$ satisfy the efficiency constraint [10]:

$$
f(\mathbf{x}_i) = \mathbb{E}[f(\mathbf{X})] + \sum_{j=1}^{d} \phi_{ij}
$$

TreeSHAP on the fitted HistGradientBoosting model yields global rankings (Figure 14–15) and local **waterfall** decompositions per `Company_ID` (Figure 16), satisfying the abstract’s auditor-facing transparency requirement.

#### 4.5.8 Optional XGBoost gain importance

When `xgboost` is installed, gain-based importance (Figure 17) cross-checks SHAP rankings.

---

### 4.6 Performance measures (credit-scoring analogue)

Following [12, §2.5]:

| Measure | Role in TaxGuard | Notes |
|---------|------------------|-------|
| **AUROC** | Primary discrimination | 0.5 = random; >0.8 strong [12] |
| **Gini** | 2 × AUROC − 1 | Ensemble Gini ≈ **0.9896** |
| **KS statistic** | Max absolute gap between CDF of scores for High vs Not High | Class separation (cf. [12]) |
| **PR-AUC** | Imbalanced / costly FP audits | Preferred when positives are rare in deployment |
| **Brier** | Calibration quality | Lower is better; 0 = perfect |
| **F1** | Operations at chosen τ* | Balances precision/recall for queue size |

**ROC curve:** plot FPR vs TPR across thresholds. **PR curve:** precision vs recall.

At threshold $\tau$:

$$
\mathrm{Precision}(\tau) = \frac{\mathrm{TP}(\tau)}{\mathrm{TP}(\tau)+\mathrm{FP}(\tau)}, \quad
\mathrm{Recall}(\tau) = \frac{\mathrm{TP}(\tau)}{\mathrm{TP}(\tau)+\mathrm{FN}(\tau)}
$$

$$
F_1(\tau) = \frac{2 \cdot \mathrm{Precision}(\tau) \cdot \mathrm{Recall}(\tau)}{\mathrm{Precision}(\tau) + \mathrm{Recall}(\tau)}
$$

**Brier score** (calibration): Eq. in Table above. **KS statistic** [12]: $D = \sup_t \left| F_0(t) - F_1(t) \right|$ where $F_0, F_1$ are CDFs of $\hat{p}$ for $y=0$ and $y=1$.

**Why AUROC and PR-AUC?** Under class imbalance at deployment, $\mathrm{AP}$ prioritises precision at the top of the ranked queue—critical when audit capacity is binding.

### 4.7 Risk scoring function (deployment prototype)

For new filing features **x**:

1. Engineer features via `engineer_taxguard_features`.
2. Compute Isolation Forest score, Autoencoder score, and HGB probability on production models.
3. Meta-learner: $\hat{p}_i = \sigma(\mathbf{w}^{\top}\mathbf{z}_i + b)$ per Eq. (21).
4. Flag if $\hat{p}_i \geq \tau^{\star}$; attach SHAP values $\phi_{ij}$ for AMS UI.

Artefacts: `outputs/models/taxguard_ensemble.pkl`, `outputs/models/taxguard_metrics.json`.

---

## 5.0 Results and discussion

This section interprets experimental outputs against the abstract’s claims: hybrid detection, probabilistic scoring, explainability, and continuous learning readiness.

### 5.0.1 Summary of hypothesis tests

The prototype confirms four hypotheses: (H1) engineered tax-gap features outperform `Audit_Likelihood` in univariate ranking; (H2) composite masks surface policy-relevant cohorts manual rules miss; (H3) supervised gradient boosting plus stacking achieves near-perfect OOF separation of High risk tier; (H4) post-audit outcomes are not fully predictable from pre-audit fields alone, so outcomes should drive **retraining** more than **same-year queue scoring**. H3 supports deployment of **p_hat** as primary queue key; H4 supports the dual-target design in equations [4]–[5].

### 5.1 Exploratory figures

| Fig. | File | Finding |
|------|------|---------|
| 1 | `01_target_distributions.png` | Balanced risk tiers; 30.6% non-clean audits |
| 2 | `02_univariate_by_risk_tier.png` | Tier-stratified revenue, ETR, deviation |
| 3 | `03_bivariate_boxplots.png` | High tier: wider deviation, planning |
| 4 | `04_correlation_matrix.png` | Deviation ↔ underpayment; planning partly independent |
| 5 | `05_univariate_auc_ranking.png` | Deviation dominates |
| 6 | `06_roc_univariate_top_features.png` | Deviation vs baseline ROC |
| 7 | `07_risk_outcome_crosstab.png` | Tier–outcome calibration |
| 8 | `08_zimra_overlooked_indicators.png` | Composite uplifts |
| 9 | `09_pairplot_top_features.png` | Non-linear tier separation |
| 10 | `10_multiclass_ovr_roc.png` | OvR with deviation |

### 5.2 Model figures

| Fig. | File | Finding |
|------|------|---------|
| 11 | `11_model_roc_pr_comparison.png` | Ensemble dominates components |
| 12 | `12_confusion_matrix.png` | Low false positives at τ* |
| 13 | `13_calibration_curve.png` | Scores track observed rates |
| 14–16 | SHAP plots | Explainable drivers |
| 17 | `17_xgb_feature_importance.png` | Gain alignment |
| 18 | `18_noncompliance_roc.png` | Weak outcome prediction |

### 5.3 Comparison to example credit-scoring benchmarks

| Study | Domain | Best model | AUROC |
|-------|--------|------------|-------|
| Nyoni & Matshisela (2018) [12] | German credit | Lasso | 0.8048 |
| Nyoni & Matshisela (2018) [12] | German credit | Random Forest | 0.7869 |
| Nyoni & Matshisela (2018) [12] | German credit | Logit | 0.7678 |
| **TaxGuard (this study)** | Corporate tax / ZIMRA | **Stacked ensemble** | **0.9948** |

*Caution:* TaxGuard uses a structured research dataset; German Credit is UCI real data. Metrics are **not directly comparable** across domains—but methodology (10-fold/5-fold CV, AUROC, KS) is aligned.

### 5.4 Why unsupervised layers remain in the stack

Isolation Forest and autoencoder alone underperform (AUC 0.32, 0.42) but add **orthogonal errors** to boosting; the meta-learner weights them without in-sample leakage. Primary signal remains **supervised** tax-gap features.

### 5.5 Calibration and operational thresholding

The calibration curve (Figure 13) and Brier score **0.0244** indicate that **p_hat** is usable as a proportional weight when planning audit hours—e.g. expecting roughly 78% positive rate among cases scoring 0.80 if the curve is near-diagonal in that band. The F1-optimal threshold **τ* = 0.7811** is a research default; ZIMRA may lower τ* to increase recall (more audits) or raise τ* to protect compliant firms, trading off precision per examiner hour. SHAP waterfalls (Figure 16) should accompany any adverse action to meet transparency expectations under national AI ethics frameworks [4].

### 5.6 Limitations and external validity

Results are obtained on **1,900 structured prototype records**, not live ZIMRA operational data. AUROC near 0.99 may moderate when label noise, concept drift, and missing external data are introduced. Sector and VAT features in the abstract are not yet in the feature matrix. Legal defensibility requires human sign-off on audits regardless of model output. Finally, phone contact details in the author block are for correspondence on this demonstration only and should follow institutional data-protection policies in any public document version.

---

## 6.0 Policy implications for ZIMRA

| Finding | Manual gap | Recommendation |
|---------|------------|----------------|
| Tax rate deviation AUC ≈ 0.92 | Revenue-focused selection | Mandatory ETR–STR gap rules + ML score |
| First-time aggressors 2.74× uplift | Skipped (no fines) | Composite flag in AMS |
| Clean history + high planning 1.4× | Recency bias | Decay + planning override |
| Audit_Likelihood AUC 0.70 | Stale heuristic | Replace with TaxGuard p_hat |
| SHAP attributions | Opaque flags | Waterfall per case |

### 6.1 Phased AMS integration

**Phase 1 (pilot region, 0–6 months):** Batch-score corporate returns monthly; display top decile with SHAP summaries; compare against legacy `Audit_Likelihood` ranking; measure hit rate on Qualified/Adverse closures.

**Phase 2 (6–18 months):** Embed scores in workflow; add sector and VAT features; calibrate τ* by industry; train examiner workshops on reading SHAP.

**Phase 3 (18–36 months):** Near-real-time scoring on e-filing; automated retraining pipeline; publish aggregate compliance analytics (privacy-preserving).

### 6.2 Alignment with national strategy

TaxGuard directly supports **domestic revenue mobilisation** (NDS2) by improving yield per audit hour and **National AI Strategy** goals by demonstrating ethical, explainable public-sector AI. It also advances **4IR** skills transfer: Zimbabwean universities can reproduce notebooks and extend features using open artefacts in this repository.

### 6.3 Ethics, fairness, and oversight

Risk scores must not substitute for due process. Models should be monitored for disparate impact across sectors and ownership structures once sector codes exist. An appeals pathway and model version logging are recommended. Continuous learning must avoid feedback loops that permanently penalise firms once incorrectly labelled—human review of Adverse labels before they enter training data is advised.

---

## 7.0 Implementation and reproducibility

```bash
pip install -r requirements.txt
python scripts/generate_report_assets.py
jupyter notebook notebooks/01_taxguard_eda.ipynb
jupyter notebook notebooks/02_taxguard_hybrid_model.ipynb
```

| Artefact | Path |
|----------|------|
| Metrics JSON | `outputs/metrics/experiment_summary.json` |
| Figures | `notebooks/figures/*.png` |
| Model bundle | `outputs/models/taxguard_ensemble.pkl` |
| Web demo | `web/` (React + Vite) |

**Environment.** Python 3.11+ recommended; dependencies in `requirements.txt` (pandas, scikit-learn, shap, matplotlib, xgboost optional). Random seed **42** is fixed across notebooks and `generate_report_assets.py` for reproducible figures under `notebooks/figures/`.

**Replication checklist.**

1. Clone repository and create virtual environment.
2. Run `python scripts/generate_report_assets.py` — regenerates figures and `outputs/metrics/experiment_summary.json`.
3. Execute `01_taxguard_eda.ipynb` then `02_taxguard_hybrid_model.ipynb` for interactive review.
4. Compare metrics to Table 5; tolerances ±0.001 may apply if library versions differ slightly.

**Web demonstration.** The React application in `web/` provides a stakeholder-facing UI for entering filing attributes and viewing risk outputs—useful for prototype reviews, not for production scoring without backend hardening.

---

## 8.0 Conclusion

TaxGuard applies **credit-scoring discipline**—explicit scores, probabilistic calibration, ensemble ML, and AUROC/KS-style evaluation—to **corporate tax audit risk** at ZIMRA. The study answers the abstract’s call for a **hybrid, explainable, continuously learnable** framework: Isolation Forest and autoencoder supply multivariate anomaly signals; HistGradientBoosting captures non-linear tax-gap structure; logistic stacking calibrates a single **p_hat** for queue management; SHAP satisfies auditor-facing transparency.

Empirically, **tax rate deviation** and **planning × deviation** dominate univariate rankings; **composite policy rules** expose cohorts with up to **2.74×** baseline high-risk rates; the **stacked ensemble** achieves OOF **ROC-AUC 0.9948**, far above **`Audit_Likelihood` (0.6965)**. Deployment should proceed via phased AMS integration with human oversight, sector/VAT enrichment, and quarterly retraining on **Qualified** and **Adverse** closures—turning each completed audit into institutional learning consistent with Zimbabwe’s fiscal digitalisation agenda.

Future work: live ZIMRA operational data validation, sector-normalised thresholds, VAT-to-turnover and macroeconomic covariates, and field trials measuring incremental revenue per additional audit hour attributable to TaxGuard ranking.

---

## 9.0 References

[1] Harwood, J.R. et al. (1999). *Managing Risk in Farming: Concepts, Research and Analysis.* USDA ERS Agricultural Economic Report No. 774.

[2] Khan, T. and Ahmed, H. (2001). *Risk Management: An Analysis of Issues in Islamic Financial Industry.* IRTI/IDB Occasional Paper No. 5.

[3] Basel Committee on Banking Supervision (2006). *International Convergence of Capital Measurement and Capital Standards: A Revised Framework.* Bank for International Settlements, Basel.

[4] Government of Zimbabwe (2026). *National AI Strategy 2026–2030* (policy alignment cited in prototype documentation).

[5] Government of Zimbabwe. *National Development Strategy 2 (NDS2)* — domestic revenue mobilisation pillar.

[6] Rose, P.S. (2002). *Commercial Bank Management.* 5th ed. McGraw-Hill/Irwin.

[7] BIS (2000). *Principles for the Management of Credit Risk.* Basel Committee document series.

[8] OECD (2015). *BEPS Action 13: Transfer Pricing Documentation and Country-by-Country Reporting* — offshore / transfer-pricing risk context.

[9] Kohavi, R. (1995). A study of cross-validation and bootstrap for accuracy estimation and model selection. *IJCAI.*

[10] Lundberg, S.M. and Lee, S.-I. (2017). A unified approach to interpreting model predictions. *NeurIPS* (SHAP).

[11] Nyathi, K.T. et al. (2014). Optimisation of the linear probability model for credit risk management. *International Journal of Computer and Information Technology,* 3(6), 1340–1345. *(Example document in `docs/`.)*

[12] Nyoni, E.E.T. and Matshisela, N. (2018). Credit scoring using machine learning algorithms. *Zimbabwe Journal of Science & Technology,* 13, 26–34. *(Example document in `docs/`.)*

[13] Verstraeten, G. and Van den Poel, D. (2004). The impact of sample bias on consumer credit scoring performance. *Journal of the Operational Research Society,* 56(8), 981–992.

[14] Breiman, L. (2001). Random forests. *Machine Learning,* 45(1), 5–32.

[15] Pedregosa, F. et al. (2011). Scikit-learn: Machine learning in Python. *JMLR,* 12, 2825–2830. — `HistGradientBoostingClassifier`, `IsolationForest`, `roc_auc_score`, `calibration_curve`.

[16] Liu, F.T., Ting, K.M., and Zhou, Z.-H. (2008). Isolation Forest. *IEEE ICDM.*

[17] Hinton, G.E. and Salakhutdinov, R.R. (2006). Reducing the dimensionality of data with neural networks. *Science,* 313(5786), 504–507. (Autoencoder foundation.)

[18] Friedman, J.H. (2001). Greedy function approximation: a gradient boosting machine. *Annals of Statistics,* 29(5), 1189–1232.

[19] Tabachnick, B.G. and Fidell, L.S. (2013). *Using Multivariate Statistics.* 6th ed. Pearson. (Chi-square, MANOVA context.)

[20] Field, A., Miles, J., and Field, Z. (2012). *Discovering Statistics Using R.* Sage. (Log-likelihood, model fit.)

[21] Hanley, J.A. and McNeil, B.J. (1982). The meaning and use of the area under a receiver operating characteristic (ROC) curve. *Radiology,* 143(1), 29–36.

[22] Thomas, L.C. (2000). A survey of credit and behavioural scoring: forecasting financial risk of lending to consumers. *International Journal of Forecasting,* 16(2), 149–172.

[23] Crook, J.N., Edelman, D.B., and Thomas, L.C. (2007). Recent developments in consumer credit risk assessment. *European Journal of Operational Research,* 183(3), 1447–1465.

[24] Yu, L., Wang, S., and Lai, K.K. (2008). *Bio-Inspired Credit Risk Analysis.* Springer. (SVM / intelligent credit analysis.)

[25] Chen, T. and Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *KDD* (optional TaxGuard component).

[26] Zimbabwe Revenue Authority (ZIMRA). Corporate income tax administration and audit management context (institutional framing for prototype).

[27] TaxGuard Project (2026). `corporate_tax_risk_dataset.csv`, `src/taxguard_features.py`, `scripts/generate_report_assets.py` — reproducible artefacts in repository `tax-compliance-risk-prediction`.

---

*TaxGuard Research Prototype — Edith Muyambiri & Andile Bhebhe, 2026*  
*Supporting intelligent, accountable, data-driven fiscal governance at ZIMRA.*
