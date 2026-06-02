# TaxGuard: AI-Based Corporate Tax Return Risk Scoring

[![Theme](https://img.shields.io/badge/Theme-4IR%20Intelligent%20Systems-blue)]()
[![Category](https://img.shields.io/badge/Category-Prototype%20Demonstration-green)]()
[![ZIMRA](https://img.shields.io/badge/Context-Zimbabwe%20Revenue%20Authority-red)]()

> **TaxGuard** — An AI-Based Anomaly Detection Framework for Corporate Tax Return Risk Scoring at ZIMRA

**Authors:** Edith Muyambiri (078685719) & Andile Bhebhe (0777303324)  
**Research Theme:** Area 1 — Data, Automation and Intelligent Systems in the 4IR Era

---

## Overview

Corporate tax non-compliance poses a significant threat to Zimbabwe's fiscal sustainability. The Zimbabwe Revenue Authority (ZIMRA) currently relies heavily on **manual audit selection** — resource-intensive, inconsistent, and unable to scale with the growing volume and complexity of corporate tax filings.

**TaxGuard** proposes a hybrid machine learning framework that integrates unsupervised and supervised techniques to:

- Detect anomalies in corporate tax returns
- Assign **probabilistic risk scores** to each filing
- Provide **SHAP-based explainability** for auditor-facing flag justification
- Support **continuous learning** from completed audit outcomes

This repository contains the research prototype: exploratory analysis, feature engineering, and a production-grade hybrid ML pipeline.

📄 **[Full Research Summary Report](TAXGUARD_RESEARCH_SUMMARY.md)** — Complete findings, graphs, ZIMRA policy implications, and national development alignment.

---

## Problem Statement

| Challenge | Impact |
|-----------|--------|
| Manual audit selection | High-risk filings may evade detection |
| No data-driven prioritisation | Compliant firms face disproportionate scrutiny |
| Subjective heuristics | Inconsistent audit allocation across regions |
| Scale limitations | Cannot keep pace with filing volume growth |

TaxGuard addresses these gaps through intelligent, accountable, data-driven fiscal governance aligned with Zimbabwe's **National AI Strategy (2026–2030)** and **National Development Strategy 2**.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     TaxGuard Hybrid Pipeline                  │
├──────────────────────────────────────────────────────────────┤
│  UNSUPERVISED                    SUPERVISED                   │
│  ├── Isolation Forest            ├── HistGradientBoosting     │
│  └── Autoencoder (MLP)           ├── XGBoost                  │
│                                  └── LightGBM                 │
├──────────────────────────────────────────────────────────────┤
│  STACKED ENSEMBLE → Calibrated Risk Score → SHAP Explanation  │
└──────────────────────────────────────────────────────────────┘
```

### Key Detection Features

- Income-to-expense ratio deviations (`Profit_Margin`)
- Inter-period tax rate inconsistencies (`Tax_Rate_Deviation`, `ETR_to_STR_Ratio`)
- Offshore / transfer pricing exposure (`Offshore_Intensity`)
- Sector-based outlier patterns (engineered interactions)
- Aggressive tax planning composites (`Planning_Deviation_Interaction`)
- Governance weakness indicators (`Control_Risk`, ownership concentration)

---

## Repository Structure

```
tax-compliance-risk-prediction/
├── corporate_tax_risk_dataset.csv   # 1,900 corporate tax filings
├── notebooks/
│   ├── 01_taxguard_eda.ipynb        # Research-grade exploratory analysis
│   └── 02_taxguard_hybrid_model.ipynb  # Hybrid ML pipeline
├── src/
│   └── taxguard_features.py         # Shared feature engineering
├── scripts/
│   └── build_notebooks.py           # Notebook generator
├── outputs/
│   ├── figures/                     # Generated plots (after running notebooks)
│   └── models/                      # Saved model artefacts
├── requirements.txt
└── README.md
```

---

## Dataset

**File:** `corporate_tax_risk_dataset.csv`  
**Records:** 1,900 corporate entities  
**Features:** 15 raw columns + 16 engineered features

| Column | Description |
|--------|-------------|
| `Company_ID` | Unique corporate identifier |
| `Revenue_Million` | Annual revenue (USD millions) |
| `Profit_Before_Tax_Million` | Pre-tax profit |
| `Statutory_Tax_Rate` | Applicable statutory rate (%) |
| `Effective_Tax_Rate` | Reported effective rate (%) |
| `Offshore_Transactions_Million` | Offshore transaction value |
| `Ownership_Concentration_Percent` | Ownership concentration index |
| `Internal_Control_Score` | Internal control quality (1–5) |
| `Audit_Likelihood` | Existing ZIMRA heuristic score |
| `Audit_Outcome` | Clean / Qualified / Adverse |
| `History_Fines_Million` | Prior penalty history |
| `Offshore_Subsidiaries` | Count of offshore entities |
| `Aggressive_Tax_Planning_Score` | Planning aggressiveness index |
| `Tax_Rate_Deviation` | ETR vs statutory gap |
| `Tax_Risk_Label` | Target: Low / Medium / High |

---

## Quick Start

### 1. Clone and install dependencies

```bash
git clone https://github.com/your-org/tax-compliance-risk-prediction.git
cd tax-compliance-risk-prediction
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Run exploratory analysis

```bash
jupyter notebook notebooks/01_taxguard_eda.ipynb
```

### 3. Train the hybrid model

```bash
jupyter notebook notebooks/02_taxguard_hybrid_model.ipynb
```

Model artefacts are saved to `outputs/models/taxguard_ensemble.pkl`.

---

## Key Findings (EDA)

1. **`Tax_Rate_Deviation`** is the strongest univariate predictor of high-risk filings (AUC > 0.90).
2. **Composite indicators** (offshore intensity × weak controls, planning × deviation) surface cohorts invisible to single-variable manual rules.
3. ZIMRA's existing **`Audit_Likelihood`** heuristic underperforms top engineered features — validating ML investment.
4. **Risk labels correlate significantly** with audit outcomes — supporting supervised retraining on completed audits.
5. Firms with **clean audit history but high planning scores** remain elevated risk — a pattern manual selection often misses.

---

## Model Performance

Metrics are computed via **5-fold stratified cross-validation** with out-of-fold predictions (no data leakage):

| Metric | Description |
|--------|-------------|
| **ROC-AUC** | High-risk filing discrimination |
| **PR-AUC** | Precision-recall for imbalanced audit costing |
| **Brier Score** | Probability calibration quality |
| **F1 @ optimal threshold** | Operational audit queue sizing |

Run `02_taxguard_hybrid_model.ipynb` for full results. Metrics are exported to `outputs/models/taxguard_metrics.json`.

---

## ZIMRA Deployment Recommendations

1. **Integrate TaxGuard scores** into the Audit Management System (AMS) workflow.
2. **Display SHAP waterfall charts** alongside flagged cases for auditor transparency.
3. **Retrain quarterly** on Adverse and Qualified audit closures (continuous learning loop).
4. **Expand data ingestion** to VAT-to-turnover ratios and sector classification codes.
5. **Calibrate thresholds** by sector and entity size to ensure equitable audit allocation.

---

## Research Alignment

| Framework | Alignment |
|-----------|-----------|
| National AI Strategy (2026–2030) | Intelligent automation for public sector |
| NDS2 | Domestic revenue mobilisation |
| 4IR Theme | Data-driven decision-making in fiscal governance |

---

## License

This research prototype is provided for academic and demonstration purposes. Contact the authors for ZIMRA deployment licensing discussions.

---

## Citation

```bibtex
@misc{muyambiri2026taxguard,
  title={TaxGuard: An AI-Based Anomaly Detection Framework for Corporate Tax Return Risk Scoring at ZIMRA},
  author={Muyambiri, Edith and Bhebhe, Andile},
  year={2026},
  note={Prototype Demonstration — Area 1: Data, Automation and Intelligent Systems in the 4IR Era}
}
```

---

## Contact

| Author | Contact |
|--------|---------|
| Edith Muyambiri 
| Andile Bhebhe 