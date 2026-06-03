# TaxGuard notebook figures

Plots are saved here when you run `01_taxguard_eda.ipynb`, `02_taxguard_hybrid_model.ipynb`, or:

```bash
python scripts/generate_report_assets.py
```

| File | Description |
|------|-------------|
| `01_target_distributions.png` | Risk labels, audit outcomes, audit likelihood |
| `02_univariate_by_risk_tier.png` | Feature histograms by risk tier |
| `03_bivariate_boxplots.png` | Key indicators vs risk tier |
| `04_correlation_matrix.png` | Feature correlation heatmap |
| `05_univariate_auc_ranking.png` | Top features by univariate AUC |
| `06_roc_univariate_top_features.png` | Univariate ROC vs ZIMRA baseline |
| `07_risk_outcome_crosstab.png` | P(outcome \| risk tier) |
| `08_zimra_overlooked_indicators.png` | Composite indicator uplift |
| `09_pairplot_top_features.png` | Pairwise relationships by tier |
| `10_multiclass_ovr_roc.png` | Multiclass OvR ROC |
| `11_model_roc_pr_comparison.png` | Hybrid model ROC & PR curves |
| `12_confusion_matrix.png` | Confusion matrix @ optimal threshold |
| `13_calibration_curve.png` | Probability calibration |
| `14_shap_summary.png` | SHAP beeswarm |
| `15_shap_bar.png` | Mean \|SHAP\| bar chart |
| `16_shap_waterfall_example.png` | Single-filing waterfall |
| `17_xgb_feature_importance.png` | XGBoost gain importance |
| `18_noncompliance_roc.png` | Non-compliance ROC |

Referenced in [README.md](../../README.md) and [TAXGUARD_RESEARCH_SUMMARY.md](../../TAXGUARD_RESEARCH_SUMMARY.md).
