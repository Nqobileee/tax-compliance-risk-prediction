import experimentStats from "../data/experimentStats.json";

export const stats = experimentStats;

export const modelMetrics = experimentStats.model_oof;

export const riskDistribution = [
  { tier: "Low", count: experimentStats.targets.risk_label_distribution.Low, fill: "#86efac" },
  { tier: "Medium", count: experimentStats.targets.risk_label_distribution.Medium, fill: "#4ade80" },
  { tier: "High", count: experimentStats.targets.risk_label_distribution.High, fill: "#059669" },
];

export const auditOutcomes = [
  { outcome: "Clean", count: experimentStats.targets.audit_outcome_distribution.Clean },
  { outcome: "Qualified", count: experimentStats.targets.audit_outcome_distribution.Qualified },
  { outcome: "Adverse", count: experimentStats.targets.audit_outcome_distribution.Adverse },
];

export const modelComparison = [
  { model: "ZIMRA Baseline", auc: experimentStats.model_oof.audit_likelihood_auc_baseline },
  { model: "Isolation Forest", auc: experimentStats.model_oof.isolation_forest_auc },
  { model: "Autoencoder", auc: experimentStats.model_oof.autoencoder_auc },
  { model: "Gradient Boosting", auc: experimentStats.model_oof.hist_gradient_boosting_auc },
  { model: "TaxGuard Ensemble", auc: experimentStats.model_oof.ensemble_roc_auc },
];

export const featureAuc = Object.entries(experimentStats.top_univariate_auc.AUC_High_Risk)
  .slice(0, 8)
  .map(([name, auc]) => ({
    feature: name.replace(/_/g, " "),
    auc: auc as number,
  }));

export const overlookedIndicators = experimentStats.zimra_overlooked_indicators;
