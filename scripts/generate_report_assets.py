"""Generate all TaxGuard figures and export summary metrics for research report."""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, label_binarize

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from taxguard_features import engineer_taxguard_features, get_model_feature_columns

FIGURES = ROOT / "outputs" / "figures"
METRICS = ROOT / "outputs" / "metrics"
FIGURES.mkdir(parents=True, exist_ok=True)
METRICS.mkdir(parents=True, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")
RANDOM_STATE = 42
N_SPLITS = 5
risk_order = ["Low", "Medium", "High"]
outcome_order = ["Clean", "Qualified", "Adverse"]


class AutoencoderAnomalyScorer:
    def __init__(self, hidden=(48, 24, 48), max_iter=500, random_state=42):
        self.hidden = hidden
        self.max_iter = max_iter
        self.random_state = random_state
        self.scaler_ = StandardScaler()
        self.model_ = None

    def fit(self, X):
        Xs = self.scaler_.fit_transform(X)
        self.model_ = MLPRegressor(
            hidden_layer_sizes=self.hidden,
            max_iter=self.max_iter,
            random_state=self.random_state,
            early_stopping=True,
            validation_fraction=0.1,
        )
        self.model_.fit(Xs, Xs)
        return self

    def score_samples(self, X):
        Xs = self.scaler_.transform(X)
        recon = self.model_.predict(Xs)
        return -np.mean((Xs - recon) ** 2, axis=1)


def main():
    df_raw = pd.read_csv(ROOT / "corporate_tax_risk_dataset.csv")
    df = engineer_taxguard_features(df_raw)
    df["High_Risk"] = (df["Tax_Risk_Label"] == "High").astype(int)
    df["Non_Compliant"] = df["Audit_Outcome"].isin(["Adverse", "Qualified"]).astype(int)

    summary = {
        "dataset": {
            "records": int(len(df)),
            "features_raw": int(df_raw.shape[1]),
            "features_engineered": int(len(get_model_feature_columns(df))),
            "missing_values": int(df_raw.isnull().sum().sum()),
            "duplicate_ids": int(df_raw["Company_ID"].duplicated().sum()),
        },
        "targets": {
            "risk_label_distribution": df["Tax_Risk_Label"].value_counts().to_dict(),
            "audit_outcome_distribution": df["Audit_Outcome"].value_counts().to_dict(),
            "high_risk_rate": float(df["High_Risk"].mean()),
            "non_compliant_rate": float(df["Non_Compliant"].mean()),
        },
    }

    # --- Figure 1: Target distributions ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    sns.countplot(data=df, x="Tax_Risk_Label", order=risk_order, ax=axes[0], hue="Tax_Risk_Label", legend=False)
    axes[0].set_title("Tax Risk Label Distribution")
    sns.countplot(data=df, x="Audit_Outcome", order=outcome_order, ax=axes[1], hue="Audit_Outcome", legend=False)
    axes[1].set_title("Audit Outcome Distribution")
    axes[2].hist(df["Audit_Likelihood"], bins=40, edgecolor="white", alpha=0.85, color="#3498db")
    axes[2].axvline(df["Audit_Likelihood"].median(), color="crimson", ls="--", label="Median")
    axes[2].set_title("ZIMRA Audit Likelihood Score")
    axes[2].legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "01_target_distributions.png", bbox_inches="tight", dpi=150)
    plt.close()

    # --- Univariate AUC ---
    exclude = {"Company_ID", "Tax_Risk_Label", "Audit_Outcome", "High_Risk", "Non_Compliant"}
    numeric_features = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]
    auc_high, auc_nc = {}, {}
    for col in numeric_features:
        auc_high[col] = float(roc_auc_score(df["High_Risk"], df[col]))
        auc_nc[col] = float(roc_auc_score(df["Non_Compliant"], df[col]))

    auc_df = pd.DataFrame({"AUC_High_Risk": auc_high, "AUC_Non_Compliant": auc_nc})
    auc_df["distance"] = (auc_df["AUC_High_Risk"] - 0.5).abs()
    auc_df = auc_df.sort_values("distance", ascending=False)
    summary["top_univariate_auc"] = auc_df.head(10).round(4).to_dict()

    top_n = 12
    top = auc_df.head(top_n).sort_values("AUC_High_Risk")
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ["#27ae60" if v > 0.5 else "#c0392b" for v in top["AUC_High_Risk"]]
    ax.barh(top.index.str.replace("_", " "), top["AUC_High_Risk"], color=colors, alpha=0.85)
    ax.axvline(0.5, color="black", ls="--", lw=1)
    ax.axvline(0.8, color="#e67e22", ls=":", lw=1.5, label="Strong signal (0.80)")
    ax.set_xlabel("AUC-ROC (High Risk vs Others)")
    ax.set_title("Top Features by Univariate Discriminative Power")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "05_univariate_auc_ranking.png", bbox_inches="tight", dpi=150)
    plt.close()

    # --- ROC univariate ---
    top_features = auc_df.head(5).index.tolist() + ["Audit_Likelihood"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for col in top_features:
        fpr, tpr, _ = roc_curve(df["High_Risk"], df[col])
        axes[0].plot(fpr, tpr, lw=2, label=f"{col.replace('_', ' ')} ({roc_auc_score(df['High_Risk'], df[col]):.3f})")
        fpr2, tpr2, _ = roc_curve(df["Non_Compliant"], df[col])
        axes[1].plot(fpr2, tpr2, lw=2, label=f"{col.replace('_', ' ')} ({roc_auc_score(df['Non_Compliant'], df[col]):.3f})")
    for ax in axes:
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(fontsize=7, loc="lower right")
    axes[0].set_title("ROC — High Risk Detection")
    axes[1].set_title("ROC — Non-Compliance Detection")
    plt.tight_layout()
    plt.savefig(FIGURES / "06_roc_univariate_top_features.png", bbox_inches="tight", dpi=150)
    plt.close()

    # --- Boxplots ---
    key_features = [
        "Tax_Rate_Deviation", "Aggressive_Tax_Planning_Score", "Offshore_Intensity",
        "Profit_Margin", "Control_Risk", "Fine_Intensity",
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, col in zip(axes.ravel(), key_features):
        sns.boxplot(data=df, x="Tax_Risk_Label", y=col, order=risk_order, ax=ax, hue="Tax_Risk_Label", legend=False)
        ax.set_title(col.replace("_", " "))
    plt.suptitle("Risk Tier Separation — Key Audit Indicators", y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES / "03_bivariate_boxplots.png", bbox_inches="tight", dpi=150)
    plt.close()

    # --- Correlation ---
    corr_features = [
        "Revenue_Million", "Profit_Margin", "Effective_Tax_Rate", "Tax_Rate_Deviation",
        "Offshore_Intensity", "Aggressive_Tax_Planning_Score", "Audit_Likelihood",
        "Control_Risk", "Planning_Deviation_Interaction", "Tax_Underpayment_Ratio",
    ]
    corr = df[corr_features].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    plt.figure(figsize=(11, 9))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r", center=0, square=True)
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig(FIGURES / "04_correlation_matrix.png", bbox_inches="tight", dpi=150)
    plt.close()

    # --- Risk vs outcome crosstab ---
    cross = pd.crosstab(df["Tax_Risk_Label"], df["Audit_Outcome"], normalize="index")
    chi2, p, dof, _ = stats.chi2_contingency(pd.crosstab(df["Tax_Risk_Label"], df["Audit_Outcome"]))
    summary["chi_square"] = {"statistic": float(chi2), "p_value": float(p), "df": int(dof)}
    summary["risk_outcome_crosstab_pct"] = (cross * 100).round(1).to_dict()

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(cross, annot=True, fmt=".1%", cmap="YlOrRd", ax=ax)
    ax.set_title("P(Audit Outcome | Risk Tier)")
    plt.tight_layout()
    plt.savefig(FIGURES / "07_risk_outcome_crosstab.png", bbox_inches="tight", dpi=150)
    plt.close()

    # --- ZIMRA overlooked indicators ---
    baseline = df["High_Risk"].mean()
    insights = []
    masks = {
        "Low profit margin + High revenue": (
            (df["Profit_Margin"] < df["Profit_Margin"].quantile(0.20))
            & (df["Revenue_Million"] > df["Revenue_Million"].quantile(0.75))
        ),
        "High offshore + Weak controls": df["High_Offshore_Low_Control"] == 1,
        "Top-decile tax deviation + Zero fines": (
            (df["Tax_Rate_Deviation"] > df["Tax_Rate_Deviation"].quantile(0.90))
            & (df["History_Fines_Million"] == 0)
        ),
        "High planning score + Clean audit history": (
            (df["Aggressive_Tax_Planning_Score"] > 0.7) & (df["Audit_Outcome"] == "Clean")
        ),
        "Concentrated ownership + ≥3 offshore subsidiaries": (
            (df["Ownership_Concentration_Percent"] > 80) & (df["Offshore_Subsidiaries"] >= 3)
        ),
    }
    for name, mask in masks.items():
        flagged = int(mask.sum())
        rate = float(df.loc[mask, "High_Risk"].mean()) if flagged else 0.0
        insights.append({
            "indicator": name,
            "flagged_firms": flagged,
            "high_risk_rate": round(rate, 3),
            "baseline_rate": round(baseline, 3),
            "risk_uplift_x": round(rate / baseline, 2) if baseline else 0,
        })
    summary["zimra_overlooked_indicators"] = insights

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(insights))
    rates = [i["high_risk_rate"] for i in insights]
    ax.bar(x - 0.2, rates, 0.4, label="Flagged cohort", color="#c0392b")
    ax.bar(x + 0.2, [baseline] * len(insights), 0.4, label="Population baseline", color="#2980b9")
    ax.set_xticks(x)
    ax.set_xticklabels([i["indicator"][:35] for i in insights], rotation=20, ha="right")
    ax.set_ylabel("High Risk Rate")
    ax.set_title("Overlooked Composite Indicators — Risk Uplift vs Baseline")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "08_zimra_overlooked_indicators.png", bbox_inches="tight", dpi=150)
    plt.close()

    # --- Model pipeline ---
    X = df[get_model_feature_columns(df)]
    y = df["High_Risk"]
    y2 = df["Non_Compliant"]
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    def oof_iso():
        s = np.zeros(len(X))
        for tr, te in skf.split(X, y):
            m = IsolationForest(n_estimators=300, contamination=0.33, random_state=RANDOM_STATE)
            m.fit(X.iloc[tr])
            s[te] = m.score_samples(X.iloc[te])
        return s

    def oof_ae():
        s = np.zeros(len(X))
        for tr, te in skf.split(X, y):
            m = AutoencoderAnomalyScorer()
            m.fit(X.iloc[tr])
            s[te] = m.score_samples(X.iloc[te])
        return s

    def oof_hgb():
        o = np.zeros(len(X))
        for tr, te in skf.split(X, y):
            c = HistGradientBoostingClassifier(
                max_depth=8, learning_rate=0.05, max_iter=600,
                min_samples_leaf=10, l2_regularization=1.0, random_state=RANDOM_STATE,
            )
            c.fit(X.iloc[tr], y.iloc[tr])
            o[te] = c.predict_proba(X.iloc[te])[:, 1]
        return o

    iso_oof, ae_oof, hgb_oof = oof_iso(), oof_ae(), oof_hgb()
    stack = np.column_stack([iso_oof, ae_oof, hgb_oof])
    meta = LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_STATE)
    ensemble_oof = np.zeros(len(X))
    for tr, te in skf.split(stack, y):
        meta.fit(stack[tr], y.iloc[tr])
        ensemble_oof[te] = meta.predict_proba(stack[te])[:, 1]

    prec, rec, thr = precision_recall_curve(y, ensemble_oof)
    f1s = 2 * prec * rec / (prec + rec + 1e-12)
    best_idx = int(np.argmax(f1s[:-1]))

    summary["model_oof"] = {
        "isolation_forest_auc": round(float(roc_auc_score(y, iso_oof)), 4),
        "autoencoder_auc": round(float(roc_auc_score(y, ae_oof)), 4),
        "hist_gradient_boosting_auc": round(float(roc_auc_score(y, hgb_oof)), 4),
        "ensemble_roc_auc": round(float(roc_auc_score(y, ensemble_oof)), 4),
        "ensemble_pr_auc": round(float(average_precision_score(y, ensemble_oof)), 4),
        "ensemble_brier": round(float(brier_score_loss(y, ensemble_oof)), 4),
        "ensemble_f1_optimal": round(float(f1s[best_idx]), 4),
        "optimal_threshold": round(float(thr[best_idx]), 4),
        "audit_likelihood_auc_baseline": round(float(roc_auc_score(y, df["Audit_Likelihood"])), 4),
    }

    # Non-compliance ensemble
    stack2 = np.column_stack([iso_oof, ae_oof, hgb_oof])
    meta2 = LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_STATE)
    ens2 = np.zeros(len(X))
    for tr, te in skf.split(stack2, y2):
        meta2.fit(stack2[tr], y2.iloc[tr])
        ens2[te] = meta2.predict_proba(stack2[te])[:, 1]
    summary["model_oof"]["non_compliance_ensemble_auc"] = round(float(roc_auc_score(y2, ens2)), 4)

    # --- Model ROC comparison ---
    all_preds = {
        "Isolation Forest": iso_oof,
        "Autoencoder": ae_oof,
        "HistGradientBoosting": hgb_oof,
        "TaxGuard Ensemble": ensemble_oof,
    }
    fig, ax = plt.subplots(figsize=(9, 7))
    for name, preds in all_preds.items():
        fpr, tpr, _ = roc_curve(y, preds)
        lw = 2.5 if name == "TaxGuard Ensemble" else 1.5
        ax.plot(fpr, tpr, lw=lw, label=f"{name} (AUC={roc_auc_score(y, preds):.3f})")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("TaxGuard Hybrid Model — ROC Curve Comparison")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(FIGURES / "11_model_roc_comparison.png", bbox_inches="tight", dpi=150)
    plt.close()

    # --- PR curve ---
    fig, ax = plt.subplots(figsize=(9, 7))
    for name, preds in all_preds.items():
        p, r, _ = precision_recall_curve(y, preds)
        ax.plot(r, p, lw=2.5 if name == "TaxGuard Ensemble" else 1.5,
                label=f"{name} (AP={average_precision_score(y, preds):.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("TaxGuard Hybrid Model — Precision-Recall Curves")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(FIGURES / "11_model_pr_comparison.png", bbox_inches="tight", dpi=150)
    plt.close()

    with open(METRICS / "experiment_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
