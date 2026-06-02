import { Link } from "react-router-dom";
import { modelMetrics } from "../lib/stats";

export function Home() {
  return (
    <div className="page">
      <section style={{ marginBottom: "3.5rem" }}>
        <p className="eyebrow">Prototype Demonstration — ZIMRA / 4IR Research</p>
        <h1 style={{ fontSize: "clamp(2rem, 5vw, 3rem)", maxWidth: "18ch", margin: "0.75rem 0 1rem" }}>
          Intelligent corporate tax risk scoring for Zimbabwe
        </h1>
        <p className="lead">
          TaxGuard is a hybrid machine learning framework that detects anomalies in corporate tax
          returns and assigns probabilistic risk scores — replacing manual, inconsistent audit
          selection with data-driven prioritisation at scale.
        </p>
        <div style={{ display: "flex", gap: "0.75rem", marginTop: "1.75rem", flexWrap: "wrap" }}>
          <Link to="/assess" className="btn btn-primary">
            Assess a Corporate Filing
          </Link>
          <Link to="/stats" className="btn btn-secondary">
            View System Performance
          </Link>
        </div>
      </section>

      <div className="grid-3" style={{ marginBottom: "3rem" }}>
        <div className="card-glow">
          <p className="eyebrow">Model ROC-AUC</p>
          <p className="stat-value">{modelMetrics.ensemble_roc_auc.toFixed(4)}</p>
          <p className="stat-label">Out-of-fold ensemble discrimination</p>
        </div>
        <div className="card-glow">
          <p className="eyebrow">vs ZIMRA Baseline</p>
          <p className="stat-value">
            +{(((modelMetrics.ensemble_roc_auc - modelMetrics.audit_likelihood_auc_baseline) / modelMetrics.audit_likelihood_auc_baseline) * 100).toFixed(0)}%
          </p>
          <p className="stat-label">Relative improvement over Audit_Likelihood heuristic</p>
        </div>
        <div className="card-glow">
          <p className="eyebrow">F1 @ Optimal Threshold</p>
          <p className="stat-value">{modelMetrics.ensemble_f1_optimal.toFixed(4)}</p>
          <p className="stat-label">Operational audit queue classification</p>
        </div>
      </div>

      <section style={{ marginBottom: "3rem" }}>
        <h2 className="section-title">Why this matters to Zimbabwe</h2>
        <p className="section-sub">
          Corporate tax non-compliance directly undermines fiscal sustainability. Every undetected
          high-risk filing represents revenue diverted from national development priorities.
        </p>
        <div className="grid-2">
          <div className="card">
            <h3 style={{ marginTop: 0, fontSize: "1.05rem" }}>For ZIMRA operations</h3>
            <div className="info-block">
              <p>
                Manual audit selection is resource-intensive and cannot scale with filing volume.
                TaxGuard prioritises the audit queue using calibrated probabilistic scores with
                SHAP-based explanations — reducing subjective bias and wasted audits on compliant firms.
              </p>
            </div>
            <div className="info-block">
              <p>
                The existing Audit_Likelihood heuristic achieves AUC 0.70. The TaxGuard ensemble
                reaches AUC 0.99 — auditors reach genuinely risky firms substantially earlier in
                the selection process.
              </p>
            </div>
          </div>
          <div className="card">
            <h3 style={{ marginTop: 0, fontSize: "1.05rem" }}>For Zimbabwean corporates & citizens</h3>
            <div className="info-block">
              <p>
                Compliant corporations compete on a level playing field when aggressive tax avoiders
                are identified systematically. Recovered revenue funds infrastructure, healthcare,
                education, and agricultural support under NDS2.
              </p>
            </div>
            <div className="info-block">
              <p>
                TaxGuard aligns with the National AI Strategy (2026–2030) — demonstrating practical,
                explainable AI in public sector revenue administration rather than opaque automation.
              </p>
            </div>
          </div>
        </div>
      </section>

      <section style={{ marginBottom: "3rem" }}>
        <h2 className="section-title">Hybrid ML architecture</h2>
        <p className="section-sub">
          Unsupervised anomaly detection fused with supervised gradient boosting and stacked
          meta-learning — the approach specified in the TaxGuard research proposal.
        </p>
        <pre className="pipeline">{`┌─────────────────────────────────────────────────────────────┐
│                   TaxGuard Hybrid Pipeline                   │
├─────────────────────────────────────────────────────────────┤
│  UNSUPERVISED              │  SUPERVISED                    │
│  · Isolation Forest        │  · HistGradientBoosting          │
│  · Autoencoder (MLP)       │  · XGBoost / LightGBM          │
├─────────────────────────────────────────────────────────────┤
│  STACKED ENSEMBLE  →  Calibrated Risk Score  →  SHAP Audit   │
└─────────────────────────────────────────────────────────────┘

Detection features: tax rate deviation · offshore intensity ·
planning aggressiveness · internal control weakness · penalty history`}</pre>
      </section>

      <section>
        <h2 className="section-title">Research context</h2>
        <div className="card">
          <p style={{ margin: "0 0 1rem", color: "var(--text-muted)", fontSize: "0.92rem" }}>
            <strong>TaxGuard: An AI-Based Anomaly Detection Framework for Corporate Tax Return
            Risk Scoring at ZIMRA</strong> — Authors: Edith Muyambiri & Andile Bhebhe. Theme: Area 1,
            Data, Automation and Intelligent Systems in the 4IR Era. Category: Prototype Demonstration.
          </p>
          <p style={{ margin: 0, color: "var(--text-muted)", fontSize: "0.92rem" }}>
            This demo accepts corporate filing inputs manually. In production deployment, the assessment
            form will be automatically populated via ZIMRA data pipelines integrating e-filing submissions,
            audited financial statements, enforcement history, and transfer pricing registers.
          </p>
        </div>
      </section>
    </div>
  );
}
