import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  auditOutcomes,
  featureAuc,
  modelComparison,
  modelMetrics,
  overlookedIndicators,
  riskDistribution,
  stats,
} from "../lib/stats";

export function Stats() {
  const totalRecords = stats.dataset.records;

  return (
    <div className="page">
      <p className="eyebrow">Population Analytics & Model Validation</p>
      <h1 style={{ margin: "0.5rem 0 0.75rem" }}>System performance statistics</h1>
      <p className="lead" style={{ marginBottom: "2.5rem" }}>
        Aggregate metrics from the TaxGuard research experiment — 1,900 corporate tax filings,
        5-fold stratified cross-validation, out-of-fold predictions. These statistics underpin
        the case for ZIMRA deployment and national revenue mobilisation under NDS2.
      </p>

      <div className="grid-3" style={{ marginBottom: "2.5rem" }}>
        <div className="card-glow">
          <p className="stat-label">Corporates analysed</p>
          <p className="stat-value">{totalRecords.toLocaleString()}</p>
        </div>
        <div className="card-glow">
          <p className="stat-label">Ensemble ROC-AUC</p>
          <p className="stat-value">{modelMetrics.ensemble_roc_auc.toFixed(4)}</p>
        </div>
        <div className="card-glow">
          <p className="stat-label">Non-compliance rate</p>
          <p className="stat-value">{(stats.targets.non_compliant_rate * 100).toFixed(1)}%</p>
        </div>
        <div className="card-glow">
          <p className="stat-label">PR-AUC</p>
          <p className="stat-value">{modelMetrics.ensemble_pr_auc.toFixed(4)}</p>
        </div>
        <div className="card-glow">
          <p className="stat-label">Brier Score</p>
          <p className="stat-value">{modelMetrics.ensemble_brier.toFixed(4)}</p>
        </div>
        <div className="card-glow">
          <p className="stat-label">Engineered features</p>
          <p className="stat-value">{stats.dataset.features_engineered}</p>
        </div>
      </div>

      <div className="grid-2" style={{ marginBottom: "2.5rem" }}>
        <div className="card">
          <h3 className="section-title" style={{ fontSize: "1.05rem" }}>Risk tier distribution</h3>
          <p className="section-sub">Balanced three-tier classification across research dataset</p>
          <div className="chart-wrap-sm">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={riskDistribution}
                  dataKey="count"
                  nameKey="tier"
                  cx="50%"
                  cy="50%"
                  innerRadius={55}
                  outerRadius={90}
                  paddingAngle={3}
                >
                  {riskDistribution.map((entry, i) => (
                    <Cell key={i} fill={entry.fill} stroke="white" strokeWidth={2} />
                  ))}
                </Pie>
                <Tooltip formatter={(v: number) => [v, "Filings"]} />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="card">
          <h3 className="section-title" style={{ fontSize: "1.05rem" }}>Audit outcome distribution</h3>
          <p className="section-sub">Post-audit ground truth for continuous learning loop</p>
          <div className="chart-wrap-sm">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={auditOutcomes}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis dataKey="outcome" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="count" fill="#10b981" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      <section style={{ marginBottom: "2.5rem" }}>
        <h2 className="section-title">Model comparison — ROC-AUC</h2>
        <p className="section-sub">
          TaxGuard ensemble vs component models and ZIMRA's existing Audit_Likelihood baseline
        </p>
        <div className="card">
          <div className="chart-wrap">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={modelComparison} layout="vertical" margin={{ left: 20, right: 30 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" horizontal={false} />
                <XAxis type="number" domain={[0, 1]} tick={{ fontSize: 11 }} />
                <YAxis type="category" dataKey="model" width={130} tick={{ fontSize: 11 }} />
                <Tooltip formatter={(v: number) => [v.toFixed(4), "ROC-AUC"]} />
                <Bar dataKey="auc" radius={[0, 4, 4, 0]}>
                  {modelComparison.map((entry, i) => (
                    <Cell
                      key={i}
                      fill={entry.model === "TaxGuard Ensemble" ? "#059669" : "#cbd5e1"}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </section>

      <section style={{ marginBottom: "2.5rem" }}>
        <h2 className="section-title">Top feature discriminative power</h2>
        <p className="section-sub">Univariate AUC-ROC — features ZIMRA should prioritise in audit checklists</p>
        <div className="card">
          <div className="chart-wrap">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={featureAuc} margin={{ bottom: 60 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis
                  dataKey="feature"
                  tick={{ fontSize: 9 }}
                  interval={0}
                  height={80}
                  angle={-35}
                  textAnchor="end"
                />
                <YAxis domain={[0, 1]} tick={{ fontSize: 11 }} />
                <Tooltip formatter={(v: number) => [v.toFixed(4), "AUC-ROC"]} />
                <Line
                  type="monotone"
                  dataKey="auc"
                  stroke="#059669"
                  strokeWidth={2.5}
                  dot={{ fill: "#10b981", r: 4 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </section>

      <section style={{ marginBottom: "2.5rem" }}>
        <h2 className="section-title">Overlooked indicators — population analysis</h2>
        <p className="section-sub">
          Composite risk patterns manual selection misses. Risk uplift vs 33.4% baseline high-risk rate.
        </p>
        <div className="card" style={{ overflowX: "auto" }}>
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.88rem" }}>
            <thead>
              <tr style={{ borderBottom: "2px solid var(--border)", textAlign: "left" }}>
                <th style={{ padding: "0.75rem 0.5rem" }}>Indicator</th>
                <th style={{ padding: "0.75rem 0.5rem" }}>Flagged</th>
                <th style={{ padding: "0.75rem 0.5rem" }}>High-Risk Rate</th>
                <th style={{ padding: "0.75rem 0.5rem" }}>Uplift</th>
              </tr>
            </thead>
            <tbody>
              {overlookedIndicators.map((row) => (
                <tr key={row.indicator} style={{ borderBottom: "1px solid var(--border)" }}>
                  <td style={{ padding: "0.65rem 0.5rem", color: "var(--text-muted)" }}>{row.indicator}</td>
                  <td style={{ padding: "0.65rem 0.5rem", fontFamily: "var(--mono)" }}>{row.flagged_firms}</td>
                  <td style={{ padding: "0.65rem 0.5rem", fontFamily: "var(--mono)" }}>
                    {(row.high_risk_rate * 100).toFixed(1)}%
                  </td>
                  <td style={{ padding: "0.65rem 0.5rem", fontFamily: "var(--mono)", color: "var(--green-700)" }}>
                    {row.risk_uplift_x}x
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section>
        <h2 className="section-title">National fiscal impact</h2>
        <div className="grid-2">
          <div className="card">
            <h3 style={{ marginTop: 0, fontSize: "1rem" }}>Revenue mobilisation</h3>
            <p style={{ color: "var(--text-muted)", fontSize: "0.92rem", margin: 0 }}>
              With 30.6% of filings showing Qualified or Adverse audit outcomes in the research
              dataset, even marginal improvements in audit targeting translate to substantial
              recovered revenue. TaxGuard's 43% relative AUC improvement over manual heuristics
              means auditors reach non-compliant entities earlier — before revenue is permanently
              lost to aggressive planning structures.
            </p>
          </div>
          <div className="card">
            <h3 style={{ marginTop: 0, fontSize: "1rem" }}>Equity for compliant corporates</h3>
            <p style={{ color: "var(--text-muted)", fontSize: "0.92rem", margin: 0 }}>
              Zimbabwean corporates that meet their statutory obligations should not bear the
              indirect cost of under-audited competitors. TaxGuard reduces disproportionate scrutiny
              of compliant firms (F1 0.96) while elevating entities with top-decile tax deviation
              and zero fine history — a cohort with 91.4% high-risk rate that manual processes
              routinely overlook.
            </p>
          </div>
        </div>
      </section>
    </div>
  );
}
