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
import { useIsMobile } from "../lib/useMediaQuery";
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
  const isMobile = useIsMobile();
  const totalRecords = stats.dataset.records;
  const tickSm = isMobile ? 9 : 11;
  const pieInner = isMobile ? 40 : 55;
  const pieOuter = isMobile ? 70 : 90;

  return (
    <div className="page">
      <p className="eyebrow">Population Analytics & Model Validation</p>
      <h1 className="hero-title" style={{ maxWidth: "none", fontSize: "clamp(1.5rem, 5vw, 2.25rem)" }}>
        System performance statistics
      </h1>
      <p className="lead mb-2-5">
        Aggregate metrics from the TaxGuard research experiment — 1,900 corporate tax filings,
        5-fold stratified cross-validation, out-of-fold predictions. These statistics underpin
        the case for ZIMRA deployment and national revenue mobilisation under NDS2.
      </p>

      <div className="grid-3 mb-2-5">
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

      <div className="grid-2 mb-2-5">
        <div className="card">
          <h3 className="card-title-lg">Risk tier distribution</h3>
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
                  innerRadius={pieInner}
                  outerRadius={pieOuter}
                  paddingAngle={3}
                >
                  {riskDistribution.map((entry, i) => (
                    <Cell key={i} fill={entry.fill} stroke="white" strokeWidth={2} />
                  ))}
                </Pie>
                <Tooltip formatter={(v: number) => [v, "Filings"]} />
                <Legend wrapperStyle={{ fontSize: isMobile ? 11 : 12 }} />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="card">
          <h3 className="card-title-lg">Audit outcome distribution</h3>
          <p className="section-sub">Post-audit ground truth for continuous learning loop</p>
          <div className="chart-wrap-sm">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={auditOutcomes} margin={{ left: isMobile ? -10 : 0, right: 8 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis dataKey="outcome" tick={{ fontSize: tickSm }} />
                <YAxis tick={{ fontSize: tickSm }} width={isMobile ? 32 : 40} />
                <Tooltip />
                <Bar dataKey="count" fill="#10b981" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      <section className="mb-2-5">
        <h2 className="section-title">Model comparison — ROC-AUC</h2>
        <p className="section-sub">
          TaxGuard ensemble vs component models and ZIMRA's existing Audit_Likelihood baseline
        </p>
        <div className="card">
          <div className={isMobile ? "chart-wrap-tall" : "chart-wrap"}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={modelComparison}
                layout="vertical"
                margin={{ left: isMobile ? 4 : 20, right: isMobile ? 12 : 30, top: 8, bottom: 8 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" horizontal={false} />
                <XAxis type="number" domain={[0, 1]} tick={{ fontSize: tickSm }} />
                <YAxis
                  type="category"
                  dataKey="model"
                  width={isMobile ? 88 : 130}
                  tick={{ fontSize: isMobile ? 8 : 11 }}
                />
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

      <section className="mb-2-5">
        <h2 className="section-title">Top feature discriminative power</h2>
        <p className="section-sub">Univariate AUC-ROC — features ZIMRA should prioritise in audit checklists</p>
        <div className="card">
          <div className="chart-wrap">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={featureAuc} margin={{ bottom: isMobile ? 72 : 60, left: 4, right: 8 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis
                  dataKey="feature"
                  tick={{ fontSize: isMobile ? 7 : 9 }}
                  interval={0}
                  height={isMobile ? 64 : 80}
                  angle={-35}
                  textAnchor="end"
                />
                <YAxis domain={[0, 1]} tick={{ fontSize: tickSm }} width={isMobile ? 28 : 36} />
                <Tooltip formatter={(v: number) => [v.toFixed(4), "AUC-ROC"]} />
                <Line
                  type="monotone"
                  dataKey="auc"
                  stroke="#059669"
                  strokeWidth={2.5}
                  dot={{ fill: "#10b981", r: isMobile ? 3 : 4 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </section>

      <section className="mb-2-5">
        <h2 className="section-title">Overlooked indicators — population analysis</h2>
        <p className="section-sub">
          Composite risk patterns manual selection misses. Risk uplift vs 33.4% baseline high-risk rate.
        </p>
        <div className="card table-scroll">
          <table className="stats-table">
            <thead>
              <tr>
                <th>Indicator</th>
                <th>Flagged</th>
                <th>High-Risk Rate</th>
                <th>Uplift</th>
              </tr>
            </thead>
            <tbody>
              {overlookedIndicators.map((row) => (
                <tr key={row.indicator}>
                  <td style={{ color: "var(--text-muted)", maxWidth: isMobile ? 200 : undefined }}>
                    {row.indicator}
                  </td>
                  <td>{row.flagged_firms}</td>
                  <td>{(row.high_risk_rate * 100).toFixed(1)}%</td>
                  <td style={{ color: "var(--green-700)" }}>{row.risk_uplift_x}x</td>
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
            <h3 className="card-title">Revenue mobilisation</h3>
            <p style={{ color: "var(--text-muted)", fontSize: "0.92rem", margin: 0 }}>
              With 30.6% of filings showing Qualified or Adverse audit outcomes in the research
              dataset, even marginal improvements in audit targeting translate to substantial
              recovered revenue. TaxGuard's 43% relative AUC improvement over manual heuristics
              means auditors reach non-compliant entities earlier — before revenue is permanently
              lost to aggressive planning structures.
            </p>
          </div>
          <div className="card">
            <h3 className="card-title">Equity for compliant corporates</h3>
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
