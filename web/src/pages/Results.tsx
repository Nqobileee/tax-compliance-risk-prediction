import { Link, Navigate } from "react-router-dom";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  PolarAngleAxis,
  PolarGrid,
  PolarRadiusAxis,
  Radar,
  RadarChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { RiskGauge } from "../components/RiskGauge";
import { useFiling } from "../context/FilingContext";
import { formatPercent } from "../lib/scoring";

export function Results() {
  const { filing, assessment } = useFiling();

  if (!assessment) {
    return <Navigate to="/assess" replace />;
  }

  const topCauses = assessment.contributions.slice(0, 8).map((c) => ({
    name: c.label.length > 22 ? c.label.slice(0, 20) + "…" : c.label,
    fullName: c.label,
    contribution: Math.round(c.contribution * 1000) / 10,
    fill: c.contribution > 0.08 ? "#059669" : "#94a3b8",
  }));

  const radarData = assessment.contributions.slice(0, 6).map((c) => ({
    subject: c.label.split(" ")[0],
    value: Math.round(c.value * 100),
    fullMark: 100,
  }));

  const tierClass =
    assessment.tier === "High" ? "tag-high" : assessment.tier === "Medium" ? "tag-medium" : "tag-low";

  return (
    <div className="page">
      <p className="eyebrow">Risk Assessment Report</p>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", flexWrap: "wrap", gap: "1rem", marginBottom: "2rem" }}>
        <div>
          <h1 style={{ margin: "0.5rem 0" }}>
            {filing.companyId || "Corporate Entity"}
          </h1>
          <p className="lead" style={{ margin: 0 }}>
            TaxGuard hybrid ensemble output — probabilistic risk score with feature attribution
          </p>
        </div>
        <span className={`tag ${tierClass}`}>{assessment.tier} Risk Tier</span>
      </div>

      <div className="grid-2" style={{ marginBottom: "2rem" }}>
        <div className="card-glow" style={{ textAlign: "center" }}>
          <p className="eyebrow" style={{ marginBottom: "0.5rem" }}>TaxGuard Risk Score</p>
          <RiskGauge score={assessment.score} tier={assessment.tier} />
          <p style={{ margin: "1rem 0 0", fontSize: "0.88rem", color: "var(--text-muted)" }}>
            Audit queue flag:{" "}
            <strong style={{ color: assessment.flaggedForAudit ? "#b91c1c" : "var(--green-700)" }}>
              {assessment.flaggedForAudit ? "PRIORITY REVIEW" : "Standard monitoring"}
            </strong>
          </p>
        </div>

        <div className="card">
          <h3 style={{ marginTop: 0, fontSize: "1rem" }}>Key filing indicators</h3>
          <table style={{ width: "100%", fontSize: "0.85rem", borderCollapse: "collapse" }}>
            <tbody>
              {[
                ["Tax Rate Deviation", `${assessment.features.taxRateDeviation.toFixed(2)}%`],
                ["Profit Margin", formatPercent(assessment.features.profitMargin)],
                ["Offshore Intensity", formatPercent(assessment.features.offshoreIntensity, 2)],
                ["Tax Underpayment Ratio", formatPercent(Math.max(0, assessment.features.taxUnderpaymentRatio))],
                ["Control Risk Index", String(assessment.features.controlRisk)],
                ["Planning × Deviation", assessment.features.planningDeviationInteraction.toFixed(2)],
              ].map(([k, v]) => (
                <tr key={k} style={{ borderBottom: "1px solid var(--border)" }}>
                  <td style={{ padding: "0.5rem 0", color: "var(--text-muted)" }}>{k}</td>
                  <td style={{ padding: "0.5rem 0", textAlign: "right", fontFamily: "var(--mono)" }}>{v}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <section style={{ marginBottom: "2.5rem" }}>
        <h2 className="section-title">Leading risk drivers</h2>
        <p className="section-sub">
          SHAP-style feature attribution — the financial indicators contributing most to this filing's
          elevated risk score. ZIMRA auditors use these explanations to justify audit selection.
        </p>
        <div className="grid-2">
          <div className="card">
            <div className="chart-wrap">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={topCauses} layout="vertical" margin={{ left: 10, right: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" horizontal={false} />
                  <XAxis type="number" domain={[0, "auto"]} tick={{ fontSize: 11 }} />
                  <YAxis type="category" dataKey="name" width={110} tick={{ fontSize: 10 }} />
                  <Tooltip
                    formatter={(v: number) => [`${v}`, "Contribution index"]}
                    labelFormatter={(_, payload) =>
                      payload?.[0]?.payload?.fullName ?? ""
                    }
                  />
                  <Bar dataKey="contribution" radius={[0, 4, 4, 0]}>
                    {topCauses.map((entry, i) => (
                      <Cell key={i} fill={entry.fill} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="card">
            <div className="chart-wrap">
              <ResponsiveContainer width="100%" height="100%">
                <RadarChart data={radarData}>
                  <PolarGrid stroke="#e2e8f0" />
                  <PolarAngleAxis dataKey="subject" tick={{ fontSize: 11 }} />
                  <PolarRadiusAxis angle={30} domain={[0, 100]} tick={{ fontSize: 10 }} />
                  <Radar
                    name="Risk signal"
                    dataKey="value"
                    stroke="#059669"
                    fill="#10b981"
                    fillOpacity={0.35}
                  />
                  <Tooltip />
                </RadarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      </section>

      {assessment.overlookedFlags.length > 0 && (
        <section style={{ marginBottom: "2.5rem" }}>
          <h2 className="section-title">ZIMRA overlooked indicators detected</h2>
          <p className="section-sub">
            Composite patterns that manual audit selection typically misses — identified in TaxGuard
            research as high-yield audit targets.
          </p>
          <ul className="alert-list">
            {assessment.overlookedFlags.map((flag) => (
              <li key={flag}>{flag}</li>
            ))}
          </ul>
        </section>
      )}

      <section style={{ marginBottom: "2rem" }}>
        <h2 className="section-title">Implications for this entity</h2>
        <div className="card">
          <p style={{ margin: "0 0 1rem", color: "var(--text-muted)", fontSize: "0.92rem" }}>
            {assessment.tier === "High" ? (
              <>
                This filing exceeds the TaxGuard optimal audit threshold (0.781). Primary concern:{" "}
                <strong>{assessment.contributions[0]?.label}</strong> contributes the largest share
                of elevated risk. ZIMRA should prioritise this entity for desk review or field audit,
                with particular attention to statutory vs effective tax rate reconciliation and
                transfer pricing documentation.
              </>
            ) : assessment.tier === "Medium" ? (
              <>
                Moderate risk profile — monitor in subsequent filing periods. Deviation trends and
                offshore transaction growth should be tracked. Consider for secondary audit queue if
                resources permit.
              </>
            ) : (
              <>
                Low immediate audit priority based on current indicators. Continue standard
                compliance monitoring. Score may shift if planning aggressiveness or offshore
                exposure increases in future filings.
              </>
            )}
          </p>
          <p style={{ margin: 0, fontSize: "0.85rem", color: "var(--text-subtle)" }}>
            Score computed client-side using TaxGuard feature weights calibrated to research AUC
            rankings. Production deployment uses full hybrid ensemble (Isolation Forest + Autoencoder
            + Gradient Boosting stack).
          </p>
        </div>
      </section>

      <div style={{ display: "flex", gap: "0.75rem", flexWrap: "wrap" }}>
        <Link to="/assess" className="btn btn-secondary">
          Assess Another Filing
        </Link>
        <Link to="/stats" className="btn btn-ghost">
          View Population Statistics
        </Link>
      </div>
    </div>
  );
}
