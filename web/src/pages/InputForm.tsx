import { FormEvent } from "react";
import { useNavigate } from "react-router-dom";
import { useFiling } from "../context/FilingContext";
import { assessRisk } from "../lib/scoring";
import type { CompanyFiling } from "../types";

type FormField = keyof CompanyFiling;

const fields: {
  key: FormField;
  label: string;
  hint: string;
  type?: string;
  min?: number;
  max?: number;
  step?: number;
}[] = [
  { key: "companyId", label: "Company ID / TIN", hint: "ZIMRA taxpayer identifier", type: "text" },
  { key: "revenueMillion", label: "Revenue (USD Millions)", hint: "Annual turnover from CIT return", min: 0, step: 0.01 },
  { key: "profitBeforeTaxMillion", label: "Profit Before Tax (USD M)", hint: "Pre-tax profit from audited accounts", step: 0.01 },
  { key: "statutoryTaxRate", label: "Statutory Tax Rate (%)", hint: "Applicable CIT rate for entity class", min: 0, max: 100, step: 0.1 },
  { key: "effectiveTaxRate", label: "Effective Tax Rate (%)", hint: "Tax paid as percentage of profit", min: 0, max: 100, step: 0.01 },
  { key: "offshoreTransactionsMillion", label: "Offshore Transactions (USD M)", hint: "Related-party / cross-border value", min: 0, step: 0.01 },
  { key: "offshoreSubsidiaries", label: "Offshore Subsidiaries", hint: "Count of foreign group entities", min: 0, step: 1 },
  { key: "ownershipConcentrationPercent", label: "Ownership Concentration (%)", hint: "Beneficial ownership concentration index", min: 0, max: 100, step: 0.1 },
  { key: "internalControlScore", label: "Internal Control Score (1–5)", hint: "5 = strongest governance", min: 1, max: 5, step: 1 },
  { key: "historyFinesMillion", label: "History of Fines (USD M)", hint: "Prior ZIMRA penalty total", min: 0, step: 1 },
  { key: "aggressiveTaxPlanningScore", label: "Aggressive Planning Score (0–1)", hint: "Planning aggressiveness index", min: 0, max: 1, step: 0.01 },
  { key: "auditLikelihood", label: "Prior Audit Likelihood (0–1)", hint: "Existing ZIMRA AMS heuristic", min: 0, max: 1, step: 0.01 },
];

export function InputForm() {
  const { filing, setFiling, setAssessment } = useFiling();
  const navigate = useNavigate();

  const update = (key: FormField, raw: string) => {
    if (key === "companyId") {
      setFiling({ ...filing, companyId: raw });
      return;
    }
    const num = raw === "" ? 0 : Number(raw);
    setFiling({ ...filing, [key]: num });
  };

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    const result = assessRisk(filing);
    setAssessment(result);
    navigate("/results");
  };

  const loadSample = () => {
    setFiling({
      companyId: "C0847",
      revenueMillion: 1317.14,
      profitBeforeTaxMillion: 197.66,
      statutoryTaxRate: 30,
      effectiveTaxRate: 14.85,
      offshoreTransactionsMillion: 33.29,
      ownershipConcentrationPercent: 61.32,
      internalControlScore: 5,
      auditLikelihood: 0.79,
      historyFinesMillion: 10,
      offshoreSubsidiaries: 2,
      aggressiveTaxPlanningScore: 0.86,
    });
  };

  return (
    <div className="page-narrow">
      <p className="eyebrow">Corporate Filing Input</p>
      <h1 style={{ margin: "0.5rem 0 0.75rem" }}>Assess tax return risk</h1>
      <p className="lead" style={{ marginBottom: "2rem" }}>
        Enter corporate filing details below. In production, these fields will be auto-populated from
        ZIMRA data pipelines — e-filing API, financial statement registry, enforcement database, and
        transfer pricing records.
      </p>

      <div className="card-glow" style={{ marginBottom: "1.5rem", padding: "1rem 1.25rem" }}>
        <p style={{ margin: 0, fontSize: "0.85rem", color: "var(--text-muted)" }}>
          <span className="mono" style={{ color: "var(--green-600)" }}>PIPELINE STATUS:</span>{" "}
          Manual input mode — ZIMRA integration pending. Tax rate deviation will be computed as
          |Statutory Rate − Effective Rate| during scoring.
        </p>
      </div>

      <form onSubmit={handleSubmit}>
        <div
          className="card"
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))",
            gap: "1.25rem",
            marginBottom: "1.5rem",
          }}
        >
          {fields.map(({ key, label, hint, type = "number", min, max, step }) => (
            <div className="form-group" key={key}>
              <label htmlFor={key}>{label}</label>
              <input
                id={key}
                type={type}
                value={key === "companyId" ? filing.companyId : filing[key]}
                onChange={(e) => update(key, e.target.value)}
                min={min}
                max={max}
                step={step}
                required={key === "companyId"}
              />
              <small>{hint}</small>
            </div>
          ))}
        </div>

        <div style={{ display: "flex", gap: "0.75rem", flexWrap: "wrap" }}>
          <button type="submit" className="btn btn-primary">
            Run TaxGuard Assessment
          </button>
          <button type="button" className="btn btn-ghost" onClick={loadSample}>
            Load High-Risk Sample
          </button>
        </div>
      </form>
    </div>
  );
}
