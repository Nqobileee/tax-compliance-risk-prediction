import type { CompanyFiling, FeatureContribution, RiskAssessment } from "../types";
import thresholds from "../data/thresholds.json";
import { detectOverlookedFlags, engineerFeatures } from "./features";

/** Feature weights derived from univariate AUC rankings in TaxGuard research. */
const WEIGHTS: { key: keyof ReturnType<typeof buildNormalized>; label: string; weight: number }[] = [
  { key: "taxRateDeviation", label: "Tax Rate Deviation", weight: 0.417 },
  { key: "planningDeviationInteraction", label: "Planning × Deviation", weight: 0.366 },
  { key: "taxUnderpaymentRatio", label: "Tax Underpayment Ratio", weight: 0.280 },
  { key: "taxGapMillion", label: "Estimated Tax Gap (USD M)", weight: 0.245 },
  { key: "auditPlanningSynergy", label: "Audit–Planning Synergy", weight: 0.211 },
  { key: "auditLikelihood", label: "Prior Audit Likelihood", weight: 0.197 },
  { key: "offshoreIntensity", label: "Offshore Intensity", weight: 0.165 },
  { key: "aggressivePlanning", label: "Aggressive Tax Planning", weight: 0.160 },
  { key: "controlRisk", label: "Internal Control Weakness", weight: 0.140 },
  { key: "fineIntensity", label: "Penalty History Intensity", weight: 0.110 },
  { key: "ownershipOffshore", label: "Ownership × Offshore", weight: 0.105 },
  { key: "highOffshoreLowControl", label: "Offshore + Weak Controls", weight: 0.095 },
];

function clamp01(v: number): number {
  return Math.min(1, Math.max(0, v));
}

function norm(value: number, max: number): number {
  return clamp01(value / max);
}

function buildNormalized(filing: CompanyFiling, f: ReturnType<typeof engineerFeatures>) {
  return {
    taxRateDeviation: norm(f.taxRateDeviation, 25),
    planningDeviationInteraction: norm(f.planningDeviationInteraction, 20),
    taxUnderpaymentRatio: norm(Math.max(0, f.taxUnderpaymentRatio), 1),
    taxGapMillion: norm(Math.max(0, f.taxGapMillion), 80),
    auditPlanningSynergy: norm(f.auditPlanningSynergy, 1),
    auditLikelihood: clamp01(filing.auditLikelihood),
    offshoreIntensity: norm(f.offshoreIntensity, 0.08),
    aggressivePlanning: clamp01(filing.aggressiveTaxPlanningScore),
    controlRisk: norm(f.controlRisk, 5),
    fineIntensity: norm(f.fineIntensity, 0.02),
    ownershipOffshore: norm(f.ownershipOffshoreInteraction, 3),
    highOffshoreLowControl: f.highOffshoreLowControl,
  };
}

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

function tierFromScore(score: number): "Low" | "Medium" | "High" {
  if (score < 0.33) return "Low";
  if (score < 0.66) return "Medium";
  return "High";
}

export function assessRisk(filing: CompanyFiling): RiskAssessment {
  const features = engineerFeatures(filing);
  const normalized = buildNormalized(filing, features);

  let logit = -1.85;
  const contributions: FeatureContribution[] = [];

  for (const { key, label, weight } of WEIGHTS) {
    const value = normalized[key];
    const contribution = weight * value;
    logit += contribution * 4.2;
    contributions.push({
      feature: key,
      label,
      value,
      contribution,
      direction: contribution > 0.02 ? "increases" : "decreases",
    });
  }

  const score = sigmoid(logit);
  const sorted = [...contributions].sort((a, b) => b.contribution - a.contribution);

  return {
    score,
    tier: tierFromScore(score),
    flaggedForAudit: score >= thresholds.optimal_threshold,
    contributions: sorted,
    features,
    overlookedFlags: detectOverlookedFlags(filing, features),
  };
}

export function formatPercent(value: number, digits = 1): string {
  return `${(value * 100).toFixed(digits)}%`;
}

export function formatScore(value: number): string {
  return (value * 100).toFixed(1);
}
