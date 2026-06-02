export interface CompanyFiling {
  companyId: string;
  revenueMillion: number;
  profitBeforeTaxMillion: number;
  statutoryTaxRate: number;
  effectiveTaxRate: number;
  offshoreTransactionsMillion: number;
  ownershipConcentrationPercent: number;
  internalControlScore: number;
  auditLikelihood: number;
  historyFinesMillion: number;
  offshoreSubsidiaries: number;
  aggressiveTaxPlanningScore: number;
}

export interface EngineeredFeatures {
  profitMargin: number;
  offshoreIntensity: number;
  offshorePerSubsidiary: number;
  etrToStrRatio: number;
  taxGapMillion: number;
  fineIntensity: number;
  controlRisk: number;
  ownershipOffshoreInteraction: number;
  planningDeviationInteraction: number;
  auditPlanningSynergy: number;
  taxUnderpaymentRatio: number;
  taxRateDeviation: number;
  lowProfitHighRevenue: number;
  highOffshoreLowControl: number;
}

export interface FeatureContribution {
  feature: string;
  label: string;
  value: number;
  contribution: number;
  direction: "increases" | "decreases";
}

export interface RiskAssessment {
  score: number;
  tier: "Low" | "Medium" | "High";
  flaggedForAudit: boolean;
  contributions: FeatureContribution[];
  features: EngineeredFeatures;
  overlookedFlags: string[];
}

export const DEFAULT_FILING: CompanyFiling = {
  companyId: "",
  revenueMillion: 1000,
  profitBeforeTaxMillion: 180,
  statutoryTaxRate: 25,
  effectiveTaxRate: 12,
  offshoreTransactionsMillion: 15,
  ownershipConcentrationPercent: 65,
  internalControlScore: 3,
  auditLikelihood: 0.5,
  historyFinesMillion: 0,
  offshoreSubsidiaries: 2,
  aggressiveTaxPlanningScore: 0.45,
};
