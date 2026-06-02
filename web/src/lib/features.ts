import type { CompanyFiling, EngineeredFeatures } from "../types";
import thresholds from "../data/thresholds.json";

export function engineerFeatures(filing: CompanyFiling): EngineeredFeatures {
  const revenue = Math.max(filing.revenueMillion, 1e-6);
  const profit = filing.profitBeforeTaxMillion;
  const statutory = Math.max(filing.statutoryTaxRate, 1);
  const effective = filing.effectiveTaxRate;

  const profitMargin = profit / revenue;
  const offshoreIntensity = filing.offshoreTransactionsMillion / revenue;
  const offshorePerSubsidiary =
    filing.offshoreTransactionsMillion / (filing.offshoreSubsidiaries + 1);
  const etrToStrRatio = effective / statutory;
  const taxGapMillion = profit * (statutory / 100) * (1 - effective / statutory);
  const fineIntensity = filing.historyFinesMillion / revenue;
  const controlRisk = 6 - filing.internalControlScore;
  const taxRateDeviation = Math.abs(filing.statutoryTaxRate - filing.effectiveTaxRate);

  const ownershipOffshoreInteraction =
    filing.ownershipConcentrationPercent * offshoreIntensity;
  const planningDeviationInteraction =
    filing.aggressiveTaxPlanningScore * taxRateDeviation;
  const auditPlanningSynergy =
    filing.auditLikelihood * filing.aggressiveTaxPlanningScore;

  const impliedTaxLiability = profit * (effective / 100);
  const expectedTaxLiability = profit * (statutory / 100);
  const taxUnderpaymentRatio =
    (expectedTaxLiability - impliedTaxLiability) /
    Math.max(expectedTaxLiability, 1e-6);

  const lowProfitHighRevenue =
    profitMargin < thresholds.profit_margin_q25 &&
    filing.revenueMillion > thresholds.revenue_median
      ? 1
      : 0;

  const highOffshoreLowControl =
    offshoreIntensity > thresholds.offshore_intensity_q75 &&
    filing.internalControlScore <= 2
      ? 1
      : 0;

  return {
    profitMargin,
    offshoreIntensity,
    offshorePerSubsidiary,
    etrToStrRatio,
    taxGapMillion,
    fineIntensity,
    controlRisk,
    ownershipOffshoreInteraction,
    planningDeviationInteraction,
    auditPlanningSynergy,
    taxUnderpaymentRatio,
    taxRateDeviation,
    lowProfitHighRevenue,
    highOffshoreLowControl,
  };
}

export function detectOverlookedFlags(
  filing: CompanyFiling,
  features: EngineeredFeatures
): string[] {
  const flags: string[] = [];

  if (
    features.taxRateDeviation > thresholds.tax_rate_deviation_q90 &&
    filing.historyFinesMillion === 0
  ) {
    flags.push(
      "First-time aggressor: top-decile tax rate deviation with zero enforcement history"
    );
  }

  if (
    filing.aggressiveTaxPlanningScore > 0.7 &&
    filing.auditLikelihood < 0.4
  ) {
    flags.push(
      "High aggressive planning score with low prior audit attention — recency bias risk"
    );
  }

  if (features.highOffshoreLowControl === 1) {
    flags.push(
      "High offshore transaction intensity combined with weak internal controls (score ≤ 2)"
    );
  }

  if (
    filing.ownershipConcentrationPercent > 80 &&
    filing.offshoreSubsidiaries >= 3
  ) {
    flags.push(
      "Highly concentrated ownership with three or more offshore subsidiaries"
    );
  }

  if (features.lowProfitHighRevenue === 1) {
    flags.push(
      "Low profit margin relative to revenue scale — potential profit shifting indicator"
    );
  }

  return flags;
}
