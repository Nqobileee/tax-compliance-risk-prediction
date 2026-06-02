"""Shared feature engineering for TaxGuard corporate tax risk models."""

from __future__ import annotations

import pandas as pd


def engineer_taxguard_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create domain-informed features aligned with ZIMRA audit risk indicators."""
    d = df.copy()

    revenue = d["Revenue_Million"].clip(lower=1e-6)
    profit = d["Profit_Before_Tax_Million"]
    statutory = d["Statutory_Tax_Rate"].clip(lower=1)
    effective = d["Effective_Tax_Rate"]

    d["Profit_Margin"] = profit / revenue
    d["Offshore_Intensity"] = d["Offshore_Transactions_Million"] / revenue
    d["Offshore_per_Subsidiary"] = d["Offshore_Transactions_Million"] / (
        d["Offshore_Subsidiaries"] + 1
    )
    d["ETR_to_STR_Ratio"] = effective / statutory
    d["Tax_Gap_Million"] = profit * (statutory / 100) * (1 - effective / statutory)
    d["Fine_Intensity"] = d["History_Fines_Million"] / revenue
    d["Control_Risk"] = 6 - d["Internal_Control_Score"]
    d["Ownership_Offshore_Interaction"] = (
        d["Ownership_Concentration_Percent"] * d["Offshore_Intensity"]
    )
    d["Planning_Deviation_Interaction"] = (
        d["Aggressive_Tax_Planning_Score"] * d["Tax_Rate_Deviation"]
    )
    d["Audit_Planning_Synergy"] = d["Audit_Likelihood"] * d["Aggressive_Tax_Planning_Score"]
    d["Revenue_per_Control_Point"] = revenue / d["Internal_Control_Score"].clip(lower=1)
    d["Low_Profit_High_Revenue"] = (
        (d["Profit_Margin"] < d["Profit_Margin"].quantile(0.25))
        & (d["Revenue_Million"] > d["Revenue_Million"].median())
    ).astype(int)
    d["High_Offshore_Low_Control"] = (
        (d["Offshore_Intensity"] > d["Offshore_Intensity"].quantile(0.75))
        & (d["Internal_Control_Score"] <= 2)
    ).astype(int)
    d["Statutory_Effective_Gap"] = d["Tax_Rate_Deviation"].abs()
    d["Implied_Tax_Liability_Million"] = profit * (effective / 100)
    d["Expected_Tax_Liability_Million"] = profit * (statutory / 100)
    d["Tax_Underpayment_Ratio"] = (
        d["Expected_Tax_Liability_Million"] - d["Implied_Tax_Liability_Million"]
    ) / d["Expected_Tax_Liability_Million"].clip(lower=1e-6)

    return d


def get_model_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return columns suitable for model training (excludes identifiers and labels)."""
    exclude = {
        "Company_ID",
        "Tax_Risk_Label",
        "Audit_Outcome",
        "High_Risk",
        "Non_Compliant",
    }
    return [c for c in df.columns if c not in exclude]
