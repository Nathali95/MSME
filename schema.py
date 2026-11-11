"""Exact Phase 2 input schema transcribed from notebook cell 40.

YEAR_SUFFIXES refer to the training window 2020-2023. For rolling years,
form inputs retain semantic order (year 1..4); they do not imply that an
existing 2024 model has been validated for future prediction years.
"""
from __future__ import annotations

YEARS_TRAINED = (2020, 2021, 2022, 2023)
CONTINUOUS_GROUPS = {
    "Owner resources": ["Owner_age", "Owner_family_size", "Revenue_percent_withdrawn", "Owned_total_asset_market_value", "Rented_total_asset_market_value"],
    "Staffing": [f"{prefix}_{year}" for prefix in ("Owners", "Admin_Staff", "Tech_Staff") for year in YEARS_TRAINED],
    "Operating costs": [f"{prefix}_{year}" for prefix in ("Employee_compensation", "Cost_of_raw_materials", "Utility_bills", "Finance_Cost", "Rent", "Transport_cost", "Maintenance_cost", "Other_expenses") for year in YEARS_TRAINED],
    "Business performance": [*[f"Gross_annual_revenue_{y}" for y in YEARS_TRAINED], "Minimum_2023_Monthly_profit", "Maximum_2023_Monthly_profit", *[f"Other_income_total_{y}" for y in YEARS_TRAINED]],
    "Investment and debt": [f"{prefix}_{year}" for prefix in ("Cash_inflow_investing", "Cash_inflow_financial_activities", "total_investment", "non-performing_debt") for year in YEARS_TRAINED],
}
CATEGORICAL_COLUMNS = [
    "Owner_gender", "Main_Income_Source_Other", "Main_Income_Source_MSME", "Main_Income_Source_Non_Agriculture", "Main_Income_Source_PVT_Sector", "Main_Income_Source_Government", "Business_Sole_Ownership", "District_Kandy", "District_Matale", "District_NuwaraEliya", "Raw_material_access_Local", "Raw_material_access_Regional", "Raw_material_access_National", "Raw_material_access_International", "Deposit_daily", "Deposit_weekly", "Deposit_monthly", "Withdraw_daily", "Withdraw_weekly", "Withdraw_monthly", "daily_recording", "annual_recording", "formal_sector_concessionary_loans", "foraml_sector_bank_loans", "formal_sector_other_loans", "formal_sector_micro_finance", "Informal_sector_informal_money", "Informal_sector_micro_finance", "Informal_sector_other_loans",
]
CONTINUOUS_COLUMNS = [item for group in CONTINUOUS_GROUPS.values() for item in group]
PHASE2_COLUMNS = CONTINUOUS_COLUMNS + CATEGORICAL_COLUMNS

DISPLAY_CATEGORIES = {
    "Owner resources": "Owner & resources",
    "Staffing": "People & employment",
    "Operating costs": "Operating expenses",
    "Business performance": "Sales & performance",
    "Investment and debt": "Investment & borrowings",
}

def display_label(column: str, actual_years=YEARS_TRAINED) -> str:
    value = column
    for original, actual in zip(YEARS_TRAINED, actual_years):
        if column.endswith(f"_{original}"):
            value = column[:-5] + f" ({actual})"
            break
    return value.replace("_", " ").replace("foraml", "formal").strip().capitalize()


def detailed_row(values: dict, actual_years=YEARS_TRAINED) -> dict:
    """Export inputs in original notebook column names; actual-year mapping in metadata."""
    return {column: values.get(column, None) for column in PHASE2_COLUMNS}
