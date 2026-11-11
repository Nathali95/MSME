"""Phase 2 notebook-compatible preprocessing and prediction.

IMPORTANT: The notebook pickled Phase2FeatureBuilder from __main__. We expose the
same class under __main__ before loading to support that artifact. Only load a
.joblib file from a trusted source (joblib/pickle can execute arbitrary code).
"""
from __future__ import annotations

import __main__
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from factor_analyzer import FactorAnalyzer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from schema import CONTINUOUS_COLUMNS, CATEGORICAL_COLUMNS, CONTINUOUS_GROUPS, PHASE2_COLUMNS, YEARS_TRAINED

NON_FA_CONTINUOUS_COLUMNS = [
    *CONTINUOUS_GROUPS['Owner resources'],
    'Admin_Staff_2020',
    *(f'Finance_Cost_{y}' for y in YEARS_TRAINED),
    'Rent_2023',
    *(f'Cash_inflow_investing_{y}' for y in YEARS_TRAINED),
    *(f'Cash_inflow_financial_activities_{y}' for y in YEARS_TRAINED),
    'non-performing_debt_2020',
    *(f'total_investment_{y}' for y in YEARS_TRAINED),
]
NON_FA_CONTINUOUS_COLUMNS = list(dict.fromkeys(NON_FA_CONTINUOUS_COLUMNS))
REVISED_FA_GROUPS = {
    'staffing': [c for c in CONTINUOUS_GROUPS['Staffing'] if c not in NON_FA_CONTINUOUS_COLUMNS],
    'operating_costs': [c for c in CONTINUOUS_GROUPS['Operating costs'] if c not in NON_FA_CONTINUOUS_COLUMNS],
    'business_performance': [c for c in CONTINUOUS_GROUPS['Business performance'] if c not in NON_FA_CONTINUOUS_COLUMNS],
    'investment_and_debt': ['non-performing_debt_2021','non-performing_debt_2022','non-performing_debt_2023'],
}
SELECTED_N_FACTORS = {'staffing':3,'operating_costs':4,'business_performance':2,'investment_and_debt':1}
FINAL_FACTOR_NAMES = {
    'staffing':['FA_Owner_Staffing_Capacity','FA_Technical_Staffing_Capacity','FA_Administrative_Staffing_Capacity'],
    'operating_costs':['FA_General_Operating_and_Logistics_Costs','FA_Utility_Cost_Burden','FA_Employee_and_Raw_Material_Costs','FA_Rental_Cost_Burden'],
    'business_performance':['FA_Revenue_and_Profit_Performance','FA_Other_Income_Capacity'],
    'investment_and_debt':['FA_Nonperforming_Debt_Burden'],
}

class Phase2FeatureBuilder:
    """Exact interface and feature ordering of the builder saved by notebook cell 83."""
    def fit(self, df: pd.DataFrame):
        missing = sorted(set(CONTINUOUS_COLUMNS + CATEGORICAL_COLUMNS) - set(df.columns))
        if missing:
            raise ValueError(f'Missing Phase 2 predictor columns: {missing}')
        self.continuous_imputer_ = SimpleImputer(strategy='median', keep_empty_features=True)
        self.categorical_imputer_ = SimpleImputer(strategy='most_frequent', keep_empty_features=True)
        continuous = pd.DataFrame(self.continuous_imputer_.fit_transform(df[CONTINUOUS_COLUMNS]), columns=CONTINUOUS_COLUMNS, index=df.index)
        self.categorical_imputer_.fit(df[CATEGORICAL_COLUMNS])
        self.factor_models_ = {}
        for group_name, variables in REVISED_FA_GROUPS.items():
            n_factors = SELECTED_N_FACTORS[group_name]
            model = FactorAnalyzer(n_factors=n_factors, method='minres', rotation='varimax' if n_factors > 1 else None, use_smc=True)
            model.fit(continuous[variables]); self.factor_models_[group_name] = model
        self.non_fa_scaler_ = StandardScaler()
        self.non_fa_scaler_.fit(continuous[NON_FA_CONTINUOUS_COLUMNS])
        self.feature_columns_ = self.transform(df).columns.tolist()
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        continuous = pd.DataFrame(self.continuous_imputer_.transform(df[CONTINUOUS_COLUMNS]), columns=CONTINUOUS_COLUMNS, index=df.index)
        categorical = pd.DataFrame(self.categorical_imputer_.transform(df[CATEGORICAL_COLUMNS]), columns=CATEGORICAL_COLUMNS, index=df.index, dtype=float)
        factor_frames = [pd.DataFrame(self.factor_models_[group_name].transform(continuous[variables]), columns=FINAL_FACTOR_NAMES[group_name], index=df.index) for group_name, variables in REVISED_FA_GROUPS.items()]
        non_fa = pd.DataFrame(self.non_fa_scaler_.transform(continuous[NON_FA_CONTINUOUS_COLUMNS]), columns=NON_FA_CONTINUOUS_COLUMNS, index=df.index)
        result = pd.concat([*factor_frames, non_fa, categorical], axis=1)
        if result.columns.duplicated().any() or result.isna().any().any() or not np.isfinite(result.to_numpy(dtype=float)).all():
            raise ValueError('Invalid values in Phase 2 feature transformation')
        return result

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)


def load_phase2_model(path: str | Path):
    """Load only the original Phase 2 artifact; never silently use Phase 1."""
    path = Path(path)
    if not path.is_file():
        return None
    # Original notebook saved a custom class defined in its __main__ namespace.
    __main__.Phase2FeatureBuilder = Phase2FeatureBuilder
    artifact = joblib.load(path)
    required = {'feature_builder','model','phase1_closed_threshold'}
    if not isinstance(artifact, dict) or not required.issubset(artifact):
        raise ValueError('This is not the expected phase2_surrogate_model.joblib artifact.')
    for key in ('feature_builder','model'):
        if not hasattr(artifact[key], 'transform' if key == 'feature_builder' else 'predict'):
            raise ValueError(f'Model artifact has an invalid {key}.')
    return artifact


def raw_frame(values: dict) -> pd.DataFrame:
    missing = [c for c in PHASE2_COLUMNS if c not in values or pd.isna(values[c])]
    if missing:
        raise ValueError('Missing required inputs: ' + ', '.join(missing[:12]) + (' ...' if len(missing) > 12 else ''))
    frame = pd.DataFrame([{column: values[column] for column in PHASE2_COLUMNS}], columns=PHASE2_COLUMNS)
    for col in CONTINUOUS_COLUMNS:
        frame[col] = pd.to_numeric(frame[col], errors='raise')
        if not np.isfinite(frame[col].iloc[0]):
            raise ValueError(f'Invalid value: {col}')
    for col in CATEGORICAL_COLUMNS:
        if frame[col].iloc[0] not in (0,1):
            raise ValueError(f'{col} must be 0 or 1.')
    return frame


def predict_phase2(values: dict, artifact: dict) -> dict:
    data = raw_frame(values)
    transformed = artifact['feature_builder'].transform(data)
    active = float(np.clip(artifact['model'].predict(transformed)[0], 0., 1.))
    closed = 1. - active
    threshold = float(artifact['phase1_closed_threshold'])
    if not (0 <= threshold <= 1):
        raise ValueError('Invalid closure threshold in saved artifact.')
    return {'active': active, 'closed': closed, 'threshold': threshold, 'higher_risk': closed >= threshold}


EXPENSE_TYPES = ['Employee_compensation','Cost_of_raw_materials','Utility_bills','Finance_Cost','Rent','Transport_cost','Maintenance_cost','Other_expenses']

def what_if(values: dict, revenue_change: float, expense_change: float, debt_change: float) -> dict:
    """Only adjust three meaningful 2023 inputs/groups; preserve all others."""
    scenario = dict(values)
    scenario['Gross_annual_revenue_2023'] = max(0., values['Gross_annual_revenue_2023'] * (1 + revenue_change / 100))
    for prefix in EXPENSE_TYPES:
        k = f'{prefix}_2023'
        scenario[k] = max(0., values[k] * (1 + expense_change / 100))
    scenario['non-performing_debt_2023'] = max(0., values['non-performing_debt_2023'] * (1 + debt_change / 100))
    return scenario


def snapshot(values: dict):
    rev = float(values['Gross_annual_revenue_2023'])
    cost = sum(float(values[f'{k}_2023']) for k in EXPENSE_TYPES)
    debt = float(values['non-performing_debt_2023'])
    prev = float(values['Gross_annual_revenue_2022'])
    return {'revenue': rev, 'costs':cost, 'balance': rev-cost, 'margin':(rev-cost)/rev if rev else None, 'growth':(rev-prev)/prev if prev else None, 'debt':debt}
