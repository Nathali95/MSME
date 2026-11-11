# iMEWS+ — Phase 2 business dashboard

This Streamlit application uses **only `phase2_surrogate_model.joblib`** from notebook `Full_methodology (7) (1).ipynb` (cell 83). It does **not** switch to Phase 1 if the artifact is missing.

## Get started

1. Copy your **trusted, notebook-generated** artifact into `iMEWS_plus/model/phase2_surrogate_model.joblib`.
2. In a terminal inside `iMEWS_plus`, run:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install -r requirements.txt
   python -m streamlit run app.py
   ```

3. Complete the Phase 2 form. All **104 original input variables** are collected: **75 continuous/numeric variables and 29 binary indicators**. Four-year data is entered in four compact, editable tables. No prediction is shown without a real saved model.

The `.joblib` file is **not included** because it was not part of the uploaded notebook; a notebook describing a saved model is not the same as the saved model. The app displays an explicit notice until you supply it. A custom `Phase2FeatureBuilder` is provided to support the class originally defined in the notebook's `__main__` environment. The required FactorAnalyzer, sklearn, XGBoost, and LightGBM packages are in requirements.txt; actual model compatibility may additionally require matching the training environment versions.

## What the model actually predicts

The Phase 2 surrogate is a **regression model for the Phase 1 estimated probability that a business remains active**, not a separate independently trained survival classifier. Its output is clipped to [0,1]; `1 - active_probability` is used for the displayed closure estimate, and the saved `phase1_closed_threshold` determines high/low risk. The original underlying training period was 2020–2023 with a status-at-2024 outcome. The website lets you display any four-year input window by *positionally mapping* the years to the model's 2020–2023 features, but **using it for future years is unvalidated** without further testing and calibration. Missing/after-closure history may also introduce label leakage in the original research.

## What-if analysis

Only **three levers** are exposed: latest-year gross annual revenue, latest-year eight reported operating-expense categories (changed together proportionally), and latest-year non-performing debt. The full Phase 2 feature preprocessing and model are rerun for every scenario. The estimate is descriptive sensitivity, **not a causal prediction** of the effect of business interventions. Debt percentages cannot change a zero starting debt value.

## Input details and caution

- Monetary units must match the original training data's currency/unit conventions. Selecting the currency label does not convert it.
- All 29 categorical predictors use the original binary 0/1 coding (the `Owner_gender` encoding must be checked against the training dictionary).
- Original predictor names, including `foraml_sector_bank_loans`, are intentionally preserved.
- Gross annual revenue is the actual Phase 2 revenue input (`Gross_annual_revenue_YEAR`); Phase 1's `Total_Revenue_YEAR` is NOT a Phase 2 input.
- The website does not persist user inputs in an external database; Streamlit session state and in-memory uploaded data are still subject to your deployment's security.
- Only load `.joblib` files you trust: Python pickle/joblib artifacts can execute code during loading.
