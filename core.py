"""Features and prediction logic adapted from the uploaded iMEWS methodology.

The model uses a four-year observation window and a binary next-cutoff status.
Class 1 means CLOSED; class 0 means ACTIVE.
"""
from __future__ import annotations

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

DEFAULT_YEARS = ('4 years before prediction', '3 years before prediction', '2 years before prediction', '1 year before prediction')
METRICS = ("revenue", "expenditure", "staff")
FEATURES = (
    "Revenue_latest", "Revenue_mean", "Revenue_std",
    "Expenditure_latest", "Expenditure_mean", "Expenditure_std",
    "Staff_latest", "Staff_mean", "Staff_std",
    "Profit_latest", "Profit_mean", "Profit_std",
    "Profit_margin_latest", "Profit_margin_mean",
    "Revenue_trend", "Expenditure_trend", "Staff_trend", "Loss_years",
)


def source_columns(years=DEFAULT_YEARS):
    return {
        "revenue": [f"Total_Revenue_{y}" for y in years],
        "expenditure": [f"Total_expenditure_{y}" for y in years],
        "staff": [f"Staff_total_{y}" for y in years],
    }


def _trend(frame: pd.DataFrame, years) -> pd.Series:
    """Same OLS slope logic as the notebook, supporting missing yearly values."""
    x = np.asarray(years, dtype=float)
    result = np.full(len(frame), np.nan)
    for i, values in enumerate(frame.to_numpy(dtype=float)):
        observed = np.isfinite(values)
        if observed.sum() >= 2:
            result[i] = float(np.polyfit(x[observed] - x[observed].mean(), values[observed], 1)[0])
    return pd.Series(result, index=frame.index)


def _latest(frame: pd.DataFrame) -> pd.Series:
    values = frame.to_numpy(dtype=float)
    out = np.full(len(frame), np.nan)
    for i, row in enumerate(values):
        valid = row[np.isfinite(row)]
        if valid.size:
            out[i] = valid[-1]
    return pd.Series(out, index=frame.index)


def build_features(df: pd.DataFrame, years=DEFAULT_YEARS) -> pd.DataFrame:
    """Reproduces notebook cell 13 features, with a configurable 4-year window."""
    if len(years) != 4:
        raise ValueError("A four-year observation window is required.")
    cols = source_columns(years)
    missing = [c for values in cols.values() for c in values if c not in df.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    revenue, expense, staff = (
        df[cols[key]].apply(pd.to_numeric, errors="coerce").set_axis(years, axis=1)
        for key in METRICS
    )
    profit = revenue - expense
    margin = profit.div(revenue.replace(0, np.nan))
    out = pd.DataFrame(index=df.index)
    for prefix, frame in (
        ("Revenue", revenue), ("Expenditure", expense),
        ("Staff", staff), ("Profit", profit),
    ):
        out[f"{prefix}_latest"] = _latest(frame)
        out[f"{prefix}_mean"] = frame.mean(axis=1, skipna=True)
        out[f"{prefix}_std"] = frame.std(axis=1, skipna=True)
    out["Profit_margin_latest"] = _latest(margin)
    out["Profit_margin_mean"] = margin.mean(axis=1, skipna=True)
    for prefix, frame in (("Revenue", revenue), ("Expenditure", expense), ("Staff", staff)):
        out[f"{prefix}_trend"] = _trend(frame, years)
    # Important: missing profit years do not count as loss-making years.
    out["Loss_years"] = profit.lt(0).where(profit.notna()).sum(axis=1)
    return out[list(FEATURES)].replace([np.inf, -np.inf], np.nan)


def row_from_entries(entries: dict, years) -> pd.DataFrame:
    row = {}
    for y in years:
        for metric, prefix in (("revenue", "Total_Revenue"), ("expenditure", "Total_expenditure"), ("staff", "Staff_total")):
            val = float(entries[y][metric])
            if not np.isfinite(val) or val < 0:
                raise ValueError(f"{metric.title()} in {y} must be a non-negative number.")
            if metric == "staff" and val != int(val):
                raise ValueError(f"Staff count in {y} must be a whole number.")
            row[f"{prefix}_{y}"] = val
    return pd.DataFrame([row])


def load_bundle(folder: str | Path = "model") -> dict | None:
    folder = Path(folder)
    model_file, meta_file = folder / "model.joblib", folder / "metadata.json"
    if not (model_file.is_file() and meta_file.is_file()):
        return None
    with meta_file.open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    if metadata.get("features") != list(FEATURES):
        raise ValueError("Model has an incompatible feature list.")
    if len(metadata.get("years", [])) != 4:
        raise ValueError("Invalid four-year model window.")
    return {"model": joblib.load(model_file), "metadata": metadata}


def predict(entries: dict, bundle: dict) -> dict:
    years = bundle["metadata"]["years"]
    features = build_features(row_from_entries(entries, years), years)
    model = bundle["model"]
    classes = list(model.classes_)
    if 0 not in classes or 1 not in classes:
        raise ValueError("Expected model classes 0 (active) and 1 (closed).")
    closed = float(model.predict_proba(features)[0, classes.index(1)])
    threshold = float(bundle["metadata"]["threshold"])
    return {
        "closed": closed,
        "active": 1.0 - closed,
        "status": "Higher closure risk" if closed >= threshold else "Lower closure risk",
        "flag": bool(closed >= threshold),
        "threshold": threshold,
        "features": features,
    }


def scenario(entries: dict, year: int, revenue_pct: float, expenditure_pct: float, staff_pct: float) -> dict:
    """Edit one latest-year observation; recalculate ALL derived model features."""
    copy = {y: dict(v) for y, v in entries.items()}
    copy[year]["revenue"] = max(0., entries[year]["revenue"] * (1 + revenue_pct / 100))
    copy[year]["expenditure"] = max(0., entries[year]["expenditure"] * (1 + expenditure_pct / 100))
    copy[year]["staff"] = max(0, int(round(entries[year]["staff"] * (1 + staff_pct / 100))))
    return copy


def financial_summary(entries: dict, years) -> dict:
    y = years[-1]
    revenue = entries[y]["revenue"]
    cost = entries[y]["expenditure"]
    profit = revenue - cost
    prior = entries[years[-2]]["revenue"]
    return {
        "revenue": revenue, "cost": cost, "profit": profit,
        "margin": profit / revenue * 100 if revenue else None,
        "growth": (revenue - prior) / prior * 100 if prior else None,
        "staff": entries[y]["staff"],
    }
