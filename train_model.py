"""Train an iMEWS+ classification bundle from a real labeled cohort.

Example:
 python train_model.py --csv data/final_dataset.csv --end-year 2023

For genuine prospective forecasting, provide a NEW cohort with predictor data
ending in the chosen year and labels at the beginning of the FOLLOWING year.
Changing --end-year without corresponding valid labels does not establish that.
"""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from core import FEATURES, build_features, source_columns


def fit_bundle(csv: str, output: str = "model", end_year: int = 2023):
    df = pd.read_csv(csv)
    df.columns = df.columns.str.strip()
    years = list(range(end_year - 3, end_year + 1))
    required = ["Business_Status", *sum(source_columns(years).values(), [])]
    missing = [c for c in required if c not in df]
    if missing:
        raise ValueError("Training CSV missing: " + ", ".join(missing))
    df = df.drop_duplicates().copy()
    if "Active_upto_year" in df.columns:
        closure_year = pd.to_numeric(df["Active_upto_year"], errors="coerce")
        df = df[closure_year.isna() | (closure_year >= years[0])].copy()
        print("Caution: closure-history missingness can still leak outcome information.")
    y = pd.to_numeric(df["Business_Status"], errors="coerce")
    if y.isna().any() or not set(y.unique()).issubset({0, 1}):
        raise ValueError("Business_Status must be coded 0=active, 1=closed and have no missing labels.")
    if y.nunique() != 2 or y.value_counts().min() < 5:
        raise ValueError("Need both classes with at least five examples each for stratified train/validation/test.")
    # Do not fill absent year values with fabricated zeros.
    X = build_features(df, years)
    X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=.20, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_dev, y_dev, test_size=.25, stratify=y_dev, random_state=43)
    candidates = {
        "Logistic regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(class_weight="balanced", max_iter=3000, random_state=42)),
        ]),
        "Random forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(n_estimators=350, max_depth=9, min_samples_leaf=3, class_weight="balanced", random_state=42, n_jobs=-1)),
        ]),
    }
    best = None
    thresholds = np.linspace(.10, .90, 161)
    for name, candidate in candidates.items():
        candidate.fit(X_train, y_train)
        p = candidate.predict_proba(X_val)[:, list(candidate.classes_).index(1)]
        scores = [f1_score(y_val, p >= threshold, average="macro") for threshold in thresholds]
        idx = int(np.argmax(scores))
        result = (float(scores[idx]), name, float(thresholds[idx]), candidate)
        print(f"{name}: validation macro-F1={result[0]:.3f}, threshold={result[2]:.3f}")
        if best is None or result[0] > best[0]:
            best = result
    _, name, threshold, selected = best
    p_test = selected.predict_proba(X_test)[:, list(selected.classes_).index(1)]
    metrics = {
        "test_macro_f1": round(float(f1_score(y_test, p_test >= threshold, average="macro")), 4),
        "test_balanced_accuracy": round(float(balanced_accuracy_score(y_test, p_test >= threshold)), 4),
        "test_brier_score": round(float(brier_score_loss(y_test, p_test)), 4),
        "test_roc_auc": round(float(roc_auc_score(y_test, p_test)), 4),
        "test_sample_size": int(len(y_test)),
    }
    # Preserve holdout integrity: save selected model exactly as evaluated.
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    metadata = {
        "model_name": name, "years": years, "prediction_year": end_year + 1,
        "threshold": threshold, "features": list(FEATURES),
        "target": "Business_Status", "class_1": "closed", "class_0": "active",
        "metrics": metrics,
        "limitation": "Historical status classification; not validated as a future-year forecast. Scenario differences are associations, not causal effects.",
    }
    joblib.dump(selected, out / "model.joblib")
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("Saved", out.resolve(), "\n", json.dumps(metadata, indent=2))
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output", default="model")
    parser.add_argument("--end-year", type=int, default=2023)
    args = parser.parse_args()
    fit_bundle(args.csv, args.output, args.end_year)
