#!/usr/bin/env python3
"""Model next-quarter Medicaid GLP-1 prescription-demand surges."""

from __future__ import annotations

import json
import math
import platform
import sys
import warnings
import hashlib
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import statsmodels.api as sm
from scipy import stats
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, log_loss, precision_score,
    recall_score, roc_auc_score, roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.utils.validation import check_is_fitted
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.outliers_influence import variance_inflation_factor
from xgboost import XGBClassifier


SEED = 214
RUN_DATE = datetime.now(timezone.utc).isoformat()
ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "data"
OUT = ROOT / "outputs"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
MODELS = OUT / "models"
EVIDENCE = OUT / "evidence"
for folder in [TABLES, FIGURES, MODELS, EVIDENCE]:
    folder.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="notebook")
warnings.filterwarnings("ignore", message="Found unknown categories")
BLUE = "#1F4E79"
GOLD = "#C69214"
ORANGE = "#D97706"
GRAY = "#6B7280"
PALETTE = [BLUE, GOLD, ORANGE, "#708238", "#B04A6F"]

NUMERIC_FEATURES = [
    "rate_per_100k", "rate_lag_1q", "rate_lag_2q", "rate_lag_4q",
    "qoq_rate_growth", "yoy_rate_growth", "reimbursement_per_rx",
    "units_per_rx", "medicaid_share", "mco_share", "medicaid_enrollment",
    "product_age_qtrs",
]
CATEGORICAL_FEATURES = ["state", "brand", "quarter", "market_era"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "surge_20pct"
INFERENCE_NUMERIC = [
    "log_rate_per_100k", "qoq_rate_growth", "yoy_rate_growth",
    "log_reimbursement_per_rx", "log_units_per_rx", "medicaid_share",
    "mco_share", "log_medicaid_enrollment", "log_product_age_qtrs",
]
INFERENCE_CATEGORICAL = CATEGORICAL_FEATURES
INFERENCE_FEATURES = INFERENCE_NUMERIC + INFERENCE_CATEGORICAL

PRODUCT_START = {
    "BYETTA": (2005, 2), "BYDUREON": (2012, 1), "VICTOZA": (2010, 1),
    "TRULICITY": (2014, 4), "SAXENDA": (2015, 1), "ADLYXIN": (2016, 3),
    "OZEMPIC": (2018, 1), "RYBELSUS": (2019, 4), "WEGOVY": (2021, 3),
    "MOUNJARO": (2022, 2), "ZEPBOUND": (2023, 4),
}


def save_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def market_era(row: pd.Series) -> str:
    idx = int(row["year"]) * 4 + int(row["quarter"])
    if idx < 2021 * 4 + 3:
        return "Pre-Wegovy"
    if idx < 2023 * 4 + 4:
        return "Wegovy expansion"
    return "Zepbound expansion"


def product_age(row: pd.Series) -> int:
    year, quarter = PRODUCT_START[str(row["brand"])]
    return max(0, int(row["year"]) * 4 + int(row["quarter"]) - (year * 4 + quarter))


def load_and_prepare() -> tuple[pd.DataFrame, pd.DataFrame]:
    panel_path = RUN_ROOT / "processed" / "analytic_panel_2018_2025.csv.gz"
    panel = pd.read_csv(panel_path, dtype={"state": "string", "ndc11": "string", "brand": "string"})
    panel["market_era"] = panel.apply(market_era, axis=1)
    panel["product_age_qtrs"] = panel.apply(product_age, axis=1)
    panel["quarter"] = panel["quarter"].astype(str)
    candidates = panel.loc[panel["strict_history_eligible"].eq(True)].copy()
    candidates = candidates.dropna(subset=[TARGET]).copy()
    candidates[TARGET] = candidates[TARGET].astype(int)
    for source, target in [
        ("rate_per_100k", "log_rate_per_100k"),
        ("reimbursement_per_rx", "log_reimbursement_per_rx"),
        ("units_per_rx", "log_units_per_rx"),
        ("medicaid_enrollment", "log_medicaid_enrollment"),
        ("product_age_qtrs", "log_product_age_qtrs"),
    ]:
        candidates[target] = np.log1p(candidates[source].clip(lower=0))
    candidates["split"] = np.select(
        [candidates["year"].le(2023), candidates["year"].eq(2024), candidates["year"].eq(2025)],
        ["Train 2018-2023", "Validation 2024", "Test 2025"],
        default="Excluded",
    )
    if candidates["split"].eq("Excluded").any():
        raise AssertionError("Unexpected modeling years were found.")
    return panel, candidates


def split_frames(
    df: pd.DataFrame,
    target: str = TARGET,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = df[df["year"].le(2023)].copy()
    validation = df[df["year"].eq(2024)].copy()
    test = df[df["year"].eq(2025)].copy()
    for name, part in [("train", train), ("validation", validation), ("test", test)]:
        if part.empty:
            raise AssertionError(f"{name} split is empty")
    return train, validation, test


def make_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", RobustScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric, NUMERIC_FEATURES),
        ("cat", categorical, CATEGORICAL_FEATURES),
    ], verbose_feature_names_out=False)


def make_inference_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", RobustScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric, INFERENCE_NUMERIC),
        ("cat", categorical, INFERENCE_CATEGORICAL),
    ], verbose_feature_names_out=False)


def choose_threshold(y_true: pd.Series, probabilities: np.ndarray) -> tuple[float, pd.DataFrame]:
    rows = []
    for threshold in np.arange(0.10, 0.901, 0.01):
        pred = (probabilities >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        specificity = tn / (tn + fp) if tn + fp else np.nan
        rows.append({
            "threshold": round(float(threshold), 2),
            "accuracy": accuracy_score(y_true, pred),
            "recall": recall_score(y_true, pred, zero_division=0),
            "specificity": specificity,
            "f1": f1_score(y_true, pred, zero_division=0),
        })
    scores = pd.DataFrame(rows)
    eligible = scores[scores["recall"].ge(0.20)]
    ranked = eligible if not eligible.empty else scores
    best = ranked.sort_values(["accuracy", "f1", "specificity"], ascending=False).iloc[0]
    return float(best["threshold"]), scores


def classification_metrics(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    model: str,
    split: str,
) -> dict[str, float | int | str]:
    pred = (np.asarray(probabilities) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    n = len(y_true)
    correct = int((pred == np.asarray(y_true)).sum())
    ci = stats.binomtest(correct, n).proportion_ci(confidence_level=0.95, method="wilson")
    p_value = stats.binomtest(correct, n, p=0.70, alternative="greater").pvalue
    return {
        "model": model, "split": split, "threshold": threshold, "n": n,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "accuracy": accuracy_score(y_true, pred),
        "accuracy_ci_low": ci.low, "accuracy_ci_high": ci.high,
        "accuracy_vs_70_p_value": p_value,
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "specificity": tn / (tn + fp) if tn + fp else np.nan,
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "log_loss": log_loss(y_true, probabilities, labels=[0, 1]),
    }


def hosmer_lemeshow(
    y_true: pd.Series,
    probabilities: np.ndarray,
    groups: int = 10,
) -> dict[str, float | int]:
    frame = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(probabilities)})
    frame["group"] = pd.qcut(frame["p"], q=min(groups, frame["p"].nunique()), duplicates="drop")
    grouped = frame.groupby("group", observed=True).agg(observed=("y", "sum"), expected=("p", "sum"), n=("y", "size"))
    grouped["expected_no"] = grouped["n"] - grouped["expected"]
    grouped["observed_no"] = grouped["n"] - grouped["observed"]
    denom_yes = grouped["expected"].clip(lower=1e-9)
    denom_no = grouped["expected_no"].clip(lower=1e-9)
    statistic = (((grouped["observed"] - grouped["expected"]) ** 2 / denom_yes) +
                 ((grouped["observed_no"] - grouped["expected_no"]) ** 2 / denom_no)).sum()
    dof = max(len(grouped) - 2, 1)
    return {"statistic": float(statistic), "degrees_freedom": dof, "p_value": float(stats.chi2.sf(statistic, dof))}


def fit_models(candidates: pd.DataFrame) -> tuple[Any, ...]:
    train, validation, test = split_frames(candidates)
    X_train, y_train = train[FEATURES], train[TARGET]
    X_val, y_val = validation[FEATURES], validation[TARGET]
    X_test, y_test = test[FEATURES], test[TARGET]

    logistic = Pipeline([
        ("preprocess", make_preprocessor()),
        ("model", LogisticRegression(max_iter=3000, random_state=SEED, solver="lbfgs")),
    ])
    logistic.fit(X_train, y_train)
    val_prob = logistic.predict_proba(X_val)[:, 1]
    threshold, threshold_table = choose_threshold(y_val, val_prob)
    test_prob = logistic.predict_proba(X_test)[:, 1]
    logistic_metrics = classification_metrics(y_test, test_prob, threshold, "Logistic regression", "Test 2025")

    xgb_candidates = [
        {"max_depth": 2, "learning_rate": 0.05, "n_estimators": 300, "subsample": 0.85, "colsample_bytree": 0.85},
        {"max_depth": 3, "learning_rate": 0.05, "n_estimators": 300, "subsample": 0.85, "colsample_bytree": 0.85},
        {"max_depth": 3, "learning_rate": 0.08, "n_estimators": 250, "subsample": 0.90, "colsample_bytree": 0.90},
        {"max_depth": 4, "learning_rate": 0.05, "n_estimators": 250, "subsample": 0.85, "colsample_bytree": 0.85},
    ]
    xgb_runs = []
    fitted = []
    for params in xgb_candidates:
        model = Pipeline([
            ("preprocess", make_preprocessor()),
            ("model", XGBClassifier(
                **params, min_child_weight=5, reg_lambda=2.0, reg_alpha=0.1,
                objective="binary:logistic", eval_metric="logloss", n_jobs=4,
                random_state=SEED,
            )),
        ])
        model.fit(X_train, y_train)
        prob = model.predict_proba(X_val)[:, 1]
        th, _ = choose_threshold(y_val, prob)
        metrics = classification_metrics(y_val, prob, th, "XGBoost", "Validation 2024")
        xgb_runs.append({**params, **metrics})
        fitted.append((metrics["roc_auc"], metrics["accuracy"], model, th))
    fitted.sort(key=lambda x: (x[0], x[1]), reverse=True)
    _, _, xgb, xgb_threshold = fitted[0]
    xgb_prob = xgb.predict_proba(X_test)[:, 1]
    xgb_metrics = classification_metrics(y_test, xgb_prob, xgb_threshold, "XGBoost", "Test 2025")

    threshold_table.to_csv(TABLES / "logistic_validation_thresholds.csv", index=False)
    pd.DataFrame(xgb_runs).to_csv(TABLES / "xgboost_validation_tuning.csv", index=False)
    pd.DataFrame([logistic_metrics, xgb_metrics]).to_csv(TABLES / "model_comparison_test_2025.csv", index=False)
    joblib.dump(logistic, MODELS / "logistic_pipeline.joblib")
    joblib.dump(xgb, MODELS / "xgboost_pipeline.joblib")
    save_json(MODELS / "selected_thresholds.json", {"logistic": threshold, "xgboost": xgb_threshold})
    return (train, validation, test), logistic, xgb, (test_prob, xgb_prob), (logistic_metrics, xgb_metrics)


def inferential_logistic(train: pd.DataFrame) -> tuple[Any, ...]:
    # Candidate 1 mirrors the broader predictive feature set.
    full_preprocessor = make_preprocessor()
    X_full = full_preprocessor.fit_transform(train[FEATURES])
    full_names = full_preprocessor.get_feature_names_out().tolist()
    full_keep = np.nanvar(X_full, axis=0) > 1e-12
    full_names = [name for name, flag in zip(full_names, full_keep) if flag]
    full_glm = sm.GLM(
        train[TARGET].to_numpy(),
        sm.add_constant(X_full[:, full_keep], prepend=True, has_constant="add"),
        family=sm.families.Binomial(),
    ).fit(maxiter=200)

    # Candidate 2 uses log transforms and removes redundant raw lag terms.
    preprocessor = make_inference_preprocessor()
    X = preprocessor.fit_transform(train[INFERENCE_FEATURES])
    feature_names = preprocessor.get_feature_names_out().tolist()
    # Remove zero-variance columns before fitting the unpenalized GLM.
    variance = np.nanvar(X, axis=0)
    keep = np.isfinite(variance) & (variance > 1e-12)
    X = X[:, keep]
    feature_names = [name for name, flag in zip(feature_names, keep) if flag]
    X_const = sm.add_constant(X, prepend=True, has_constant="add")
    names = ["Intercept"] + feature_names
    glm = sm.GLM(train[TARGET].to_numpy(), X_const, family=sm.families.Binomial()).fit(maxiter=200)
    # Select from the computed AIC values rather than a hard-coded prior result.
    select_full = full_glm.aic <= glm.aic
    selected_glm = full_glm if select_full else glm
    selected_names = ["Intercept"] + (full_names if select_full else feature_names)
    null = sm.GLM(train[TARGET].to_numpy(), np.ones((len(train), 1)), family=sm.families.Binomial()).fit()
    lr_stat = 2 * (selected_glm.llf - null.llf)
    lr_df = int(selected_glm.df_model - null.df_model)
    lr_p = stats.chi2.sf(lr_stat, lr_df)
    conf = selected_glm.conf_int()
    coef = pd.DataFrame({
        "feature": selected_names,
        "coefficient": selected_glm.params,
        "standard_error": selected_glm.bse,
        "wald_z": selected_glm.tvalues,
        "wald_p_value": selected_glm.pvalues,
        "ci_low_logit": conf[:, 0],
        "ci_high_logit": conf[:, 1],
    })
    coef["adjusted_odds_ratio"] = np.exp(coef["coefficient"].clip(-20, 20))
    coef["or_ci_low"] = np.exp(coef["ci_low_logit"].clip(-20, 20))
    coef["or_ci_high"] = np.exp(coef["ci_high_logit"].clip(-20, 20))
    coef.to_csv(TABLES / "logistic_coefficients_odds_ratios.csv", index=False)
    save_json(TABLES / "logistic_overall_fit.json", {
        "n": len(train), "log_likelihood": selected_glm.llf, "null_log_likelihood": null.llf,
        "likelihood_ratio_chi_square": lr_stat, "likelihood_ratio_df": lr_df,
        "likelihood_ratio_p_value": lr_p, "aic": selected_glm.aic, "pseudo_r2_mcfadden": 1 - selected_glm.llf / null.llf,
    })
    pd.DataFrame([
        {"candidate": "Broad predictive specification", "aic": full_glm.aic, "parameters": int(full_glm.df_model + 1), "selected": bool(select_full)},
        {"candidate": "Transformed parsimonious inferential specification", "aic": glm.aic, "parameters": int(glm.df_model + 1), "selected": bool(not select_full)},
    ]).to_csv(TABLES / "logistic_aic_comparison.csv", index=False)
    joblib.dump({"preprocessor": full_preprocessor if select_full else preprocessor,
                 "keep_columns": full_keep if select_full else keep,
                 "feature_names": selected_names, "model": selected_glm},
                MODELS / "inferential_model_bundle.joblib")
    return selected_glm, coef


def assumption_checks(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    numeric = train[NUMERIC_FEATURES].replace([np.inf, -np.inf], np.nan).copy()
    numeric = numeric.fillna(numeric.median(numeric_only=True))
    # VIF is calculated on standardized continuous predictors only.
    z = (numeric - numeric.mean()) / numeric.std(ddof=0).replace(0, 1)
    vif = pd.DataFrame({
        "feature": z.columns,
        "vif": [variance_inflation_factor(z.to_numpy(), i) for i in range(z.shape[1])],
    })
    vif.to_csv(TABLES / "variance_inflation_factors.csv", index=False)
    bt_rows = []
    positive_features = ["rate_per_100k", "rate_lag_1q", "rate_lag_2q", "rate_lag_4q", "reimbursement_per_rx", "units_per_rx", "medicaid_enrollment", "product_age_qtrs"]
    diagnostic_preprocessor = make_preprocessor()
    design_values = diagnostic_preprocessor.fit_transform(train[FEATURES])
    design_values = design_values[:, np.var(design_values, axis=0) > 1e-12]
    base = sm.add_constant(design_values, has_constant="add")
    for feature in positive_features:
        # x*log(x) checks on strictly positive values, controlling for the
        # full predictor design. Add one only if the feature contains zeros.
        offset = 1.0 if numeric[feature].min() <= 0 else 0.0
        raw = numeric[feature] + offset
        interaction = RobustScaler().fit_transform((raw*np.log(raw)).to_numpy().reshape(-1,1))
        design = np.column_stack([base, interaction])
        try:
            result = sm.GLM(train[TARGET].to_numpy(), design, family=sm.families.Binomial()).fit(maxiter=100)
            bt_rows.append({"feature": feature, "positive_offset": offset,
                            "interaction_coefficient": result.params[-1],
                            "p_value": result.pvalues[-1],
                            "bonferroni_p_value": min(1.0, result.pvalues[-1]*len(positive_features))})
        except Exception as exc:
            bt_rows.append({"feature": feature, "interaction_coefficient": np.nan, "p_value": np.nan, "note": str(exc)[:120]})
    box_tidwell = pd.DataFrame(bt_rows)
    box_tidwell.to_csv(TABLES / "box_tidwell_checks.csv", index=False)
    return vif, box_tidwell


def build_quality_tables(panel: pd.DataFrame, candidates: pd.DataFrame) -> None:
    source_rows = []
    log_path = RUN_ROOT / "logs" / "sdud_requests.csv"
    enrollment_path = RUN_ROOT / "raw" / "enrollment" / "medicaid_chip_enrollment_all.csv.gz"
    if log_path.exists() and enrollment_path.exists():
        sdud_logs = pd.read_csv(log_path)
        for year, group in sdud_logs.groupby("year"):
            source_rows.append({"source": "CMS SDUD", "year": int(year), "dataset_id": group["dataset_id"].iloc[0], "retrieved_rows": int(group["retrieved_count"].sum()), "requests": len(group)})
        source_rows.append({"source": "CMS Medicaid and CHIP enrollment", "year": "All", "dataset_id": "6165f45b-ca93-5bb5-9d06-db29c692a360", "retrieved_rows": len(pd.read_csv(enrollment_path)), "requests": 1})
        inventory = pd.DataFrame(source_rows)
        inventory["verification_status"] = "Recomputed from retained source logs and extract"
    else:
        inventory = pd.read_csv(ROOT / "reference_outputs" / "tables" / "source_inventory.csv")
        inventory["verification_status"] = "Historical summary only; original raw files/logs absent"
    inventory.to_csv(TABLES / "source_inventory.csv", index=False)

    split = candidates.groupby("split", observed=True).agg(rows=(TARGET, "size"), positives=(TARGET, "sum"), prevalence=(TARGET, "mean")).reset_index()
    split["majority_baseline_accuracy"] = 1 - split["prevalence"]
    split.to_csv(TABLES / "split_counts_prevalence.csv", index=False)

    row_flow = pd.DataFrame([
        {"stage": "Channel rows represented in retained audit panel (not raw retrieval count)", "rows": int(panel["channel_rows"].sum())},
        {"stage": "State-NDC-quarter audit panel", "rows": len(panel)},
        {"stage": "Unsuppressed current-quarter grains", "rows": int((~panel["suppression_any"]).sum())},
        {"stage": "Target eligible with >=50 prescriptions", "rows": int(panel["target_eligible"].sum())},
        {"stage": "Strict 1-, 2-, and 4-quarter history", "rows": len(candidates)},
        {"stage": "Locked 2025 test observations", "rows": int(candidates["year"].eq(2025).sum())},
    ])
    row_flow.to_csv(TABLES / "row_flow.csv", index=False)

    missing = candidates[FEATURES].isna().mean().sort_values(ascending=False).rename("missing_rate").reset_index().rename(columns={"index": "variable"})
    missing.to_csv(TABLES / "model_feature_missingness.csv", index=False)

    variable_rows = []
    for variable in FEATURES + [TARGET]:
        variable_rows.append({
            "variable": variable,
            "role": "Dependent variable" if variable == TARGET else "Predictor",
            "type": "Categorical" if variable in CATEGORICAL_FEATURES or variable == TARGET else "Continuous",
            "description": {
                "rate_per_100k": "Current prescriptions per 100,000 Medicaid enrollees",
                "rate_lag_1q": "Prescription rate one quarter earlier",
                "rate_lag_2q": "Prescription rate two quarters earlier",
                "rate_lag_4q": "Prescription rate four quarters earlier",
                "qoq_rate_growth": "Current quarter-over-quarter rate change",
                "yoy_rate_growth": "Current year-over-year rate change",
                "reimbursement_per_rx": "Total Medicaid reimbursement per prescription",
                "units_per_rx": "Units reimbursed per prescription",
                "medicaid_share": "Medicaid share of total reimbursement",
                "mco_share": "Managed-care share of prescriptions",
                "medicaid_enrollment": "Mean monthly total_medicaid_enrollment; CHIP not added",
                "product_age_qtrs": "Quarters since product launch",
                "state": "Two-character jurisdiction code",
                "brand": "GLP-1 product brand",
                "quarter": "Calendar quarter",
                "market_era": "Pre-Wegovy, Wegovy expansion, or Zepbound expansion",
                TARGET: "1 when next-quarter enrollment-standardized prescriptions increase at least 20%; otherwise 0",
            }[variable],
        })
    pd.DataFrame(variable_rows).to_csv(TABLES / "variable_dictionary.csv", index=False)


def sensitivity_analysis(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target_pct in [10, 20, 25, 30]:
        target = f"surge_{target_pct}pct"
        # The retained target flags exist only for rows with at least 50 prescriptions.
        for floor in [50, 100, 250]:
            history_ok = panel[["rate_lag_1q", "rate_lag_2q", "rate_lag_4q"]].notna().all(axis=1)
            base_ok = (
                ~panel["suppression_any"] & panel["next_suppression_any"].eq(False)
                & panel["consecutive_next_quarter"].eq(True)
                & panel["rate_per_100k"].gt(0) & panel["next_rate_per_100k"].notna()
                & panel["medicaid_enrollment"].notna()
            )
            eligible = history_ok & base_ok & panel["prescriptions"].ge(floor) & panel[target].notna()
            data = panel.loc[eligible].copy()
            if data["year"].nunique() < 3 or data[target].nunique() < 2:
                continue
            data["market_era"] = data.apply(market_era, axis=1)
            data["product_age_qtrs"] = data.apply(product_age, axis=1)
            data["quarter"] = data["quarter"].astype(str)
            for source, transformed in [
                ("rate_per_100k", "log_rate_per_100k"),
                ("reimbursement_per_rx", "log_reimbursement_per_rx"),
                ("units_per_rx", "log_units_per_rx"),
                ("medicaid_enrollment", "log_medicaid_enrollment"),
                ("product_age_qtrs", "log_product_age_qtrs"),
            ]:
                data[transformed] = np.log1p(data[source].clip(lower=0))
            train, validation, test = split_frames(data, target)
            model = Pipeline([("preprocess", make_preprocessor()), ("model", LogisticRegression(max_iter=2500, random_state=SEED))])
            model.fit(train[FEATURES], train[target].astype(int))
            val_prob = model.predict_proba(validation[FEATURES])[:, 1]
            threshold, _ = choose_threshold(validation[target].astype(int), val_prob)
            test_prob = model.predict_proba(test[FEATURES])[:, 1]
            metric = classification_metrics(test[target].astype(int), test_prob, threshold, "Logistic regression", "Test 2025")
            rows.append({"surge_threshold_percent": target_pct, "prescription_floor": floor, "rows": len(data), "test_prevalence": test[target].mean(), **{k: metric[k] for k in ["threshold", "accuracy", "recall", "specificity", "f1", "roc_auc"]}})
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "sensitivity_analysis.csv", index=False)
    return out


def make_figures(
    panel: pd.DataFrame,
    candidates: pd.DataFrame,
    splits: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    logistic: Any,
    xgb: Any,
    probabilities: dict[str, np.ndarray],
    metrics: pd.DataFrame,
    coef: pd.DataFrame,
) -> None:
    train, validation, test = splits
    test_prob, xgb_prob = probabilities
    logistic_metrics, xgb_metrics = metrics

    # National quarterly utilization: numerator across all NDCs; denominator once per state-quarter.
    usable = panel[~panel["suppression_any"] & panel["prescriptions"].notna() & panel["medicaid_enrollment"].notna()].copy()
    num = usable.groupby(["year", "quarter"], as_index=False)["prescriptions"].sum()
    denom = usable[["state", "year", "quarter", "medicaid_enrollment"]].drop_duplicates().groupby(["year", "quarter"], as_index=False)["medicaid_enrollment"].sum()
    trend = num.merge(denom, on=["year", "quarter"], validate="one_to_one")
    trend["period"] = pd.PeriodIndex(
        trend["year"].astype(str) + "Q" + trend["quarter"].astype(str), freq="Q"
    ).to_timestamp()
    trend["rx_per_100k"] = trend["prescriptions"] / trend["medicaid_enrollment"] * 100000
    trend.to_csv(TABLES / "quarterly_national_utilization.csv", index=False)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(trend["period"], trend["rx_per_100k"], color=BLUE, linewidth=2.4, marker="o", markersize=3)
    ax.axvline(pd.Timestamp("2021-07-01"), color=GOLD, linestyle="--", label="Wegovy era")
    ax.axvline(pd.Timestamp("2023-10-01"), color=ORANGE, linestyle=":", label="Zepbound era")
    ax.set(title="Quarterly Medicaid GLP-1 Prescription Utilization", ylabel="Prescriptions per 100,000 enrollees", xlabel="Quarter")
    ax.legend(frameon=False)
    sns.despine()
    fig.tight_layout(); fig.savefig(FIGURES / "figure_1_utilization_trend.png", dpi=220); plt.close(fig)

    suppression = panel.groupby("year", as_index=False)["suppression_any"].mean()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(data=suppression, x="year", y="suppression_any", color=BLUE, ax=ax)
    ax.set(title="Suppression-Affected State-Product Quarters", xlabel="Year", ylabel="Share of audit grains")
    ax.yaxis.set_major_formatter(lambda x, pos: f"{x:.0%}")
    sns.despine(); fig.tight_layout(); fig.savefig(FIGURES / "figure_2_suppression_by_year.png", dpi=220); plt.close(fig)

    prevalence = candidates.groupby(["split", "brand"], as_index=False)[TARGET].mean()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=prevalence, y="brand", x=TARGET, hue="split", palette=PALETTE[:3], ax=ax)
    ax.set(title="Next-Quarter Surge Prevalence by Product and Split", xlabel="Share with at least 20% growth", ylabel="Product")
    ax.xaxis.set_major_formatter(lambda x, pos: f"{x:.0%}")
    ax.legend(title="", frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3)
    sns.despine(); fig.tight_layout(); fig.savefig(FIGURES / "figure_3_target_prevalence.png", dpi=220, bbox_inches="tight"); plt.close(fig)

    forest = coef[(coef["feature"] != "Intercept") & np.isfinite(coef["adjusted_odds_ratio"])].copy()
    forest["distance"] = np.abs(np.log(forest["adjusted_odds_ratio"].clip(1e-6, 1e6)))
    forest = forest.sort_values(["wald_p_value", "distance"]).head(15).sort_values("adjusted_odds_ratio")
    fig, ax = plt.subplots(figsize=(9, 6.5))
    y = np.arange(len(forest))
    ax.errorbar(forest["adjusted_odds_ratio"], y,
                xerr=[forest["adjusted_odds_ratio"] - forest["or_ci_low"], forest["or_ci_high"] - forest["adjusted_odds_ratio"]],
                fmt="o", color=BLUE, ecolor=GRAY, capsize=3)
    ax.axvline(1, color="black", linewidth=1, linestyle="--")
    ax.set_yticks(y, forest["feature"].str.replace("_", " ", regex=False))
    ax.set_xscale("log")
    ax.set(title="Adjusted Odds Ratios for Selected Logistic Predictors", xlabel="Adjusted odds ratio on log scale", ylabel="")
    sns.despine(); fig.tight_layout(); fig.savefig(FIGURES / "figure_4_odds_ratio_forest.png", dpi=220); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    threshold = logistic_metrics["threshold"]
    pred = (test_prob >= threshold).astype(int)
    matrix = confusion_matrix(test[TARGET], pred, labels=[0, 1])
    sns.heatmap(matrix, annot=True, fmt=",d", cmap="Blues", cbar=False, ax=axes[0], square=True)
    axes[0].set(title="Logistic Confusion Matrix", xlabel="Predicted class", ylabel="Actual class", xticklabels=["No surge", "Surge"], yticklabels=["No surge", "Surge"])
    for prob, label, color, style in [(test_prob, "Logistic", BLUE, "-"), (xgb_prob, "XGBoost", ORANGE, "--")]:
        fpr, tpr, _ = roc_curve(test[TARGET], prob)
        axes[1].plot(fpr, tpr, label=f"{label} AUC={roc_auc_score(test[TARGET], prob):.3f}", color=color, linestyle=style, linewidth=2)
    axes[1].plot([0, 1], [0, 1], color=GRAY, linestyle=":")
    axes[1].set(title="Locked 2025 ROC Curves", xlabel="False-positive rate", ylabel="True-positive rate")
    axes[1].legend(frameon=False)
    fig.tight_layout(); fig.savefig(FIGURES / "figure_5_confusion_roc.png", dpi=220); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for prob, label, color, marker in [(test_prob, "Logistic", BLUE, "o"), (xgb_prob, "XGBoost", ORANGE, "s")]:
        observed, predicted = calibration_curve(test[TARGET], prob, n_bins=10, strategy="quantile")
        ax.plot(predicted, observed, marker=marker, color=color, label=label)
    ax.plot([0, 1], [0, 1], color=GRAY, linestyle="--", label="Ideal")
    ax.set(title="Predicted and Observed Surge Probabilities", xlabel="Mean predicted probability", ylabel="Observed surge rate")
    ax.legend(frameon=False); sns.despine(); fig.tight_layout(); fig.savefig(FIGURES / "figure_6_calibration.png", dpi=220); plt.close(fig)

    # SHAP on a bounded, deterministic test sample.
    preprocess = xgb.named_steps["preprocess"]
    model = xgb.named_steps["model"]
    X_test_transformed = preprocess.transform(test[FEATURES])
    names = preprocess.get_feature_names_out()
    rng = np.random.default_rng(SEED)
    sample_idx = np.sort(rng.choice(len(test), size=min(600, len(test)), replace=False))
    X_sample = X_test_transformed[sample_idx]
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    plt.figure(figsize=(9, 6.5))
    shap.summary_plot(shap_values, X_sample, feature_names=names, max_display=15, show=False, color_bar=True)
    plt.title("XGBoost SHAP Feature Contributions on the 2025 Test Set")
    plt.tight_layout(); plt.savefig(FIGURES / "figure_7_shap_summary.png", dpi=220, bbox_inches="tight"); plt.close()
    importance = pd.DataFrame({"feature": names, "mean_abs_shap": np.abs(shap_values).mean(axis=0)}).sort_values("mean_abs_shap", ascending=False)
    importance.to_csv(TABLES / "xgboost_shap_importance.csv", index=False)


def main() -> None:
    np.random.seed(SEED)
    from audit_inputs import audit
    audit_result = audit()
    panel, candidates = load_and_prepare()
    build_quality_tables(panel, candidates)
    splits, logistic, xgb, probabilities, metrics = fit_models(candidates)
    glm, coef = inferential_logistic(splits[0])
    vif, box_tidwell = assumption_checks(splits[0])
    sensitivity = sensitivity_analysis(panel)
    make_figures(panel, candidates, splits, logistic, xgb, probabilities, metrics, coef)

    test = splits[2]
    logistic_prob, xgb_prob = probabilities
    hl_logistic = hosmer_lemeshow(test[TARGET], logistic_prob)
    hl_xgb = hosmer_lemeshow(test[TARGET], xgb_prob)
    save_json(TABLES / "hosmer_lemeshow_test_2025.json", {"logistic": hl_logistic, "xgboost": hl_xgb})

    baseline_pred = np.zeros(len(test), dtype=int)
    logistic_pred = (logistic_prob >= metrics[0]["threshold"]).astype(int)
    baseline_correct = baseline_pred == test[TARGET].to_numpy()
    logistic_correct = logistic_pred == test[TARGET].to_numpy()
    table = [[int((~baseline_correct & ~logistic_correct).sum()), int((~baseline_correct & logistic_correct).sum())],
             [int((baseline_correct & ~logistic_correct).sum()), int((baseline_correct & logistic_correct).sum())]]
    mc = mcnemar(table, exact=False, correction=True)
    save_json(TABLES / "mcnemar_logistic_vs_majority_baseline.json", {"table": table, "statistic": mc.statistic, "p_value": mc.pvalue})

    summary = {
        "run_date": RUN_DATE,
        "python": platform.python_version(),
        "rows": {"panel": len(panel), "modeling": len(candidates), "train": len(splits[0]), "validation": len(splits[1]), "test": len(splits[2])},
        "suppression_rate": float(panel["suppression_any"].mean()),
        "overall_positive_rate": float(candidates[TARGET].mean()),
        "test_positive_rate": float(test[TARGET].mean()),
        "test_majority_baseline_accuracy": float(1 - test[TARGET].mean()),
        "logistic": metrics[0], "xgboost": metrics[1],
        "hosmer_lemeshow": {"logistic": hl_logistic, "xgboost": hl_xgb},
        "hypothesis_supported": bool(metrics[0]["accuracy"] > 0.70 and metrics[0]["accuracy_vs_70_p_value"] < 0.05),
        "beats_majority_baseline": bool(metrics[0]["accuracy"] > (1 - test[TARGET].mean())),
    }
    save_json(OUT / "analysis_summary.json", summary)
    from robustness_checks import run as run_robustness
    run_robustness(candidates, logistic, logistic_prob)
    # Export row-level predictions so metrics and ranking claims can be audited.
    predictions = test[["state", "ndc11", "year", "quarter", TARGET]].copy()
    predictions["logistic_probability"] = logistic_prob
    predictions["xgboost_probability"] = xgb_prob
    predictions.to_csv(TABLES / "test_predictions.csv", index=False)
    ranked = predictions.sort_values(["xgboost_probability","state","ndc11","year","quarter"], ascending=[False,True,True,True,True])
    top = ranked.head(math.ceil(len(ranked)*.20))
    save_json(TABLES / "top_20_percent_review.json", {
        "reviewed_rows":len(top), "total_rows":len(ranked),
        "surges_captured":int(top[TARGET].sum()), "total_surges":int(test[TARGET].sum()),
        "capture_rate":float(top[TARGET].sum()/test[TARGET].sum()),
        "precision":float(top[TARGET].mean()),
        "lift":float(top[TARGET].mean()/test[TARGET].mean()),
        "status":"Retrospective ranking evaluation; not demonstrated inventory savings"})
    save_json(OUT / "run_manifest.json", {
        "completed_at_utc":datetime.now(timezone.utc).isoformat(),
        "panel_sha256":hashlib.sha256((RUN_ROOT/'processed/analytic_panel_2018_2025.csv.gz').read_bytes()).hexdigest(),
        "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "packages":{p:version(p) for p in ['numpy','pandas','scipy','scikit-learn','statsmodels','xgboost','shap','matplotlib','seaborn','joblib']},
        "input_audit":audit_result,
        "limitations":["Raw files are a fresh 2026-09-15 retrieval, whose panel matches the original; original raw bytes were not available.","2025 data have already been inspected; reruns do not create a new untouched test set.","Input-quarter split permits next-quarter labels to cross year boundaries; separate sensitivity is provided."]})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
