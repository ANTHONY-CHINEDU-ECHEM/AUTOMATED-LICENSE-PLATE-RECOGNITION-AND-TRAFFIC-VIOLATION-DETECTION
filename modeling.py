"""False positive triage model feasibility study.

Question: can the capture metadata predict which detections reviewers will mark
as false positives, so that likely false positives are removed before they
reach the review queue or, worse, an automated citation?

The study is deliberately conservative. Candidate models are compared against
a prior only baseline with stratified cross validation, and the best model is
checked with a label permutation test. A model is only recommended for
deployment when it clears a minimum discrimination bar.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate, permutation_test_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config as cfg

NUMERIC_FEATURES = [
    "detection_confidence", "ocr_confidence_score", "image_quality_score",
    "processing_latency_ms", "speed_detected_mph", "legal_speed_limit_mph", "band_start_hour",
]
CATEGORICAL_FEATURES = [
    "model_name", "lighting_condition", "weather_at_capture", "vehicle_type",
    "road_type", "district_zone", "corridor_name",
]
TARGET = "false_positive_flag"
MIN_DEPLOYABLE_AUC = 0.70
PERMUTATIONS = 30


def _preprocessor(scale):
    numeric_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))
    return ColumnTransformer(
        [
            ("numeric", Pipeline(numeric_steps), NUMERIC_FEATURES),
            ("categorical", OneHotEncoder(handle_unknown="ignore", min_frequency=20, sparse_output=False), CATEGORICAL_FEATURES),
        ]
    )


def candidate_models():
    return {
        "Prior baseline": Pipeline(
            [("prep", _preprocessor(scale=False)), ("model", DummyClassifier(strategy="prior"))]
        ),
        "Logistic regression": Pipeline(
            [("prep", _preprocessor(scale=True)),
             ("model", LogisticRegression(max_iter=2000, class_weight="balanced"))]
        ),
        "Gradient boosting": Pipeline(
            [("prep", _preprocessor(scale=False)),
             ("model", HistGradientBoostingClassifier(
                 max_iter=200, learning_rate=0.05, max_depth=4,
                 class_weight="balanced", random_state=cfg.RANDOM_STATE))]
        ),
    }


def build_dataset(df):
    data = df[df[TARGET].notna()].copy()
    features = data[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    for column in CATEGORICAL_FEATURES:
        features[column] = features[column].astype(str)
    target = data[TARGET].astype(bool).astype(int)
    return features, target


def evaluate(df):
    """Cross validated comparison of candidate models plus a permutation test."""
    features, target = build_dataset(df)
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg.RANDOM_STATE)
    rows = []
    for name, model in candidate_models().items():
        scores = cross_validate(model, features, target, cv=folds,
                                scoring=["roc_auc", "average_precision"])
        rows.append(
            {
                "model": name,
                "roc_auc_mean": round(float(np.mean(scores["test_roc_auc"])), 4),
                "roc_auc_std": round(float(np.std(scores["test_roc_auc"])), 4),
                "average_precision_mean": round(float(np.mean(scores["test_average_precision"])), 4),
            }
        )
    results = pd.DataFrame(rows)

    best_name = results[results["model"].ne("Prior baseline")].sort_values(
        "roc_auc_mean", ascending=False)["model"].iloc[0]
    best_model = candidate_models()[best_name]
    score, _, p_value = permutation_test_score(
        best_model, features, target, cv=folds, scoring="roc_auc",
        n_permutations=PERMUTATIONS, random_state=cfg.RANDOM_STATE,
    )
    prevalence = float(target.mean())
    summary = {
        "records": int(len(target)),
        "false_positive_prevalence_pct": round(prevalence * 100, 2),
        "best_model": best_name,
        "best_roc_auc": round(float(score), 4),
        "permutation_p_value": round(float(p_value), 4),
        "permutations": PERMUTATIONS,
        "deployable": bool(score >= MIN_DEPLOYABLE_AUC and p_value < 0.05),
        "deployment_bar_auc": MIN_DEPLOYABLE_AUC,
    }
    return results, summary
