"""Perception stage analysis: vehicle detection, plate localisation and OCR.

The capture log records the confidence outputs of the two stage perception
pipeline (a detector such as YOLOv5, YOLOv8 or Faster RCNN followed by plate
OCR) together with capture conditions and a false positive label from review.
This module answers four operational questions:

1. How does each model perform on confidence, false positives and gate pass rates?
2. Do lighting and weather degrade OCR and image quality?
3. Do the confidence scores actually separate true detections from false positives?
4. Where should the OCR and detection thresholds sit, given review workload?
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import config as cfg
from .stats_utils import chi_square_independence, kruskal_by_group, rate_table


def model_scorecard(df):
    """Per model perception scorecard."""
    gates = cfg.EVIDENCE_GATES
    work = df.assign(
        fp=df["false_positive_flag"].fillna(False).astype(bool),
        ocr_gate=df["ocr_confidence_score"] >= gates["ocr_confidence_min"],
        det_gate=df["detection_confidence"] >= gates["detection_confidence_min"],
        img_gate=df["image_quality_score"] >= gates["image_quality_min"],
        review=df["manual_review_flag"].fillna(False).astype(bool),
    )
    card = work.groupby("model_name").agg(
        captures=("capture_id", "count"),
        mean_detection_conf=("detection_confidence", "mean"),
        mean_ocr_conf=("ocr_confidence_score", "mean"),
        mean_image_quality=("image_quality_score", "mean"),
        detection_gate_pass_pct=("det_gate", "mean"),
        ocr_gate_pass_pct=("ocr_gate", "mean"),
        all_gates_pass_pct=("evidence_gates_passed", "mean"),
        manual_review_pct=("review", "mean"),
    )
    for column in ["detection_gate_pass_pct", "ocr_gate_pass_pct", "all_gates_pass_pct", "manual_review_pct"]:
        card[column] = (card[column] * 100).round(2)
    card = card.round(4)
    fp = rate_table(work, "model_name", "fp").set_index("model_name")
    card["false_positive_pct"] = fp["rate_pct"]
    card["fp_ci_low_pct"] = fp["ci_low_pct"]
    card["fp_ci_high_pct"] = fp["ci_high_pct"]
    return card.reset_index()


def condition_effects(df):
    """Effect of capture conditions on OCR confidence and image quality."""
    rows = []
    for factor in ["lighting_condition", "weather_at_capture", "vehicle_type", "road_type", "model_name"]:
        subset = df[df[factor].ne("Unknown")]
        for measure in ["ocr_confidence_score", "image_quality_score", "detection_confidence"]:
            test = kruskal_by_group(subset, factor, measure)
            rows.append({"factor": factor, "measure": measure, **test})
    out = pd.DataFrame(rows)
    out["significant_at_5pct"] = out["p_value"] < 0.05
    return out


def ocr_condition_matrix(df):
    """Mean OCR confidence by lighting and weather at capture."""
    subset = df[df["lighting_condition"].ne("Unknown") & df["weather_at_capture"].ne("Unknown")]
    return subset.pivot_table(
        index="lighting_condition", columns="weather_at_capture",
        values="ocr_confidence_score", aggfunc="mean",
    ).round(3)


def false_positive_separability(df):
    """How well does each quality signal separate false positives (ROC AUC).

    An AUC of 0.5 means the score carries no information about whether a
    detection is false. A low score should indicate risk, so each score is
    reversed before the AUC is computed.
    """
    fp = df["false_positive_flag"].fillna(False).astype(bool)
    rows = []
    for signal in ["detection_confidence", "ocr_confidence_score", "image_quality_score", "evidence_strength"]:
        mask = df[signal].notna()
        if fp[mask].nunique() < 2:
            continue
        risk = df.loc[mask, signal].rsub(df.loc[mask, signal].max())
        auc = roc_auc_score(fp[mask], risk)
        rows.append(
            {
                "signal": signal,
                "records": int(mask.sum()),
                "false_positives": int(fp[mask].sum()),
                "roc_auc": round(float(auc), 4),
                "mean_when_false_positive": round(float(df.loc[mask & fp, signal].mean()), 4),
                "mean_when_true_detection": round(float(df.loc[mask & ~fp, signal].mean()), 4),
            }
        )
    return pd.DataFrame(rows)


def threshold_sweep(df):
    """Trade off between automation and review workload across gate thresholds.

    Only enforcement candidates (captures with a rules layer violation and no
    hotlist alert) are considered because they are the records that consume
    review effort.
    """
    candidates = df[
        df["rules_violation_type"].ne("No Violation")
        & df["decision"].ne(cfg.DECISION_PRIORITY_ALERT)
    ]
    fp = candidates["false_positive_flag"].fillna(False).astype(bool)
    matched = candidates["plate_match_database_flag"].fillna(False).astype(bool)
    image_ok = candidates["image_quality_score"] >= cfg.EVIDENCE_GATES["image_quality_min"]
    phase_block = candidates["rules_violation_type"].eq("Red Light") & ~candidates["red_light_phase_available"]
    total = len(candidates)
    rows = []
    for det_t in cfg.DETECTION_THRESHOLD_GRID:
        for ocr_t in cfg.OCR_THRESHOLD_GRID:
            accepted = (
                (candidates["ocr_confidence_score"] >= ocr_t)
                & (candidates["detection_confidence"] >= det_t)
                & image_ok & matched & ~phase_block
            )
            auto = accepted & ~fp
            review = ~accepted & ~fp
            fp_in_accepted = (accepted & fp).sum()
            rows.append(
                {
                    "detection_threshold": det_t,
                    "ocr_threshold": ocr_t,
                    "auto_issue": int(auto.sum()),
                    "auto_issue_pct": round(auto.sum() / total * 100, 2) if total else np.nan,
                    "manual_review": int(review.sum()),
                    "false_positive_share_of_accepted_pct": round(
                        fp_in_accepted / accepted.sum() * 100, 2) if accepted.sum() else np.nan,
                    "review_hours_per_10k_captures": round(
                        review.sum() / len(df) * 10000 * cfg.REVIEW_MINUTES_PER_CASE / 60, 1),
                }
            )
    return pd.DataFrame(rows)


def perception_tests(df):
    """Headline hypothesis tests for the perception stage."""
    work = df.assign(fp=df["false_positive_flag"].fillna(False).astype(bool))
    return {
        "false_positive_rate_by_model": chi_square_independence(work, "model_name", "fp"),
        "false_positive_rate_by_lighting": chi_square_independence(
            work[work["lighting_condition"].ne("Unknown")], "lighting_condition", "fp"),
        "ocr_confidence_by_lighting": kruskal_by_group(
            work[work["lighting_condition"].ne("Unknown")], "lighting_condition", "ocr_confidence_score"),
        "ocr_confidence_by_weather": kruskal_by_group(
            work[work["weather_at_capture"].ne("Unknown")], "weather_at_capture", "ocr_confidence_score"),
    }
