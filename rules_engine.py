"""Violation classification rules layer.

The rules layer sits between perception (vehicle detection, plate localisation
and OCR) and enforcement. It turns measured evidence into a deterministic,
explainable decision for every capture:

    Priority Alert   stolen vehicle or Amber Alert hit, dispatched immediately
    Auto Issue       a violation supported by evidence that passes every gate
    Manual Review    a violation that is plausible but not provable automatically
    Reject           a candidate violation on a capture marked as a false positive
    No Action        no violation supported by the evidence

Every decision carries machine readable reason codes so that an enforcement
officer, an auditor or an appeals panel can see exactly why it was made.
"""

import numpy as np
import pandas as pd

from . import config as cfg

SIGNAL_PHASE_COLUMN = "red_phase_elapsed_s"


def speed_assessment(df, tolerance=cfg.SPEED_TOLERANCE_MPH):
    """Measure speed excess against the legal limit and assign a severity tier."""
    excess = df["speed_detected_mph"].sub(df["legal_speed_limit_mph"]).clip(lower=0)
    measured = df["speed_detected_mph"].notna() & df["legal_speed_limit_mph"].notna()
    speeding = measured & (excess > tolerance)

    tiers = sorted(cfg.SPEED_TIERS, key=lambda item: item[1], reverse=True)
    conditions = [speeding & (excess > floor) for _, floor in tiers]
    labels = [label for label, _ in tiers]
    tier = np.select(conditions, labels, default="None")
    tier = pd.Series(tier, index=df.index).where(speeding, "None")
    return pd.DataFrame(
        {
            "speed_measured": measured,
            "speed_excess_mph": excess.where(measured),
            "measured_speeding": speeding,
            "speed_tier": tier,
        }
    )


def evidence_gates(df, gates=cfg.EVIDENCE_GATES):
    """Evaluate the evidence quality gates and build reason codes for failures."""
    checks = {
        "Low OCR confidence": ~(df["ocr_confidence_score"] >= gates["ocr_confidence_min"]),
        "Low detection confidence": ~(df["detection_confidence"] >= gates["detection_confidence_min"]),
        "Low image quality": ~(df["image_quality_score"] >= gates["image_quality_min"]),
        "Plate not matched to registry": ~df["plate_match_database_flag"].fillna(False).astype(bool),
    }
    failures = pd.DataFrame(checks, index=df.index)
    reasons = failures.apply(
        lambda row: "; ".join(name for name, failed in row.items() if failed), axis=1
    )
    passed = ~failures.any(axis=1)

    ocr = df["ocr_confidence_score"].fillna(0)
    det = df["detection_confidence"].fillna(0)
    img = df["image_quality_score"].fillna(0) / 100.0
    strength = (ocr + det + img) / 3.0
    return pd.DataFrame(
        {
            "evidence_gates_passed": passed,
            "evidence_gates_failed_count": failures.sum(axis=1),
            "evidence_failure_reasons": reasons,
            "evidence_strength": strength.round(4),
        }
    )


def candidate_violations(df, speed):
    """Derive the violations that the evidence supports, one column per type.

    Speeding is derived from measurement. Red light, expired registration and
    toll evasion are detector events that must pass a type specific validity
    check before they can become enforcement candidates.
    """
    recorded = df["violation_type"]

    red_light_event = recorded.eq("Red Light")
    if SIGNAL_PHASE_COLUMN in df.columns:
        red_light_valid = red_light_event & (df[SIGNAL_PHASE_COLUMN] > cfg.RED_LIGHT_AMBER_GRACE_S)
        red_light_phase_available = df[SIGNAL_PHASE_COLUMN].notna()
    else:
        red_light_valid = red_light_event
        red_light_phase_available = pd.Series(False, index=df.index)

    registration_valid = recorded.eq("Expired Registration") & df["plate_match_database_flag"].fillna(False).astype(bool)
    toll_valid = recorded.eq("Toll Evasion") & df["toll_gantry_present"].astype(bool)

    return pd.DataFrame(
        {
            "cand_speeding": speed["measured_speeding"],
            "cand_red_light": red_light_valid,
            "cand_expired_registration": registration_valid,
            "cand_toll_evasion": toll_valid,
            "red_light_phase_available": red_light_phase_available,
        }
    )


def _primary_violation(candidates, speed):
    """Choose the single highest value violation to cite for each capture."""
    speeding_label = "Speeding " + speed["speed_tier"].astype(str)
    fines = pd.DataFrame(
        {
            "Speeding": np.where(candidates["cand_speeding"], speeding_label.map(cfg.FINE_SCHEDULE_USD), 0.0),
            "Red Light": np.where(candidates["cand_red_light"], cfg.FINE_SCHEDULE_USD["Red Light"], 0.0),
            "Expired Registration": np.where(
                candidates["cand_expired_registration"], cfg.FINE_SCHEDULE_USD["Expired Registration"], 0.0
            ),
            "Toll Evasion": np.where(candidates["cand_toll_evasion"], cfg.FINE_SCHEDULE_USD["Toll Evasion"], 0.0),
        },
        index=candidates.index,
    ).fillna(0.0)
    any_candidate = fines.gt(0).any(axis=1)
    primary = fines.idxmax(axis=1).where(any_candidate, "No Violation")
    fine = fines.max(axis=1).where(any_candidate, 0.0)
    count = fines.gt(0).sum(axis=1)
    return primary, fine, count


def reconcile(recorded_type, rules_type):
    """Compare the rules layer outcome to the violation recorded in the export."""
    recorded_none = recorded_type.isin(["No Violation", "Unrecorded"])
    rules_none = rules_type.eq("No Violation")
    conditions = [
        recorded_none & rules_none,
        ~recorded_none & ~rules_none & recorded_type.eq(rules_type),
        ~recorded_none & ~rules_none & recorded_type.ne(rules_type),
        ~recorded_none & rules_none,
        recorded_none & ~rules_none,
    ]
    labels = [
        "Agree no violation",
        "Agree same violation",
        "Different violation type",
        "Recorded but unsupported",
        "Supported but not recorded",
    ]
    return pd.Series(np.select(conditions, labels, default="Unclassified"), index=recorded_type.index)


def apply_rules(df):
    """Run the full rules layer and return the input frame with decision columns."""
    out = df.copy()
    speed = speed_assessment(out)
    gates = evidence_gates(out)
    candidates = candidate_violations(out, speed)
    out = pd.concat([out, speed, gates, candidates], axis=1)

    primary, fine, count = _primary_violation(candidates, speed)
    out["rules_violation_type"] = primary
    out["rules_violation_detail"] = np.where(
        primary.eq("Speeding"), "Speeding " + speed["speed_tier"].astype(str), primary
    )
    out["rules_fine_usd"] = fine
    out["rules_violation_count"] = count

    alert = (
        out["stolen_vehicle_alert_flag"].fillna(False).astype(bool)
        | out["amber_alert_match_flag"].fillna(False).astype(bool)
    )
    false_positive = out["false_positive_flag"].fillna(False).astype(bool)
    has_candidate = primary.ne("No Violation")
    red_light_needs_phase = primary.eq("Red Light") & ~candidates["red_light_phase_available"]

    decision = np.select(
        [
            alert,
            ~has_candidate,
            has_candidate & false_positive,
            has_candidate & gates["evidence_gates_passed"] & ~red_light_needs_phase,
        ],
        [
            cfg.DECISION_PRIORITY_ALERT,
            cfg.DECISION_NO_ACTION,
            cfg.DECISION_REJECT,
            cfg.DECISION_AUTO_ISSUE,
        ],
        default=cfg.DECISION_MANUAL_REVIEW,
    )
    out["decision"] = decision

    reasons = gates["evidence_failure_reasons"].copy()
    reasons = reasons.where(~red_light_needs_phase, (reasons + "; Signal phase data unavailable").str.strip("; "))
    reasons = reasons.where(~false_positive, "Detection marked false positive")
    reasons = reasons.where(~alert, np.where(
        out["stolen_vehicle_alert_flag"].fillna(False).astype(bool), "Stolen vehicle hotlist match", "Amber Alert hotlist match"
    ))
    reasons = reasons.where(has_candidate | alert, "")
    out["decision_reasons"] = reasons

    unsupported_speeding = out["violation_type"].eq("Speeding") & speed["speed_measured"] & ~speed["measured_speeding"]
    out["recorded_speeding_unsupported_flag"] = unsupported_speeding

    out["reconciliation"] = reconcile(out["violation_type"], out["rules_violation_type"])

    # Review priority: alerts first, then value at stake and evidence strength
    severity = out["rules_fine_usd"] / max(cfg.FINE_SCHEDULE_USD.values())
    out["review_priority_score"] = (
        (severity * 60 + out["evidence_strength"] * 40).round(2)
        + np.where(alert, 1000, 0)
    )
    return out


def decision_summary(df):
    """Counts and fine value by decision."""
    summary = (
        df.groupby("decision")
        .agg(captures=("capture_id", "count"), fine_value_usd=("rules_fine_usd", "sum"))
        .reindex([
            cfg.DECISION_PRIORITY_ALERT, cfg.DECISION_AUTO_ISSUE, cfg.DECISION_MANUAL_REVIEW,
            cfg.DECISION_REJECT, cfg.DECISION_NO_ACTION,
        ])
        .fillna(0)
    )
    summary["share_pct"] = (summary["captures"] / summary["captures"].sum() * 100).round(2)
    return summary
