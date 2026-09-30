"""Enforcement analytics: hotspots, label integrity, citations and appeals."""

import numpy as np
import pandas as pd

from . import config as cfg
from .rules_engine import speed_assessment
from .stats_utils import chi_square_independence, rate_table


def corridor_summary(df):
    """Violation profile and enforcement value per corridor."""
    work = df.assign(
        violation=df["rules_violation_type"].ne("No Violation"),
        severe=df["speed_tier"].eq("Severe"),
        auto=df["decision"].eq(cfg.DECISION_AUTO_ISSUE),
        review=df["decision"].eq(cfg.DECISION_MANUAL_REVIEW),
    )
    out = work.groupby("corridor_name").agg(
        captures=("capture_id", "count"),
        violation_rate_pct=("violation", "mean"),
        speeding_rate_pct=("measured_speeding", "mean"),
        severe_speeding=("severe", "sum"),
        mean_excess_when_speeding_mph=("speed_excess_mph", lambda s: s[s > cfg.SPEED_TOLERANCE_MPH].mean()),
        auto_issue=("auto", "sum"),
        manual_review=("review", "sum"),
        fine_value_usd=("rules_fine_usd", "sum"),
    )
    out["violation_rate_pct"] = (out["violation_rate_pct"] * 100).round(2)
    out["speeding_rate_pct"] = (out["speeding_rate_pct"] * 100).round(2)
    out["mean_excess_when_speeding_mph"] = out["mean_excess_when_speeding_mph"].round(2)
    return out.sort_values("fine_value_usd", ascending=False).reset_index()


def hotspot_matrix(df):
    """Speeding rate (percent) by corridor and time band."""
    matrix = df.pivot_table(
        index="corridor_name", columns="time_of_day", values="measured_speeding", aggfunc="mean"
    )
    ordered = [band for band in cfg.TIME_BAND_ORDER if band in matrix.columns]
    return (matrix[ordered] * 100).round(1)


def segment_rate_tables(df):
    """Rules layer violation rates with confidence intervals per segment type."""
    work = df.assign(violation=df["rules_violation_type"].ne("No Violation"))
    tables = {}
    tests = {}
    for factor in ["district_zone", "road_type", "time_of_day", "jurisdiction", "vehicle_type", "day_of_week"]:
        subset = work[work[factor].ne("Unknown")]
        tables[factor] = rate_table(subset, factor, "violation")
        tests[factor] = chi_square_independence(subset, factor, "violation")
    return tables, tests


def monthly_trend(df):
    """Monthly capture volume, rules layer violations and recorded violations."""
    dated = df[df["observation_month"].notna()].assign(
        rules_violation=lambda d: d["rules_violation_type"].ne("No Violation"),
        recorded_violation=lambda d: ~d["violation_type"].isin(["No Violation", "Unrecorded"]),
        cited=lambda d: d["citation_issued_flag"].fillna(False).astype(bool),
    )
    trend = dated.groupby("observation_month").agg(
        captures=("capture_id", "count"),
        rules_violations=("rules_violation", "sum"),
        recorded_violations=("recorded_violation", "sum"),
        citations=("cited", "sum"),
    ).reset_index()
    trend["rules_violation_rate_pct"] = (trend["rules_violations"] / trend["captures"] * 100).round(2)
    return trend


def limit_source_sensitivity(df):
    """How much the speeding decision depends on which speed limit field is trusted."""
    posted = speed_assessment(df.assign(legal_speed_limit_mph=df["posted_speed_limit_mph"]))
    segment = speed_assessment(df.assign(legal_speed_limit_mph=df["speed_limit_mph"]))
    both = posted["speed_measured"] & segment["speed_measured"]
    agree = (posted["measured_speeding"] == segment["measured_speeding"]) & both
    recorded_speeding = df["violation_type"].eq("Speeding")
    return {
        "captures_with_both_limits": int(both.sum()),
        "limit_fields_disagree_pct": round(float(df["speed_limit_conflict_flag"].mean() * 100), 2),
        "speeding_decision_agreement_pct": round(float(agree.sum() / both.sum() * 100), 2),
        "speeding_under_posted_limit": int(posted["measured_speeding"].sum()),
        "speeding_under_segment_limit": int(segment["measured_speeding"].sum()),
        "recorded_speeding_supported_by_posted_limit_pct": round(float(
            posted.loc[recorded_speeding & posted["speed_measured"], "measured_speeding"].mean() * 100), 2),
        "recorded_speeding_supported_by_segment_limit_pct": round(float(
            segment.loc[recorded_speeding & segment["speed_measured"], "measured_speeding"].mean() * 100), 2),
        "no_violation_records_measured_speeding_pct": round(float(
            posted.loc[df["violation_type"].eq("No Violation") & posted["speed_measured"], "measured_speeding"].mean() * 100), 2),
    }


def label_integrity(df):
    """Scale of the internal contradictions in the recorded enforcement labels."""
    flags = {
        "Violation flag contradicts violation type": "violation_label_conflict_flag",
        "Citation issued on a No Violation capture": "citation_without_violation_flag",
        "Citation amount with no citation": "orphan_citation_amount_flag",
        "Citation with no amount": "citation_amount_missing_flag",
        "Appeal outcome with no appeal filed": "appeal_outcome_orphan_flag",
        "Appeal filed with no citation": "appeal_without_citation_flag",
        "Toll evasion with no toll gantry": "toll_evasion_without_gantry_flag",
        "Recorded speeding contradicted by measured speed": "recorded_speeding_unsupported_flag",
        "Segment and posted speed limits disagree": "speed_limit_conflict_flag",
        "Recorded weekday contradicts date": "weekday_conflict_flag",
        "Reviewer assigned without review flag": "review_assignment_conflict_flag",
        "Record identifier collision repaired": "record_id_repaired",
    }
    rows = []
    for label, column in flags.items():
        count = int(df[column].astype(bool).sum())
        rows.append({"issue": label, "records": count, "share_pct": round(count / len(df) * 100, 2)})
    return pd.DataFrame(rows).sort_values("records", ascending=False).reset_index(drop=True)


def citation_and_appeals(df):
    """Citation coverage, appeal rates and appeal outcomes by evidence quality."""
    cited = df["citation_issued_flag"].fillna(False).astype(bool)
    filed = df["appeal_filed_flag"].fillna(False).astype(bool) & cited
    decided = df["appeal_outcome"].isin(["Upheld", "Overturned"]) & filed
    overturned = df["appeal_outcome"].eq("Overturned") & decided
    by_evidence = pd.DataFrame(
        {
            "evidence_gates_passed": df.loc[decided, "evidence_gates_passed"],
            "overturned": overturned[decided],
        }
    ).groupby("evidence_gates_passed")["overturned"].agg(["sum", "count"]).rename(
        columns={"sum": "overturned", "count": "decided_appeals"}
    ).reset_index()
    return {
        "citations_recorded": int(cited.sum()),
        "citations_with_supported_violation": int((cited & df["rules_violation_type"].ne("No Violation")).sum()),
        "citations_without_supported_violation": int((cited & df["rules_violation_type"].eq("No Violation")).sum()),
        "appeals_filed_on_citations": int(filed.sum()),
        "appeal_rate_pct": round(float(filed.sum() / cited.sum() * 100), 2) if cited.sum() else np.nan,
        "appeals_decided": int(decided.sum()),
        "appeals_overturned": int(overturned.sum()),
        "overturn_rate_pct": round(float(overturned.sum() / decided.sum() * 100), 2) if decided.sum() else np.nan,
        "overturn_by_evidence_gate": by_evidence.to_dict(orient="records"),
    }


def workload_and_value(df):
    """Review workload and fine value at stake under the current policy."""
    review = df["decision"].eq(cfg.DECISION_MANUAL_REVIEW)
    auto = df["decision"].eq(cfg.DECISION_AUTO_ISSUE)
    missed = df["reconciliation"].eq("Supported but not recorded")
    unsupported = df["reconciliation"].eq("Recorded but unsupported")
    review_hours = review.sum() * cfg.REVIEW_MINUTES_PER_CASE / 60
    return {
        "auto_issue_cases": int(auto.sum()),
        "auto_issue_value_usd": round(float(df.loc[auto, "rules_fine_usd"].sum()), 2),
        "manual_review_cases": int(review.sum()),
        "manual_review_value_usd": round(float(df.loc[review, "rules_fine_usd"].sum()), 2),
        "manual_review_hours": round(float(review_hours), 1),
        "manual_review_shifts": round(float(review_hours / cfg.REVIEWER_HOURS_PER_SHIFT), 1),
        "review_hours_per_10k_captures": round(float(review_hours / len(df) * 10000), 1),
        "supported_not_recorded_cases": int(missed.sum()),
        "supported_not_recorded_value_usd": round(float(df.loc[missed, "rules_fine_usd"].sum()), 2),
        "recorded_but_unsupported_cases": int(unsupported.sum()),
        "priority_alerts": int(df["decision"].eq(cfg.DECISION_PRIORITY_ALERT).sum()),
    }
