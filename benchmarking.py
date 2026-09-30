"""End to end pipeline benchmarking: processing speed and accuracy.

Speed is measured from processing latency per capture (camera frame to
decision). Accuracy is measured with the proxies available in production logs:
false positive rate from reviewer labels, evidence gate pass rate, and the
agreement between the rules layer and the recorded violation.
"""

import numpy as np
import pandas as pd

from . import config as cfg
from .stats_utils import bootstrap_quantile_ci, kruskal_by_group


def latency_profile(df, group="model_name"):
    """Latency percentiles, SLA compliance and implied throughput per group."""
    rows = []
    for key, part in df.groupby(group):
        latency = part["processing_latency_ms"].dropna()
        if latency.empty:
            continue
        p95_low, p95_high = bootstrap_quantile_ci(latency, 0.95, iterations=500)
        rows.append(
            {
                group: key,
                "captures": int(len(latency)),
                "p50_ms": round(float(latency.quantile(0.50)), 1),
                "p90_ms": round(float(latency.quantile(0.90)), 1),
                "p95_ms": round(float(latency.quantile(0.95)), 1),
                "p95_ci_low_ms": round(p95_low, 1),
                "p95_ci_high_ms": round(p95_high, 1),
                "p99_ms": round(float(latency.quantile(0.99)), 1),
                "mean_ms": round(float(latency.mean()), 1),
                "sla_compliance_pct": round(float((latency <= cfg.LATENCY_SLA_MS).mean() * 100), 2),
                "throughput_fps_per_stream": round(1000.0 / float(latency.mean()), 2),
            }
        )
    return pd.DataFrame(rows)


def accuracy_profile(df, group="model_name"):
    """Accuracy proxies per group."""
    work = df.assign(
        fp=df["false_positive_flag"].fillna(False).astype(bool),
        agree=df["reconciliation"].isin(["Agree no violation", "Agree same violation"]),
        auto=df["decision"].eq(cfg.DECISION_AUTO_ISSUE),
        review=df["decision"].eq(cfg.DECISION_MANUAL_REVIEW),
    )
    out = work.groupby(group).agg(
        captures=("capture_id", "count"),
        false_positive_pct=("fp", "mean"),
        evidence_gate_pass_pct=("evidence_gates_passed", "mean"),
        rules_label_agreement_pct=("agree", "mean"),
        auto_issue_pct=("auto", "mean"),
        manual_review_pct=("review", "mean"),
    )
    for column in out.columns.drop("captures"):
        out[column] = (out[column] * 100).round(2)
    return out.reset_index()


def benchmark_scorecard(df):
    """Combined speed and accuracy scorecard per model with a composite rank."""
    latency = latency_profile(df).set_index("model_name")
    accuracy = accuracy_profile(df).set_index("model_name").drop(columns="captures")
    card = latency.join(accuracy)
    # Rank: lower p95 latency and false positive rate are better, higher gate pass is better
    card["rank_latency"] = card["p95_ms"].rank(method="min")
    card["rank_false_positive"] = card["false_positive_pct"].rank(method="min")
    card["rank_evidence"] = card["evidence_gate_pass_pct"].rank(method="min", ascending=False)
    card["composite_rank"] = (
        card[["rank_latency", "rank_false_positive", "rank_evidence"]].mean(axis=1).rank(method="min")
    )
    return card.reset_index().sort_values("composite_rank")


def latency_drivers(df):
    """Test whether latency depends on model, conditions or vehicle type."""
    rows = []
    for factor in ["model_name", "lighting_condition", "weather_at_capture", "vehicle_type", "road_type", "time_of_day"]:
        subset = df[df[factor].ne("Unknown")]
        rows.append({"factor": factor, **kruskal_by_group(subset, factor, "processing_latency_ms")})
    out = pd.DataFrame(rows)
    out["significant_at_5pct"] = out["p_value"] < 0.05
    return out


def overall_benchmark(df):
    """Headline end to end benchmark figures."""
    latency = df["processing_latency_ms"].dropna()
    fp = df["false_positive_flag"].fillna(False).astype(bool)
    agree = df["reconciliation"].isin(["Agree no violation", "Agree same violation"])
    p95_low, p95_high = bootstrap_quantile_ci(latency, 0.95, iterations=500)
    return {
        "captures": int(len(df)),
        "latency_p50_ms": round(float(latency.quantile(0.5)), 1),
        "latency_p95_ms": round(float(latency.quantile(0.95)), 1),
        "latency_p95_ci_ms": [round(p95_low, 1), round(p95_high, 1)],
        "latency_p99_ms": round(float(latency.quantile(0.99)), 1),
        "latency_sla_ms": cfg.LATENCY_SLA_MS,
        "latency_sla_compliance_pct": round(float((latency <= cfg.LATENCY_SLA_MS).mean() * 100), 2),
        "false_positive_rate_pct": round(float(fp.mean() * 100), 2),
        "evidence_gate_pass_pct": round(float(df["evidence_gates_passed"].mean() * 100), 2),
        "rules_label_agreement_pct": round(float(agree.mean() * 100), 2),
        "throughput_fps_per_stream": round(1000.0 / float(latency.mean()), 2),
    }
